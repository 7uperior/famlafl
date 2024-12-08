from pathlib import Path
from typing import Union

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Constants for time conversions
NANOSECOND = 1
MICROSECOND = 1000
MILLISECOND = 1000000
SECOND = 1000000000

# Feature engineering functions
def trades_balance(trades_df: pd.DataFrame, window: Union[str, int]) -> pd.Series:
    sells = trades_df["ask_amount"].rolling(window=window, min_periods=1).sum()
    buys = trades_df["bid_amount"].rolling(window=window, min_periods=1).sum()
    return (sells - buys) / (sells + buys + 1e-8)

def calc_imbalance(lobs: pd.DataFrame) -> pd.Series:
    """Calculate order book imbalance for the first level."""
    bid_amount = lobs["bids[0].amount"]
    ask_amount = lobs["asks[0].amount"]
    return (bid_amount - ask_amount) / (bid_amount + ask_amount + 1e-8)

def vwap(books_df: pd.DataFrame, lvl_count: int) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)."""
    ask_weighted_price = sum(
        books_df[f"asks[{i}].price"] * books_df[f"asks[{i}].amount"]
        for i in range(lvl_count)
    )
    ask_volume = sum(books_df[f"asks[{i}].amount"] for i in range(lvl_count))

    bid_weighted_price = sum(
        books_df[f"bids[{i}].price"] * books_df[f"bids[{i}].amount"]
        for i in range(lvl_count)
    )
    bid_volume = sum(books_df[f"bids[{i}].amount"] for i in range(lvl_count))

    total_weighted_price = ask_weighted_price + bid_weighted_price
    total_volume = ask_volume + bid_volume

    return total_weighted_price / (total_volume + 1e-8)

def calc_features(
    lobs: pd.DataFrame,
    agg_trades: pd.DataFrame,
    lobs_embedding: pd.DataFrame,
    target_data: pd.DataFrame,
) -> pd.DataFrame:
    lobs = lobs.copy()
    lobs["mid_price"] = (lobs["asks[0].price"] + lobs["bids[0].price"]) / 2

    # Filter and sort lobs_embedding
    btcusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "BTCUSDT"][["mid_price"]].sort_index()
    ethusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "ETHUSDT"][["mid_price"]].sort_index()

    # Ensure lobs is sorted
    lobs = lobs.sort_index()

    # Merge lobs with btcusdt_mid_price
    lobs = pd.merge_asof(
        lobs.reset_index(),
        btcusdt_mid_price.reset_index(),
        on='timestamp_ns',
        direction='backward',
        suffixes=('', '_btcusdt')
    )
    lobs = pd.merge_asof(
        lobs,
        ethusdt_mid_price.reset_index(),
        on='timestamp_ns',
        direction='backward',
        suffixes=('', '_ethusdt')
    )

    # Set index back to timestamp_ns
    lobs.set_index('timestamp_ns', inplace=True)

    # Calculate deviations
    lobs['main_btcusdt_dev'] = (
        lobs["mid_price"] / (lobs["mid_price_btcusdt"] + 1e-6)
    ) * target_data['side'].reindex(lobs.index, method='ffill')

    lobs['main_ethusdt_dev'] = (
        lobs["mid_price"] / (lobs["mid_price_ethusdt"] + 1e-6)
    ) * target_data['side'].reindex(lobs.index, method='ffill')

    # Continue with other feature calculations...

    # Return the features aligned with target_data
    features = lobs[['main_btcusdt_dev', 'main_ethusdt_dev']]
    features = features.reindex(target_data.index, method='ffill')

    return features

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
DATA_DIR = BASE_DIR / "cmf/data/train"

datasets = {
    "agg_trades": pd.read_parquet(DATA_DIR / "agg_trades.parquet"),
    "orderbook_embedding": pd.read_parquet(DATA_DIR / "orderbook_embedding.parquet"),
    "orderbook_solusdt": pd.read_parquet(DATA_DIR / "orderbook_solusdt.parquet"),
    "target_data_solusdt": pd.read_parquet(DATA_DIR / "target_data_solusdt.parquet"),
    "target_solusdt": pd.read_parquet(DATA_DIR / "target_solusdt.parquet"),
}

agg_trades = datasets["agg_trades"]
orderbook_embedding = datasets["orderbook_embedding"]
orderbook_solusdt = datasets["orderbook_solusdt"]
target_data_solusdt = datasets["target_data_solusdt"]
target_solusdt = datasets["target_solusdt"]

# Ensure correct time index
for df in [agg_trades, orderbook_embedding, orderbook_solusdt, target_data_solusdt, target_solusdt]:
    df.index = pd.to_datetime(df.index)

# Generate features and target
X = calc_features(orderbook_solusdt, agg_trades, orderbook_embedding, target_data_solusdt)
y = target_solusdt["target"]

# Extract weights
weights = target_solusdt["weight"]

# Handle any NaN values (optional but recommended)
X = X.fillna(method='ffill').fillna(method='bfill')
y = y.loc[X.index]
weights = weights.loc[X.index]

# Train-test split (include weights)
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.2, shuffle=False
)

# Train model with sample weights
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, verbose=0)
model.fit(X_train, y_train, sample_weight=weights_train)

# Predictions and log loss with sample weights
y_pred_proba = model.predict_proba(X_test)[:, 1]
log_loss_value = log_loss(y_test, y_pred_proba, sample_weight=weights_test)

# Output log loss
print(f"Log Loss: {log_loss_value}")

model.save_model("model_M2.cbm")
