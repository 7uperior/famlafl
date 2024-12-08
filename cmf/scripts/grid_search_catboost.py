
from typing import Union

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

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
    bid_amount = lobs["bids[0].amount"]
    ask_amount = lobs["asks[0].amount"]
    return (bid_amount - ask_amount) / (bid_amount + ask_amount + 1e-8)

def vwap(books_df: pd.DataFrame, lvl_count: int) -> pd.Series:
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
    lobs["mid_price"] = (lobs["asks[0].price"] + lobs["bids[0].price"]) / 2

    btcusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "BTCUSDT"]["mid_price"]
    ethusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "ETHUSDT"]["mid_price"]

    main_btcusdt_dev = (
        lobs["mid_price"] / (btcusdt_mid_price.asof(lobs.index) + 1e-6)
    ).asof(target_data.index) * target_data.side

    main_ethusdt_dev = (
        lobs["mid_price"] / (ethusdt_mid_price.asof(lobs.index) + 1e-6)
    ).asof(target_data.index) * target_data.side

    distance_to_mid_price = (
        target_data.price / (lobs["mid_price"].asof(target_data.index) + 1e-6) - 1
    ) * target_data.side

    imbalance_series = calc_imbalance(lobs).asof(target_data.index) * target_data.side

    depth = 5
    vwap_series = vwap(lobs, depth).asof(target_data.index) * target_data.side

    solusdt_agg_trades = agg_trades[agg_trades["instrument"] == "SOLUSDT"]
    solusdt_agg_trades.index = pd.to_datetime(solusdt_agg_trades.index)
    trades_ratio_series = (
        trades_balance(solusdt_agg_trades, 10 * SECOND).asof(target_data.index)
        * target_data.side
    )
    sol_mid_price = lobs["mid_price"].asof(target_data.index)

    return pd.concat(
        [
            target_data.side,
            vwap_series.rename("vwap"),
            trades_ratio_series.rename("trades_ratio"),
            distance_to_mid_price.rename("distance_to_mid_price"),
            main_ethusdt_dev.rename("main_ethusdt_dev"),
            main_btcusdt_dev.rename("main_btcusdt_dev"),
            imbalance_series.rename("imbalance"),
            sol_mid_price.rename("sol_mid_price"),
        ],
        axis=1,
    )

# Load data
path_to_data_folder = "./cmf/data/train/"
agg_trades = pd.read_parquet(f'{path_to_data_folder}agg_trades.parquet')
orderbook_embedding = pd.read_parquet(f'{path_to_data_folder}orderbook_embedding.parquet')
orderbook_solusdt = pd.read_parquet(f'{path_to_data_folder}orderbook_solusdt.parquet')
target_data_solusdt = pd.read_parquet(f'{path_to_data_folder}target_data_solusdt.parquet')
target_solusdt = pd.read_parquet(f'{path_to_data_folder}target_solusdt.parquet')

for df in [agg_trades, orderbook_embedding, orderbook_solusdt, target_data_solusdt, target_solusdt]:
    df.index = pd.to_datetime(df.index)

def adjust_weight(weight: float) -> float:
    if weight > 10:
        return weight / 100
    elif weight > 1:
        return weight / 10
    return weight

target_solusdt_preprocessed = target_solusdt.copy()
target_solusdt_preprocessed['weight'] = target_solusdt_preprocessed['weight'].map(adjust_weight)

X = calc_features(orderbook_solusdt, agg_trades, orderbook_embedding, target_data_solusdt)
y = target_solusdt_preprocessed["target"]
weights = target_solusdt_preprocessed["weight"]

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

param_grid = {
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.02, 0.03],
    "l2_leaf_reg": [10, 50, 100],
    "iterations": [1200, 6000, 8000],
}

model = CatBoostClassifier(
    task_type="CPU",
    eval_metric="Logloss",
    use_best_model=True,
    early_stopping_rounds=50
)

cv = TimeSeriesSplit(n_splits=3)

# Grid search with time-series split
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_log_loss",
    cv=cv,  # Use time-series split
    verbose=3
)

grid_search.fit(X_train, y_train, sample_weight=weights_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Log-Loss Score:", -grid_search.best_score_)
