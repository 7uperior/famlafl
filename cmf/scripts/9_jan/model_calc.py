
from datetime import datetime, timedelta, timezone
from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

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


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR) indicator for market candlestick data.

    Parameters:
    - data: A pd.DataFrame with columns ['high', 'low', 'close']
    - period: The period for ATR calculation (default is 14)

    Returns:
    - pd.Series with the calculated ATR
    """
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()

    return atr


def calc_features(
    lobs: pd.DataFrame,
    agg_trades: pd.DataFrame,
    lobs_embedding: pd.DataFrame,
    target_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate features and align them with the target data."""
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


    # Calculate ATR
    lobs["high"] = lobs[["asks[0].price", "bids[0].price"]].max(axis=1)
    lobs["low"] = lobs[["asks[0].price", "bids[0].price"]].min(axis=1)
    lobs["close"] = lobs["mid_price"]
    atr_series = calculate_atr(lobs, period=14).asof(target_data.index)

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
            atr_series.rename("atr")
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

# Ensure correct time index
for df in [agg_trades, orderbook_embedding, orderbook_solusdt, target_data_solusdt, target_solusdt]:
    df.index = pd.to_datetime(df.index)

# # Preprocess weights
# def adjust_weight(weight: float) -> float:
#     if weight > 10:
#         return weight / 100
#     elif weight > 1:
#         return weight / 10
#     return weight

# Create a preprocessed version of target_solusdt
target_solusdt_preprocessed = target_solusdt.copy()
# target_solusdt_preprocessed['weight'] = target_solusdt_preprocessed['weight'].map(adjust_weight)

# Generate features and target
X = calc_features(orderbook_solusdt, agg_trades, orderbook_embedding, target_data_solusdt)
y = target_solusdt_preprocessed["target"]
weights = target_solusdt_preprocessed["weight"]
# TimeSeriesSplit configuration
tscv = TimeSeriesSplit(n_splits=5, gap = 10)

params_research = {
    "use_best_model": True,
    "eval_metric": "Logloss",
    "iterations": 6000,
    "thread_count": 16,
    "loss_function": "Logloss",
    "l2_leaf_reg": 50,
    "task_type": "CPU",
    "depth": 8,
    "early_stopping_rounds": 50,
    "learning_rate": 0.01,
}

# params_boyko_model = {
#     "use_best_model": True,
#     "eval_metric": "Logloss",
#     "verbose": 50,
#     "iterations": 1200,
#     "thread_count": 13,
#     "loss_function": "Logloss",
#     "l2_leaf_reg": 50,
#     "task_type": "CPU",
#     "depth": 5,
#     "learning_rate": 0.01
# }




log_loss_values = []
weighted_log_loss_values = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

    model = CatBoostClassifier(**params_research)
    model.fit(
        X_train,
        y_train,
        sample_weight=weights_train,
        eval_set=(X_test, y_test),
        verbose=25
    )

    # Evaluate model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    log_loss_value = log_loss(y_test, y_pred_proba)
    log_loss_value_weighted = log_loss(y_test, y_pred_proba, sample_weight=weights_test)

    print(f"Fold {fold + 1} Log Loss: {log_loss_value}")
    print(f"Fold {fold + 1} Weighted Log Loss: {log_loss_value_weighted}")

    # Collect metrics
    log_loss_values.append(log_loss_value)
    weighted_log_loss_values.append(log_loss_value_weighted)



final_log_loss = round(sum(log_loss_values) / len(log_loss_values),6)
final_weighted_log_loss = round(sum(weighted_log_loss_values) / len(weighted_log_loss_values),6)

print(f"Final Log Loss (Average Across Folds): {final_log_loss}")
print(f"Final Weighted Log Loss (Average Across Folds): {final_weighted_log_loss}")


#Saving file
features_count = X.shape[1]

# Generate the model name with the feature count and timestamp

gmt_plus_3 = timezone(timedelta(hours=3))
now_gmt_plus_3 = datetime.now(gmt_plus_3)
timestamp = now_gmt_plus_3.strftime("%Y-%b-%d_%H-%M")
model_name = f"model_count{features_count}_loglloss{final_log_loss}_{timestamp}.cbm"

path_to_model_folder = "./cmf/models/"
model.save_model(f'{path_to_model_folder}{model_name}')
print(f"Model saved as: {model_name}")

#0.679794-0.6788005280989209
