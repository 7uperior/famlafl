from datetime import datetime, timedelta, timezone
from typing import Union
import polars as pl

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

import time

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

def calc_imbalance(lobs: pd.DataFrame, lvl_count: int = 20) -> pd.Series:
    """
    Calculate order book imbalance across multiple levels for orderbook_solusdt
    
    Parameters:
    - lobs: DataFrame orderbook_solusdt(lobs) containing order book data (with asks[0-19] and bids[0-19])
    - lvl_count: Number of levels to include in imbalance calculation (default: 20 for all levels)
    
    Returns:
    - Series containing the imbalance calculation
    """
    if lvl_count > 20:
        lvl_count = 20  # Safeguard against requesting more levels than available
        
    bid_amount = sum(lobs[f"bids[{i}].amount"] for i in range(lvl_count))
    ask_amount = sum(lobs[f"asks[{i}].amount"] for i in range(lvl_count))
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

def calculate_open_interest(data: pd.DataFrame) -> pd.Series:
    open_interest = data['bids[0].amount'] + data['asks[0].amount']
    return open_interest

def calculate_volume(trades_df: pd.DataFrame, window: Union[str, int]) -> pd.Series:
    volume = trades_df['ask_amount'].rolling(window=window, min_periods=1).sum() + trades_df['bid_amount'].rolling(window=window, min_periods=1).sum()
    return volume

#12
def calculate_large_density(lobs: pd.DataFrame, volume_series: pd.Series) -> pd.Series:

    density = lobs['bids[0].amount'] + lobs['asks[0].amount']
    density, volume = density.align(volume_series, join='outer', fill_value=0)
    large_density = density[density > volume]
    return large_density

#13  ATR for volume using volume series

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

    imbalance_series_lvl_20 = calc_imbalance(lobs).asof(target_data.index) * target_data.side

    imbalance_series_lvl_10 = calc_imbalance(lobs, lvl_count=10).asof(target_data.index) * target_data.side

    imbalance_series_lvl_5 = calc_imbalance(lobs, lvl_count=5).asof(target_data.index) * target_data.side


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

    # Calculate Open Interest
    open_interest_series = calculate_open_interest(lobs).asof(target_data.index)


    # Calculate Volume
    volume_series = calculate_volume(solusdt_agg_trades, window='5min').asof(target_data.index)

    # Calculate Large Density
    large_density_series = calculate_large_density(lobs, volume_series).asof(target_data.index)

    # Calculate ATR for volume
    #13
    lobs["high_volume"] = lobs[["asks[0].amount", "bids[0].amount"]].max(axis=1)
    lobs["low_volume"] = lobs[["asks[0].amount", "bids[0].amount"]].min(axis=1)
    lobs["close_volume"] = (lobs["asks[0].amount"] + lobs["bids[0].amount"]) / 2
    atr_volume_series = calculate_atr(lobs, period=14).asof(target_data.index)

    
    return pd.concat(
        [
            target_data.side,
            vwap_series.rename("vwap"),
            trades_ratio_series.rename("trades_ratio"),
            distance_to_mid_price.rename("distance_to_mid_price"),
            main_ethusdt_dev.rename("main_ethusdt_dev"),
            main_btcusdt_dev.rename("main_btcusdt_dev"),
            imbalance_series_lvl_20.rename("imbalance_lvl_20"),
            imbalance_series_lvl_10.rename("imbalance_lvl_10"),
            imbalance_series_lvl_5.rename("imbalance_lvl_5"),
            sol_mid_price.rename("sol_mid_price"),
            atr_series.rename("atr"),
            open_interest_series.rename("open_interest"),
            volume_series.rename("volume"),
            large_density_series.rename("large_density"),
            atr_volume_series.rename("atr_volume")
        ],
        axis=1,
    )

def print_time_taken(start_time, section_name):
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for {section_name}: {time_taken:.2f} seconds")

# Start timing for data loading
print("\n=== Starting Data Loading ===")
data_load_start = time.time()

# Load data
path_to_data_folder = "./cmf/data/train/"
agg_trades = pl.read_parquet(f'{path_to_data_folder}agg_trades.parquet').with_columns(
    pl.col('timestamp_ns').cast(pl.Datetime)
).to_pandas().set_index('timestamp_ns')
print("agg_trades")
print(agg_trades.head(3))
print()

orderbook_embedding = pl.read_parquet(f'{path_to_data_folder}orderbook_embedding.parquet').with_columns(
    pl.col('timestamp_ns').cast(pl.Datetime)
).to_pandas().set_index('timestamp_ns')
print("orderbook_embedding")
print(orderbook_embedding.head(3))
print()

orderbook_solusdt = pl.read_parquet(f'{path_to_data_folder}orderbook_solusdt.parquet').with_columns(
    pl.col('timestamp_ns').cast(pl.Datetime)
).to_pandas().set_index('timestamp_ns')
print("orderbook_solusdt")
print(orderbook_solusdt.head(3))
print()

target_data_solusdt = pl.read_parquet(f'{path_to_data_folder}target_data_solusdt.parquet').with_columns(
    pl.col('timestamp_ns').cast(pl.Datetime)
).to_pandas().set_index('timestamp_ns')
print("target_data_solusdt")
print(target_data_solusdt.head(3))
print()

target_solusdt = pl.read_parquet(f'{path_to_data_folder}target_solusdt.parquet').with_columns(
    pl.col('timestamp_ns').cast(pl.Datetime)
).to_pandas().set_index('timestamp_ns')
print("target_solusdt")
print(target_solusdt.head(3))
print()

print_time_taken(data_load_start, "Data Loading")


# Start timing for feature generation
print("\n=== Starting Feature Generation ===")
feature_gen_start = time.time()
target_solusdt_preprocessed = target_solusdt.copy()

# Generate features and target
X = calc_features(orderbook_solusdt, agg_trades, orderbook_embedding, target_data_solusdt)
y = target_solusdt_preprocessed["target"]
weights = target_solusdt_preprocessed["weight"]

print_time_taken(feature_gen_start, "Feature Generation")  # Add this line

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

# Start timing for model training
print("\n=== Starting Model Training ===")
training_start = time.time()

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
        verbose=100
    )

    # Evaluate model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    log_loss_value = log_loss(y_test, y_pred_proba)
    log_loss_value_weighted = log_loss(y_test, y_pred_proba, sample_weight=weights_test)

    print(f"Fold {fold + 1} Log Loss: {round(log_loss_value,6)}")
    print(f"Fold {fold + 1} Weighted Log Loss: {round(log_loss_value_weighted,6)}")

    log_loss_values.append(log_loss_value)
    weighted_log_loss_values.append(log_loss_value_weighted)

print_time_taken(training_start, "Total Model Training")

# Start timing for model saving
print("\n=== Starting Model Saving ===")
saving_start = time.time()


#Saving file
features_count = X.shape[1]

# Generate the model name with the feature count and timestamp
final_log_loss = round(sum(log_loss_values) / len(log_loss_values),6)
final_weighted_log_loss = round(sum(weighted_log_loss_values) / len(weighted_log_loss_values),6)

gmt_plus_3 = timezone(timedelta(hours=3))
now_gmt_plus_3 = datetime.now(gmt_plus_3)
timestamp = now_gmt_plus_3.strftime("%Y-%b-%d_%H-%M")
model_name = f"model_count{features_count}_loglloss{final_log_loss}_{timestamp}.cbm"

path_to_model_folder = "./cmf/models/"
model.save_model(f'{path_to_model_folder}{model_name}')
print(f"Model saved as: {model_name}")
print_time_taken(saving_start, "Model Saving")

# Print total execution time
print("\n=== Total Execution Summary ===")
print_time_taken(data_load_start, "Total Script Execution")


print(f"Final Log Loss (Average Across Folds): {final_log_loss}")
print(f"Final Weighted Log Loss (Average Across Folds): {final_weighted_log_loss}")



