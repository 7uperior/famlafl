from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

NANOSECOND = 1
MICROSECOND = 1000
MILLISECOND = 1000000
SECOND = 1000000000

def trades_balance(trades_df: pd.DataFrame, window: Union[str, int]) -> pd.Series:
    sells = trades_df["ask_amount"].rolling(window=window, min_periods=1).sum()
    buys = trades_df["bid_amount"].rolling(window=window, min_periods=1).sum()
    return (sells - buys) / (sells + buys + 1e-8)

def calc_imbalance(lobs):
    """
    Computes the order book imbalance.

    Parameters:
    - lob: pd.DataFrame row containing LOB data.

    Returns:
    - imbalance_value: float
    """
    bid_amount = lobs["bids[0].amount"]
    ask_amount = lobs["asks[0].amount"]
    imbalance_value = (bid_amount - ask_amount) / (bid_amount + ask_amount)
    return imbalance_value

def vwap(books_df: pd.DataFrame, lvl_count: int) -> pd.Series:
    """Volume-weighted average price."""
    ask_weighted_price = sum(books_df[f"asks[{i}].price"] * books_df[f"asks[{i}].amount"] for i in range(lvl_count))
    ask_volume = sum(books_df[f"asks[{i}].amount"] for i in range(lvl_count))

    bid_weighted_price = sum(books_df[f"bids[{i}].price"] * books_df[f"bids[{i}].amount"] for i in range(lvl_count))
    bid_volume = sum(books_df[f"bids[{i}].amount"] for i in range(lvl_count))

    total_weighted_price = ask_weighted_price + bid_weighted_price
    total_volume = ask_volume + bid_volume

    vwap = total_weighted_price / total_volume

    return vwap / books_df["mid_price"]

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()

    return atr

def adjust_weight(weight: float) -> float:
    if weight > 10:
        return weight / 100
    elif weight > 1:
        return weight / 10
    return weight

def calculate_open_interest(data: pd.DataFrame) -> pd.Series:
    open_interest = data['bids[0].amount'] + data['asks[0].amount']
    return open_interest

def calculate_volume(trades_df: pd.DataFrame, window: Union[str, int]) -> pd.Series:
    volume = trades_df['ask_amount'].rolling(window=window, min_periods=1).sum() + trades_df['bid_amount'].rolling(window=window, min_periods=1).sum()
    return volume

def calculate_large_density(lobs: pd.DataFrame, volume_series: pd.Series) -> pd.Series:
    density = lobs['bids[0].amount'] + lobs['asks[0].amount']
    large_density = density[density > volume_series]
    return large_density

def calculate_atr_volume(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data['high_volume'] - data['low_volume']
    high_close = np.abs(data['high_volume'] - data['close_volume'].shift())
    low_close = np.abs(data['low_volume'] - data['close_volume'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()

    return atr

class Predictor:
    def __init__(self, full_model_path: Union[str, list[str]]):
        self.model = CatBoostClassifier()
        self.model.load_model(full_model_path, format="cbm")

    @staticmethod
    def model_name() -> Union[str, list[str]]:
        return "20241206-192402_model_with_tscv_weights_no_preprocessed_20241206-191738.cbm"

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Method is called once every time new submission received
            Params:
                Your features returned from `calc_features` method

            Returns: pd.Series[float]
                Array of predicted returns (price_{n + 1} / price_{n} - 1).
                One value must be generated for every bbo dataframe timestamp
                so that len(Series) == len(bbos)
        """

        predict = pd.Series(self.model.predict_proba(features)[:, 1])

        return predict

    def calc_features(
        lobs: pd.DataFrame,
        agg_trades: pd.DataFrame,
        lobs_embedding: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate features and align them with the target data."""
        # Add time-based features
        timestamp = pd.to_datetime(target_data.index)
        day_of_week = pd.Series(timestamp.dayofweek, index=timestamp)
        hour_of_day = pd.Series(timestamp.hour, index=timestamp)

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

        # Calculate Open Interest
        open_interest_series = calculate_open_interest(lobs).asof(target_data.index)

        # Calculate Volume
        volume_series = calculate_volume(solusdt_agg_trades, window='5min').asof(target_data.index)

        # Calculate Large Density
        large_density_series = calculate_large_density(lobs, volume_series).asof(target_data.index)

        # Calculate ATR for volume
        lobs["high_volume"] = lobs[["asks[0].amount", "bids[0].amount"]].max(axis=1)
        lobs["low_volume"] = lobs[["asks[0].amount", "bids[0].amount"]].min(axis=1)
        lobs["close_volume"] = (lobs["asks[0].amount"] + lobs["bids[0].amount"]) / 2
        atr_volume_series = calculate_atr_volume(lobs, period=14).asof(target_data.index)

        # Name all series before concat
        side_series = pd.Series(target_data.side, name="side")
        vwap_series.name = "vwap"
        trades_ratio_series.name = "trades_ratio"
        distance_to_mid_price.name = "distance_to_mid_price"
        main_ethusdt_dev.name = "main_ethusdt_dev"
        main_btcusdt_dev.name = "main_btcusdt_dev"
        imbalance_series.name = "imbalance"
        sol_mid_price.name = "sol_mid_price"
        atr_series.name = "atr"
        atr_volume_series.name = "atr_volume"
        open_interest_series.name = "open_interest"
        volume_series.name = "volume"
        large_density_series.name = "large_density"
        day_of_week.name = "day_of_week"
        hour_of_day.name = "hour_of_day"

        return pd.concat(
            [
                target_data_preprocessed.side,
                vwap_series,
                trades_ratio_series,
                distance_to_mid_price,
                main_ethusdt_dev,
                main_btcusdt_dev,
                imbalance_series,
                sol_mid_price,
                atr_series,
                open_interest_series,
                volume_series,
                large_density_series,
            ],
            axis=1,
        )
