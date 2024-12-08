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

def adjust_weight(weight: float) -> float:
    if weight > 10:
        return weight / 100
    elif weight > 1:
        return weight / 10
    return weight

class Predictor:
    def __init__(self, full_model_path: Union[str, list[str]]):
        self.model = CatBoostClassifier()
        self.model.load_model(full_model_path, format="cbm")

    @staticmethod
    def model_name() -> Union[str, list[str]]:
        return "20241208-154128_model_count9_loglloss0.6783156809620212_2024-Dec-08_18-39.cbm"

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
        self,
        lobs: pd.DataFrame | None,
        agg_trades: pd.DataFrame | None,
        lobs_embedding: pd.DataFrame | None,
        target_data: pd.DataFrame | None,
    ) -> pd.DataFrame:
        # Preprocess weights
        target_data_preprocessed = target_data.copy()
        if 'weight' in target_data.columns:
            target_data_preprocessed['weight'] = target_data_preprocessed['weight'].map(adjust_weight)

        lobs["mid_price"] = (lobs["asks[0].price"] + lobs["bids[0].price"]) / 2
        sol_mid_price = lobs["mid_price"].asof(target_data_preprocessed.index)
        sol_mid_price.name = "sol_mid_price"

        btcusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "BTCUSDT"]["mid_price"]
        ethusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "ETHUSDT"]["mid_price"]

        main_btcusdt_dev = (lobs["mid_price"] / (btcusdt_mid_price.asof(lobs.index) + 1e-6)).asof(
            target_data_preprocessed.index
        ) * target_data_preprocessed.side
        main_btcusdt_dev.name = "main_btcusdt_dev"

        main_ethusdt_dev = (lobs["mid_price"] / (ethusdt_mid_price.asof(lobs.index) + 1e-6)).asof(
            target_data_preprocessed.index
        ) * target_data_preprocessed.side
        main_ethusdt_dev.name = "main_ethusdt_dev"

        distance_to_mid_price = (
            target_data_preprocessed.price / (lobs["mid_price"].asof(target_data_preprocessed.index) + 1e-6) - 1
        ) * target_data_preprocessed.side
        distance_to_mid_price.name = "distance_to_mid_price"

        imbalance_series = calc_imbalance(lobs).asof(target_data_preprocessed.index) * target_data_preprocessed.side
        imbalance_series.name = "imbalance"

        depth = 5
        vwap_series = vwap(lobs, depth).asof(target_data_preprocessed.index) * target_data_preprocessed.side
        vwap_series.name = "vwap"

        solusdt_agg_trades = agg_trades[agg_trades["instrument"] == "SOLUSDT"]
        solusdt_agg_trades.index = pd.to_datetime(solusdt_agg_trades.index)
        trades_ratio_series = trades_balance(solusdt_agg_trades, 10 * SECOND).asof(
            target_data_preprocessed.index
        ) * target_data_preprocessed.side
        trades_ratio_series.name = "trades_ratio"

        # Calculate ATR
        lobs["high"] = lobs[["asks[0].price", "bids[0].price"]].max(axis=1)
        lobs["low"] = lobs[["asks[0].price", "bids[0].price"]].min(axis=1)
        lobs["close"] = lobs["mid_price"]
        atr_series = calculate_atr(lobs, period=14).asof(target_data_preprocessed.index)
        atr_series.name = "atr"

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
            ],
            axis=1,
        )
