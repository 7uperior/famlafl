import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

from typing import Union
from numba import njit

NANOSECOND = 1
MICROSECOND = 1000
MILLISECOND = 1000000
SECOND = 1000000000

from typing import Union


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

    vwap = total_weighted_price / total_volume

    return vwap / books_df["mid_price"]


class Predictor:
    def __init__(self, full_model_path: Union[str, list[str]]):
        self.model = CatBoostClassifier()
        self.model.load_model(full_model_path, format="cbm")

    @staticmethod
    def model_name() -> Union[str, list[str]]:
        return "20241120-134956_model_20241120-033256_model_baseline.cbm"

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
        """
        Calculates features using provided functions and aligns them with target_data.

        Parameters:
        - lobs: pd.DataFrame of limit orderbooks.
        - agg_trades: pd.DataFrame of aggregated trades.
        - lobs_embedding: pd.DataFrame of embedding over limit orderbooks.
        - target_data: pd.DataFrame with target timestamps.

        Returns:
        - features: pd.DataFrame with features aligned to target_data.index.
        """
        lobs["mid_price"] = (lobs["asks[0].price"] + lobs["bids[0].price"]) / 2

        btcusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "BTCUSDT"][
            "mid_price"
        ]
        ethusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "ETHUSDT"][
            "mid_price"
        ]

        main_btcusdt_dev = (
            lobs["mid_price"] / (btcusdt_mid_price.asof(lobs.index) + 1e-6)
        ).asof(target_data.index) * target_data.side
        main_btcusdt_dev.name = "main_btcusdt_dev"

        main_ethusdt_dev = (
            lobs["mid_price"] / (ethusdt_mid_price.asof(lobs.index) + 1e-6)
        ).asof(target_data.index) * target_data.side
        main_ethusdt_dev.name = "main_ethusdt_dev"

        distance_to_mid_price = (
            target_data.price / (lobs["mid_price"].asof(target_data.index) + 1e-6) - 1
        ) * target_data.side
        distance_to_mid_price.name = "distance_to_mid_price"

        imbalance_series = (
            calc_imbalance(lobs).asof(target_data.index) * target_data.side
        )
        imbalance_series.name = "imbalance"

        depth = 5
        vwap_series = vwap(lobs, depth).asof(target_data.index) * target_data.side
        vwap_series.name = "vwap"

        solusdt_agg_trades = agg_trades[agg_trades["instrument"] == "SOLUSDT"]
        solusdt_agg_trades.index = pd.to_datetime(solusdt_agg_trades.index)
        trades_ratio_series = (
            trades_balance(solusdt_agg_trades, 10 * SECOND).asof(target_data.index)
            * target_data.side
        )
        trades_ratio_series.name = "trades_ratio"

        return pd.concat(
            [
                target_data.side,
                vwap_series,
                trades_ratio_series,
                distance_to_mid_price,
                main_ethusdt_dev,
                main_btcusdt_dev,
            ],
            axis=1,
        )