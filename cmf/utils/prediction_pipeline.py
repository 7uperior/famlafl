import math
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier

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

def slice_by_first(*args):
    return tuple(a[a.index < args[0].index[-1]] for a in args)


def validate(
    predictor,
    book: pd.DataFrame | None,
    agg_trades: pd.DataFrame | None,
    lobs_embedding: pd.DataFrame | None,
    target_data: pd.DataFrame | None,
    checks_cnt: int = 50,
    checks_fraq: float = 0.4,
):
    """
    Checks your features do not look ahead of time
    """
    target_check_size = int(target_data.shape[0] * checks_fraq)
    one_time_check = 2

    # Sample some indexes to check features values (sort to ensure correctness of procedure)
    checks_idxs = np.random.randint(one_time_check, target_check_size, size=checks_cnt)
    checks_idxs[::-1].sort()

    target_data_, book_, agg_trades_, lobs_embedding_ = slice_by_first(
        target_data.iloc[:target_check_size],
        book,
        agg_trades,
        lobs_embedding,
    )
    features_ = predictor.calc_features(
        lobs=book_,
        agg_trades=agg_trades_,
        lobs_embedding=lobs_embedding_,
        target_data=target_data_,
    )
    for idx in checks_idxs:
        if idx >= len(target_data.index) or (
            target_data.index[idx] == target_data.index[idx + 1]
        ):
            continue
        target_data_, book_, agg_trades_, lobs_embedding_ = slice_by_first(
            target_data.iloc[:idx],
            book,
            agg_trades,
            lobs_embedding,
        )
        # Calc features for validation
        features = predictor.calc_features(
            lobs=book_,
            agg_trades=agg_trades_,
            lobs_embedding=lobs_embedding_,
            target_data=target_data_,
        )

        # Get the indices we want to compare
        compare_indices = target_data_.index[-one_time_check:]
        
        # Extract exactly those indices from both feature sets
        features_cur = features.loc[compare_indices]
        features_prev = features_.loc[compare_indices]
        
        if not np.allclose(
            features_cur.values, features_prev.values, equal_nan=True, rtol=0, atol=1e-7
        ):
            validation = features_cur.values != features_prev.values
            where_is_leak = np.argwhere(validation)
            leaked_features_idxs = set(where_is_leak[:, 1])
            raise RuntimeError(
                f"Lookahead validation didn't pass for features with idxs: {leaked_features_idxs}"
            )

        features_ = features


def prediction_simulation(
    user_models_path: Union[str, list[str]],
    predictor_cls,
    book: pd.DataFrame | None,
    agg_trades: pd.DataFrame | None,
    lobs_embedding: pd.DataFrame | None,
    target_data: pd.DataFrame | None,
    target: pd.DataFrame | None,
    validate_fn=validate,
):
    import os

    model_path = predictor_cls.model_name()

    if isinstance(model_path, str):
        full_model_path = f"{user_models_path}/{model_path}"

    elif isinstance(model_path, list):
        full_model_path = [f"{user_models_path}/{mp}" for mp in model_path]
        for mp in full_model_path:
            if not os.path.exists(mp):
                raise Exception(f"model with name {mp} does not exist")

    else:
        raise Exception("passed model name is of wrong type. allowed: str, list[str]")
    predictor = predictor_cls(full_model_path)

    validate_fn(predictor, book, agg_trades, lobs_embedding, target_data)
    features = predictor.calc_features(book, agg_trades, lobs_embedding, target_data)
    y_pred = predictor.predict(features)
    y_true = target["target"].values
    if len(y_pred) != len(target):
        raise Exception(
            f"len of generated prediction array is not equal to target len: pred - {len(y_pred)} target - {len(y_true)}"
        )

    score = log_loss(y_true, y_pred)

    if math.isnan(score):
        raise Exception("score value is nan")

    return score


class Predictor:
    def __init__(self, full_model_path: Union[str, list[str]]):
        self.model = CatBoostClassifier()
        self.model.load_model(full_model_path, format="cbm")

    @staticmethod
    def model_name() -> Union[str, list[str]]:
        MOD_DIR = "./cmf/models/"
        model = f"{MOD_DIR}model_count9_loglloss0.678801_2025-Jan-09_09-15.cbm"
        return model

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Method is called once every time new submission received
            Params:
                Your features returned from `calc_features` method

            Returns: pd.Series[float]
                Array of predicted probabilities for the positive class.
                One value must be generated for every bbo dataframe timestamp
                so that len(Series) == len(bbos)
        """
        # Use the CatBoost model to predict probabilities
        return pd.Series(self.model.predict_proba(features)[:, 1], index=features.index)

    def calc_features(
        self,
        lobs: pd.DataFrame,
        agg_trades: pd.DataFrame,
        lobs_embedding: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate features and align them with the target data."""
        # Create a copy of lobs to avoid SettingWithCopyWarning
        lobs = lobs.copy()
        
        lobs.loc[:, "mid_price"] = (lobs["asks[0].price"] + lobs["bids[0].price"]) / 2

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
        lobs.loc[:, "high"] = lobs[["asks[0].price", "bids[0].price"]].max(axis=1)
        lobs.loc[:, "low"] = lobs[["asks[0].price", "bids[0].price"]].min(axis=1)
        lobs.loc[:, "close"] = lobs["mid_price"]
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


if __name__ == "__main__":
    MD_DIR = "./cmf/data/train/"
    agg_trades = pd.read_parquet(f"{MD_DIR}/agg_trades.parquet")
    orderbook_embedding = pd.read_parquet(f"{MD_DIR}/orderbook_embedding.parquet")
    orderbook_solusdt = pd.read_parquet(f"{MD_DIR}/orderbook_solusdt.parquet")
    target_data_solusdt = pd.read_parquet(f"{MD_DIR}/target_data_solusdt.parquet")
    target_solusdt = pd.read_parquet(f"{MD_DIR}/target_solusdt.parquet")

    for df in [
        agg_trades,
        orderbook_embedding,
        orderbook_solusdt,
        target_data_solusdt,
        target_solusdt,
    ]:
        df.index = pd.to_datetime(df.index)
    score = prediction_simulation(
        "./",
        Predictor,
        orderbook_solusdt,
        agg_trades,
        orderbook_embedding,
        target_data_solusdt,
        target_solusdt,
    )

    print("score is", score)
