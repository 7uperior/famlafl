import math
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


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

        # Compare one_time_check rows with previous calculated features
        # here sorting of checks_idxs ensures that we check correctly
        features_cur = features.iloc[-one_time_check:]
        features_prev = features_.loc[features_cur.index]
        if not np.array_equal(features_cur.values, features_prev.values, equal_nan=True):
            validation = features_cur.values != features_prev.values
            where_is_leak = np.argwhere(validation)
            leaked_features_idxs = set(where_is_leak[:, 1])
            raise RuntimeError(f"Lookahead validation didn't pass for features with idxs: {leaked_features_idxs}")

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
    def __init__(self, full_model_name: Union[str, list[str]]):
        pass

    @staticmethod
    def model_name() -> Union[str, list[str]]:
        return []

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

        return features["pi"]

    def calc_features(
        self,
        lobs: pd.DataFrame | None,
        agg_trades: pd.DataFrame | None,
        lobs_embedding: pd.DataFrame | None,
        target_data: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Params
        ----------
        lobs: pd.DataFrame
            Index - timestamp - datetime
            columns - ["asks[0].price", "asks[0].amount", "bids[0].price", "bids[0].amount", ..., "asks[10].price", "asks[10].amount", "bids[10].price", "bids[10].amount"]    # noqa: E501
            dtypes -  float
            level price or vol can be empty

        trades: pd.DataFrame
            Index - timestamp - datetime
            columns - ["side", "price", "size"]
            dtypes -  [int   , float  , float ]
            side: 1 for bid and 0 for ask

        Returns:
            Pandas DataFrame of all your features of format `target_data timestamps` x `your features`
            so that len(DataFrame) == len(target_data)
        """
        features = pd.DataFrame()
        features.index = target_data.index
        features["pi"] = 3.1415
        return features


# this Predictor does not use any external model that can be loaded from the file


if __name__ == "__main__":
    MD_DIR = "./data/md/test"
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
