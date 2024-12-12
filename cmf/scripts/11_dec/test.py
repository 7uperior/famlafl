import time
from datetime import datetime, timedelta, timezone
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

path_to_data_folder = "./cmf/data/train/"
agg_trades = pl.read_parquet(f'{path_to_data_folder}agg_trades.parquet').with_columns(
    pl.col('timestamp_ns').cast(pl.Datetime)
).to_pandas().set_index('timestamp_ns')
print("agg_trades")
print(agg_trades.columns)
print(agg_trades.info())

