#famlafl/data_structures/standard_data_structures.py
"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al.

Many of the projects going forward will require Dollar and Volume bars.
"""

from typing import Union, Iterable, Optional

import numpy as np
import polars as pl

from famlafl.data_structures.base_bars import BaseBars


class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, threshold: Union[float, pl.Series] = 50000, batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "dollar_run"
        :param threshold: (float or pl.Series) Threshold at which to sample. If Series, then at each sampling time
                        the closest previous threshold is used.
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        BaseBars.__init__(self, metric, batch_size)

        # Threshold at which to sample
        self.threshold = threshold

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for standard bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: np.ndarray) -> list:
        """
        For loop which compiles the various bars: dollar, volume, or tick.
        We investigated using a vectorized approach, but a for-loop worked well in practice.

        :param data: (np.ndarray) A 2D array of shape (N, 3), 
                    where each row is [date_time, price, volume].
        :return: (list) A list of bars formed by this method.
        """
        list_bars = []

        for row in data:
            date_time = row[0]
            self.tick_num += 1
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            # 1) Determine the threshold for the current row
            if isinstance(self.threshold, (int, float)):
                # Fixed threshold
                threshold_val = self.threshold
            else:
                # Dynamic threshold: find the most recent threshold <= date_time
                filtered = (
                    self.threshold
                    .filter(pl.col("date_time") <= date_time)
                    .select("threshold")
                    .tail(1)
                )
                if filtered.is_empty():
                    # No valid threshold yet; decide your default
                    # e.g. float("inf") means "no bar forms yet"
                    threshold_val = float("inf")
                else:
                    # Convert 1x1 DataFrame -> float
                    threshold_val = filtered.item(0, 0)

            # 2) Initialize open price if needed
            if self.open_price is None:
                self.open_price = price

            # 3) Update high/low
            self.high_price, self.low_price = self._update_high_low(price)

            # 4) Update cumulative statistics
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # 5) Check if we've crossed the threshold
            if self.cum_statistics[self.metric] >= threshold_val:
                self._create_bars(
                    date_time=date_time,
                    close_price=price,
                    high_price=self.high_price,
                    low_price=self.low_price,
                    list_bars=list_bars
                )
                self._reset_cache()

        return list_bars



def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pl.DataFrame], threshold: Union[float, pl.Series] = 70000000,
                    batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the dollar bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
    properties.

    :param file_path_or_df: (str, iterable of str, or pl.DataFrame) Path to the csv file(s) or Polars Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float, or pl.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pl.DataFrame) DataFrame of dollar bars
    """
    bars = StandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size)
    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars


def get_volume_bars(file_path_or_df: Union[str, Iterable[str], pl.DataFrame], threshold: Union[float, pl.Series] = 70000000,
                    batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the volume bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.

    :param file_path_or_df: (str, iterable of str, or pl.DataFrame) Path to the csv file(s) or Polars Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float, or pl.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pl.DataFrame) DataFrame of volume bars
    """
    bars = StandardBars(metric='cum_volume', threshold=threshold, batch_size=batch_size)
    volume_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars


def get_tick_bars(file_path_or_df: Union[str, Iterable[str], pl.DataFrame], threshold: Union[float, pl.Series] = 70000000,
                  batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the tick bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pl.DataFrame) Path to the csv file(s) or Polars Data Frame containing raw tick data
                             in the format[date_time, price, volume]
    :param threshold: (float, or pl.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pl.DataFrame) DataFrame of tick bars
    """
    bars = StandardBars(metric='cum_ticks', threshold=threshold, batch_size=batch_size)
    tick_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars
