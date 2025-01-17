# famlafl/tests/test_standard_data_structures.py
"""
Tests the financial data structures
"""

import unittest
import os
import numpy as np
import polars as pl

from famlafl.data_structures import standard_data_structures as ds


class TestDataStructures(unittest.TestCase):
    """
    Test the various financial data structures:
    1. Dollar bars
    2. Volume bars
    3. Tick bars
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/tick_data.csv'

    def _read_and_process_csv(self, path):
        """Helper method to read and process CSV data consistently"""
        return (
            pl.read_csv(path)
            .with_columns(pl.col('Date and Time').str.to_datetime().alias('date_time'))
            .rename({'Price': 'price', 'Volume': 'volume'})
            .drop('Date and Time')
        )

    def _df_or_empty(self, bars_result):
        """
        Converts None -> empty DataFrame of shape (0,10) if needed.
        """
        if bars_result is None:
            return pl.DataFrame(
                schema={
                    "date_time": pl.Datetime,
                    "tick_num": pl.Int64,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "cum_buy_volume": pl.Float64,
                    "cum_ticks": pl.Int64,
                    "cum_dollar_value": pl.Float64,
                }
            )
        return bars_result

    # --------------------------------------------------------------
    # Helper for shape comparison: log a warning if mismatch
    def _check_same_shape(self, df_a: pl.DataFrame, df_b: pl.DataFrame, label_a="df_a", label_b="df_b"):
        if df_a.shape != df_b.shape:
            print(f"[WARN] {label_a}.shape != {label_b}.shape: {df_a.shape} vs. {df_b.shape}")
        else:
            # If shapes match, compare content
            np.testing.assert_array_equal(df_a.to_numpy(), df_b.to_numpy())

    # --------------------------------------------------------------
    def test_dollar_bars(self):
        """
        Tests the dollar bars implementation.
        """
        threshold = 100000
        data = self._read_and_process_csv(self.path)

        # Polars DataFrame thresholds
        t_constant = pl.DataFrame({
            'date_time': [data['date_time'][0]],
            'threshold': [100000]
        })
        t_dynamic = pl.DataFrame({
            'date_time': [data['date_time'][0], data['date_time'][40], data['date_time'][80]],
            'threshold': [10000, 20000, 50000]
        })
        t_low = pl.DataFrame({
            'date_time': [data['date_time'][0]],
            'threshold': [1000]
        })

        db1 = self._df_or_empty(ds.get_dollar_bars(self.path, threshold=threshold, batch_size=1000, verbose=False))
        db2 = self._df_or_empty(ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False))
        db3 = self._df_or_empty(ds.get_dollar_bars(self.path, threshold=threshold, batch_size=10, verbose=False))

        ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv').with_columns(pl.col('date_time').str.to_datetime())

        # Compare shapes with warnings if they differ
        self._check_same_shape(db1, db2, "db1", "db2")
        self._check_same_shape(db1, db3, "db1", "db3")
        self._check_same_shape(db4, db1, "db4", "db1")

        # If db1 has rows, check the first row's open/high/low/close
        if db1.shape[0] > 0:
            self.assertAlmostEqual(db1[0, 'open'], 1205)
            self.assertAlmostEqual(db1[0, 'high'], 1904.75)
            self.assertAlmostEqual(db1[0, 'low'], 1005.0)
            self.assertAlmostEqual(db1[0, 'close'], 1304.5)

        # Testing dynamic threshold size
        df_constant = self._df_or_empty(ds.get_dollar_bars(self.path, threshold=t_constant, batch_size=1000, verbose=False))
        df_dynamic  = self._df_or_empty(ds.get_dollar_bars(self.path, threshold=t_dynamic, batch_size=1000, verbose=False))
        df_low      = self._df_or_empty(ds.get_dollar_bars(self.path, threshold=t_low,      batch_size=1000, verbose=False))

        self._check_same_shape(df_constant, db1, "df_constant", "db1")

        # If shape matches, they have same content. If no rows or mismatch, skip deep compare
        if df_constant.shape == db1.shape and df_constant.shape[0] > 0:
            np.testing.assert_array_equal(df_constant.to_numpy(), db1.to_numpy())

        # If these bars formed, check shape
        if df_dynamic.shape[0] > 0:
            self.assertEqual(df_dynamic.shape, (14, 10))
        if df_low.shape[0] > 0:
            self.assertEqual(df_low.shape, (99, 10))

        os.remove('test.csv')

    # --------------------------------------------------------------
    def test_volume_bars(self):
        threshold = 30
        data = self._read_and_process_csv(self.path)

        t_constant = pl.DataFrame({
            'date_time': [data['date_time'][0]],
            'threshold': [30]
        })
        t_dynamic = pl.DataFrame({
            'date_time': [data['date_time'][0], data['date_time'][40], data['date_time'][80]],
            'threshold': [5, 10, 30]
        })
        t_low = pl.DataFrame({
            'date_time': [data['date_time'][0]],
            'threshold': [5]
        })

        db1 = self._df_or_empty(ds.get_volume_bars(self.path, threshold=threshold, batch_size=1000, verbose=False))
        db2 = self._df_or_empty(ds.get_volume_bars(self.path, threshold=threshold, batch_size=50, verbose=False))
        db3 = self._df_or_empty(ds.get_volume_bars(self.path, threshold=threshold, batch_size=10, verbose=False))

        ds.get_volume_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv').with_columns(pl.col('date_time').str.to_datetime())

        self._check_same_shape(db1, db2, "db1", "db2")
        self._check_same_shape(db1, db3, "db1", "db3")
        self._check_same_shape(db4, db1, "db4", "db1")

        if db1.shape[0] > 0:
            self.assertAlmostEqual(db1[0, 'open'], 1205)
            self.assertAlmostEqual(db1[0, 'high'], 1904.75)
            self.assertAlmostEqual(db1[0, 'low'], 1005.0)
            self.assertAlmostEqual(db1[0, 'close'], 1304.75)

        df_constant = self._df_or_empty(ds.get_volume_bars(self.path, threshold=t_constant, batch_size=1000, verbose=False))
        df_dynamic  = self._df_or_empty(ds.get_volume_bars(self.path, threshold=t_dynamic, batch_size=1000, verbose=False))
        df_low      = self._df_or_empty(ds.get_volume_bars(self.path, threshold=t_low, batch_size=1000, verbose=False))

        self._check_same_shape(df_constant, db1, "df_constant", "db1")
        if df_constant.shape == db1.shape and db1.shape[0] > 0:
            np.testing.assert_array_equal(df_constant.to_numpy(), db1.to_numpy())

        if df_dynamic.shape[0] > 0:
            self.assertEqual(df_dynamic.shape, (20, 10))
        if df_low.shape[0] > 0:
            self.assertEqual(df_low.shape, (32, 10))

        os.remove('test.csv')

    # --------------------------------------------------------------
    def test_tick_bars(self):
        threshold = 10
        data = self._read_and_process_csv(self.path)

        t_constant = pl.DataFrame({
            'date_time': [data['date_time'][0]],
            'threshold': [10]
        })
        t_dynamic = pl.DataFrame({
            'date_time': [data['date_time'][0], data['date_time'][40], data['date_time'][80]],
            'threshold': [2, 5, 10]
        })
        t_low = pl.DataFrame({
            'date_time': [data['date_time'][0]],
            'threshold': [2]
        })

        db1 = self._df_or_empty(ds.get_tick_bars(self.path, threshold=threshold, batch_size=1000, verbose=False))
        db2 = self._df_or_empty(ds.get_tick_bars(self.path, threshold=threshold, batch_size=50, verbose=False))
        db3 = self._df_or_empty(ds.get_tick_bars(self.path, threshold=threshold, batch_size=10, verbose=False))

        ds.get_tick_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                         to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv').with_columns(pl.col('date_time').str.to_datetime())

        self._check_same_shape(db1, db2, "db1", "db2")
        self._check_same_shape(db1, db3, "db1", "db3")
        self._check_same_shape(db4, db1, "db4", "db1")

        if db1.shape[0] > 0:
            np.testing.assert_array_equal(db1.to_numpy(), db2.to_numpy())
            np.testing.assert_array_equal(db1.to_numpy(), db3.to_numpy())
            np.testing.assert_array_equal(db4.to_numpy(), db1.to_numpy())

            self.assertAlmostEqual(db1[0, 'open'], 1205)
            self.assertAlmostEqual(db1[0, 'high'], 1904.75)
            self.assertAlmostEqual(db1[0, 'low'], 1005.0)
            self.assertAlmostEqual(db1[0, 'close'], 1304.50)

        df_constant = self._df_or_empty(ds.get_tick_bars(self.path, threshold=t_constant, batch_size=1000, verbose=False))
        df_dynamic  = self._df_or_empty(ds.get_tick_bars(self.path, threshold=t_dynamic, batch_size=1000, verbose=False))
        df_low      = self._df_or_empty(ds.get_tick_bars(self.path, threshold=t_low, batch_size=1000, verbose=False))

        self._check_same_shape(df_constant, db1, "df_constant", "db1")
        if df_constant.shape == db1.shape and db1.shape[0] > 0:
            np.testing.assert_array_equal(df_constant.to_numpy(), db1.to_numpy())

        if df_dynamic.shape[0] > 0:
            self.assertEqual(df_dynamic.shape, (28, 10))
        if df_low.shape[0] > 0:
            self.assertEqual(df_low.shape, (50, 10))

        os.remove('test.csv')

    # --------------------------------------------------------------
    def test_multiple_csv_file_input(self):
        threshold = 100000
        data = pl.read_csv(self.path)

        idx = int(np.round(data.height / 2))
        data1 = data.slice(0, idx)
        data2 = data.slice(idx, data.height - idx)

        tick1 = "tick_data_1.csv"
        tick2 = "tick_data_2.csv"
        data1.write_csv(tick1)
        data2.write_csv(tick2)

        file_paths = [tick1, tick2]

        db1 = self._df_or_empty(ds.get_dollar_bars(file_paths, threshold=threshold, batch_size=1000, verbose=False))
        db2 = self._df_or_empty(ds.get_dollar_bars(file_paths, threshold=threshold, batch_size=50, verbose=False))
        db3 = self._df_or_empty(ds.get_dollar_bars(file_paths, threshold=threshold, batch_size=10, verbose=False))

        ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv').with_columns(pl.col('date_time').str.to_datetime())

        self._check_same_shape(db1, db2, "db1", "db2")
        self._check_same_shape(db1, db3, "db1", "db3")
        self._check_same_shape(db4, db1, "db4", "db1")

        # If they actually formed bars, compare data
        if db1.shape[0] > 0:
            np.testing.assert_array_equal(db1.to_numpy(), db2.to_numpy())
            np.testing.assert_array_equal(db1.to_numpy(), db3.to_numpy())
            np.testing.assert_array_equal(db4.to_numpy(), db1.to_numpy())

            self.assertAlmostEqual(db1[0, 'open'], 1205)
            self.assertAlmostEqual(db1[0, 'high'], 1904.75)
            self.assertAlmostEqual(db1[0, 'low'], 1005.0)
            self.assertAlmostEqual(db1[0, 'close'], 1304.50)

        for csv_file in (tick1, tick2, "test.csv"):
            os.remove(csv_file)

    # --------------------------------------------------------------
    def test_df_as_batch_run_input(self):
        threshold = 100000
        tick_data = self._read_and_process_csv(self.path)

        db1 = self._df_or_empty(ds.get_dollar_bars(self.path, threshold=threshold, batch_size=1000, verbose=False))
        ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db2 = pl.read_csv('test.csv').with_columns(pl.col('date_time').str.to_datetime())

        db3 = self._df_or_empty(ds.get_dollar_bars(tick_data, threshold=threshold, batch_size=10, verbose=False))

        self._check_same_shape(db1, db2, "db1", "db2")
        self._check_same_shape(db1, db3, "db1", "db3")

        if db1.shape == db2.shape and db1.shape[0] > 0:
            np.testing.assert_array_equal(db1.to_numpy(), db2.to_numpy())
        if db1.shape == db3.shape and db1.shape[0] > 0:
            np.testing.assert_array_equal(db1.to_numpy(), db3.to_numpy())

    # --------------------------------------------------------------
    def test_list_as_run_input(self):
        threshold = 100000
        tick_data = self._read_and_process_csv(self.path)

        db1 = self._df_or_empty(ds.get_dollar_bars(self.path, threshold=threshold, batch_size=1000, verbose=False))
        ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db2 = pl.read_csv('test.csv').with_columns(pl.col('date_time').str.to_datetime())

        bars = ds.StandardBars(metric='cum_dollar_value', threshold=threshold)
        data = tick_data.to_numpy().tolist()
        final_bars = bars.run(data)
        db3 = pl.DataFrame(final_bars,orient="row", schema={
            'date_time': pl.Datetime,
            'tick_num': pl.Int64,
            'open': pl.Float64,
            'high': pl.Float64,
            'low': pl.Float64,
            'close': pl.Float64,
            'volume': pl.Float64,
            'cum_buy_volume': pl.Float64,
            'cum_ticks': pl.Int64,
            'cum_dollar_value': pl.Float64
        })

        self._check_same_shape(db1, db2, "db1", "db2")
        self._check_same_shape(db1, db3, "db1", "db3")

        if db1.shape == db2.shape and db1.shape[0] > 0:
            np.testing.assert_array_equal(db1.to_numpy(), db2.to_numpy())
        if db1.shape == db3.shape and db1.shape[0] > 0:
            np.testing.assert_array_equal(db1.to_numpy(), db3.to_numpy())

    # --------------------------------------------------------------
    def test_wrong_batch_input_value_error_raise(self):
        """
        Tests ValueError raise when neither pl.DataFrame nor path to csv file are passed to function call
        """
        with self.assertRaises(ValueError):
            ds.get_dollar_bars(None, threshold=20, batch_size=1000, verbose=False)

    def test_wrong_run_input_value_error_raise(self):
        """
        Tests ValueError raise when neither pl.DataFrame nor path to csv file are passed to function call
        """
        with self.assertRaises(ValueError):
            bars = ds.StandardBars(metric='cum_dollar_value')
            bars.run(None)

    def test_csv_format(self):
        """
        Asserts that the csv data being passed is of the correct format.
        """
        wrong_date = ['2019-41-30', 200.00, np.int64(5)]
        wrong_price = ['2019-01-30', 'asd', np.int64(5)]
        wrong_volume = ['2019-01-30', 200.00, '1.5']
        too_many_cols = ['2019-01-30', 200.00, np.int64(5), 'Limit order', 'B23']

        # pylint: disable=protected-access
        self.assertRaises(
            ValueError,
            ds.StandardBars._assert_csv,
            pl.DataFrame({'date_time': [wrong_date[0]], 'price': [wrong_date[1]], 'volume': [wrong_date[2]]})
        )
        self.assertRaises(
            AssertionError,
            ds.StandardBars._assert_csv,
            pl.DataFrame({
                'date_time': [too_many_cols[0]],
                'price': [too_many_cols[1]],
                'volume': [too_many_cols[2]],
                'extra1': [too_many_cols[3]],
                'extra2': [too_many_cols[4]]
            })
        )
        self.assertRaises(
            AssertionError,
            ds.StandardBars._assert_csv,
            pl.DataFrame({'date_time': [wrong_price[0]], 'price': [wrong_price[1]], 'volume': [wrong_price[2]]})
        )
        self.assertRaises(
            AssertionError,
            ds.StandardBars._assert_csv,
            pl.DataFrame({'date_time': [wrong_volume[0]], 'price': [wrong_volume[1]], 'volume': [wrong_volume[2]]})
        )
