# famlafl/tests/test_time_data_structures.py
"""
Tests the financial data structures
"""

import unittest
import os
import numpy as np
import polars as pl

from famlafl.data_structures import time_data_structures as ds


class TestTimeDataStructures(unittest.TestCase):
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
        self.path = project_path + '/test_data/tick_data_time_bars.csv'

    def _parse_test_csv(self, csv_path: str) -> pl.DataFrame:
        """
        Helper function to ensure we parse date_time properly
        """
        df = pl.read_csv(csv_path)
        # Manually parse the string date_time as a datetime with fractional seconds:
        df = df.with_columns(
            pl.col("date_time").str.strptime(
                pl.Datetime, 
                format="%Y-%m-%d %H:%M:%S.%3f",
                strict=False  # or True if all rows definitely match
            )
        )
        return df

    def test_day_bars(self):
        """
        Tests daily bars.
        """
        db1 = ds.get_time_bars(self.path, resolution='D', num_units=1, batch_size=1000, verbose=False)
        db2 = ds.get_time_bars(self.path, resolution='D', num_units=1, batch_size=50, verbose=False)
        db3 = ds.get_time_bars(self.path, resolution='D', num_units=1, batch_size=10, verbose=False)
        ds.get_time_bars(
            self.path,
            resolution='D',
            num_units=1,
            batch_size=50,
            verbose=False,
            to_csv=True,
            output_path='test.csv'
        )
        # Use the helper parser, rather than `try_parse_dates=True`
        db4 = self._parse_test_csv('test.csv')

        # Assert diff batch sizes have same number of bars
        self.assertEqual(db1.shape[0], 1)
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db4.to_numpy()))

        # Assert OHLC is correct
        self.assertTrue(db1["open"][0] == 1200.0)
        self.assertTrue(db1["high"][0] == 1249.75)
        self.assertTrue(db1["low"][0] == 1200.0)
        self.assertTrue(db1["close"][0] == 1249.75)

        # Assert date_time is correct
        expected_timestamp = pl.Datetime(2011, 8, 1, 0, 0, 0).timestamp()
        self.assertTrue(db1["date_time"][0] == expected_timestamp)

        # delete generated csv file
        os.remove('test.csv')

    def test_hour_bars(self):
        """
        Tests hourly bars.
        """
        db1 = ds.get_time_bars(self.path, resolution='H', num_units=1, batch_size=1000, verbose=False)
        db2 = ds.get_time_bars(self.path, resolution='H', num_units=1, batch_size=50, verbose=False)
        db3 = ds.get_time_bars(self.path, resolution='H', num_units=1, batch_size=10, verbose=False)
        ds.get_time_bars(
            self.path,
            resolution='H',
            num_units=1,
            batch_size=50,
            verbose=False,
            to_csv=True,
            output_path='test.csv'
        )
        db4 = self._parse_test_csv('test.csv')

        # Assert diff batch sizes have same number of bars
        self.assertEqual(db1.shape[0], 3)
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db4.to_numpy() == db1.to_numpy()))

        # Assert OHLC is correct
        self.assertTrue(db1["open"][1] == 1225.0)
        self.assertTrue(db1["high"][1] == 1249.75)
        self.assertTrue(db1["low"][1] == 1225.0)
        self.assertTrue(db1["close"][1] == 1249.75)

        # Assert date_time is correct
        expected_timestamp = pl.Datetime(2011, 8, 1, 0, 0, 0).timestamp()
        self.assertTrue(db1["date_time"][1] == expected_timestamp)

        os.remove('test.csv')

    def test_minute_bars(self):
        """
        Tests the minute bars implementation.
        """
        db1 = ds.get_time_bars(self.path, resolution='MIN', num_units=1, batch_size=1000, verbose=False)
        db2 = ds.get_time_bars(self.path, resolution='MIN', num_units=1, batch_size=50, verbose=False)
        db3 = ds.get_time_bars(self.path, resolution='MIN', num_units=1, batch_size=10, verbose=False)
        ds.get_time_bars(
            self.path,
            resolution='MIN',
            num_units=1,
            batch_size=50,
            verbose=False,
            to_csv=True,
            output_path='test.csv'
        )
        db4 = self._parse_test_csv('test.csv')

        # Assert diff batch sizes have same number of bars
        self.assertEqual(db1.shape[0], 11)
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db4.to_numpy() == db1.to_numpy()))

        # Assert OHLC is correct
        self.assertTrue(db1["open"][9] == 1275.0)
        self.assertTrue(db1["high"][9] == 1277.0)
        self.assertTrue(db1["low"][9] == 1275.0)
        self.assertTrue(db1["close"][9] == 1277.0)

        # Assert date_time is correct
        expected_timestamp = pl.Datetime(2011, 8, 1, 23, 39, 0).timestamp()
        self.assertTrue(db1["date_time"][9] == expected_timestamp)

        os.remove('test.csv')

    def test_second_bars(self):
        """
        Tests the seconds bars implementation.
        """
        db1 = ds.get_time_bars(self.path, resolution='S', num_units=10, batch_size=1000, verbose=False)
        db2 = ds.get_time_bars(self.path, resolution='S', num_units=10, batch_size=50, verbose=False)
        db3 = ds.get_time_bars(self.path, resolution='S', num_units=10, batch_size=10, verbose=False)
        ds.get_time_bars(
            self.path,
            resolution='S',
            num_units=10,
            batch_size=50,
            verbose=False,
            to_csv=True,
            output_path='test.csv'
        )
        db4 = self._parse_test_csv('test.csv')

        # Assert diff batch sizes have same number of bars
        self.assertEqual(db1.shape[0], 47)
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db4.to_numpy() == db1.to_numpy()))

        # Assert OHLC is correct
        self.assertTrue(db1["open"][1] == 1201.0)
        self.assertTrue(db1["high"][1] == 1202.0)
        self.assertTrue(db1["low"][1] == 1201.0)
        self.assertTrue(db1["close"][1] == 1202.0)

        # Assert date_time is correct
        expected_timestamp = pl.Datetime(2011, 7, 31, 22, 39, 0).timestamp()
        self.assertTrue(db1["date_time"][1] == expected_timestamp)

        os.remove('test.csv')

    def test_wrong_input_value_error_raise(self):
        """
        Tests ValueError raise when neither pl.DataFrame nor path to csv file are passed.
        """
        with self.assertRaises(ValueError):
            ds.get_time_bars(None, resolution='MIN', num_units=1, batch_size=1000, verbose=False)

    def test_csv_format(self):
        """
        Asserts that the csv data being passed is of the correct format.
        """
        wrong_date = pl.DataFrame({
            'date_time': ['2019-41-30'],
            'price': [200.00],
            'volume': [np.int64(5)]
        })
        wrong_price = pl.DataFrame({
            'date_time': ['2019-01-30'],
            'price': ['asd'],
            'volume': [np.int64(5)]
        })
        wrong_volume = pl.DataFrame({
            'date_time': ['2019-01-30'],
            'price': [200.00],
            'volume': ['1.5']
        })
        too_many_cols = pl.DataFrame({
            'date_time': ['2019-01-30'],
            'price': [200.00],
            'volume': [np.int64(5)],
            'type': ['Limit order'],
            'id': ['B23']
        })

        # pylint: disable=protected-access
        self.assertRaises(ValueError, ds.TimeBars._assert_csv, wrong_date)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError, ds.TimeBars._assert_csv, too_many_cols)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError, ds.TimeBars._assert_csv, wrong_price)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError, ds.TimeBars._assert_csv, wrong_volume)
