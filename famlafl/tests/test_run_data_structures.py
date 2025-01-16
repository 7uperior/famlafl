"""
Tests the financial data structures
"""

import unittest
import os
import numpy as np
import polars as pl

from famlafl.data_structures import run_data_structures as ds


class TestDataStructures(unittest.TestCase):
    """
    Test the various financial data structures:
    1. Run Dollar bars
    2. Run Volume bars
    3. Run Tick bars
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/imbalance_sample_data_small.csv'

    def test_ema_run_dollar_bars(self):
        """
        Tests the EMA run dollar bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                   analyse_thresholds=True)
        db2, thresh_2 = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                   analyse_thresholds=True)
        db3, _ = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   expected_imbalance_window=10000,
                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                   to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv', try_parse_dates=True)

        self.assertEqual(db1.shape, (2, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db4.to_numpy()))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("high").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("low").item(), 1303.00)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("close").item(), 1305.75)

        # Assert OHLC is correct (the second value)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("open").item(), 1305.75)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("high").item(), 1308.75)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("low").item(), 1305.25)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("close").item(), 1307.25)

        # Assert conditions across all rows
        self.assertTrue((db1.select("high") >= db1.select("low")).all())
        self.assertTrue((db1.select("volume") >= db1.select("cum_buy_volume")).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_ema_run_volume_bars(self):
        """
        Tests the EMA run volume bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_ema_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                   analyse_thresholds=True)
        db2, thresh_2 = ds.get_ema_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                   analyse_thresholds=True)
        db3, _ = ds.get_ema_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_ema_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   expected_imbalance_window=10000,
                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                   to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv', try_parse_dates=True)

        self.assertEqual(db1.shape, (2, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db4.to_numpy()))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("high").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("low").item(), 1303.00)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("close").item(), 1305.75)

        # Assert OHLC is correct (the second value)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("open").item(), 1305.75)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("high").item(), 1308.75)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("low").item(), 1305.25)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("close").item(), 1307.25)

        # Assert conditions across all rows
        self.assertTrue((db1.select("high") >= db1.select("low")).all())
        self.assertTrue((db1.select("volume") >= db1.select("cum_buy_volume")).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_ema_run_tick_bars(self):
        """
        Tests the EMA run tick bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_ema_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                 expected_imbalance_window=10000,
                                                 num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                 analyse_thresholds=True)
        db2, thresh_2 = ds.get_ema_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                 expected_imbalance_window=10000,
                                                 num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                 analyse_thresholds=True)
        db3, _ = ds.get_ema_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                          expected_imbalance_window=10000,
                                          num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_ema_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                 expected_imbalance_window=10000,
                                 num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                 to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv', try_parse_dates=True)

        self.assertEqual(db1.shape, (2, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db4.to_numpy()))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("high").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("low").item(), 1303.00)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("close").item(), 1305.75)

        # Assert OHLC is correct (the second value)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("open").item(), 1305.75)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("high").item(), 1308.75)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("low").item(), 1305.25)
        self.assertEqual(db1.filter(pl.col("index") == 1).select("close").item(), 1307.25)

        # Assert conditions across all rows
        self.assertTrue((db1.select("high") >= db1.select("low")).all())
        self.assertTrue((db1.select("volume") >= db1.select("cum_buy_volume")).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_ema_run_dollar_bars_with_constraints(self):
        """
        Test the EMA Dollar Run bars with expected number of ticks max and min constraints
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3
        exp_num_ticks_constraints = [100, 1000]

        db1, _ = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            exp_num_ticks_constraints=exp_num_ticks_constraints,
                                            num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False)
        db2, _ = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            exp_num_ticks_constraints=exp_num_ticks_constraints,
                                            num_prev_bars=num_prev_bars, batch_size=50, verbose=False)
        db3, _ = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            exp_num_ticks_constraints=exp_num_ticks_constraints,
                                            num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   expected_imbalance_window=10000,
                                   exp_num_ticks_constraints=exp_num_ticks_constraints,
                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                   to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv', try_parse_dates=True)

        self.assertEqual(db1.shape, (5, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db4.to_numpy()))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("high").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("low").item(), 1303.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("close").item(), 1305.75)

        # Assert conditions across all rows
        self.assertTrue((db1.select("high") >= db1.select("low")).all())

        # Assert OHLC is correct (last value)
        self.assertEqual(db1.filter(pl.col("index") == 4).select("open").item(), 1306)
        self.assertEqual(db1.filter(pl.col("index") == 4).select("high").item(), 1306.75)
        self.assertEqual(db1.filter(pl.col("index") == 4).select("low").item(), 1303.5)
        self.assertEqual(db1.filter(pl.col("index") == 4).select("close").item(), 1303.5)

        self.assertTrue((db1.select("volume") >= db1.select("cum_buy_volume")).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_const_run_dollar_bars(self):
        """
        Tests the Const run dollar bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_const_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                     expected_imbalance_window=10000,
                                                     num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                     analyse_thresholds=True)
        db2, thresh_2 = ds.get_const_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                     expected_imbalance_window=10000,
                                                     num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                     analyse_thresholds=True)
        db3, _ = ds.get_const_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                              expected_imbalance_window=10000,
                                              num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_const_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     expected_imbalance_window=10000,
                                     num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                     to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv', try_parse_dates=True)

        self.assertEqual(db1.shape, (5, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db4.to_numpy()))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("high").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("low").item(), 1303.00)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("close").item(), 1305.75)

        # Assert OHLC is correct (last value)
        self.assertEqual(db1.filter(pl.col("index") == 4).select("open").item(), 1306)
        self.assertEqual(db1.filter(pl.col("index") == 4).select("high").item(), 1306.75)
        self.assertEqual(db1.filter(pl.col("index") == 4).select("low").item(), 1303.5)
        self.assertEqual(db1.filter(pl.col("index") == 4).select("close").item(), 1303.5)

        # Assert conditions across all rows
        self.assertTrue((db1.select("high") >= db1.select("low")).all())
        self.assertTrue((db1.select("volume") >= db1.select("cum_buy_volume")).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_const_run_volume_bars(self):
        """
        Tests the Const run volume bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_const_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                     expected_imbalance_window=10000,
                                                     num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                     analyse_thresholds=True)
        db2, thresh_2 = ds.get_const_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                     expected_imbalance_window=10000,
                                                     num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                     analyse_thresholds=True)
        db3, _ = ds.get_const_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                              expected_imbalance_window=10000,
                                              num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_const_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     expected_imbalance_window=10000,
                                     num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                     to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv', try_parse_dates=True)

        self.assertEqual(db1.shape, (5, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db4.to_numpy()))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("high").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("low").item(), 1303.00)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("close").item(), 1305.75)

        # Assert OHLC is correct (third value)
        self.assertEqual(db1.filter(pl.col("index") == 2).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 2).select("high").item(), 1307.75)
        self.assertEqual(db1.filter(pl.col("index") == 2).select("low").item(), 1305.75)
        self.assertEqual(db1.filter(pl.col("index") == 2).select("close").item(), 1307.75)

        # Assert conditions across all rows
        self.assertTrue((db1.select("high") >= db1.select("low")).all())
        self.assertTrue((db1.select("volume") >= db1.select("cum_buy_volume")).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_const_run_tick_bars(self):
        """
        Tests the Const run tick bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_const_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                   analyse_thresholds=True)
        db2, thresh_2 = ds.get_const_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                   analyse_thresholds=True)
        db3, _ = ds.get_const_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_const_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   expected_imbalance_window=10000,
                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                   to_csv=True, output_path='test.csv')
        db4 = pl.read_csv('test.csv', try_parse_dates=True)

        self.assertEqual(db1.shape, (5, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.to_numpy() == db2.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db3.to_numpy()))
        self.assertTrue(np.all(db1.to_numpy() == db4.to_numpy()))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("high").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("low").item(), 1303.00)
        self.assertEqual(db1.filter(pl.col("index") == 0).select("close").item(), 1305.75)

        # Assert OHLC is correct (third value)
        self.assertEqual(db1.filter(pl.col("index") == 2).select("open").item(), 1306.0)
        self.assertEqual(db1.filter(pl.col("index") == 2).select("high").item(), 1307.5)
        self.assertEqual(db1.filter(pl.col("index") == 2).select("low").item(), 1305.75)
        self.assertEqual(db1.filter(pl.col("index") == 2).select("close").item(), 1307.5)

        # Assert conditions across all rows
        self.assertTrue((db1.select("high") >= db1.select("low")).all())
        self.assertTrue((db1.select("volume") >= db1.select("cum_buy_volume")).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

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
        self.assertRaises(ValueError, ds.BaseRunBars._assert_csv, wrong_date)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError, ds.BaseRunBars._assert_csv, too_many_cols)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError, ds.BaseRunBars._assert_csv, wrong_price)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError, ds.BaseRunBars._assert_csv, wrong_volume)
