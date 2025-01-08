# famlafl/tests/test_multiprocess.py
"""
Test various functions regarding multiprocessing utilities.
"""

import unittest
import pandas as pd
import numpy as np
import io
import time
from unittest.mock import patch

from famlafl.util.multiprocess import (
    mp_pandas_obj,
    process_jobs,
    process_jobs_,
    process_jobs_in_batches,
    lin_parts,
    nested_parts,
    report_progress,
)


def simple_function(molecule):
    """Simple function that returns the molecule unchanged."""
    return pd.Series(index=molecule)

def none_function(molecule):
    """Function that returns None."""
    return None

def int_function(molecule):
    """Function that returns a constant integer."""
    return 42

def fail_on_second_call(molecule):
    """Function that succeeds on first call but fails on second."""
    if not hasattr(fail_on_second_call, 'called'):
        fail_on_second_call.called = True
        return pd.Series(index=molecule)
    raise RuntimeError("Second call fails")


def series_function(molecule):
    """Function that returns a Series with molecule as index."""
    return pd.Series(1, index=molecule)


def dataframe_function(molecule):
    """Function that returns a DataFrame with molecule as index."""
    return pd.DataFrame({'value': 1}, index=molecule)


class TestMultiprocess(unittest.TestCase):
    """
    Test multiprocessing functions.
    """

    def setUp(self):
        """
        Set up test data.
        """
        self.index = pd.date_range('2020-01-01', periods=100, freq='D')

    def test_lin_parts(self):
        """
        Test linear partitioning of atoms.
        """
        # Test with num_atoms > num_threads
        parts = lin_parts(100, 10)
        self.assertEqual(len(parts), 11)  # num_threads + 1 parts
        self.assertEqual(parts[0], 0)
        self.assertEqual(parts[-1], 100)

        # Test with num_atoms < num_threads
        parts = lin_parts(5, 10)
        self.assertEqual(len(parts), 6)  # num_atoms + 1 parts
        self.assertEqual(parts[0], 0)
        self.assertEqual(parts[-1], 5)

    def test_nested_parts(self):
        """
        Test nested partitioning of atoms.
        """
        # Test regular nested partitioning
        parts = nested_parts(100, 10)
        self.assertTrue(len(parts) > 1)
        self.assertEqual(parts[0], 0)
        self.assertTrue(all(parts[i] <= parts[i + 1] for i in range(len(parts) - 1)))

        # Test upper triangle
        parts = nested_parts(100, 10, upper_triangle=True)
        self.assertTrue(len(parts) > 1)
        self.assertEqual(parts[0], 0)

    def test_process_jobs_single_thread(self):
        """
        Test processing jobs with a single thread.
        """
        jobs = [
            {'molecule': self.index[:10], 'func': simple_function},
            {'molecule': self.index[10:20], 'func': simple_function},
        ]
        result = process_jobs(jobs, num_threads=1)
        self.assertEqual(len(result), 2)

    def test_process_jobs_multi_thread(self):
        """
        Test processing jobs with multiple threads.
        """
        jobs = [
            {'molecule': self.index[:10], 'func': simple_function},
            {'molecule': self.index[10:20], 'func': simple_function},
        ]
        result = process_jobs(jobs, num_threads=2)
        self.assertEqual(len(result), 2)

    def test_mp_pandas_obj_empty_input(self):
        """
        Test mp_pandas_obj with empty input.
        """
        empty_index = pd.Index([])
        result = mp_pandas_obj(simple_function, ('molecule', empty_index))
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(len(result), 0)

    def test_mp_pandas_obj_series_output(self):
        """
        Test mp_pandas_obj with Series output.
        """
        result = mp_pandas_obj(series_function, ('molecule', self.index), num_threads=1)
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(len(result), len(self.index))
        self.assertTrue(all(result == 1))

    def test_mp_pandas_obj_dataframe_output(self):
        """
        Test mp_pandas_obj with DataFrame output.
        """
        result = mp_pandas_obj(dataframe_function, ('molecule', self.index), num_threads=1)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(len(result), len(self.index))
        self.assertTrue(all(result['value'] == 1))

    def test_mp_pandas_obj_linear_partition(self):
        """
        Test mp_pandas_obj with linear partitioning.
        """
        result = mp_pandas_obj(simple_function, ('molecule', self.index), lin_mols=True, num_threads=1)
        self.assertEqual(len(result), len(self.index))

    def test_mp_pandas_obj_nested_partition(self):
        """
        Test mp_pandas_obj with nested partitioning.
        """
        result = mp_pandas_obj(simple_function, ('molecule', self.index), lin_mols=False, num_threads=1)
        self.assertEqual(len(result), len(self.index))

    def test_mp_pandas_obj_single_thread(self):
        """
        Test mp_pandas_obj with single thread.
        """
        result = mp_pandas_obj(simple_function, ('molecule', self.index), num_threads=1)
        self.assertEqual(len(result), len(self.index))

    def test_mp_pandas_obj_multi_thread(self):
        """
        Test mp_pandas_obj with multiple threads.
        """
        result = mp_pandas_obj(simple_function, ('molecule', self.index), num_threads=2)
        self.assertEqual(len(result), len(self.index))

    def test_mp_pandas_obj_batches(self):
        """
        Test mp_pandas_obj with multiple batches.
        """
        result = mp_pandas_obj(simple_function, ('molecule', self.index), mp_batches=2, num_threads=1)
        self.assertEqual(len(result), len(self.index))

    def test_process_jobs_empty_jobs(self):
        """
        Test process_jobs with empty jobs list.
        """
        result = process_jobs([])
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(len(result), 0)

    def test_process_jobs_error_handling(self):
        """
        Test process_jobs error handling.
        """
        def error_function(molecule):
            raise ValueError("Test error")

        jobs = [
            {'molecule': self.index[:10], 'func': error_function},
        ]
        result = process_jobs(jobs, num_threads=1)
        # When all jobs fail, process_jobs should return an empty DataFrame with default columns
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(len(result), 0)
        self.assertTrue(all(col in result.columns for col in ['t1', 'pt', 'sl']))

    # 1) Test process_jobs_ (which returns both outputs and failed jobs)
    def test_process_jobs_variant_success_fail(self):
        """
        Test process_jobs_ variant that returns (outputs, failed).
        """
        jobs = [
            {'molecule': self.index[:10], 'func': simple_function},
            {'molecule': self.index[10:20], 'func': none_function},  # returns None => fail
        ]
        outputs, failed = process_jobs_(jobs, task="test_task", num_threads=1)

        # The first job produces a valid (empty) Series, the second job returns None => goes to 'failed'
        self.assertEqual(len(outputs), 1)  # Only one successful output
        self.assertEqual(len(failed), 1)   # One failed job

    def test_process_jobs_variant_multi_thread_fail(self):
        """
        Test process_jobs_ with multiple threads and a failing job.
        """
        def failing_func(molecule):
            raise RuntimeError("Failing job.")

        jobs = [
            {'molecule': self.index[:10], 'func': simple_function},
            {'molecule': self.index[10:20], 'func': failing_func},
        ]
        outputs, failed = process_jobs_(jobs, num_threads=2)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(failed), 1)

    # 2) Test process_jobs_in_batches
    def test_process_jobs_in_batches_success(self):
        """
        Test process_jobs_in_batches with multiple batches and success.
        """
        jobs = []
        for i in range(0, 100, 10):
            jobs.append({'molecule': self.index[i:i+10], 'func': simple_function})

        results, failed = process_jobs_in_batches(
            jobs,
            task="test_batch_task",
            num_threads=2,
            batch_size=2,  # process in batches of 2
            verbose=False  # Turn off printing to keep test output clean
        )
        # Results should be concatenated Series or list of Series
        self.assertTrue(isinstance(results, (pd.Series, list)))
        self.assertEqual(len(failed), 0)

    def test_process_jobs_in_batches_partial_fail(self):
        """
        Test process_jobs_in_batches with some jobs failing.
        """
        jobs = []
        # The first batch will succeed, the second batch fails on second job
        for i in range(0, 20, 10):
            jobs.append({'molecule': self.index[i:i+5], 'func': simple_function})
            jobs.append({'molecule': self.index[i+5:i+10], 'func': none_function})  # returns None => fail

        results, failed = process_jobs_in_batches(
            jobs,
            task="test_batch_task",
            num_threads=1,
            batch_size=2,
            verbose=False
        )
        self.assertTrue(len(failed) > 0)
        # Because the successful ones are Series, the final results might be
        # a single concatenated Series or a list of Series:
        if isinstance(results, pd.Series):
            self.assertGreater(len(results), 0)
        else:
            self.assertGreater(len(results), 0)

    # 3) Edge cases for process_jobs
    def test_process_jobs_non_pandas_outputs(self):
        """
        Test process_jobs with a function returning non-pandas (an integer).
        This should preserve the job structure by returning a list of integers.
        """
        jobs = [
            {'molecule': self.index[:10], 'func': int_function},
            {'molecule': self.index[10:20], 'func': int_function},
        ]
        results = process_jobs(jobs, num_threads=1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], 42)
        self.assertEqual(results[1], 42)

    def test_process_jobs_none_output_single_job(self):
        """
        Test process_jobs with a single job returning None.
        """
        jobs = [{'molecule': self.index[:10], 'func': none_function}]
        results = process_jobs(jobs, num_threads=1)
        # Because the only job returned None, outputs is empty => returns default DataFrame
        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertEqual(len(results), 0)  # no rows
        self.assertTrue(all(col in results.columns for col in ['t1', 'pt', 'sl']))

    def test_process_jobs_partial_outputs(self):
        """
        Test the partial outputs scenario: one job fails in the middle, or returns None,
        so the length of 'outputs' doesn't match the length of jobs.
        """
        jobs = [
            {'molecule': self.index[:10], 'func': fail_on_second_call},  # first call succeeds, second fails
            {'molecule': self.index[10:20], 'func': fail_on_second_call},
        ]
        results = process_jobs(jobs, num_threads=1)
        # Because the second call fails, process_jobs will catch the exception and
        # return an empty DataFrame with default columns
        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertEqual(len(results), 0)
        self.assertTrue(all(col in results.columns for col in ['t1', 'pt', 'sl']))

    def test_process_jobs_task_none(self):
        """
        Test that if 'task' is None, the default is set from jobs[0]['func'].__name__.
        We'll just ensure no exception is raised and coverage is triggered.
        """
        jobs = [
            {'molecule': self.index[:10], 'func': simple_function},
            {'molecule': self.index[10:20], 'func': simple_function},
        ]
        # Capture stderr to ensure coverage of report_progress
        with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
            results = process_jobs(jobs, task=None, num_threads=2)
            self.assertEqual(len(results), 2)
            # Check that the default task name (simple_function) got printed
            self.assertIn("simple_function", fake_stderr.getvalue())

    # 4) Directly test coverage of report_progress (optional)
    # Usually covered by process_jobs in multi-thread mode, but here's a direct test if needed:
    def test_report_progress_direct(self):
        """
        Test report_progress by calling it directly with a small example.
        """
        from famlafl.util.multiprocess import report_progress
        with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
            t0 = time.time()
            report_progress(job_number=1, num_jobs=2, time0=t0, task="direct_test")
            output = fake_stderr.getvalue()
            self.assertIn("direct_test", output)
            self.assertIn("% done after", output)
            # Also test final call (job_number == num_jobs)
            with patch('sys.stderr', new=io.StringIO()) as fake_stderr2:
                report_progress(job_number=2, num_jobs=2, time0=t0, task="direct_test")
                self.assertIn("done after", fake_stderr2.getvalue())