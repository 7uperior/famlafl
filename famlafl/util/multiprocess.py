# famlafl/util/multiprocess.py
"""
Contains functionality for multiprocessing.
"""

import sys
import datetime as dt
import time
from multiprocessing import Process, Queue, cpu_count
import numpy as np
import pandas as pd


def expand_call(kwargs):
    """
    Expand the arguments of a callback function, python.apply_async allows only
    one argument, tuple.
    """
    func = kwargs['func']
    del kwargs['func']
    if 'queue' in kwargs:
        queue = kwargs.pop('queue')
        try:
            out = func(**kwargs)
            if queue is not None:
                queue.put(('success', out))
        except Exception as e:
            if queue is not None:
                queue.put(('error', str(e)))
        return None
    return func(**kwargs)


def report_progress(job_number, num_jobs, time0, task):
    """
    Report progress as asynch jobs are completed.

    :param job_number: (int) Number of jobs completed
    :param num_jobs: (int) Total number of jobs
    :param time0: (time) Start time
    :param task: (str) Task description
    """
    # 1) Compute progress
    msg = [float(job_number) / num_jobs, (time.time() - time0) / 60.0]
    # 2) Add message
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))

    # 3) Report progress
    if job_number < num_jobs:
        sys.stderr.write(
            time_stamp + ' ' + task + ' ' +
            str(round(msg[0] * 100, 2)) + '% done after ' + str(
                round(msg[1], 2)) + ' minutes. Remaining ' + str(
                    round(msg[2], 2)) + ' minutes.')
    else:
        sys.stderr.write(
            time_stamp + ' ' + task + ' done after ' + str(round(msg[1], 2)) +
            ' minutes.')


def process_jobs(jobs, task=None, num_threads=cpu_count()):
    """
    Run in parallel. jobs must contain a 'func' callback, for expand_call

    :param jobs: (list) Jobs (each job is a dict)
    :param task: (str) Task description
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :return: (pd.Series or pd.DataFrame) Returns a pandas object with the results
    """
    if not jobs:
        # Return empty Series for empty jobs list
        return pd.Series(dtype=float)
        
    if num_threads == 1:
        # Run jobs sequentially for better error handling
        outputs = []
        for job in jobs:
            try:
                result = expand_call(job)
                if result is not None:
                    outputs.append(result)
            except Exception as e:
                print(f"Error in job: {str(e)}")
                # Return empty DataFrame with default columns for error handling
                return pd.DataFrame(columns=['t1', 'pt', 'sl'])
    else:
        if task is None:
            task = jobs[0]['func'].__name__
        queue = Queue()
        processes = []
        for job in jobs:
            job_copy = job.copy()
            job_copy['queue'] = queue
            process = Process(target=expand_call, args=(job_copy,))
            processes.append(process)

        # Additional variables for controlling the number of concurrent processes
        processes_alive = 0  # Counter of alive processes
        process_idx = 0  # Index of the next process to start
        outputs = []  # Stores outputs as they arrive
        time0 = time.time()

        # Start processes
        while True:
            # Start processes
            while processes_alive < num_threads and process_idx < len(processes):
                processes[process_idx].start()
                process_idx += 1
                processes_alive += 1

            # Get output
            if not queue.empty():
                status, result = queue.get()
                if status == 'success':
                    if result is not None:
                        outputs.append(result)
                else:  # status == 'error'
                    print(f"Error in job: {result}")
                report_progress(len(outputs), len(processes), time0, task)

            # Check if any process is done
            processes_alive = sum([process.is_alive() for process in processes])

            # Exit if all processes are done
            if processes_alive == 0 and process_idx == len(processes):
                break

            # Sleep for a short time before checking again
            time.sleep(0.1)

    # Handle outputs
    if len(outputs) > 0:
        if isinstance(outputs[0], pd.DataFrame):
            # For DataFrames, preserve job structure
            return outputs
        elif isinstance(outputs[0], pd.Series):
            # For Series, concatenate all outputs
            return pd.concat(outputs)
        # For non-pandas outputs, always preserve job structure
        return outputs
    # Return empty DataFrame with default columns for empty outputs
    return pd.DataFrame(columns=['t1', 'pt', 'sl'])


def process_jobs_(jobs, task=None, num_threads=cpu_count()):
    """
    Run in parallel. jobs must contain a 'func' callback, for expand_call.
    This is a variant of process_jobs that returns both outputs and failed jobs.

    :param jobs: (list) Jobs (each job is a dict)
    :param task: (str) Task description
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :return: (tuple) The first element is the outputs list, the second element is the failed jobs list
    """
    if task is None:
        task = jobs[0]['func'].__name__
    queue = Queue()
    processes = []
    for job in jobs:
        job_copy = job.copy()
        job_copy['queue'] = queue
        process = Process(target=expand_call, args=(job_copy,))
        processes.append(process)

    # Additional variables for controlling the number of concurrent processes
    processes_alive = 0  # Counter of alive processes
    process_idx = 0  # Index of the next process to start
    outputs = []  # Stores outputs as they arrive
    failed_indices = set()  # Stores indices of failed jobs
    time0 = time.time()

    # Start processes
    while True:
        # Start processes
        while processes_alive < num_threads and process_idx < len(processes):
            processes[process_idx].start()
            process_idx += 1
            processes_alive += 1

        # Get output
        if not queue.empty():
            try:
                status, result = queue.get()
                job_idx = len(outputs) + len(failed_indices)  # Current job index
                if status == 'success':
                    if result is not None:
                        outputs.append(result)
                    else:
                        failed_indices.add(job_idx)
                else:  # status == 'error'
                    failed_indices.add(job_idx)
                    print(f"Error in job: {result}")
                report_progress(job_idx + 1, len(processes), time0, task)
            except Exception as e:
                # If we can't get the result, mark the current job as failed
                failed_indices.add(len(outputs) + len(failed_indices))
                print(f"Error getting result: {str(e)}")

        # Check if any process is done
        processes_alive = sum([process.is_alive() for process in processes])

        # Exit if all processes are done
        if processes_alive == 0 and process_idx == len(processes):
            # Ensure all remaining jobs are marked as failed
            while len(outputs) + len(failed_indices) < len(jobs):
                failed_indices.add(len(outputs) + len(failed_indices))
            break

        # Sleep for a short time before checking again
        time.sleep(0.1)

    # Convert failed indices to failed jobs list
    failed = [jobs[i] for i in sorted(failed_indices)]

    # Handle outputs
    if len(outputs) > 0:
        if isinstance(outputs[0], pd.DataFrame):
            outputs = pd.concat(outputs, axis=0)
        elif isinstance(outputs[0], pd.Series):
            # For Series, preserve job structure (one Series per successful job)
            outputs = outputs  # Keep all successful outputs
    elif len(jobs) > 0:
        # If no outputs but jobs exist, create empty Series for successful jobs
        outputs = [pd.Series(index=jobs[i]['molecule']) for i in range(len(jobs)) if i not in failed_indices]
    
    return outputs, failed


def process_jobs_in_batches(jobs, task=None, num_threads=cpu_count(), batch_size=1, verbose=False):
    """
    Run in parallel. jobs must contain a 'func' callback, for expand_call. The jobs will be processed in batches.

    :param jobs: (list) Jobs (each job is a dict)
    :param task: (str) Task description
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :param batch_size: (int) Number of jobs to process in each batch
    :param verbose: (bool) Flag to report progress on jobs or not
    """
    results = []  # All results
    failed = []  # Failed jobs

    if task is None:
        task = jobs[0]['func'].__name__

    # Process jobs in batches
    for batch_num, i in enumerate(range(0, len(jobs), batch_size)):
        batch_jobs = jobs[i:i + batch_size]
        if verbose:
            print(f"Processing batch {batch_num + 1}")

        queue = Queue()
        processes = []
        for job in batch_jobs:
            job_copy = job.copy()
            job_copy['queue'] = queue
            process = Process(target=expand_call, args=(job_copy,))
            processes.append(process)

        # Additional variables for controlling the number of concurrent processes
        processes_alive = 0  # Counter of alive processes
        process_idx = 0  # Process index
        batch_outputs = []  # Stores outputs for current batch
        time0 = time.time()

        # Start processes
        while True:
            # Start processes
            while processes_alive < num_threads and process_idx < len(processes):
                processes[process_idx].start()
                process_idx += 1
                processes_alive += 1

            # Get output
            if not queue.empty():
                status, result = queue.get()
                if status == 'success':
                    if result is not None:
                        batch_outputs.append(result)
                    else:
                        failed.append(batch_jobs[len(batch_outputs)])
                else:  # status == 'error'
                    failed.append(batch_jobs[len(batch_outputs)])
                    print(f"Error in job: {result}")

                if verbose:
                    report_progress(len(batch_outputs), len(processes), time0, task)

            # Check if any process is done
            processes_alive = sum([process.is_alive() for process in processes])

            # Exit if all processes are done
            if processes_alive == 0 and process_idx == len(processes):
                break

            # Sleep for a short time before checking again
            time.sleep(0.1)

        # Handle batch outputs
        if len(batch_outputs) > 0:
            if isinstance(batch_outputs[0], pd.DataFrame):
                results.append(pd.concat(batch_outputs, axis=0))
            elif isinstance(batch_outputs[0], pd.Series):
                results.append(pd.concat(batch_outputs))
            else:
                results.extend(batch_outputs)

    # Combine all results
    if len(results) > 0:
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, axis=0), failed
        elif isinstance(results[0], pd.Series):
            return pd.concat(results), failed
    return results, failed


def lin_parts(num_atoms, num_threads):
    """
    Partition of atoms with a linear logic.

    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of threads that will be used in parallel (one processor per thread)
    :return: (np.array) Partition of atoms
    """
    # Partition of atoms with a linear logic
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nested_parts(num_atoms, num_threads, upper_triangle=False):
    """
    Partition of atoms with a nested logic.

    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of threads that will be used in parallel (one processor per thread)
    :param upper_triangle: (bool) Flag to partition an upper triangular matrix
    :return: (np.array) Partition of atoms
    """
    # Partition of atoms with a nested logic
    parts = [0]
    num_threads_ = min(num_threads, num_atoms)

    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + part ** 0.5) / 2.0
        parts.append(part)

    parts = np.round(parts).astype(int)

    if upper_triangle:  # Partition for an upper triangular matrix
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)

    return parts


def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, **kargs):
    """
    Parallelize jobs, return a dataframe or series.
    Example: df1=mp_pandas_obj(func,('molecule',df0.index),24,**kwds)

    :param func: (function) A function to be parallelized
    :param pd_obj: (tuple) Element 0: The name of the argument used to pass the molecule
                          Element 1: A pandas object
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :param mp_batches: (int) Number of batches
    :param lin_mols: (bool) Tells if the method should use linear or nested partitioning
    :param kargs: (var args) Keyword arguments to be passed to the function
    :return: (pd.DataFrame) Returns a pandas object with the results
    """
    # Handle empty input case
    if len(pd_obj[1]) == 0:
        return pd.Series(dtype=float)

    # Ensure at least one batch and valid number of threads
    mp_batches = max(1, min(mp_batches, len(pd_obj[1])))
    num_threads = min(num_threads, len(pd_obj[1]))

    # Create parts for the entire dataset
    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads)

    # Create jobs based on parts
    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)

    if not jobs:  # If no jobs were created
        # Handle single item case
        if len(pd_obj[1]) == 1:
            job = {pd_obj[0]: pd_obj[1], 'func': func}
            job.update(kargs)
            jobs.append(job)
        else:
            return pd.Series(dtype=float)

    # Process all jobs at once
    result = process_jobs(jobs, num_threads=num_threads)
    
    # Handle different types of results
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return result
    elif isinstance(result, list):
        if all(isinstance(x, pd.DataFrame) for x in result):
            return pd.concat(result, axis=0)
        elif all(isinstance(x, pd.Series) for x in result):
            # For Series results, concatenate and ensure index is preserved
            concatenated = pd.concat(result)
            # If the original index was split across jobs, ensure we get all values
            if len(concatenated) < len(pd_obj[1]):
                return pd.Series(index=pd_obj[1])
            return concatenated
        # For non-pandas outputs, preserve original structure
        return result
    
    return pd.Series(dtype=float)