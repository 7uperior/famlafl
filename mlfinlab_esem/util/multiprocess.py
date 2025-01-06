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
    out = func(**kwargs)
    return out


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
    """
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
                continue
    else:
        if task is None:
            task = jobs[0]['func'].__name__
        queue = Queue()
        processes = []
        for job in jobs:
            job['queue'] = queue
            process = Process(target=expand_call, args=(job,))
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
                result = queue.get()
                if result is not None:
                    outputs.append(result)
                report_progress(len(outputs), len(processes), time0, task)

            # Check if any process is done
            processes_alive = sum([process.is_alive() for process in processes])

            # Exit if all processes are done
            if processes_alive == 0 and process_idx == len(processes):
                break

            # Sleep for a short time before checking again
            time.sleep(0.1)

    # Case of output being a dataframe
    if len(outputs) > 0:
        if isinstance(outputs[0], pd.DataFrame):
            return pd.concat(outputs, axis=0)
        return outputs
    # Return empty DataFrame with correct columns and index from the job
    if len(jobs) > 0 and 'molecule' in jobs[0]:
        return pd.DataFrame(columns=['t1', 'pt', 'sl'], index=jobs[0]['molecule'])
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
        job['queue'] = queue
        process = Process(target=expand_call, args=(job,))
        processes.append(process)

    # Additional variables for controlling the number of concurrent processes
    processes_alive = 0  # Counter of alive processes
    process_idx = 0  # Index of the next process to start
    outputs = []  # Stores outputs as they arrive
    failed = []  # Stores failed jobs
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
            result = queue.get()
            if result is not None:
                outputs.append(result)
            else:
                failed.append(jobs[len(outputs)])
            report_progress(len(outputs), len(processes), time0, task)

        # Check if any process is done
        processes_alive = sum([process.is_alive() for process in processes])

        # Exit if all processes are done
        if processes_alive == 0 and process_idx == len(processes):
            break

        # Sleep for a short time before checking again
        time.sleep(0.1)

    # Case of output being a dataframe
    if len(outputs) > 0:
        if isinstance(outputs[0], pd.DataFrame):
            outputs = pd.concat(outputs, axis=0)
    return outputs, failed


def process_jobs_in_batches(jobs, task=None, num_threads=cpu_count(), batch_size=1, verbose=True):
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
            job['queue'] = queue
            process = Process(target=expand_call, args=(job,))
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
                result = queue.get()
                if result is not None:
                    batch_outputs.append(result)
                else:
                    failed.append(batch_jobs[len(batch_outputs)])

                if verbose:
                    report_progress(len(batch_outputs), len(processes), time0, task)

            # Check if any process is done
            processes_alive = sum([process.is_alive() for process in processes])

            # Exit if all processes are done
            if processes_alive == 0 and process_idx == len(processes):
                break

            # Sleep for a short time before checking again
            time.sleep(0.1)

        # Collect results from batch
        if len(batch_outputs) > 0:
            if isinstance(batch_outputs[0], pd.DataFrame):
                results.append(pd.concat(batch_outputs, axis=0))
            else:
                results.extend(batch_outputs)

    # Combine all results
    if len(results) > 0:
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, axis=0), failed
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

    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)

    if isinstance(out, (pd.Series, pd.DataFrame)):
        return out
    else:
        return pd.concat(out)
