"""
Contains the logic from chapter 20 on multiprocessing and vectorization.
+ 
async
"""

import sys
import time
import datetime as dt
import asyncio
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
from functools import partial
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Snippet 20.5 (page 306), the lin_parts function
def lin_parts(num_atoms, num_threads):
    """
    Advances in Financial Machine Learning, Snippet 20.5, page 306.

    The lin_parts function

    The simplest way to form molecules is to partition a list of atoms in subsets of equal size,
    where the number of subsets is the minimum between the number of processors and the number
    of atoms. For N subsets we need to find the N+1 indices that enclose the partitions.
    This logic is demonstrated in Snippet 20.5.

    This function partitions a list of atoms in subsets (molecules) of equal size.
    An atom is a set of indivisible set of tasks.

    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of processors
    :return: (np.array) Partition of atoms
    """
    # Partition of atoms with a single loop
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


# Snippet 20.6 (page 308), The nested_parts function
def nested_parts(num_atoms, num_threads, upper_triangle=False):
    """
    Advances in Financial Machine Learning, Snippet 20.6, page 308.

    The nested_parts function

    This function enables parallelization of nested loops.
    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of processors
    :param upper_triangle: (bool) Flag to order atoms as an upper triangular matrix (including the main diagonal)
    :return: (np.array) Partition of atoms
    """
    # Partition of atoms with an inner loop
    parts = [0]
    num_threads_ = min(num_threads, num_atoms)

    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + part ** 0.5) / 2.0
        parts.append(part)

    parts = np.round(parts).astype(int)

    if upper_triangle:  # The first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)

    return parts


# Snippet 20.7 (page 310), The mpPandasObj, used at various points in the book
def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, verbose=True, **kargs):
    """
    Advances in Financial Machine Learning, Snippet 20.7, page 310.

    The mpPandasObj, used at various points in the book

    Parallelize jobs, return a dataframe or series.
    Example: df1=mp_pandas_obj(func,('molecule',df0.index),24,**kwds)

    First, atoms are grouped into molecules, using linParts (equal number of atoms per molecule)
    or nestedParts (atoms distributed in a lower-triangular structure). When mpBatches is greater
    than 1, there will be more molecules than cores. Suppose that we divide a task into 10 molecules,
    where molecule 1 takes twice as long as the rest. If we run this process in 10 cores, 9 of the
    cores will be idle half of the runtime, waiting for the first core to process molecule 1.
    Alternatively, we could set mpBatches=10 so as to divide that task in 100 molecules. In doing so,
    every core will receive equal workload, even though the first 10 molecules take as much time as the
    next 20 molecules. In this example, the run with mpBatches=10 will take half of the time consumed by
    mpBatches=1.

    Second, we form a list of jobs. A job is a dictionary containing all the information needed to process
    a molecule, that is, the callback function, its keyword arguments, and the subset of atoms that form
    the molecule.

    Third, we will process the jobs sequentially if numThreads==1 (see Snippet 20.8), and in parallel
    otherwise (see Section 20.5.2). The reason that we want the option to run jobs sequentially is for
    debugging purposes. It is not easy to catch a bug when programs are run in multiple processors.
    Once the code is debugged, we will want to use numThreads>1.

    Fourth, we stitch together the output from every molecule into a single list, series, or dataframe.

    :param func: (function) A callback function, which will be executed in parallel
    :param pd_obj: (tuple) Element 0: The name of the argument used to pass molecules to the callback function
                    Element 1: A list of indivisible tasks (atoms), which will be grouped into molecules
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :param mp_batches: (int) Number of parallel batches (jobs per core)
    :param lin_mols: (bool) Tells if the method should use linear or nested partitioning
    :param verbose: (bool) Flag to report progress on asynch jobs
    :param kargs: (var args) Keyword arguments needed by func
    :return: (pd.DataFrame) of results
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
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads, verbose=verbose)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series(dtype='float64')
    else:
        return out

    for i in out:
        df0 = df0.append(i)

    df0 = df0.sort_index()
    return df0


# Snippet 20.8, pg 311, Single thread execution, for debugging
def process_jobs_(jobs):
    """
    Advances in Financial Machine Learning, Snippet 20.8, page 311.

    Single thread execution, for debugging

    Run jobs sequentially, for debugging

    :param jobs: (list) Jobs (molecules)
    :return: (list) Results of jobs
    """
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)

    return out


# Snippet 20.10 Passing the job (molecule) to the callback function
def expand_call(kargs):
    """
    Advances in Financial Machine Learning, Snippet 20.10.

    Passing the job (molecule) to the callback function

    Expand the arguments of a callback function, kargs['func']

    :param kargs: Job (molecule)
    :return: Result of a job
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


# Snippet 20.9.1, pg 312, Example of Asynchronous call to pythons multiprocessing library
def report_progress(job_num, num_jobs, time0, task):
    """
    Advances in Financial Machine Learning, Snippet 20.9.1, pg 312.

    Example of Asynchronous call to pythons multiprocessing library

    :param job_num: (int) Number of current job
    :param num_jobs: (int) Total number of jobs
    :param time0: (time) Start time
    :param task: (str) Task description
    :return: (None)
    """
    # Report progress as asynch jobs are completed
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))

    msg = time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

    if job_num < num_jobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')


# Snippet 20.9.2, pg 312, Example of Asynchronous call to pythons multiprocessing library
def process_jobs(jobs, task=None, num_threads=24, verbose=True):
    """
    Advances in Financial Machine Learning, Snippet 20.9.2, page 312.

    Example of Asynchronous call to pythons multiprocessing library

    Run in parallel. jobs must contain a 'func' callback, for expand_call

    :param jobs: (list) Jobs (molecules)
    :param task: (str) Task description
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (None)
    """

    if task is None:
        task = jobs[0]['func'].__name__

    pool = mp.Pool(processes=num_threads)
    outputs = pool.imap_unordered(expand_call, jobs)
    out = []
    time0 = time.time()

    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        if verbose:
            report_progress(i, len(jobs), time0, task)

    pool.close()
    pool.join()  # This is needed to prevent memory leaks
    return out

class ProgressState:
    """Thread-safe state management for progress reporting"""
    last_report = 0
    lock = asyncio.Lock()

async def report_progress_async(job_num: int, num_jobs: int, time0: float, 
                              task: str, job_info: str = None, 
                              min_interval: float = 1.0) -> None:
    """Thread-safe async progress reporting with rate limiting"""
    try:
        current_time = time.time()
        async with ProgressState.lock:
            if (current_time - ProgressState.last_report) < min_interval:
                return

            elapsed = (current_time - time0) / 60.0
            completed = job_num / num_jobs
            remaining = elapsed * (1 / completed - 1) if completed > 0 else 0
            
            progress_msg = (f"{round(completed * 100, 2)}% {task} done after "
                          f"{round(elapsed, 2)} minutes. "
                          f"Remaining {round(remaining, 2)} minutes.")
            
            if job_info:
                progress_msg += f" Current job: {job_info}"
                
            logger.info(progress_msg)
            ProgressState.last_report = current_time
    except Exception as e:
        logger.error(f"Error in progress reporting: {str(e)}")

async def safe_expand_call_async(job: dict, max_retries: int = 3) -> Any:
    """Safely execute job with retries and proper error handling"""
    func = job.pop('func')
    retries = 0
    
    try:
        while retries < max_retries:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(**job)
                return await asyncio.to_thread(func, **job)
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    logger.error(f"Error executing job after {max_retries} retries: {e}")
                    return None
                logger.warning(f"Retry {retries}/{max_retries} after error: {e}")
                await asyncio.sleep(1)  # Backoff before retry
    finally:
        job['func'] = func

async def process_jobs_async(jobs: list[dict], task: str = None, 
                           num_threads: int = None, verbose: bool = True,
                           batch_size: int = 100) -> list:
    """Enhanced async version with improved error handling and batching"""
    assert jobs and isinstance(jobs, list), "jobs must be a non-empty list"
    assert all(callable(job['func']) for job in jobs), "all jobs must have callable 'func'"
    
    if task is None:
        task = jobs[0]['func'].__name__
    
    if num_threads is None:
        num_threads = min(mp.cpu_count(), len(jobs))
    
    assert num_threads > 0, "num_threads must be greater than 0"

    async def process_batch(batch_jobs: list, batch_num: int) -> tuple[list, list]:
        results = []
        failed = []
        
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(safe_expand_call_async(job.copy())) 
                    for job in batch_jobs]
            
            for i, task in enumerate(tasks, 1):
                try:
                    result = await task
                    if result is not None:
                        results.append(result)
                    else:
                        failed.append(batch_jobs[i-1])
                    
                    if verbose and (i % 10 == 0 or i == len(tasks)):
                        job_info = f"Batch {batch_num}, Job {i}/{len(tasks)}"
                        await report_progress_async(
                            i + (batch_num * batch_size), 
                            len(jobs), time0, task, job_info
                        )
                except* asyncio.CancelledError:
                    logger.warning(f"Job {i} in batch {batch_num} cancelled")
                    failed.append(batch_jobs[i-1])
                except* Exception as e:
                    logger.error(f"Error in job {i}, batch {batch_num}: {e.exceptions[0]}")
                    failed.append(batch_jobs[i-1])
        
        return results, failed

    async def run_jobs():
        time0 = time.time()
        all_results = []
        all_failed = []
        
        # Process jobs in batches
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            results, failed = await process_batch(batch, i // batch_size)
            all_results.extend(results)
            all_failed.extend(failed)
        
        if all_failed:
            logger.warning(f"Failed jobs: {len(all_failed)}")
            # Optionally implement retry logic here
        
        return all_results

    return await run_jobs()
