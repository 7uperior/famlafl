import asyncio
import logging
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import aiofiles
import numpy as np
import pandas as pd
from mlfinlab.data_structures import (
    get_const_dollar_run_bars,
    get_const_tick_run_bars,
    get_const_volume_run_bars,
    get_dollar_bars,
    # Imbalance bars
    get_ema_dollar_imbalance_bars,
    # Run bars
    get_ema_tick_imbalance_bars,
    get_ema_volume_imbalance_bars,
    get_tick_bars,
    # Standard bars
    get_time_bars,
    get_volume_bars,
)
from mlfinlab.util.multi_asyn_process import process_jobs, report_progress

from cmf.scripts.import_data import datasets

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Create logs directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"bars_creation_agg_trades_{timestamp}.log"

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will also print to console
    ]
)
logger = logging.getLogger(__name__)

def prepare_data_for_standard_bars(df: pd.DataFrame, symbol: str, output_dir: Path) -> str:
    """Prepare data for standard bars"""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_file = output_dir / f"{symbol}_standard_bars_input.csv"
    df.to_csv(temp_file, index=False)
    return str(temp_file)

def prepare_data_for_imbalance_bars(df: pd.DataFrame, symbol: str, output_dir: Path) -> str:
    """Prepare data for imbalance bars"""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_file = output_dir / f"{symbol}_imbalance_bars_input.csv"
    df.to_csv(temp_file, index=False)
    return str(temp_file)

def prepare_data_for_run_bars(df: pd.DataFrame, symbol: str, output_dir: Path) -> str:
    """Prepare data for run bars"""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_file = output_dir / f"{symbol}_run_bars_input.csv"
    df.to_csv(temp_file, index=False)
    return str(temp_file)

async def save_bars_async(bars_df: pd.DataFrame, thresholds_df: pd.DataFrame,
                         bars_path: Path, thresholds_path: Path):
    """Asynchronously save bars and thresholds to files"""
    try:
        async with aiofiles.open(bars_path, 'w') as f:
            await f.write(bars_df.to_csv(index=False))
        if thresholds_df is not None:
            async with aiofiles.open(thresholds_path, 'w') as f:
                await f.write(thresholds_df.to_csv(index=False))
        logger.info(f"Successfully saved bars to {bars_path}")
    except Exception as e:
        logger.error(f"Error saving bars: {str(e)}")
        raise

def process_jobs_async(jobs, task=None, num_threads=24, verbose=True):
    """Modified version of MLFinLab's process_jobs using ThreadPoolExecutor"""
    if task is None:
        task = jobs[0]['func'].__name__

    async def process_job(job):
        try:
            if job['bar_type'] == 'standard_bars':
                bars_df = job['func']()
                await save_bars_async(bars_df, None,
                                    job['output_path'], None)
            else:
                bars_df, thresholds_df = job['func'](
                    analyse_thresholds=job.get('analyse_thresholds', False)
                )
                await save_bars_async(bars_df, thresholds_df,
                                    job['bars_path'],
                                    job['thresholds_path'])
            return f"Successfully processed {job['name']}"
        except Exception as e:
            logger.error(f"Error processing {job['name']}: {str(e)}")
            return f"Error processing {job['name']}: {str(e)}"

    async def run_jobs():
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            loop = asyncio.get_event_loop()
            tasks = [loop.create_task(process_job(job)) for job in jobs]
            out = []
            time0 = time.time()

            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                result = await task
                out.append(result)
                if verbose:
                    report_progress(i, len(jobs), time0, task)
            return out

    return asyncio.run(run_jobs())

def create_and_save_bars(data: pd.DataFrame, symbol: str, base_path: Path):
    """Create different types of bars and save them to respective folders"""
    logger.info(f"Starting bar creation for {symbol}")

    # Create temporary directory for input files
    temp_dir = base_path / "temp_inputs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Calculate thresholds using the prepared data
        tick_threshold = 1000
        volume_threshold = np.percentile(data['volume'], 99)
        dollar_threshold = np.percentile(data['price'] * data['volume'], 99)

        logger.info(f"Thresholds for {symbol}: tick={tick_threshold}, "
                   f"volume={int(volume_threshold)}, dollar={int(dollar_threshold)}")

        # Prepare data for each bar type
        standard_file = prepare_data_for_standard_bars(data, symbol, temp_dir)
        imbalance_file = prepare_data_for_imbalance_bars(data, symbol, temp_dir)
        run_file = prepare_data_for_run_bars(data, symbol, temp_dir)

        # Standard bars configurations
        standard_bars_configs = {
            'time_bars': [
                ('1min', lambda: get_time_bars(standard_file, resolution='MIN', num_units=1)),
                ('5min', lambda: get_time_bars(standard_file, resolution='MIN', num_units=5)),
                ('15min', lambda: get_time_bars(standard_file, resolution='MIN', num_units=15)),
                ('1H', lambda: get_time_bars(standard_file, resolution='H', num_units=1))
            ],
            'tick_bars': [
                ('default', lambda: get_tick_bars(standard_file, threshold=tick_threshold))
            ],
            'volume_bars': [
                ('default', lambda: get_volume_bars(standard_file, threshold=volume_threshold))
            ],
            'dollar_bars': [
                ('default', lambda: get_dollar_bars(standard_file, threshold=dollar_threshold))
            ]
        }

        # Imbalance bars configurations
        imbalance_bars_configs = {
            'ema': {
                'dollar': [
                    ('default', lambda: get_ema_dollar_imbalance_bars(
                        imbalance_file,
                        expected_imbalance_window=10000,
                        exp_num_ticks_init=1000,
                        batch_size=2e7
                    ))
                ],
                'volume': [
                    ('default', lambda: get_ema_volume_imbalance_bars(
                        imbalance_file,
                        expected_imbalance_window=10000,
                        exp_num_ticks_init=1000,
                        batch_size=2e7
                    ))
                ],
                'tick': [
                    ('default', lambda: get_ema_tick_imbalance_bars(
                        imbalance_file,
                        expected_imbalance_window=10000,
                        exp_num_ticks_init=1000,
                        batch_size=2e7
                    ))
                ]
            }
        }

        # Run bars configurations (based on test files lines 371-426)
        run_bars_configs = {
            'const': {
                'dollar': [
                    ('default', lambda: get_const_dollar_run_bars(
                        run_file,
                        exp_num_ticks_init=1000,
                        expected_imbalance_window=10000,
                        num_prev_bars=3,
                        batch_size=2e7
                    ))
                ],
                'volume': [
                    ('default', lambda: get_const_volume_run_bars(
                        run_file,
                        exp_num_ticks_init=1000,
                        expected_imbalance_window=10000,
                        num_prev_bars=3,
                        batch_size=2e7
                    ))
                ],
                'tick': [
                    ('default', lambda: get_const_tick_run_bars(
                        run_file,
                        exp_num_ticks_init=1000,
                        expected_imbalance_window=10000,
                        num_prev_bars=3,
                        batch_size=2e7
                    ))
                ]
            }
        }

        # Create list of all bar configurations to process
        bar_configs = []

        # Add standard bars configs
        for bar_subtype, bar_configs_list in standard_bars_configs.items():
            for name, bar_func in bar_configs_list:
                output_path = base_path / 'standard_bars' / bar_subtype / name / f"{symbol}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                bar_configs.append({
                    'func': bar_func,
                    'output_path': output_path,
                    'bar_type': 'standard_bars',
                    'bar_subtype': bar_subtype,
                    'name': name
                })

        # Add imbalance and run bars configs
        for bar_type, methods in [('imbalance_bars', imbalance_bars_configs),
                                ('run_bars', run_bars_configs)]:
            for method, subtypes in methods.items():
                for bar_subtype, bar_configs_list in subtypes.items():
                    for name, bar_func in bar_configs_list:
                        bars_path = base_path / bar_type / method / bar_subtype / name / f"{symbol}.csv"
                        thresholds_path = base_path / bar_type / method / bar_subtype / name / f"{symbol}_thresholds.csv"
                        bars_path.parent.mkdir(parents=True, exist_ok=True)
                        bar_configs.append({
                            'func': bar_func,
                            'bars_path': bars_path,
                            'thresholds_path': thresholds_path,
                            'bar_type': bar_type,
                            'method': method,
                            'bar_subtype': bar_subtype,
                            'name': name,
                            'analyse_thresholds': True
                        })

        # Process jobs using ThreadPoolExecutor and async IO
        num_threads = mp.cpu_count()
        process_jobs_async(bar_configs, task=f'create_bars_{symbol}',
                         num_threads=num_threads, verbose=True)

    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        raise
    finally:
        # Robust cleanup using shutil
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory for {symbol}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")

def prepare_data_for_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Prepare data in the format required by MLFinLab"""
    logger.info(f"Preparing data for {symbol}")
    try:
        # Log original data stats
        logger.info(f"Original data columns: {df.columns}")
        logger.info(f"NA values in original data:\n{df[['bid_mean_price', 'ask_mean_price', 'bid_amount', 'ask_amount']].isna().sum()}")

        # Fill NA values with 0 for bid/ask data
        df['bid_mean_price'] = df['bid_mean_price'].fillna(0)
        df['ask_mean_price'] = df['ask_mean_price'].fillna(0)
        df['bid_amount'] = df['bid_amount'].fillna(0)
        df['ask_amount'] = df['ask_amount'].fillna(0)

        # Create basic format required by MLFinLab
        bars_df = pd.DataFrame({
            'date_time': pd.to_datetime(df.index, unit='ns'),
            'price': (df['bid_mean_price'] + df['ask_mean_price']) / 2,  # Mid price
            'volume': df['bid_amount'] + df['ask_amount']  # Total volume
        })

        # Drop rows where price is 0 (means both bid and ask were NA)
        original_shape = bars_df.shape
        bars_df = bars_df[bars_df['price'] != 0]
        removed_rows = original_shape[0] - bars_df.shape[0]

        logger.info(f"Removed {removed_rows} rows where both bid and ask were missing ({removed_rows/original_shape[0]*100:.2f}%)")

        if bars_df.empty:
            raise ValueError(f"No valid data remaining for {symbol} after removing rows with all NaN values")

        logger.info(f"Successfully prepared data for {symbol}. Final shape: {bars_df.shape}")
        return bars_df
    except Exception as e:
        logger.error(f"Error preparing data for {symbol}: {str(e)}")
        raise


# Create base directory for bars
BARS_DIR = Path("cmf/data/bars")

def process_symbol_job(symbol, base_path):
    """Process a single symbol job"""
    try:
        logger.info(f"Processing symbol: {symbol}")
        symbol_data = datasets['agg_trades'][
            datasets['agg_trades']['instrument'] == symbol
        ].copy()

        logger.info(f"Retrieved data for {symbol}. Shape: {symbol_data.shape}")

        prepared_data = prepare_data_for_bars(symbol_data, symbol)
        create_and_save_bars(prepared_data, symbol, base_path)
        logger.info(f"Completed processing for {symbol}")
        return f"Successfully processed {symbol}"
    except Exception as e:
        logger.error(f"Failed to process {symbol}: {str(e)}", exc_info=True)
        return f"Failed to process {symbol}: {str(e)}"

if __name__ == '__main__':
    logger.info("Starting bar creation process")

    # Create jobs list following MLFinLab pattern
    jobs = []
    for symbol in ['ETHUSDT', 'BTCUSDT']:
        job = {
            'func': process_symbol_job,
            'symbol': symbol,
            'base_path': BARS_DIR
        }
        jobs.append(job)

    num_threads = mp.cpu_count()
    process_jobs(jobs, task='create_bars', num_threads=num_threads, verbose=True)

    logger.info("Bar creation process completed")

