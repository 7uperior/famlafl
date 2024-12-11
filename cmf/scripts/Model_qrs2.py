import os
from collections import Counter
from datetime import datetime
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

# Constants for time conversions
NANOSECOND = 1
MICROSECOND = 1000
MILLISECOND = 1000000
SECOND = 1000000000

# Constants and environment setup
N_THREADS = 16
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)

# Optimized CatBoost parameters
CATBOOST_PARAMS = {
    "use_best_model": True,
    "eval_metric": "Logloss",
    "iterations": 6000,
    "thread_count": N_THREADS,  # Match CPU cores
    "loss_function": "Logloss",
    "l2_leaf_reg": 50,
    "task_type": "CPU",
    "depth": 8,
    "early_stopping_rounds": 50,
    "learning_rate": 0.01,
    "max_ctr_complexity": 2,  # Limit categorical feature combinations
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,  # Add subsampling for better generalization
    "verbose": False
}

# Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features




#Entropy calculation module (Shannon, Lempel-Ziv, Plug-In, Konto)
def get_shannon_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, page 263-264.

    Get Shannon entropy from message

    :param message: (str) Encoded message
    :return: (float) Shannon entropy
    """
    exr = {}
    entropy = 0
    for each in message:
        try:
            exr[each] += 1
        except KeyError:
            exr[each] = 1
    textlen = len(message)
    for value in exr.values():
        freq = 1.0 * value / textlen
        entropy += freq * math.log(freq) / math.log(2)
    entropy *= -1
    return entropy


def get_lempel_ziv_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.2, page 266.

    Get Lempel-Ziv entropy estimate

    :param message: (str) Encoded message
    :return: (float) Lempel-Ziv entropy
    """
    i, lib = 1, [message[0]]
    while i < len(message):
        for j in range(i, len(message)):
            message_ = message[i:j + 1]
            if message_ not in lib:
                lib.append(message_)
                break
        i = j + 1
    return len(lib) / len(message)


def _prob_mass_function(message: str, word_length: int) -> dict:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 266.

    Compute probability mass function for a one-dim discete rv

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (dict) Dict of pmf for each word from message
    """
    lib = {}
    if not isinstance(message, str):
        message = ''.join(map(str, message))
    for i in range(word_length, len(message)):
        message_ = message[i - word_length:i]
        if message_ not in lib:
            lib[message_] = [i - word_length]
        else:
            lib[message_] = lib[message_] + [i - word_length]
    pmf = float(len(message) - word_length)
    pmf = {i: len(lib[i]) / pmf for i in lib}
    return pmf


def get_plug_in_entropy(message: str, word_length: int = None) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 265.

    Get Plug-in entropy estimator

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (float) Plug-in entropy
    """
    if word_length is None:
        word_length = 1
    pmf = _prob_mass_function(message, word_length)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length
    return out


@njit()
def _match_length(message: str, start_index: int, window: int) -> Union[int, str]:    # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 18.3, page 267.

    Function That Computes the Length of the Longest Match

    :param message: (str or array) Encoded message
    :param start_index: (int) Start index for search
    :param window: (int) Window length
    :return: (int, str) Match length and matched string
    """
    # Maximum matched length+1, with overlap.
    sub_str = ''
    for length in range(window):
        msg1 = message[start_index: start_index + length + 1]
        for j in range(start_index - window, start_index):
            msg0 = message[j: j + length + 1]
            if len(msg1) != len(msg0):
                continue
            if msg1 == msg0:
                sub_str = msg1
                break  # Search for higher l.
    return len(sub_str) + 1, sub_str  # Matched length + 1


def get_konto_entropy(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Implementations of Algorithms Discussed in Gao et al.[2008]

    Get Kontoyiannis entropy

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """
    out = {
        'h': 0,
        'r': 0,
        'num': 0,
        'sum': 0,
        'sub_str': []
    }
    if window <= 0:
        points = range(1, len(message) // 2 + 1)
    else:
        window = min(window, len(message) // 2)
        points = range(window, len(message) - window + 1)
    for i in points:
        if window <= 0:
            length, msg_ = _match_length(message, i, i)
            out['sum'] += np.log2(i + 1) / length  # To avoid Doeblin condition
        else:
            length, msg_ = _match_length(message, i, window)
            out['sum'] += np.log2(window + 1) / length  # To avoid Doeblin condition
        out['sub_str'].append(msg_)
        out['num'] += 1
    try:
        out['h'] = out['sum'] / out['num']
    except ZeroDivisionError:
        out['h'] = 0
    out['r'] = 1 - out['h'] / (np.log2(len(message)) if np.log2(len(message)) > 0 else 1)  # Redundancy, 0<=r<=1
    return out['h']

#Various functions for message encoding (quantile)
def encode_tick_rule_array(tick_rule_array: list) -> str:
    """
    Encode array of tick signs (-1, 1, 0)

    :param tick_rule_array: (list) Tick rules
    :return: (str) Encoded message
    """
    message = ''
    for element in tick_rule_array:
        if element == 1:
            message += 'a'
        elif element == -1:
            message += 'b'
        elif element == 0:
            message += 'c'
        else:
            raise ValueError('Unknown value for tick rule: {}'.format(element))
    return message


def _get_ascii_table() -> list:
    """
    Get all ASCII symbols

    :return: (list) ASCII symbols
    """
    # ASCII table consists of 256 characters
    table = []
    for i in range(256):
        table.append(chr(i))
    return table


def quantile_mapping(array: list, num_letters: int = 26) -> dict:
    """
    Generate dictionary of quantile-letters based on values from array and dictionary length (num_letters).

    :param array: (list) Values to split on quantiles
    :param num_letters: (int) Number of letters(quantiles) to encode
    :return: (dict) Dict of quantile-symbol
    """
    encoding_dict = {}
    ascii_table = _get_ascii_table()
    alphabet = ascii_table[:num_letters]
    for quant, letter in zip(np.linspace(0.01, 1, len(alphabet)), alphabet):
        encoding_dict[np.quantile(array, quant)] = letter
    return encoding_dict


def sigma_mapping(array: list, step: float = 0.01) -> dict:
    """
    Generate dictionary of sigma encoded letters based on values from array and discretization step.

    :param array: (list) Values to split on quantiles
    :param step: (float) Discretization step (sigma)
    :return: (dict) Dict of value-symbol
    """
    i = 0
    ascii_table = _get_ascii_table()
    encoding_dict = {}
    encoding_steps = np.arange(min(array), max(array), step)
    for element in encoding_steps:
        try:
            encoding_dict[element] = ascii_table[i]
        except IndexError:
            raise ValueError(
                'Length of dictionary ceil((max(arr) - min(arr)) / step = {} is more than ASCII table lenght)'.format(
                    len(encoding_steps)))
        i += 1
    return encoding_dict


def _find_nearest(array: list, value: float) -> float:
    """
    Find the nearest element from array to value.

    :param array: (list) Values
    :param value: (float) Value for which the nearest element needs to be found
    :return: (float) The nearest to the value element in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _get_letter_from_encoding(value: float, encoding_dict: dict) -> str:
    """
    Get letter for float/int value from encoding dict.

    :param value: (float/int) Value to use
    :param encoding_dict: (dict) Used dictionary
    :return: (str) Letter from encoding dict
    """
    return encoding_dict[_find_nearest(list(encoding_dict.keys()), value)]


def encode_array(array: list, encoding_dict: dict) -> str:
    """
    Encode array with strings using encoding dict, in case of multiple occurrences of the minimum values,
    the indices corresponding to the first occurrence are returned

    :param array: (list) Values to encode
    :param encoding_dict: (dict) Dict of quantile-symbol
    :return: (str) Encoded message
    """
    message = ''
    for element in array:
        message += _get_letter_from_encoding(element, encoding_dict)
    return message



def get_roll_measure(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Roll's measure of effective spread."""
    price_diff = prices.diff()
    autocov = price_diff.rolling(window=window).apply(
        lambda x: np.cov(x[:-1], x[1:])[0, 1]
    )
    roll_measure = 2 * np.sqrt(-autocov)
    return roll_measure

def get_roll_impact(prices: pd.Series, volumes: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Roll's impact using prices and volumes."""
    roll = get_roll_measure(prices, window)
    avg_volume = volumes.rolling(window=window).mean()
    return roll * avg_volume

def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Estimate bid-ask spread using Corwin-Schultz method."""
    beta = (high / low).rolling(window=window).apply(lambda x: np.log(x).mean())
    spread = 2 * (np.exp(beta) - 1) / (1 + np.exp(beta))
    return spread

def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Bekker-Parkinson volatility estimator."""
    log_hl = np.log(high / low)
    vol = np.sqrt(log_hl.rolling(window=window).var() / (4 * np.log(2)))
    return vol

# Second Generation Features
def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Kyle's lambda using bar data."""
    returns = close.pct_change()
    return get_trades_based_kyle_lambda(returns, volume)

def get_bar_based_amihud_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Amihud's lambda using bar data."""
    returns = close.pct_change()
    return get_trades_based_amihud_lambda(returns, volume)

def get_bar_based_hasbrouck_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Hasbrouck's lambda using bar data."""
    returns = close.pct_change()
    return get_trades_based_hasbrouck_lambda(returns, volume)

def get_trades_based_kyle_lambda(price_diff: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Kyle's lambda using trade data."""
    cov_matrix = np.cov(price_diff, volume)
    if cov_matrix.size == 1:
        return pd.Series(0, index=price_diff.index)
    lambda_val = cov_matrix[0, 1] / np.var(volume)
    return pd.Series(lambda_val, index=price_diff.index)

def get_trades_based_amihud_lambda(returns: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Amihud's lambda using trade data."""
    volume = volume.replace(0, np.nan)
    return abs(returns) / volume

def get_trades_based_hasbrouck_lambda(returns: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Hasbrouck's lambda using trade data."""
    cov_matrix = np.cov(returns, volume)
    if cov_matrix.size == 1:
        return pd.Series(0, index=returns.index)
    lambda_val = cov_matrix[0, 1] / np.std(volume)
    return pd.Series(lambda_val, index=returns.index)

# Third Generation Features
def get_vpin(volume: pd.Series, buy_volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Volume-Synchronized Probability of Informed Trading."""
    total_volume = volume.rolling(window=window).sum()
    buy_volume_sum = buy_volume.rolling(window=window).sum()
    sell_volume_sum = total_volume - buy_volume_sum
    vpin = abs(buy_volume_sum - sell_volume_sum) / total_volume
    return vpin

# Misc Features
def get_avg_tick_size(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate average tick size."""
    tick_size = abs(prices.diff())
    return tick_size.rolling(window=window).mean()

def vwap(prices: pd.Series, volumes: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    return (prices * volumes).rolling(window=window).sum() / volumes.rolling(window=window).sum()

# Encoding Functions
def quantile_mapping(array: np.ndarray, num_letters: int = 10) -> Dict:
    """Create quantile-based mapping for encoding."""
    quantiles = np.linspace(0, 100, num_letters + 1)
    bins = np.percentile(array, quantiles)
    letters = [chr(i) for i in range(65, 65 + num_letters)]

    mapping = {}
    for i in range(len(array)):
        val = array[i]
        for j in range(len(bins) - 1):
            if bins[j] <= val <= bins[j + 1]:
                mapping[val] = letters[j]
                break
    return mapping

def sigma_mapping(array: np.ndarray, step_size: float = 0.01) -> Dict:
    """Create sigma-based mapping for encoding."""
    mean = np.mean(array)
    std = np.std(array)

    mapping = {}
    for val in array:
        sigma_dist = (val - mean) / std
        letter_index = int(sigma_dist / step_size)
        mapping[val] = chr(65 + abs(letter_index) % 26)
    return mapping

def encode_array(array: np.ndarray, mapping: Dict) -> str:
    """Encode array using provided mapping."""
    return ''.join(mapping.get(val, 'X') for val in array)

def encode_tick_rule_array(tick_rule_array: np.ndarray) -> str:
    """Encode tick rule array."""
    mapping = {1: 'A', -1: 'B', 0: 'C'}
    return ''.join(mapping.get(val, 'X') for val in tick_rule_array)

# Entropy Functions
def get_shannon_entropy(message: str) -> float:
    """Calculate Shannon entropy."""
    counter = Counter(message)
    probs = [count / len(message) for count in counter.values()]
    return -sum(p * np.log2(p) for p in probs)

def get_lempel_ziv_entropy(message: str) -> float:
    """Calculate Lempel-Ziv entropy."""
    i, dictionary_size = 0, len(set(message))
    dictionary = {}
    w = ''

    while i < len(message):
        w += message[i]
        if w not in dictionary:
            dictionary[w] = dictionary_size
            dictionary_size += 1
            w = ''
        i += 1

    return len(dictionary) / len(message)

def get_plug_in_entropy(message: str, word_length: int = 1) -> float:
    """Calculate plug-in entropy."""
    if word_length > len(message):
        return 0

    words = [message[i:i+word_length] for i in range(len(message)-word_length+1)]
    counter = Counter(words)
    probs = [count / len(words) for count in counter.values()]
    return -sum(p * np.log2(p) for p in probs)

def get_konto_entropy(message: str, window: int = 20) -> float:
    """Calculate Kontoyiannis entropy."""
    n = len(message)
    sum_lambda = 0

    for i in range(n):
        lambda_i = 1
        for j in range(min(i, window)):
            if message[i-j:i+1] in message[:i]:
                lambda_i += 1
        sum_lambda += np.log2(lambda_i)

    return sum_lambda / n

if __name__ == "__main__":
    # Feature engineering functions
    def trades_balance(trades_df: pd.DataFrame, window: Union[str, int]) -> pd.Series:
        sells = trades_df["ask_amount"].rolling(window=window, min_periods=1).sum()
        buys = trades_df["bid_amount"].rolling(window=window, min_periods=1).sum()
        return (sells - buys) / (sells + buys + 1e-8)

    def calc_imbalance(lobs: pd.DataFrame) -> pd.Series:
        """Calculate order book imbalance for the first level."""
        bid_amount = lobs["bids[0].amount"]
        ask_amount = lobs["asks[0].amount"]
        return (bid_amount - ask_amount) / (bid_amount + ask_amount + 1e-8)

    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()

        return atr

    def calculate_open_interest(data: pd.DataFrame) -> pd.Series:
        open_interest = data['bids[0].amount'] + data['asks[0].amount']
        return open_interest

    def calculate_volume(trades_df: pd.DataFrame, window: Union[str, int]) -> pd.Series:
        volume = trades_df['ask_amount'].rolling(window=window, min_periods=1).sum() + \
                 trades_df['bid_amount'].rolling(window=window, min_periods=1).sum()
        return volume

    def calculate_large_density(lobs: pd.DataFrame, volume_series: pd.Series) -> pd.Series:
        density = lobs['bids[0].amount'] + lobs['asks[0].amount']
        # Align series
        density, volume = density.align(volume_series, join='outer', fill_value=0)
        large_density = density[density > volume]
        return large_density

    def calculate_atr_volume(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR-like indicator but for volume/amount instead of prices."""
        high_low = data['high_volume'] - data['low_volume']
        high_close = np.abs(data['high_volume'] - data['close_volume'].shift())
        low_close = np.abs(data['low_volume'] - data['close_volume'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_volume = true_range.rolling(window=period, min_periods=1).mean()

        return atr_volume

    def calc_features(
        lobs: pd.DataFrame,
        agg_trades: pd.DataFrame,
        lobs_embedding: pd.DataFrame,
        target_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate features and align them with the target data."""
        # Validate input data
        if any(df.empty for df in [lobs, agg_trades, lobs_embedding, target_data]):
            raise ValueError("One or more input DataFrames are empty")

        # Ensure all DataFrames have datetime index
        for df in [lobs, agg_trades, lobs_embedding, target_data]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

        # Add time-based features
        timestamp = pd.to_datetime(target_data.index)
        day_of_week = pd.Series(timestamp.dayofweek, index=timestamp)
        hour_of_day = pd.Series(timestamp.hour, index=timestamp)

        # Price features
        lobs["mid_price"] = (lobs["asks[0].price"].ffill() + lobs["bids[0].price"].ffill()) / 2

        # Embedding filtering
        btcusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "BTCUSDT"]["mid_price"].copy()
        ethusdt_mid_price = lobs_embedding[lobs_embedding["instrument"] == "ETHUSDT"]["mid_price"].copy()

        # Compute main deviations
        main_btcusdt_dev = (
            (lobs["mid_price"] / (btcusdt_mid_price.asof(lobs.index) + 1e-8))
            .asof(target_data.index) * target_data.side
        ).fillna(0)

        main_ethusdt_dev = (
            (lobs["mid_price"] / (ethusdt_mid_price.asof(lobs.index) + 1e-8))
            .asof(target_data.index) * target_data.side
        ).fillna(0)

        distance_to_mid_price = (
            target_data.price / (lobs["mid_price"].asof(target_data.index) + 1e-6) - 1
        ) * target_data.side

        imbalance_series = calc_imbalance(lobs).asof(target_data.index) * target_data.side

        depth = 5
        vwap_series = vwap(lobs, depth).asof(target_data.index) * target_data.side

        solusdt_agg_trades = agg_trades[agg_trades["instrument"] == "SOLUSDT"].copy()
        solusdt_agg_trades.index = pd.to_datetime(solusdt_agg_trades.index)
        trades_ratio_series = (
            trades_balance(solusdt_agg_trades, 10 * SECOND).asof(target_data.index)
            * target_data.side
        )
        sol_mid_price = lobs["mid_price"].asof(target_data.index)

        # Calculate ATR
        lobs["high"] = lobs[["asks[0].price", "bids[0].price"]].max(axis=1)
        lobs["low"] = lobs[["asks[0].price", "bids[0].price"]].min(axis=1)
        lobs["close"] = lobs["mid_price"]
        atr_series = calculate_atr(lobs, period=14).asof(target_data.index)

        # Calculate Open Interest
        open_interest_series = calculate_open_interest(lobs).asof(target_data.index)

        # Calculate Volume
        volume_series = calculate_volume(solusdt_agg_trades, window='5min').asof(target_data.index)

        # Calculate Large Density
        large_density_series = calculate_large_density(lobs, volume_series).asof(target_data.index)

        # Calculate ATR for volume
        lobs["high_volume"] = lobs[["asks[0].amount", "bids[0].amount"]].max(axis=1)
        lobs["low_volume"] = lobs[["asks[0].amount", "bids[0].amount"]].min(axis=1)
        lobs["close_volume"] = (lobs["asks[0].amount"] + lobs["bids[0].amount"]) / 2
        atr_volume_series = calculate_atr_volume(lobs, period=14).asof(target_data.index)

        # First Generation Features
        roll_measure = get_roll_measure(lobs["mid_price"], window=20).asof(target_data.index)
        roll_impact = get_roll_impact(lobs["mid_price"], volume_series, window=20).asof(target_data.index)
        corwin_schultz = get_corwin_schultz_estimator(
            lobs["asks[0].price"], lobs["bids[0].price"], window=20
        ).asof(target_data.index)
        bekker_park = get_bekker_parkinson_vol(
            lobs["asks[0].price"], lobs["bids[0].price"], window=20
        ).asof(target_data.index)

        # Second Generation Features
        price_diffs = lobs["mid_price"].diff().asof(target_data.index)
        log_returns = np.log(lobs["mid_price"]).diff().asof(target_data.index)

        # Fill NaNs for second-generation features computation
        price_diffs = price_diffs.fillna(0)
        volume_series = volume_series.fillna(0)
        log_returns = log_returns.fillna(0)

        kyle_lambda = get_trades_based_kyle_lambda(price_diffs, volume_series).asof(target_data.index)
        amihud_lambda = get_trades_based_amihud_lambda(log_returns, volume_series).asof(target_data.index)
        hasbrouck_lambda = get_trades_based_hasbrouck_lambda(log_returns, volume_series).asof(target_data.index)

        combined_features = {
            'kyle_lambda': kyle_lambda.fillna(0),
            'amihud_lambda': amihud_lambda.fillna(0),
            'hasbrouck_lambda': hasbrouck_lambda.fillna(0)
        }

        # Entropy features
        try:
            valid_volume = volume_series[~np.isnan(volume_series) & ~np.isinf(volume_series)]
            valid_returns = log_returns[~np.isnan(log_returns) & ~np.isinf(log_returns)]

            if len(valid_volume) > 0 and len(valid_returns) > 0:
                volume_mapping = quantile_mapping(valid_volume.values, num_letters=20)
                returns_mapping = quantile_mapping(valid_returns.values, num_letters=20)

                volume_message = encode_array(volume_series.fillna(0).values, volume_mapping)
                price_message = encode_array(log_returns.fillna(0).values, returns_mapping)

                volume_entropy_val = get_shannon_entropy(volume_message)
                price_entropy_lz_val = get_lempel_ziv_entropy(price_message)
                price_entropy_konto_val = get_konto_entropy(price_message)
            else:
                volume_entropy_val = pd.Series(0, index=target_data.index)
                price_entropy_lz_val = pd.Series(0, index=target_data.index)
                price_entropy_konto_val = pd.Series(0, index=target_data.index)
        except Exception:
            volume_entropy_val = pd.Series(0, index=target_data.index)
            price_entropy_lz_val = pd.Series(0, index=target_data.index)
            price_entropy_konto_val = pd.Series(0, index=target_data.index)

        # Naming Features
        roll_measure.name = "roll_measure"
        roll_impact.name = "roll_impact"
        corwin_schultz.name = "corwin_schultz"
        bekker_park.name = "bekker_park"
        volume_entropy_val.name = "volume_entropy"
        price_entropy_lz_val.name = "price_entropy_lz"
        price_entropy_konto_val.name = "price_entropy_konto"

        return pd.concat(
            [
                pd.Series(target_data.side, name="side"),
                vwap_series.rename("vwap"),
                trades_ratio_series.rename("trades_ratio"),
                distance_to_mid_price.rename("distance_to_mid_price"),
                main_ethusdt_dev.rename("main_ethusdt_dev"),
                main_btcusdt_dev.rename("main_btcusdt_dev"),
                imbalance_series.rename("imbalance"),
                sol_mid_price.rename("sol_mid_price"),
                atr_series.rename("atr"),
                atr_volume_series.rename("atr_volume"),
                open_interest_series.rename("open_interest"),
                volume_series.rename("volume"),
                large_density_series.rename("large_density"),
                day_of_week.rename("day_of_week"),
                hour_of_day.rename("hour_of_day"),
                roll_measure,
                roll_impact,
                corwin_schultz,
                bekker_park,
                combined_features['kyle_lambda'].rename('kyle_lambda'),
                combined_features['amihud_lambda'].rename('amihud_lambda'),
                combined_features['hasbrouck_lambda'].rename('hasbrouck_lambda'),
                volume_entropy_val,
                price_entropy_lz_val,
                price_entropy_konto_val
            ],
            axis=1,
        )

    # Load data
    path_to_data_folder = "./cmf/data/train/"
    agg_trades = pd.read_parquet(f'{path_to_data_folder}agg_trades.parquet')
    orderbook_embedding = pd.read_parquet(f'{path_to_data_folder}orderbook_embedding.parquet')
    orderbook_solusdt = pd.read_parquet(f'{path_to_data_folder}orderbook_solusdt.parquet')
    target_data_solusdt = pd.read_parquet(f'{path_to_data_folder}target_data_solusdt.parquet')
    target_solusdt = pd.read_parquet(f'{path_to_data_folder}target_solusdt.parquet')

    # Ensure correct time index
    for df in [agg_trades, orderbook_embedding, orderbook_solusdt, target_data_solusdt, target_solusdt]:
        df.index = pd.to_datetime(df.index)

    target_solusdt_preprocessed = target_solusdt.copy()

    # Generate features and target
    X = calc_features(orderbook_solusdt, agg_trades, orderbook_embedding, target_data_solusdt)
    y = target_solusdt_preprocessed["target"]
    weights = target_solusdt_preprocessed["weight"]

    # TimeSeriesSplit configuration
    tscv = TimeSeriesSplit(n_splits=5)

    params = {
        "use_best_model": True,
        "eval_metric": "Logloss",
        "iterations": 6000,
        "thread_count": 13,
        "loss_function": "Logloss",
        "l2_leaf_reg": 50,
        "task_type": "CPU",
        "depth": 8,
        "early_stopping_rounds": 50,
        "learning_rate": 0.01,
    }

    log_loss_values = []
    weighted_log_loss_values = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

        model = CatBoostClassifier(**params, verbose=25)
        model.fit(
            X_train,
            y_train,
            sample_weight=weights_train,
            eval_set=(X_test, y_test),
            verbose=25
        )

        # Evaluate model
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        log_loss_value = log_loss(y_test, y_pred_proba)
        log_loss_value_weighted = log_loss(y_test, y_pred_proba, sample_weight=weights_test)

        print(f"Fold {fold + 1} Log Loss: {log_loss_value}")
        print(f"Fold {fold + 1} Weighted Log Loss: {log_loss_value_weighted}")

        log_loss_values.append(log_loss_value)
        weighted_log_loss_values.append(log_loss_value_weighted)

    final_log_loss = sum(log_loss_values) / len(log_loss_values)
    final_weighted_log_loss = sum(weighted_log_loss_values) / len(weighted_log_loss_values)

    print(f"Final Log Loss (Average Across Folds): {final_log_loss}")
    print(f"Final Weighted Log Loss (Average Across Folds): {final_weighted_log_loss}")

    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"model_with_tscv_weights_no_preprocessed_{timestamp}.cbm"

    path_to_model_folder = "./cmf/models/"
    model.save_model(f'{path_to_model_folder}{model_name}')
    print(f"Model saved as: {model_name}")

    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # SHAP Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f'{path_to_model_folder}shap_summary_{timestamp}.png')
    plt.close()

    # Feature removal impact analysis (quick evaluation)
    base_score = final_weighted_log_loss
    feature_impacts = {}

    # A quick subset evaluation approach
    train_size = int(len(X) * 0.8)
    X_train_simple = X.iloc[:train_size]
    X_test_simple = X.iloc[train_size:]
    y_train_simple = y.iloc[:train_size]
    y_test_simple = y.iloc[train_size:]
    weights_train_simple = weights.iloc[:train_size]
    weights_test_simple = weights.iloc[train_size:]

    for feature in X.columns:
        X_reduced = X.drop(columns=[feature])
        X_train_reduced = X_reduced.iloc[:train_size]
        X_test_reduced = X_reduced.iloc[train_size:]

        model_reduced = CatBoostClassifier(**params, verbose=0)
        model_reduced.fit(
            X_train_reduced,
            y_train_simple,
            sample_weight=weights_train_simple,
            eval_set=(X_test_reduced, y_test_simple),
            verbose=0
        )

        y_pred_proba = model_reduced.predict_proba(X_test_reduced)[:, 1]
        score_without_feature = log_loss(y_test_simple, y_pred_proba, sample_weight=weights_test_simple)
        impact = score_without_feature - base_score
        feature_impacts[feature] = impact

    feature_impacts_df = pd.DataFrame({
        'Feature': list(feature_impacts.keys()),
        'Impact': list(feature_impacts.values())
    }).sort_values('Impact', ascending=False)

    print("\nFeature Removal Impact (positive means feature is important):")
    print(feature_impacts_df)

    # Save feature analysis
    with open(f'{path_to_model_folder}feature_analysis_{timestamp}.txt', 'w') as f:
        f.write("Feature Importance:\n")
        f.write(feature_importance.to_string())
        f.write("\n\nFeature Removal Impact:\n")
        f.write(feature_impacts_df.to_string())
