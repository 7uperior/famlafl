#famlafl/util/fast_ewma.py
"""
This module contains an implementation of an exponentially weighted moving average based on sample size.
The inspiration and context for this code was from a blog post by writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def ewma(arr_in: np.ndarray, window: int) -> np.ndarray:
    """
    Exponentially weighted moving average specified by a decay ``window`` to provide better adjustments for
    small windows via:
        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    :param arr_in: (np.ndarray) A single dimensional numpy array
    :param window: (int) The decay window
    :return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``
    """
    arr_length = arr_in.shape[0]
    ewma_arr = np.empty(arr_length, dtype=np.float64)

    alpha = 2.0 / (window + 1)
    # For the finite series, we keep track of the cumulative weight:
    weight = 1.0
    ewma_old = arr_in[0]
    ewma_arr[0] = ewma_old

    # We'll track a multiplier = (1 - alpha)^i incrementally
    multiplier = 1.0  # i=0
    decay = (1.0 - alpha)

    for i in range(1, arr_length):
        multiplier *= decay
        weight += multiplier
        ewma_old = ewma_old * decay + arr_in[i]
        ewma_arr[i] = ewma_old / weight

    return ewma_arr

@jit(nopython=True)
def ema_infinite(arr_in: np.ndarray, window: int) -> np.ndarray:
    arr_length = arr_in.shape[0]
    out = np.empty(arr_length, dtype=np.float64)
    alpha = 2.0 / (window + 1)
    out[0] = arr_in[0]
    for i in range(1, arr_length):
        out[i] = alpha * arr_in[i] + (1 - alpha) * out[i - 1]
    return out
