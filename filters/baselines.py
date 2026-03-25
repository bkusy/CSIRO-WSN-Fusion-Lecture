"""
Baseline estimators for comparison against the Kalman filter.

All functions operate on 1D arrays of RSSI (dBm) or distance (m) values.
NaN values (dropped packets) are handled by forward-filling before filtering.
"""

import numpy as np
from collections import deque


def fill_dropped(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values (dropped packets) with the last valid reading."""
    out = x.copy()
    last = np.nan
    for i, v in enumerate(out):
        if np.isnan(v):
            out[i] = last  # stays NaN until a valid sample arrives
        else:
            last = v
    return out


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """
    Causal moving average (no lookahead). Handles NaN by forward-filling first.
    Returns same length as input; early samples use whatever valid data exists.
    """
    filled = fill_dropped(x)
    out = np.full_like(filled, np.nan)
    buf: deque[float] = deque()
    running = 0.0
    for i, v in enumerate(filled):
        if np.isnan(v):
            out[i] = np.nan
            continue
        buf.append(v)
        running += v
        if len(buf) > window:
            running -= buf.popleft()
        out[i] = running / len(buf)
    return out


def median_filter(x: np.ndarray, window: int) -> np.ndarray:
    """
    Causal median filter (no lookahead). Handles NaN by forward-filling first.
    Useful as a pre-filter before the KF to suppress burst outliers.
    """
    filled = fill_dropped(x)
    out = np.full_like(filled, np.nan)
    buf: deque[float] = deque()
    for i, v in enumerate(filled):
        if np.isnan(v):
            out[i] = np.nan
            continue
        buf.append(v)
        if len(buf) > window:
            buf.popleft()
        out[i] = float(np.median(list(buf)))
    return out


def innovation_whiteness(innovations: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Compute normalised autocorrelation of the innovation sequence up to max_lag.
    A well-tuned KF should produce near-zero values for lags > 0.

    Returns array of length max_lag+1 (lag 0 is always 1.0).
    """
    valid = innovations[~np.isnan(innovations)]
    if len(valid) < 2:
        return np.zeros(max_lag + 1)
    valid = valid - valid.mean()
    var = np.var(valid)
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.array([
        np.mean(valid[:len(valid) - lag] * valid[lag:]) / var
        for lag in range(max_lag + 1)
    ])
    return acf
