"""
1D Kalman filter over range.

State vector: [distance, velocity]  (metres, m/s)
Measurement:  [distance]            (metres, from RSSI inversion)

Uses filterpy for the core KF machinery so students can inspect the same
matrices (F, H, Q, R, P) they know from the lecture.
"""

import numpy as np
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


@dataclass
class KFResult:
    estimates: np.ndarray       # filtered distance, shape (n_steps,)
    covariances: np.ndarray     # P[0,0] (distance variance), shape (n_steps,)
    innovations: np.ndarray     # z - H @ x_prior, shape (n_steps,)  NaN where z is NaN
    gains: np.ndarray           # Kalman gain K[0], shape (n_steps,)


def build_range_kf(dt: float, Q_var: float, R_var: float, P0: float) -> KalmanFilter:
    """
    Construct a constant-velocity 1D Kalman filter over range.

    Parameters
    ----------
    dt : float
        Time step (s).
    Q_var : float
        Process noise variance.  Tuning knob: higher → filter trusts motion model less.
    R_var : float
        Measurement noise variance (m²).  Tuning knob: higher → filter trusts measurements less.
    P0 : float
        Initial state covariance (diagonal).
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # State transition: constant velocity
    kf.F = np.array([[1, dt],
                     [0, 1]])

    # Observation: we measure distance only
    kf.H = np.array([[1, 0]], dtype=float)

    # Process noise
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)

    # Measurement noise
    kf.R = np.array([[R_var]])

    # Initial state covariance
    kf.P = np.eye(2) * P0

    return kf


def run_range_kf(
    z: np.ndarray,
    dt: float,
    Q_var: float,
    R_var: float,
    P0: float,
    x0: float | None = None,
) -> KFResult:
    """
    Run the 1D range KF over a sequence of distance measurements.

    Parameters
    ----------
    z : np.ndarray
        Measured distances (m), shape (n_steps,). NaN = dropped packet (predict-only step).
    dt, Q_var, R_var, P0 : see build_range_kf.
    x0 : float, optional
        Initial distance estimate. Defaults to first non-NaN measurement.
    """
    n = len(z)
    kf = build_range_kf(dt, Q_var, R_var, P0)

    # Initialise state from first valid measurement
    first_valid = next((v for v in z if not np.isnan(v)), 1.0)
    kf.x = np.array([[x0 if x0 is not None else first_valid],
                     [0.0]])

    estimates = np.full(n, np.nan)
    covariances = np.full(n, np.nan)
    innovations = np.full(n, np.nan)
    gains = np.full(n, np.nan)

    for k in range(n):
        kf.predict()

        if not np.isnan(z[k]):
            z_k = np.array([[z[k]]])
            innov = float((z_k - kf.H @ kf.x_prior).squeeze())
            kf.update(z_k)
            innovations[k] = innov
            gains[k] = float(kf.K[0, 0])

        estimates[k] = float(kf.x[0, 0])
        covariances[k] = float(kf.P[0, 0])

    return KFResult(estimates, covariances, innovations, gains)
