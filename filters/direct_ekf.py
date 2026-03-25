"""
Direct 2D extended Kalman filter for multi-anchor range localization.

State: [x, y, vx, vy]
Measurement: ranges to a set of fixed anchors
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DirectEKFResult:
    states: np.ndarray
    covariances: np.ndarray


def _constant_velocity_transition(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _process_noise(dt: float, q_var: float) -> np.ndarray:
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    q_1d = q_var * np.array(
        [
            [dt4 / 4.0, dt3 / 2.0],
            [dt3 / 2.0, dt2],
        ]
    )
    return np.block(
        [
            [q_1d[0:1, 0:1], np.zeros((1, 1)), q_1d[0:1, 1:2], np.zeros((1, 1))],
            [np.zeros((1, 1)), q_1d[0:1, 0:1], np.zeros((1, 1)), q_1d[0:1, 1:2]],
            [q_1d[1:2, 0:1], np.zeros((1, 1)), q_1d[1:2, 1:2], np.zeros((1, 1))],
            [np.zeros((1, 1)), q_1d[1:2, 0:1], np.zeros((1, 1)), q_1d[1:2, 1:2]],
        ]
    )


def _measurement_function(state: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    pos = state[:2]
    return np.linalg.norm(pos - anchors, axis=1)


def _measurement_jacobian(state: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    pos = state[:2]
    delta = pos - anchors
    ranges = np.linalg.norm(delta, axis=1)
    safe = np.maximum(ranges, 1e-6)
    H = np.zeros((len(anchors), 4))
    H[:, 0] = delta[:, 0] / safe
    H[:, 1] = delta[:, 1] / safe
    return H


def run_direct_ekf(
    z: np.ndarray,
    anchors: np.ndarray,
    dt: float,
    q_var: float,
    r_var: float,
    p0: float,
    x0: np.ndarray | None = None,
) -> DirectEKFResult:
    """
    Run a direct EKF using multi-anchor range measurements.

    Parameters
    ----------
    z : np.ndarray
        Range measurements with shape (n_steps, n_anchors). NaN = dropped packet.
    anchors : np.ndarray
        Anchor coordinates with shape (n_anchors, 2).
    """
    n_steps = z.shape[0]
    F = _constant_velocity_transition(dt)
    Q = _process_noise(dt, q_var)
    I = np.eye(4)

    if x0 is None:
        x = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()
    P = np.eye(4) * p0

    states = np.full((n_steps, 4), np.nan)
    covariances = np.full((n_steps, 4, 4), np.nan)

    for k in range(n_steps):
        x = F @ x
        P = F @ P @ F.T + Q

        valid = ~np.isnan(z[k])
        if np.any(valid):
            anchors_k = anchors[valid]
            z_k = z[k, valid]
            h_k = _measurement_function(x, anchors_k)
            H_k = _measurement_jacobian(x, anchors_k)
            R_k = np.eye(len(z_k)) * r_var
            y_k = z_k - h_k
            S_k = H_k @ P @ H_k.T + R_k
            K_k = P @ H_k.T @ np.linalg.pinv(S_k)
            x = x + K_k @ y_k
            IKH = I - K_k @ H_k
            P = IKH @ P @ IKH.T + K_k @ R_k @ K_k.T

        states[k] = x
        covariances[k] = P

    return DirectEKFResult(states=states, covariances=covariances)
