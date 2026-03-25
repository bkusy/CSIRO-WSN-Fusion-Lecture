"""
Target motion generators.

Each function returns (positions, velocities) as arrays of shape (n_steps, 2).
Distances from the origin (anchor at [0, 0]) are computed externally.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Trajectory:
    """Result of a trajectory generator."""
    positions: np.ndarray    # (n_steps, 2) in metres
    velocities: np.ndarray   # (n_steps, 2) in m/s
    dt: float                # time step (s)
    name: str
    description: str


def constant_velocity(
    n_steps: int = 200,
    dt: float = 0.5,
    start: tuple[float, float] = (8.0, 0.0),
    velocity: tuple[float, float] = (0.0, 0.3),
) -> Trajectory:
    """
    Straight-line motion at constant velocity.
    Baseline case: filter should converge quickly and stay converged.
    """
    pos = np.zeros((n_steps, 2))
    vel = np.array(velocity)
    pos[0] = start
    for k in range(1, n_steps):
        pos[k] = pos[k - 1] + vel * dt
    velocities = np.tile(vel, (n_steps, 1))
    return Trajectory(pos, velocities, dt, "Constant velocity", "Straight-line motion, steady speed")


def piecewise_linear(
    n_steps: int = 200,
    dt: float = 0.5,
    start: tuple[float, float] = (2.0, 0.0),
    segments: list[tuple[int, tuple[float, float]]] | None = None,
) -> Trajectory:
    """
    Motion with abrupt velocity changes at specified step indices.

    segments: list of (step_index, (vx, vy)) tuples.
    At each step_index the velocity changes to the given value.

    Default path: walk away, turn, come back — exposes filter lag at turns.
    """
    if segments is None:
        segments = [
            (0,   (0.3,  0.0)),
            (60,  (0.0,  0.3)),
            (120, (-0.3, 0.0)),
            (160, (0.0, -0.2)),
        ]

    seg_map = dict(segments)
    pos = np.zeros((n_steps, 2))
    vel_arr = np.zeros((n_steps, 2))
    pos[0] = start
    current_vel = np.array(seg_map.get(0, (0.0, 0.0)), dtype=float)

    for k in range(n_steps):
        if k in seg_map:
            current_vel = np.array(seg_map[k], dtype=float)
        vel_arr[k] = current_vel
        if k > 0:
            pos[k] = pos[k - 1] + current_vel * dt

    return Trajectory(pos, vel_arr, dt, "Piecewise linear", "Turns at fixed steps — tests process model lag")


def random_walk(
    n_steps: int = 200,
    dt: float = 0.5,
    start: tuple[float, float] = (5.0, 0.0),
    sigma_v: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """
    Velocity undergoes a slow random walk (integrated Brownian motion).
    Models a person wandering — unpredictable but slow.

    Tests Q tuning: too-small Q → filter lags; too-large Q → noisy estimate.
    """
    if rng is None:
        rng = np.random.default_rng()
    pos = np.zeros((n_steps, 2))
    vel_arr = np.zeros((n_steps, 2))
    pos[0] = start
    vel = np.zeros(2)
    for k in range(n_steps):
        vel = vel + rng.normal(0, sigma_v, 2)
        vel_arr[k] = vel
        if k > 0:
            pos[k] = pos[k - 1] + vel * dt
    return Trajectory(pos, vel_arr, dt, "Random walk", "Slow drift with unpredictable turns — tests Q tuning")


def maneuvering(
    n_steps: int = 200,
    dt: float = 0.5,
    start: tuple[float, float] = (3.0, 0.0),
    accel_sigma: float = 0.4,
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """
    High-acceleration motion: velocity changes quickly and unpredictably.
    Models a running person or a robot with fast maneuvers.

    With a standard constant-velocity KF and small Q, the filter will visibly
    lag behind and the innovation sequence will be autocorrelated.
    """
    if rng is None:
        rng = np.random.default_rng()
    pos = np.zeros((n_steps, 2))
    vel_arr = np.zeros((n_steps, 2))
    pos[0] = start
    vel = np.array([0.5, 0.0])
    for k in range(n_steps):
        accel = rng.normal(0, accel_sigma, 2)
        vel = vel + accel * dt
        # Soft speed cap to keep trajectory in a reasonable range
        speed = np.linalg.norm(vel)
        if speed > 2.0:
            vel = vel * (2.0 / speed)
        vel_arr[k] = vel
        if k > 0:
            pos[k] = pos[k - 1] + vel * dt
    return Trajectory(pos, vel_arr, dt, "Maneuvering", "Fast acceleration changes — exposes filter lag when Q is too small")


# Registry for UI selection
TRAJECTORIES = {
    "constant": constant_velocity,
    "piecewise": piecewise_linear,
    "random_walk": random_walk,
    "maneuvering": maneuvering,
}


def make_trajectory(name: str, n_steps: int, dt: float, rng: np.random.Generator) -> Trajectory:
    """Construct a trajectory by name, passing rng where supported."""
    fn = TRAJECTORIES[name]
    if name in ("random_walk", "maneuvering"):
        return fn(n_steps=n_steps, dt=dt, rng=rng)
    return fn(n_steps=n_steps, dt=dt)


def distances_from_anchor(
    positions: np.ndarray,
    anchor: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Euclidean distance from each position to an anchor point."""
    a = np.array(anchor)
    return np.linalg.norm(positions - a, axis=1)
