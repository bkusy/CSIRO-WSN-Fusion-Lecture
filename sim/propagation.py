"""
RSSI propagation and noise models for three environment scenarios.

Core model:  RSSI = A - 10*n*log10(d) + noise(scenario)
Inverse:     d = 10 ** ((A - RSSI) / (10*n))
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

def gaussian_noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    return rng.normal(0, sigma, n)


def laplace_noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    # Laplace scale b = sigma / sqrt(2) to match variance = 2*b^2 = sigma^2
    return rng.laplace(0, sigma / np.sqrt(2), n)


def burst_interference(
    rng: np.random.Generator,
    n: int,
    prob: float,
    amplitude: float,
) -> np.ndarray:
    """Additive burst outliers: each sample has `prob` chance of a large dBm spike."""
    mask = rng.random(n) < prob
    signs = rng.choice([-1, 1], size=n)
    magnitudes = rng.uniform(amplitude * 0.5, amplitude * 1.5, n)
    return mask * signs * magnitudes


def ground_effect_ripple(distances: np.ndarray, wavelength: float = 2.0) -> np.ndarray:
    """
    Two-ray ground reflection bias: sinusoidal ripple in RSSI as a function of
    distance, mimicking constructive/destructive interference near the ground.
    Amplitude tapers off at large distances.
    """
    amplitude = 3.0 / (1.0 + distances / 10.0)
    return amplitude * np.sin(2 * np.pi * distances / wavelength)


def packet_loss_mask(rng: np.random.Generator, n: int, prob: float) -> np.ndarray:
    """Boolean mask: True = packet received, False = dropped."""
    return rng.random(n) >= prob


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    description: str

    # Path-loss parameters
    n: float           # path-loss exponent
    A: float           # RSSI at 1 m reference distance (dBm)

    # Noise model
    sigma_dBm: float           # base noise std dev (dBm)
    noise_type: str            # "gaussian" or "laplace"

    # Structured interference
    outlier_prob: float        # per-sample burst outlier probability
    outlier_amplitude: float   # typical outlier magnitude (dBm)
    packet_loss_prob: float    # fraction of packets dropped

    # Ground effect (outdoor only)
    ground_effect: bool        # whether to apply distance-dependent ripple bias
    ground_wavelength: float   # ripple spatial period (m)

    # Human-readable notes shown in the UI
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------

SCENARIO_A = Scenario(
    name="A — Ideal",
    description="Free-space propagation, low Gaussian noise. Textbook KF case.",
    n=2.0,
    A=-40.0,
    sigma_dBm=2.0,
    noise_type="gaussian",
    outlier_prob=0.0,
    outlier_amplitude=0.0,
    packet_loss_prob=0.0,
    ground_effect=False,
    ground_wavelength=2.0,
    notes=[
        "Path-loss exponent n=2 matches free-space (no reflections).",
        "Noise is small and Gaussian — KF assumptions hold.",
        "Expect clean filter convergence and white innovation sequence.",
    ],
)

SCENARIO_B = Scenario(
    name="B — Outdoor",
    description="Moderate path loss, ground-effect ripple, occasional fades.",
    n=2.7,
    A=-40.0,
    sigma_dBm=4.5,
    noise_type="gaussian",
    outlier_prob=0.03,
    outlier_amplitude=12.0,
    packet_loss_prob=0.02,
    ground_effect=True,
    ground_wavelength=2.5,
    notes=[
        "n=2.7 reflects ground and vegetation scattering.",
        "Distance-dependent bias from two-ray ground reflection.",
        "Rare fades and small packet loss add mild non-stationarity.",
        "Innovation sequence will show some autocorrelation — model mismatch.",
    ],
)

SCENARIO_C = Scenario(
    name="C — Indoor",
    description="Heavy multipath, burst interference, significant packet loss.",
    n=3.5,
    A=-40.0,
    sigma_dBm=9.0,
    noise_type="laplace",
    outlier_prob=0.12,
    outlier_amplitude=20.0,
    packet_loss_prob=0.08,
    ground_effect=False,
    ground_wavelength=2.0,
    notes=[
        "n=3.5 reflects wall and furniture attenuation.",
        "Laplace (heavy-tailed) noise from multipath — not Gaussian.",
        "Burst outliers from co-channel BLE/WiFi interference.",
        "Innovation sequence will be highly non-white — filter model is wrong.",
        "Robust pre-filtering (median gate) helps before the KF.",
    ],
)

SCENARIOS: dict[str, Scenario] = {
    "A": SCENARIO_A,
    "B": SCENARIO_B,
    "C": SCENARIO_C,
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_rssi(
    distances: np.ndarray,
    scenario: Scenario,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate RSSI observations from true distances under the given scenario.

    Returns
    -------
    rssi_obs : np.ndarray
        Observed RSSI values (dBm), with NaN where packets are dropped.
    rssi_clean : np.ndarray
        Clean RSSI without noise (for reference).
    """
    n_steps = len(distances)

    # Clean path-loss model
    # Clip distances to avoid log(0)
    d_clipped = np.maximum(distances, 0.1)
    rssi_clean = scenario.A - 10 * scenario.n * np.log10(d_clipped)

    # Base noise
    if scenario.noise_type == "laplace":
        noise = laplace_noise(rng, n_steps, scenario.sigma_dBm)
    else:
        noise = gaussian_noise(rng, n_steps, scenario.sigma_dBm)

    # Ground effect bias
    if scenario.ground_effect:
        noise += ground_effect_ripple(d_clipped, scenario.ground_wavelength)

    # Burst outliers
    if scenario.outlier_prob > 0:
        noise += burst_interference(rng, n_steps, scenario.outlier_prob, scenario.outlier_amplitude)

    rssi_obs = rssi_clean + noise

    # Packet loss — replace dropped packets with NaN
    if scenario.packet_loss_prob > 0:
        received = packet_loss_mask(rng, n_steps, scenario.packet_loss_prob)
        rssi_obs = np.where(received, rssi_obs, np.nan)

    return rssi_obs, rssi_clean


def rssi_to_distance(rssi: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Invert path-loss model to get distance estimate from RSSI (dBm)."""
    return 10 ** ((scenario.A - rssi) / (10 * scenario.n))
