# CSIRO WSN Fusion Lecture

Interactive Marimo lab for teaching RSSI-based ranging, Kalman filtering, and multi-anchor localization.

The notebook is designed to do two things:

- build intuition about noisy RSSI localization systems
- provide small, reusable reference implementations of the filters used in the lab

## What This Repo Contains

The main student-facing app is:

```bash
marimo run lab_part1.py
```

It walks through four sections:

1. **Section 1 — RSSI simulation**
   Simulate RSSI observations under different environments and motion types, and inspect how raw RSSI-to-range estimates degrade under noise, burst interference, and packet loss.

2. **Section 2 — 1D range Kalman filter**
   Compare raw RSSI-derived distance, moving-average baselines, and a 1D constant-velocity Kalman filter over range.

3. **Section 3 — Multi-anchor localization**
   Use multiple anchors to estimate 2D position with:
   - raw least-squares trilateration
   - Kalman-filtered per-anchor ranges followed by trilateration
   - a direct EKF with hidden state `[x, y, v_x, v_y]`

4. **Section 4 — Reusable algorithms**
   Student-facing guidance on how the provided filter implementations can be reused in future RSSI localization and tracking projects.

## Requirements

- Python 3.11 or later
- `pip`

## Installation

1. Clone the repository

```bash
git clone <repo-url>
cd CSIRO-WSN-Fusion-Lecture
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
```

- macOS / Linux:

```bash
source .venv/bin/activate
```

- Windows:

```powershell
.venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running The App

Run the notebook as an interactive app:

```bash
marimo run lab_part1.py
```

Open the notebook for editing:

```bash
marimo edit lab_part1.py
```

## Core Concepts Covered

### Propagation Scenarios

| Scenario | Conditions | Main effect |
|---|---|---|
| A — Ideal | Low Gaussian noise, no structured interference | Textbook filtering case |
| B — Outdoor | Ground-effect ripple, mild fades, small packet loss | Biased and mildly mismatched measurements |
| C — Indoor | Heavy multipath, burst interference, packet loss | Strongly non-Gaussian, non-ideal measurements |

### Motion Models

| Motion | Description | Why it matters |
|---|---|---|
| Constant velocity | Straight-line motion | Best-case tracking scenario |
| Piecewise linear | Abrupt turns | Exposes filter lag during turns |
| Random walk | Slow unpredictable drift | Good for Q tuning intuition |
| Maneuvering | Fast acceleration changes | Highlights motion-model mismatch |

### Filters Included

| File | Method | Use case |
|---|---|---|
| `filters/range_kf.py` | 1D Kalman filter on range | Smooth single-anchor distance estimates |
| `filters/direct_ekf.py` | 2D Extended Kalman Filter | Direct multi-anchor localization with nonlinear range measurements |
| `filters/baselines.py` | Simple baselines | Moving average and other lightweight comparisons |

## Recommended Student Takeaways

Students should leave the lab understanding:

- how RSSI is converted into noisy range estimates
- why naive RSSI-to-range localization is unstable
- when a linear KF is enough and when an EKF is needed
- how `Q`, `R`, `P0`, and `dt` affect responsiveness, smoothness, and stability
- how reusable filter code can be adapted to future real-world localization projects

## Project Structure

```text
CSIRO-WSN-Fusion-Lecture/
  lab_part1.py                 # Main Marimo lab notebook
  filters/
    baselines.py              # Moving-average and other simple baselines
    range_kf.py               # 1D Kalman filter over range
    direct_ekf.py             # Direct 2D EKF for multi-anchor localization
  sim/
    propagation.py            # RSSI path-loss and noise/interference models
    trajectories.py           # Motion generators and geometry helpers
  tests/
    test_filter_domain_logic.py
  requirements.txt
  LICENSE
```

## Notes

- The UI includes an `Enable quizzes` toggle so the notebook can be used either as a guided teaching flow or as a free-exploration tool.
- In Section 3, the localization methods may use more anchors than the range-observation panel displays; the range panel intentionally stays limited to the first three anchors for readability.
- The filter modules are intended to be readable reference implementations, not just hidden notebook internals.

## License

MIT. See [LICENSE](LICENSE).
