import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full", app_title="Part 1 — RSSI Ranging & Kalman Filter")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import sys, os
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from sim.propagation import SCENARIOS, simulate_rssi, rssi_to_distance
    from sim.trajectories import make_trajectory, distances_from_anchor
    from filters.direct_ekf import run_direct_ekf
    from filters.range_kf import run_range_kf
    from filters.baselines import moving_average, innovation_whiteness
    return (
        SCENARIOS, distances_from_anchor, go, innovation_whiteness,
        make_subplots, make_trajectory, mo, moving_average,
        np, run_direct_ekf, run_range_kf, rssi_to_distance, simulate_rssi,
    )


# ── Shared reactive state ─────────────────────────────────────────────────────
@app.cell
def _(mo):
    reset_count, set_reset = mo.state(0)
    s2_unlocked, set_s2   = mo.state(False)
    s3_unlocked, set_s3   = mo.state(False)
    return (reset_count, s2_unlocked, s3_unlocked, set_reset, set_s2, set_s3)


@app.cell
def _(mo, reset_count, set_reset, set_s2, set_s3):
    reset_btn = mo.ui.button(
        label="↺  Reset all parameters to defaults",
        on_click=lambda _: (set_reset(reset_count() + 1), set_s2(False), set_s3(False)),
        kind="warn",
    )
    return (reset_btn,)


@app.cell
def _(mo):
    mo.md("""
    <style>
    .js-plotly-plot,
    .plotly,
    .plot-container,
    .svg-container {
      width: 100% !important;
      max-width: 100% !important;
    }

    .js-plotly-plot .plotly .main-svg {
      width: 100% !important;
      max-width: 100% !important;
    }
    </style>
    """)
    return


@app.cell
def _(mo):
    def labeled_control_row(label: str, control, value: str):
        return mo.hstack(
            [
                mo.md(f"{label}:"),
                control,
                mo.md(f"`{value}`" if value else ""),
            ],
            justify="start",
            align="center",
        )

    return (labeled_control_row,)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Simulator
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    mo.md("""
    # Part 1 — RSSI Ranging & Kalman Filtering

    ## Context

    A **shooter localization system** (from the lecture) uses multiple anchors and acoustic
    signals to localize a target. Here we switch to **RSSI-based measurements**: each anchor
    monitors radio packet transmissions from the target user and records the noisy RSSI values.

    Converting RSSI to range works in theory, but the inverse power-law amplifies small signal
    errors into large distance errors. A **Kalman filter** recovers clean range estimates by
    blending a motion model with the noisy measurements — and those filtered ranges then feed
    into 2D position estimation via trilateration.

    ## Lab structure

    | Section | Question answered | Key concept |
    |---------|-------------------|-------------|
    | **1 — Simulator** | Why is RSSI-to-range conversion so noisy? | Inverse power-law amplifies small signal errors into large distance errors |
    | **2 — 1D Kalman filter** | How do Q and R control the filter? | Q = motion model trust · R = measurement trust · tuning trades lag for noise |
    | **3 — Multi-anchor localization** | Does pre-filtering ranges improve 2D position? | Per-anchor KF tightens the range circles before trilateration — sharper intersection |

    Each section unlocks once you answer the quiz at the bottom of the previous one.
    """)

@app.cell
def _(mo, reset_count):
    _ = reset_count()
    quiz_toggle = mo.ui.checkbox(value=True, label="Enable quizzes")
    return (quiz_toggle,)


@app.cell
def _(mo, quiz_toggle):
    mo.vstack([
        mo.md("---"),
        mo.hstack(
            [
                mo.md("## Section 1 — Simulate the environment and observe raw data"),
                quiz_toggle,
            ],
            justify="space-between",
            align="center",
        ),
        mo.md(
            """
            The target moves smoothly in 2D, but each anchor only observes a single scalar: RSSI.
            RSSI is noisy and occasionally badly corrupted. After inverting the path-loss model it
            becomes a noisy distance estimate — and that noisy distance sequence is what the Kalman
            filter in Section 2 will smooth. You don't need to understand the full radio model here;
            just watch the gap between true distance and RSSI-derived distance in the plots below.
            """
        ),
    ])
    return

@app.cell
def _(SCENARIOS, mo, reset_count):
    _ = reset_count()
    scenario_select = mo.ui.dropdown(
        options=[v.name for v in SCENARIOS.values()],
        value=list(SCENARIOS.values())[0].name,
        label=None,
    )
    motion_select = mo.ui.dropdown(
        options=["Constant velocity", "Piecewise linear (turns)",
                 "Random walk (slow drift)", "Maneuvering (fast)"],
        value="Constant velocity",
        label=None,
    )
    seed_slider   = mo.ui.slider(1, 100,  value=42,  step=1,   label=None)
    n_steps_slider = mo.ui.slider(50, 400, value=200, step=50,  label=None)
    dt_slider     = mo.ui.slider(0.1, 2.0, value=0.5, step=0.1, label=None)
    return dt_slider, motion_select, n_steps_slider, scenario_select, seed_slider


@app.cell
def _(SCENARIOS, mo, reset_count, scenario_select):
    _ = reset_count()
    _n2k = {v.name: k for k, v in SCENARIOS.items()}
    _scenario = SCENARIOS[_n2k[scenario_select.value]]
    n_override          = mo.ui.slider(1.5, 5.0, value=_scenario.n, step=0.1,  label=None)
    sigma_override      = mo.ui.slider(0.5, 15.0, value=_scenario.sigma_dBm, step=0.5, label=None)
    outlier_prob_slider = mo.ui.slider(0.0, 0.3, value=_scenario.outlier_prob, step=0.01,  label=None)
    loss_slider         = mo.ui.slider(0.0, 0.3, value=_scenario.packet_loss_prob, step=0.01,  label=None)
    return loss_slider, n_override, outlier_prob_slider, sigma_override


@app.cell
def _(
    SCENARIOS, loss_slider, n_override, outlier_prob_slider,
    scenario_select, sigma_override,
):
    from dataclasses import replace as _replace
    _N2K = {v.name: k for k, v in SCENARIOS.items()}
    sc = SCENARIOS[_N2K[scenario_select.value]]
    sc_effective = _replace(
        sc,
        n=n_override.value,
        sigma_dBm=sigma_override.value,
        outlier_prob=outlier_prob_slider.value,
        packet_loss_prob=loss_slider.value,
    )
    ground_text = (
        f"lambda={sc_effective.ground_wavelength:.1f} m"
        if sc_effective.ground_effect else
        "off"
    )
    return ground_text, sc, sc_effective


@app.cell
def _(
    distances_from_anchor, dt_slider, make_trajectory, motion_select,
    n_steps_slider, np, rssi_to_distance, sc_effective, seed_slider,
    simulate_rssi,
):
    _MOTION = {
        "Constant velocity":        "constant",
        "Piecewise linear (turns)": "piecewise",
        "Random walk (slow drift)": "random_walk",
        "Maneuvering (fast)":       "maneuvering",
    }
    _rng = np.random.default_rng(seed_slider.value)
    traj       = make_trajectory(_MOTION[motion_select.value], n_steps=n_steps_slider.value, dt=dt_slider.value, rng=_rng)
    true_dists = distances_from_anchor(traj.positions)
    t          = np.arange(n_steps_slider.value) * dt_slider.value
    rssi_obs, _ = simulate_rssi(true_dists, sc_effective, _rng)
    d_raw      = rssi_to_distance(rssi_obs, sc_effective)
    return d_raw, rssi_obs, t, traj, true_dists


# Controls layout + live stat readout
@app.cell
def _(
    dt_slider, ground_text, labeled_control_row, loss_slider, mo, motion_select, n_override,
    n_steps_slider, outlier_prob_slider, reset_btn, sc, sc_effective,
    scenario_select, seed_slider, sigma_override,
):
    model_header = mo.hstack(
        [
            mo.md("**Environment model**"),
            scenario_select,
        ],
        justify="start",
        align="center",
    )
    info_box = mo.callout(
        mo.vstack(
            [
                model_header,
                mo.md(
                    r"""
                    $$
                    r_k = A - 10n \log_{10}(d_k) + e_{base,k} + e_{ground,k} + e_{burst,k}
                    $$
                    """
                ),
                mo.md(
                    f"""
                    {sc.description}  
                    `A = {sc_effective.A:.1f}` dBm, `n = {sc_effective.n:.1f}`  
                    $e_{{base,k}}$: random base-noise term; `σ` sets its spread  
                    $e_{{ground,k}}$: ground-reflection term, `ground = {ground_text}`  
                    $e_{{burst,k}}$: occasional large interference / outlier term, $p_{{burst}} = {sc_effective.outlier_prob:.2f}$
                    """
                ),
            ],
            align="start",
        ),
        kind="info",
    )
    mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("**Simulation**"),
                labeled_control_row("Motion", motion_select, ""),
                labeled_control_row("Seed", seed_slider, f"{seed_slider.value:d}"),
                labeled_control_row("Steps", n_steps_slider, f"{n_steps_slider.value:d}"),
                labeled_control_row("dt", dt_slider, f"{dt_slider.value:.1f} s"),
                mo.md(""),
                reset_btn,
            ], align="start", gap="1.7rem"),
            mo.vstack([
                mo.md("**Propagation (override scenario defaults)**"),
                labeled_control_row("Path-loss n", n_override, f"{n_override.value:.1f}"),
                labeled_control_row("Noise σ", sigma_override, f"{sigma_override.value:.1f} dBm"),
                labeled_control_row("Burst prob", outlier_prob_slider, f"{outlier_prob_slider.value:.2f}"),
                labeled_control_row("Loss prob", loss_slider, f"{loss_slider.value:.2f}"),
            ], align="start", gap="1.7rem"),
            mo.vstack([
                info_box,
            ], align="start"),
        ], justify="start", align="start", gap="2.5rem", widths=[1.0, 1.0, 1.8]),
    ])
    return


# Side-by-side: trajectory map | distance + RSSI time series
@app.cell
def _(d_raw, go, make_subplots, mo, np, rssi_obs, t, traj, true_dists):
    _pos = traj.positions
    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(
        x=_pos[:, 0], y=_pos[:, 1], mode="lines+markers",
        line=dict(color="#2ecc71", width=2), marker=dict(size=3), name="Trajectory",
    ))
    fig_map.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(symbol="star", size=14, color="#e74c3c"),
        text=["Anchor"], textposition="top right", name="Anchor",
    ))
    fig_map.add_trace(go.Scatter(
        x=[_pos[0, 0]], y=[_pos[0, 1]], mode="markers+text",
        marker=dict(size=10, color="#3498db"),
        text=["Start"], textposition="top left", name="Start",
    ))
    _all_x = np.append(_pos[:, 0], 0.0)
    _all_y = np.append(_pos[:, 1], 0.0)
    _pad = max(_all_x.max() - _all_x.min(), _all_y.max() - _all_y.min()) * 0.12 + 1.0
    fig_map.update_layout(
        title="Target trajectory", height=400, autosize=True,
        xaxis_title="x (m)", yaxis_title="y (m)",
        xaxis=dict(range=[_all_x.min() - _pad, _all_x.max() + _pad]),
        yaxis=dict(range=[_all_y.min() - _pad, _all_y.max() + _pad]),
        margin=dict(t=40, b=10),
    )

    fig_ts = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("True distance vs raw RSSI estimate", "Raw RSSI observations"),
        vertical_spacing=0.12, row_heights=[0.55, 0.45],
    )
    fig_ts.add_trace(go.Scatter(x=t, y=true_dists, name="True distance",
        line=dict(color="#2ecc71", width=2.5)), row=1, col=1)
    fig_ts.add_trace(go.Scatter(x=t, y=d_raw, name="Raw RSSI→dist", mode="markers",
        marker=dict(color="#e74c3c", size=3, opacity=0.5)), row=1, col=1)
    fig_ts.add_trace(go.Scatter(x=t, y=rssi_obs, name="RSSI (dBm)", mode="markers",
        marker=dict(color="#9b59b6", size=3, opacity=0.6)), row=2, col=1)
    fig_ts.update_layout(height=400, autosize=True, margin=dict(t=40, b=10),
        legend=dict(orientation="h", y=-0.18))
    fig_ts.update_yaxes(title_text="Distance (m)", row=1, col=1)
    fig_ts.update_yaxes(title_text="RSSI (dBm)",   row=2, col=1)
    fig_ts.update_xaxes(title_text="Time (s)",      row=2, col=1)

    mo.hstack([fig_map, fig_ts], justify="start")
    return

@app.cell
def _(mo):
    mo.callout(mo.md("""
    **🔬 Guided experiments before the quiz**

    1. Keep *Constant velocity* and switch Environment model Scenario A → B → C.
       Watch how a clean measurement problem becomes a difficult one. Focus on the top-right plot:
       compare **true distance** against **raw RSSI-to-distance**. This gap is the problem the filter must solve.
    2. In Scenario C, increase *Burst prob*.
       Notice that a few bad RSSI samples become very large distance errors (>100m) after inversion.
    """), kind="neutral")
    return

# S1 quiz — mo.ui.form: .value only updates on explicit submit, no state timing issues
@app.cell
def _(mo, quiz_toggle, reset_count):
    mo.stop(not quiz_toggle.value)
    _ = reset_count()
    q1_form = mo.ui.form(
        mo.ui.radio(
            options=[
                "A — Ideal: clean Gaussian noise, no spikes",
                "B — Outdoor: mild fades and ripple",
                "C — Indoor: heavy-tailed noise and burst spikes",
            ],
            label="Which environment produces the most severe burst spikes in the raw RSSI signal?",
        ),
        submit_button_label="Submit answer →",
    )
    return (q1_form,)


@app.cell
def _(mo, q1_form, quiz_toggle, set_s2):
    mo.stop(not quiz_toggle.value)
    _correct = q1_form.value is not None and "Indoor" in q1_form.value
    _wrong   = q1_form.value is not None and "Indoor" not in q1_form.value
    if _correct:
        set_s2(True)
    _fb = (
        mo.callout(mo.md("✓ **Correct!** Section 2 is now unlocked. **Before tuning, read the \"Parameters Explained\" reference just below** — it explains what Q, R, and P₀ actually control."), kind="success")
        if _correct else
        mo.callout(mo.md("❌ Not quite. Hint: which environment includes co-channel interference?"), kind="danger")
        if _wrong else
        mo.md("*Select an answer and click Submit to proceed.*")
    )
    mo.vstack([mo.md("---\n### ✏️ Check your understanding"), q1_form, _fb])
    return




# ═══════════════════════════════════════════════════════════════════════════════
# Parameters Explained
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    mo.md("""---
## Parameters Explained — Read This Before Section 2

Use this reference throughout the lab to understand what Q and R do.
""")
    return

@app.cell
def _(mo):
    mo.hstack([
        mo.callout(
            mo.md("""
            ### Q — Process Noise Variance

            **"How much do I trust the motion model?"**

            **Low Q (0.001–0.01)**: Smooth, slow to react
            - Works well when range changes smoothly
            - Can lag when the true range changes more abruptly

            **Moderate / high Q (0.03–1.0)**: More responsive
            - Can reduce lag when the motion model is too rigid
            - If Q becomes too large, measurement noise dominates

            **Tune**: For this lab, the clearest Q trade-off appears in **Scenario A + Piecewise linear**. Start around `R = 20` or `40`, then compare `Q = 0.001`, `0.03`, and `0.3`.
            """),
            kind="info",
        ),
        mo.callout(
            mo.md("""
            ### R — Measurement Noise Variance

            **"How much do I trust the sensor readings?"**

            **Low R (0.1–1.0)**: Trusts measurements
            - Estimate follows sensor readings closely
            - Includes measurement noise

            **High R (10–100)**: Ignores noisy readings
            - Estimate is smoother but may lag

            **Tune**: Sluggish? Decrease R. Too noisy? Increase R.
            """),
            kind="info",
        ),
        mo.callout(
            mo.md("""
            ### P₀ and dt

            **P₀ — Initial uncertainty**

            - **Large (10–100)**: First measurement unreliable
            - **Small (1–10)**: Starting range known well

            Effect fades after a few steps — don't overthink it.

            **dt — Time step (critical)**

            Must match your actual measurement interval exactly.
            If dt is wrong, the motion model is wrong and the filter fails.
            """),
            kind="warn",
        ),
    ], justify="start", gap="2rem", widths=[0.3, 0.3, 0.3])
    return

# ═══════════════════════════════════════════════════════════════════════════════
# KF controls — always defined so reset works regardless of which section is open
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _(mo, reset_count):
    _ = reset_count()
    Q_slider    = mo.ui.slider(0.001, 10.0,  value=0.1,  step=0.001, label=None)
    R_slider    = mo.ui.slider(0.1,   100.0, value=20.0, step=0.1,   label=None)
    P0_slider   = mo.ui.slider(0.1,   100.0, value=10.0, step=0.1,   label=None)
    return (
        P0_slider, Q_slider, R_slider,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════

# S2 controls + KF plot
@app.cell
def _(
    P0_slider, Q_slider, R_slider, d_raw, dt_slider, go,
    labeled_control_row, mo, motion_select, np, quiz_toggle, run_range_kf,
    s2_unlocked, scenario_select, t, traj, true_dists,
):
    mo.stop(
        quiz_toggle.value and not s2_unlocked(),
        mo.callout(mo.md(
            "🔒 **Section 2: Kalman Filter** — answer the quiz in Section 1 to unlock."
        ), kind="warn"),
    )

    # ── compute ───────────────────────────────────────────────────────────────
    kf_result = run_range_kf(d_raw, dt=dt_slider.value,
                             Q_var=Q_slider.value, R_var=R_slider.value, P0=P0_slider.value)

    def _rmse(e, r): return np.sqrt(np.nanmean((e - r) ** 2))
    kf_rmse = _rmse(kf_result.estimates, true_dists)
    raw_rmse = _rmse(d_raw, true_dists)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig_kf = go.Figure()
    fig_kf.add_trace(go.Scatter(x=t, y=true_dists, name="True distance",
        line=dict(color="#2ecc71", width=2.5)))
    fig_kf.add_trace(go.Scatter(x=t, y=d_raw, name="Raw RSSI→distance", mode="markers",
        marker=dict(color="#e74c3c", size=3, opacity=0.4)))
    fig_kf.add_trace(go.Scatter(x=t, y=kf_result.estimates, name="Kalman filter",
        line=dict(color="#3498db", width=2.5)))
    fig_kf.update_layout(height=400, autosize=True, xaxis_title="Time (s)", yaxis_title="Distance (m)",
        legend=dict(orientation="h", y=-0.18), margin=dict(t=30, b=10))


    # ── mini map ──────────────────────────────────────────────────────────────
    _pos = traj.positions
    _all_x = np.append(_pos[:, 0], 0.0)
    _all_y = np.append(_pos[:, 1], 0.0)
    _span = max(_all_x.max() - _all_x.min(), _all_y.max() - _all_y.min())
    _pad = _span * 0.12 + 1.0
    _fig_map = go.Figure()
    _fig_map.add_trace(go.Scatter(x=_pos[:, 0], y=_pos[:, 1], mode="lines",
        line=dict(color="#2ecc71", width=2), name="Trajectory"))
    _fig_map.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
        marker=dict(symbol="star", size=14, color="#e74c3c"),
        text=["Anchor"], textposition="top right", name="Anchor"))
    _fig_map.add_trace(go.Scatter(x=[_pos[0, 0]], y=[_pos[0, 1]], mode="markers+text",
        marker=dict(size=10, color="#3498db"),
        text=["Start"], textposition="top left", name="Start"))
    _fig_map.update_layout(
        title="Target trajectory", height=320, autosize=True,
        xaxis_title="x (m)", yaxis_title="y (m)",
        xaxis=dict(range=[_all_x.min() - _pad, _all_x.max() + _pad], autorange=False),
        yaxis=dict(range=[_all_y.min() - _pad, _all_y.max() + _pad], autorange=False),
        legend=dict(orientation="h", y=-0.18), margin=dict(t=40, b=10),
    )

    # ── layout ────────────────────────────────────────────────────────────────
    mo.vstack([
        mo.md(r"""---
## Section 2 — Kalman Filter on Range

One anchor gives us one noisy scalar: **distance**. The Kalman filter tracks a 2D state
$x_k = [d_k,\ \dot{d}_k]^\top$ — distance and its rate of change — using a
constant-velocity motion model. At each step it predicts the next state, then blends
that prediction with the noisy measurement $z_k = d_k + v_k$.
How much weight goes to the prediction vs. the measurement is controlled by **Q** and **R**.
"""),
        mo.hstack([
            mo.vstack([
                mo.md("**Tune these parameters**"),
                labeled_control_row("Scenario", scenario_select, ""),
                labeled_control_row("Motion", motion_select, ""),
                labeled_control_row("Process Q", Q_slider, f"{Q_slider.value:.3f}"),
                labeled_control_row("Measure R", R_slider, f"{R_slider.value:.1f} m²"),
                labeled_control_row("Initial P₀", P0_slider, f"{P0_slider.value:.1f}"),
                mo.md("---"),
                mo.md("**Performance**"),
                mo.md(f"Raw RMSE: **{raw_rmse:.2f} m**"),
                mo.md(f"KF RMSE: **{kf_rmse:.2f} m**"),
                mo.md(f"Improvement: **{100 * (1 - kf_rmse/raw_rmse):.0f}%**"),
            ], align="start"),
            mo.vstack([_fig_map], align="start"),
            mo.vstack([fig_kf], align="start"),
        ], justify="start", align="start", gap="2rem", widths=[1.0, 1.5, 2.3]),
    ])
    return kf_result


@app.cell
def _(mo, quiz_toggle, s2_unlocked):
    mo.stop(quiz_toggle.value and not s2_unlocked())
    mo.callout(mo.md("""
    **🔬 Guided experiments**

    1. Start in **Scenario A**, *Constant velocity*. Tune `Q` and `R` until the Kalman RMSE
       is clearly lower than the raw RSSI→distance RMSE.
    2. For the clearest **Q** demonstration, switch to **Scenario A** and *Piecewise linear* motion.
       Set `R = 20` or `40`, then compare `Q = 0.001`, `0.03`, and `0.3`. With very small `Q`, the estimate is smooth but can lag after the range profile changes.
       With moderate `Q`, that lag reduces. With larger `Q`, the estimate starts following measurement noise too closely.
    3. To study **R**, keep *Constant velocity* in **Scenario A** and compare `R = 4`, `20`, and `80`
       while keeping `Q` small. This isolates how measurement trust affects smoothness. Experiment with other scenarios and motions to see how the optimal R depends on noise level and motion dynamics.
    """), kind="neutral")
    return


# S2 quiz — mo.ui.form pattern (same fix as Q1)
@app.cell
def _(mo, quiz_toggle, reset_count, s2_unlocked):
    mo.stop((not quiz_toggle.value) or (quiz_toggle.value and not s2_unlocked()))
    _ = reset_count()
    q2_form = mo.ui.form(
        mo.ui.radio(
            options=[
                "A moderate increase in Q can reduce lag, but pushing Q too high makes the estimate too noisy",
                "Increasing Q always improves the estimate because it reacts faster to every measurement",
                "There is no visible effect — Q only scales the internal covariance matrix",
            ],
            label=(
                "In **Scenario A** with *Piecewise linear* motion, what is the main effect of increasing `Q` "
                "from a very small value while keeping `R` moderate?"
            ),
        ),
        submit_button_label="Submit answer →",
    )
    return (q2_form,)


@app.cell
def _(mo, q2_form, quiz_toggle, s2_unlocked, set_s3):
    mo.stop((not quiz_toggle.value) or (quiz_toggle.value and not s2_unlocked()))
    _correct = q2_form.value is not None and "reduce lag" in q2_form.value
    _wrong   = q2_form.value is not None and "reduce lag" not in q2_form.value
    if _correct:
        set_s3(True)
    _fb = (
        mo.callout(mo.md("✓ **Correct!** In this lab, increasing `Q` from a very small value can reduce lag at first, but if you keep increasing it the estimate becomes too noisy. Section 3 is now unlocked."), kind="success")
        if _correct else
        mo.callout(mo.md("❌ Not quite. Try **Scenario A + Piecewise linear** with `R = 20` or `40`, then compare `Q = 0.001`, `0.03`, and `0.3`."), kind="danger")
        if _wrong else
        mo.md("*Select an answer and click Submit to proceed.*")
    )
    mo.vstack([mo.md("---\n### ✏️ Check your understanding"), q2_form, _fb])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Multi-Anchor Localization
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _(mo, reset_count):
    _ = reset_count()
    anchor_seed_select = mo.ui.dropdown(
        options=[str(v) for v in (7, 21, 42, 73, 99)],
        value="42",
        label=None,
    )
    anchor_count_select = mo.ui.dropdown(
        options=["3", "4", "5", "10"],
        value="3",
        label=None,
    )
    anchor_spread_select = mo.ui.dropdown(
        options=["0.5x", "1.0x", "1.5x", "2.0x", "3.0x"],
        value="1.0x",
        label=None,
    )
    show_ls_loc = mo.ui.checkbox(value=True, label="Least squares")
    show_kf_loc = mo.ui.checkbox(value=True, label="KF + trilateration")
    show_ekf_loc = mo.ui.checkbox(value=True, label="Direct EKF")
    return (
        anchor_count_select, anchor_seed_select, anchor_spread_select,
        show_ekf_loc, show_kf_loc, show_ls_loc,
    )


@app.cell
def _(
    P0_slider, Q_slider, R_slider, dt_slider, go, labeled_control_row, make_subplots,
    mo, motion_select, np, quiz_toggle, run_range_kf, scenario_select,
    s3_unlocked, sc_effective, anchor_count_select, anchor_seed_select,
    anchor_spread_select, show_ekf_loc, show_kf_loc, show_ls_loc, simulate_rssi,
    t, traj, rssi_to_distance,
):
    mo.stop(
        quiz_toggle.value and not s3_unlocked(),
        mo.callout(mo.md(
            "🔒 **Section 3: Multi-anchor localization** — answer the quiz in Section 2 to unlock."
        ), kind="warn"),
    )

    _all_anchor_names = [
        "Anchor A", "Anchor B", "Anchor C", "Anchor D", "Anchor E",
        "Anchor F", "Anchor G", "Anchor H", "Anchor I", "Anchor J",
    ]
    _all_anchor_colors = [
        "#e74c3c", "#3498db", "#27ae60", "#f39c12", "#8e44ad",
        "#16a085", "#d35400", "#2c3e50", "#c0392b", "#7f8c8d",
    ]
    _all_anchors = np.array([
        [-4.0, -3.0],
        [10.0, -2.0],
        [2.5, 9.0],
        [12.0, 8.0],
        [-6.0, 7.0],
        [6.5, -6.5],
        [-8.0, 1.5],
        [0.0, 12.0],
        [13.0, 2.0],
        [-2.0, -8.0],
    ])
    _anchor_count = int(anchor_count_select.value)
    _anchor_spread = float(anchor_spread_select.value[:-1])
    _anchor_names = _all_anchor_names[:_anchor_count]
    _anchor_colors = _all_anchor_colors[:_anchor_count]
    _anchors = _all_anchors[:_anchor_count] * _anchor_spread

    def _distances_to_anchor(anchor_xy: np.ndarray) -> np.ndarray:
        return np.linalg.norm(traj.positions - anchor_xy, axis=1)

    def _trilaterate(anchors_xy: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(ranges)
        if np.count_nonzero(valid) < 3:
            return np.array([np.nan, np.nan])
        pts = anchors_xy[valid]
        rs = ranges[valid]
        x1, y1 = pts[0]
        r1 = rs[0]
        A = []
        b = []
        for (xi, yi), ri in zip(pts[1:], rs[1:]):
            A.append([2 * (xi - x1), 2 * (yi - y1)])
            b.append(r1**2 - ri**2 - x1**2 + xi**2 - y1**2 + yi**2)
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        est, *_ = np.linalg.lstsq(A, b, rcond=None)
        return est

    _rng = np.random.default_rng(int(anchor_seed_select.value))
    _true_ranges = np.stack([_distances_to_anchor(anchor) for anchor in _anchors], axis=1)
    _rssi_obs = []
    _raw_ranges = []
    _kf_ranges = []
    for idx in range(len(_anchors)):
        rssi_obs_i, _ = simulate_rssi(_true_ranges[:, idx], sc_effective, _rng)
        _rssi_obs.append(rssi_obs_i)
        raw_i = rssi_to_distance(rssi_obs_i, sc_effective)
        _raw_ranges.append(raw_i)
        _kf_ranges.append(
            run_range_kf(
                raw_i,
                dt=dt_slider.value,
                Q_var=Q_slider.value,
                R_var=R_slider.value,
                P0=P0_slider.value,
            ).estimates
        )

    _raw_ranges = np.stack(_raw_ranges, axis=1)
    _kf_ranges = np.stack(_kf_ranges, axis=1)
    _ls_positions = np.stack([_trilaterate(_anchors, r) for r in _raw_ranges], axis=0)
    _kf_positions = np.stack([_trilaterate(_anchors, r) for r in _kf_ranges], axis=0)

    fig_geo = go.Figure()
    fig_geo.add_trace(go.Scatter(
        x=traj.positions[:, 0], y=traj.positions[:, 1],
        mode="lines", line=dict(color="#2c3e50", width=3), name="True trajectory",
    ))
    if show_ls_loc.value:
        fig_geo.add_trace(go.Scatter(
            x=_ls_positions[:, 0], y=_ls_positions[:, 1],
            mode="lines", line=dict(color="#e67e22", width=2, dash="dot"),
            name="Least squares (raw ranges)",
        ))
    if show_kf_loc.value:
        fig_geo.add_trace(go.Scatter(
            x=_kf_positions[:, 0], y=_kf_positions[:, 1],
            mode="lines", line=dict(color="#3498db", width=2.5),
            name="KF + trilateration",
        ))
    for name, color, anchor in zip(_anchor_names, _anchor_colors, _anchors):
        fig_geo.add_trace(go.Scatter(
            x=[anchor[0]], y=[anchor[1]],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol="diamond"),
            text=[name], textposition="top center", name=name,
        ))
    fig_geo.update_layout(
        title="Geometry map",
        height=390,
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        margin=dict(t=40, b=10),
    )

    fig_rssi = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Anchor A", "Anchor B", "Anchor C"),
        horizontal_spacing=0.08,
    )
    # By design this panel stays limited to the first three anchors for readability,
    # even when the localization methods below use more anchors.
    for idx, (name, color) in enumerate(zip(_anchor_names[:3], _anchor_colors[:3]), start=1):
        fig_rssi.add_trace(go.Scatter(
            x=t, y=_true_ranges[:, idx - 1],
            mode="lines", line=dict(color=color, width=2.4),
            name=f"{name} true dist",
            legendgroup=name,
        ), row=1, col=idx)
        fig_rssi.add_trace(go.Scatter(
            x=t, y=_raw_ranges[:, idx - 1],
            mode="markers", marker=dict(color=color, size=4, opacity=0.35),
            name=f"{name} RSSI→dist",
            legendgroup=name,
            showlegend=False,
        ), row=1, col=idx)
        fig_rssi.update_xaxes(title_text="Time (s)", row=1, col=idx)
        fig_rssi.update_yaxes(title_text="Distance (m)", row=1, col=idx)
    fig_rssi.update_layout(
        title="Per-anchor range observations",
        height=390,
        margin=dict(t=55, b=10),
        legend=dict(orientation="h", y=-0.18),
    )

    def _path_rmse(est_xy: np.ndarray, truth_xy: np.ndarray) -> float:
        valid = ~np.isnan(est_xy).any(axis=1)
        if not np.any(valid):
            return np.nan
        return float(np.sqrt(np.mean(np.sum((est_xy[valid] - truth_xy[valid]) ** 2, axis=1))))

    _common_valid = ~np.isnan(_ls_positions).any(axis=1)
    _ls_rmse = _path_rmse(_ls_positions[_common_valid], traj.positions[_common_valid])
    _kf_rmse = _path_rmse(_kf_positions[_common_valid], traj.positions[_common_valid])
    _core_results_table = f"""
    <div style="font-size: 1.05rem;">
      <table>
        <thead>
          <tr><th>Method</th><th>RMSE</th></tr>
        </thead>
        <tbody>
          <tr><td>Least squares</td><td>{_ls_rmse:.2f} m</td></tr>
          <tr><td>KF + trilateration</td><td>{_kf_rmse:.2f} m</td></tr>
        </tbody>
      </table>
    </div>
    """
    mo.vstack([
        mo.md(f"""---
## Section 3 — Multi-anchor localization
"""),
        mo.callout(
            mo.md(f"""This section extends the Kalman filter concept from Section 2 to multiple anchors and 2D position estimation. With {_anchor_count} anchors, 
            we now have multiple noisy range measurements. The goal is to compare: 1) **Least squares**: A baseline that ignores per-anchor range quality 
            (uses raw RSSI-to-distance conversions directly), and 2) **KF + trilateration**: The improved approach that first filters each range independently 
            (as in Section 2), then trilaterates. The core insight: Filtering per-anchor ranges before trilateration produces better position estimates than trilateration on raw ranges.

            **How trilateration works here**

            At each time step, every anchor gives one range estimate to the target.
            In 2D, each range defines a circle centered on that anchor. The target should lie
            near the point where those circles intersect.

            In real data, the circles do not meet at exactly one point because the ranges are noisy.
            So this lab uses a **least-squares trilateration step**: it finds the 2D point that best
            fits all anchor ranges at that time step.

            - **Least squares**: uses the raw RSSI→distance values directly, then solves for the best-fit 2D point
            - **KF + trilateration**: first smooths each anchor's range with the Section 2 filter, then solves the same trilateration problem

            The only difference between the two methods is the quality of the input ranges.
            Better per-anchor ranges usually produce a better 2D position estimate.

            The key question for this section: *how much does per-anchor filtering improve the final 2D RMSE?*
            """),
            kind="neutral",
        ),

        mo.hstack([
            mo.vstack([fig_geo], align="start"),
            mo.vstack([fig_rssi], align="start"),
        ], justify="start", align="start", gap="1.5rem", widths=[1.1, 1.9]),
        mo.hstack([
            mo.vstack([
                mo.md("**Convenience controls**"),
                labeled_control_row("Motion", motion_select, ""),
                labeled_control_row("Seed", anchor_seed_select, ""),
                labeled_control_row("Propagation", scenario_select, ""),
                labeled_control_row("Anchors", anchor_count_select, ""),
                labeled_control_row("Spread", anchor_spread_select, ""),
            ], align="start"),
            mo.vstack([
                mo.md("**Kalman filter parameters**"),
                mo.md("*Same Q, R as Section 2, applied independently to each anchor range.*"),
                labeled_control_row("Process Q", Q_slider, f"{Q_slider.value:.3f}"),
                labeled_control_row("Measure R", R_slider, f"{R_slider.value:.1f} m²"),
                labeled_control_row("Initial P₀", P0_slider, f"{P0_slider.value:.1f}"),
            ], align="start"),
            mo.vstack([
                mo.md("**Results**"),
                mo.md(_core_results_table),
            ], align="start"),
            mo.vstack([
                mo.md("**Map visibility**"),
                show_ls_loc,
                show_kf_loc,
            ], align="start"),
        ], justify="start", align="start", gap="2rem", widths=[1.0, 1.0, 0.8, 0.9]),
                mo.callout(
            mo.md("""
            **🔬 Guided experiments**

            1. Start with the clean baseline:
               set `Q = 0.1`, `R = 20`, `P₀ = 10`, then use `3` anchors, `1.0x` spread, **Scenario A**, *Constant velocity*.
               Compare Least squares vs. KF + trilateration and note both the path shape and RMSE.
            2. Keep those same filter settings and scenario, and switch *Constant velocity → Piecewise linear*.
               The result seems counter-intuitive - despite the more complex *Piecewise linear* motion and low Q (remember our motion model is linear), 
                the performance of KF is actually better than *Constant velocity*. The reason is that the piecewise linear path is better-conditioned 
                as it's largely confined within the area of the anchors. 
            3. Now go back to *Constant velocity* and change the anchor geometry by increasing spread to `3.0x`.
               This shows that anchor placement matters: wider geometry improves the results in one case and worsens them in another. 
                Can you reason why this is the case? Think about the shape of the circles defined by each anchor's range and how they intersect.
            4. Experiment with increasing the number of anchors, setting spread to `0.5x`, and try **Scenario B** and **Scenario C** to explore harder or failure cases.
            You may still see a large percentage improvement from KF + trilateration, but the absolute 2D error can remain poor enough that localization is not practically useful.
            """),
            kind="info",
        ),
    ])
    return fig_geo, fig_rssi


@app.cell
def _(
    P0_slider, Q_slider, R_slider, dt_slider, go, labeled_control_row, make_subplots,
    mo, motion_select, np, run_direct_ekf, run_range_kf, scenario_select,
    sc_effective, anchor_count_select, anchor_seed_select, anchor_spread_select,
    show_ekf_loc, show_kf_loc, simulate_rssi, t, traj, rssi_to_distance,
):
    _all_anchor_names = [
        "Anchor A", "Anchor B", "Anchor C", "Anchor D", "Anchor E",
        "Anchor F", "Anchor G", "Anchor H", "Anchor I", "Anchor J",
    ]
    _all_anchor_colors = [
        "#e74c3c", "#3498db", "#27ae60", "#f39c12", "#8e44ad",
        "#16a085", "#d35400", "#2c3e50", "#c0392b", "#7f8c8d",
    ]
    _all_anchors = np.array([
        [-4.0, -3.0],
        [10.0, -2.0],
        [2.5, 9.0],
        [12.0, 8.0],
        [-6.0, 7.0],
        [6.5, -6.5],
        [-8.0, 1.5],
        [0.0, 12.0],
        [13.0, 2.0],
        [-2.0, -8.0],
    ])
    _anchor_count = int(anchor_count_select.value)
    _anchor_spread = float(anchor_spread_select.value[:-1])
    _anchor_names = _all_anchor_names[:_anchor_count]
    _anchor_colors = _all_anchor_colors[:_anchor_count]
    _anchors = _all_anchors[:_anchor_count] * _anchor_spread

    def _distances_to_anchor(anchor_xy: np.ndarray) -> np.ndarray:
        return np.linalg.norm(traj.positions - anchor_xy, axis=1)

    def _trilaterate(anchors_xy: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(ranges)
        if np.count_nonzero(valid) < 3:
            return np.array([np.nan, np.nan])
        pts = anchors_xy[valid]
        rs = ranges[valid]
        x1, y1 = pts[0]
        r1 = rs[0]
        A = []
        b = []
        for (xi, yi), ri in zip(pts[1:], rs[1:]):
            A.append([2 * (xi - x1), 2 * (yi - y1)])
            b.append(r1**2 - ri**2 - x1**2 + xi**2 - y1**2 + yi**2)
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        est, *_ = np.linalg.lstsq(A, b, rcond=None)
        return est

    _rng = np.random.default_rng(int(anchor_seed_select.value))
    _true_ranges = np.stack([_distances_to_anchor(anchor) for anchor in _anchors], axis=1)
    _raw_ranges = []
    _kf_ranges = []
    for _ekf_anchor_idx in range(len(_anchors)):
        _ekf_rssi_obs_i, _ = simulate_rssi(_true_ranges[:, _ekf_anchor_idx], sc_effective, _rng)
        _ekf_raw_i = rssi_to_distance(_ekf_rssi_obs_i, sc_effective)
        _raw_ranges.append(_ekf_raw_i)
        _kf_ranges.append(
            run_range_kf(
                _ekf_raw_i,
                dt=dt_slider.value,
                Q_var=Q_slider.value,
                R_var=R_slider.value,
                P0=P0_slider.value,
            ).estimates
        )

    _raw_ranges = np.stack(_raw_ranges, axis=1)
    _kf_ranges = np.stack(_kf_ranges, axis=1)
    _kf_positions = np.stack([_trilaterate(_anchors, r) for r in _kf_ranges], axis=0)
    _first_valid = next((p for p in _kf_positions if not np.isnan(p).any()), traj.positions[0])
    _ekf_result = run_direct_ekf(
        _raw_ranges,
        _anchors,
        dt=dt_slider.value,
        q_var=Q_slider.value,
        r_var=R_slider.value,
        p0=P0_slider.value,
        x0=np.array([_first_valid[0], _first_valid[1], 0.0, 0.0]),
    )
    _ekf_positions = _ekf_result.states[:, :2]

    fig_geo_ekf = go.Figure()
    fig_geo_ekf.add_trace(go.Scatter(
        x=traj.positions[:, 0], y=traj.positions[:, 1],
        mode="lines", line=dict(color="#2c3e50", width=3), name="True trajectory",
    ))
    if show_kf_loc.value:
        fig_geo_ekf.add_trace(go.Scatter(
            x=_kf_positions[:, 0], y=_kf_positions[:, 1],
            mode="lines", line=dict(color="#3498db", width=2.5),
            name="KF + trilateration",
        ))
    if show_ekf_loc.value:
        fig_geo_ekf.add_trace(go.Scatter(
            x=_ekf_positions[:, 0], y=_ekf_positions[:, 1],
            mode="lines", line=dict(color="#8e44ad", width=2.5, dash="dot"),
            name="Direct EKF",
        ))
    for _ekf_name, _ekf_color, _ekf_anchor in zip(_anchor_names, _anchor_colors, _anchors):
        fig_geo_ekf.add_trace(go.Scatter(
            x=[_ekf_anchor[0]], y=[_ekf_anchor[1]],
            mode="markers+text",
            marker=dict(size=14, color=_ekf_color, symbol="diamond"),
            text=[_ekf_name], textposition="top center", name=_ekf_name,
        ))
    fig_geo_ekf.update_layout(
        title="Geometry map",
        height=390,
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        margin=dict(t=40, b=10),
    )

    fig_rssi_ekf = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Anchor A", "Anchor B", "Anchor C"),
        horizontal_spacing=0.08,
    )
    for _ekf_plot_idx, (_ekf_plot_name, _ekf_plot_color) in enumerate(zip(_anchor_names[:3], _anchor_colors[:3]), start=1):
        fig_rssi_ekf.add_trace(go.Scatter(
            x=t, y=_true_ranges[:, _ekf_plot_idx - 1],
            mode="lines", line=dict(color=_ekf_plot_color, width=2.4),
            name=f"{_ekf_plot_name} true dist",
            legendgroup=_ekf_plot_name,
        ), row=1, col=_ekf_plot_idx)
        fig_rssi_ekf.add_trace(go.Scatter(
            x=t, y=_raw_ranges[:, _ekf_plot_idx - 1],
            mode="markers", marker=dict(color=_ekf_plot_color, size=4, opacity=0.35),
            name=f"{_ekf_plot_name} RSSI→dist",
            legendgroup=_ekf_plot_name,
            showlegend=False,
        ), row=1, col=_ekf_plot_idx)
        fig_rssi_ekf.update_xaxes(title_text="Time (s)", row=1, col=_ekf_plot_idx)
        fig_rssi_ekf.update_yaxes(title_text="Distance (m)", row=1, col=_ekf_plot_idx)
    fig_rssi_ekf.update_layout(
        title="Per-anchor range observations",
        height=390,
        margin=dict(t=55, b=10),
        legend=dict(orientation="h", y=-0.18),
    )

    def _path_rmse(est_xy: np.ndarray, truth_xy: np.ndarray) -> float:
        valid = ~np.isnan(est_xy).any(axis=1)
        if not np.any(valid):
            return np.nan
        return float(np.sqrt(np.mean(np.sum((est_xy[valid] - truth_xy[valid]) ** 2, axis=1))))

    _common_valid = ~np.isnan(_kf_positions).any(axis=1)
    _kf_rmse = _path_rmse(_kf_positions[_common_valid], traj.positions[_common_valid])
    _ekf_rmse = _path_rmse(_ekf_positions[_common_valid], traj.positions[_common_valid])
    _results_table = f"""
    <div style="font-size: 1.05rem;">
      <table>
        <thead>
          <tr><th>Method</th><th>RMSE</th></tr>
        </thead>
        <tbody>
          <tr><td>KF + trilateration</td><td>{_kf_rmse:.2f} m</td></tr>
          <tr><td>Direct EKF</td><td>{_ekf_rmse:.2f} m</td></tr>
        </tbody>
      </table>
    </div>
    """

    mo.vstack([
        mo.md("""---
## Section 4 — Direct EKF (Advanced)
"""),
        mo.callout(
            mo.md("""
            A Kalman filter is a recursive estimator: it predicts the next state of a system using a motion model,
            then corrects that prediction using noisy measurements. In Section 2, the state was 1D range. In Section 3,
            that range filter was applied independently to each anchor before trilateration.

            This section shows a more advanced alternative: a **direct Extended Kalman Filter (EKF)** that estimates
            the full 2D state `[x, y, v_x, v_y]` in one model. The EKF keeps the same Kalman-filter idea of
            predict-then-correct, but it handles a **nonlinear measurement model**: each anchor measures Euclidean
            distance to the target, not a linear function of the state. Because the measurement model is nonlinear,
            the EKF linearizes it at each step using a Jacobian matrix.

            The interactive controls are intentionally the same as in Section 3. Follow the same experiments again,
            but now compare the performance of the two Kalman-filter approaches:

            - **KF + trilateration**: filter each anchor range first, then solve for position
            - **Direct EKF**: estimate position and velocity jointly from all anchor ranges at once
            """),
            kind="neutral",
        ),
        mo.hstack([
            mo.vstack([fig_geo_ekf], align="start"),
            mo.vstack([fig_rssi_ekf], align="start"),
        ], justify="start", align="start", gap="1.5rem", widths=[1.1, 1.9]),
        mo.hstack([
            mo.vstack([
                mo.md("**Convenience controls**"),
                labeled_control_row("Motion", motion_select, ""),
                labeled_control_row("Seed", anchor_seed_select, ""),
                labeled_control_row("Propagation", scenario_select, ""),
                labeled_control_row("Anchors", anchor_count_select, ""),
                labeled_control_row("Spread", anchor_spread_select, ""),
            ], align="start"),
            mo.vstack([
                mo.md("**Kalman filter parameters**"),
                mo.md("*Same Q, R, and P₀ as Section 3.*"),
                labeled_control_row("Process Q", Q_slider, f"{Q_slider.value:.3f}"),
                labeled_control_row("Measure R", R_slider, f"{R_slider.value:.1f} m²"),
                labeled_control_row("Initial P₀", P0_slider, f"{P0_slider.value:.1f}"),
            ], align="start"),
            mo.vstack([
                mo.md("**Results**"),
                mo.md(_results_table),
                mo.md("*Repeat the Section 3 experiments and compare the two KF-based approaches.*"),
            ], align="start"),
            mo.vstack([
                mo.md("**Map visibility**"),
                show_kf_loc,
                show_ekf_loc,
            ], align="start"),
        ], justify="start", align="start", gap="2rem", widths=[1.0, 1.0, 0.9, 0.9]),
    ])
    return fig_geo_ekf, fig_rssi_ekf


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""---
## Section 5 — Filter Code Reference

This final section gives you two reusable filtering implementations from the lab:

**`filters/range_kf.py`** — A 1D Kalman filter for smoothing RSSI-to-range estimates from a single anchor.
**`filters/direct_ekf.py`** — A 2D Extended Kalman Filter that estimates position and velocity jointly from multi-anchor ranges.

The first is the simpler pipeline used throughout the core lab. The second is the advanced formulation from Section 4.
"""),
        mo.hstack([
            mo.callout(mo.md("""
**How to understand the code**

Open `filters/range_kf.py` and inspect:
- the state vector `[distance, velocity]` — what you're tracking
- the state transition matrix `F` — how you predict the next state from the current state
- the measurement matrix `H` — how measurement (noisy range) relates to state
- the covariance matrices `Q`, `R`, `P` — your beliefs about process noise, measurement noise, and uncertainty

The key insight: changing Q and R in this lab directly translates to changing the filter's behavior in production.
"""), kind="info"),
            mo.callout(mo.md("""
**How to understand the code**

Open `filters/direct_ekf.py` and inspect:
- the state vector `[x, y, v_x, v_y]` — position and velocity in 2D
- the constant-velocity transition model `F`
- the nonlinear measurement function `h(x)` that maps state to anchor ranges
- the Jacobian `H_k` — the linearization of that nonlinear measurement model
- the covariance matrices `Q`, `R`, `P` — the same uncertainty ideas, now applied to a 2D nonlinear estimator

The key insight: the EKF keeps the Kalman-filter structure, but replaces the simple linear measurement model with a local linear approximation at each step.
"""), kind="info"),
        ], justify="start", gap="1.5rem", widths=[1.0, 1.0]),
        mo.hstack([
            mo.md("""
### When To Use range_kf.py

- You have RSSI measurements from one anchor
- You want to smooth them into cleaner range estimates
- You need a simple, interpretable baseline before trying more complex approaches

### How To Adapt It For Your Projects

1. Replace the range values with your own RSSI-derived ranges
2. Tune Q, R, and P0 using the same principles you learned in Section 2
3. Check the filtered output against your raw measurements
4. If the filter is too sluggish, increase Q; if it is too noisy, decrease R

### Tuning Reminders

| Parameter | Controls | Increase if | Decrease if |
|-----------|----------|------------|------------|
| Q | Motion model trust | target moves fast | estimate is noisy |
| R | Measurement trust | measurements unreliable | filter lags behind changes |
| P0 | Initial uncertainty | you're unsure of starting state | you know the initial condition well |
| dt | Sampling interval | — set to match your actual measurement rate; do not tune this like Q or R |
"""),
            mo.md("""
### When To Use direct_ekf.py

- You have multiple anchors and want one joint 2D state estimate
- You want to estimate position and velocity together instead of filtering each anchor separately
- You are comfortable with a nonlinear measurement model and its extra complexity

### How To Adapt It For Your Projects

1. Replace the anchor coordinates and range measurements with your own data
2. Choose a sensible initial state `[x, y, v_x, v_y]`
3. Tune Q, R, and P0 using the same principles as before, but now judge the final 2D path quality
4. Compare EKF output against the simpler KF + trilateration pipeline before deciding the extra complexity is worth it
"""),
        ], justify="start", gap="1.5rem", widths=[1.0, 1.0]),
        mo.callout(mo.md("""
*Use `range_kf.py` when you want the simpler, more interpretable pipeline. Use `direct_ekf.py` when your project benefits from a joint 2D state model and you understand the added complexity.*
"""), kind="warn"),
    ])
    return


if __name__ == "__main__":
    app.run()
