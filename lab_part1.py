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

    Work through this lab in three steps. Each section unlocks once you answer the
    question at the bottom of the previous one.
    """)
    return

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
            Configure the radio environment and target motion below, then study how the
            RSSI signature changes before any filtering is applied.
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
    fig_map.update_layout(
        title="Target trajectory", height=400,
        xaxis_title="x (m)", yaxis_title="y (m)",
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
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
    fig_ts.update_layout(height=400, margin=dict(t=40, b=10),
        legend=dict(orientation="h", y=-0.18))
    fig_ts.update_yaxes(title_text="Distance (m)", row=1, col=1)
    fig_ts.update_yaxes(title_text="RSSI (dBm)",   row=2, col=1)
    fig_ts.update_xaxes(title_text="Time (s)",      row=2, col=1)

    mo.hstack([fig_map, fig_ts], justify="start")
    return

@app.cell
def _(mo):
    mo.callout(mo.md("""
    **🔬 Guided experiments — work through these before answering the quiz:**

    1. **Compare environments**: keep *Constant velocity*, cycle through Scenarios A → B → C.
       Watch how the RSSI plot and raw distance estimate degrade.
    2. **Motion vs environment**: switch to *Maneuvering* in Scenario A. Notice the trajectory
       shape. Then switch to Scenario C — two sources of error are now combined.
    3. **Burst interference**: in Scenario C, drag *Burst outlier prob* to 0.2.
       Where do the outlier spikes appear — distance estimate or RSSI?
    4. **Path-loss exponent**: set n = 2.0 vs n = 4.0. How does the range of the raw
       distance estimate change as the target moves away from the anchor?
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
        mo.callout(mo.md("✓ **Correct!** Scroll down — Section 2 is now unlocked."), kind="success")
        if _correct else
        mo.callout(mo.md("❌ Not quite. Hint: which environment includes co-channel interference?"), kind="danger")
        if _wrong else
        mo.md("*Select an answer and click Submit to proceed.*")
    )
    mo.vstack([mo.md("---\n### ✏️ Check your understanding"), q1_form, _fb])
    return




# ═══════════════════════════════════════════════════════════════════════════════
# KF controls — always defined so reset works regardless of which section is open
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _(mo, reset_count):
    _ = reset_count()
    Q_slider    = mo.ui.slider(0.001, 10.0,  value=0.1,  step=0.001, label=None)
    R_slider    = mo.ui.slider(0.1,   100.0, value=4.0,  step=0.1,   label=None)
    P0_slider   = mo.ui.slider(0.1,   100.0, value=10.0, step=0.1,   label=None)
    ma_window   = mo.ui.slider(3, 30, value=10, step=1,              label=None)
    show_ma_dBm = mo.ui.checkbox(value=True,  label="Moving avg (dBm domain)")
    show_ma_dist= mo.ui.checkbox(value=True,  label="Moving avg (distance domain)")
    show_kf     = mo.ui.checkbox(value=True,  label="Kalman filter")
    return (
        P0_slider, Q_slider, R_slider, ma_window,
        show_kf, show_ma_dBm, show_ma_dist,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════

# S2 controls + KF plot
@app.cell
def _(
    P0_slider, Q_slider, R_slider, d_raw, dt_slider, go, labeled_control_row, ma_window,
    make_subplots, mo, moving_average, np, rssi_obs, rssi_to_distance,
    quiz_toggle, run_range_kf, s2_unlocked, sc_effective, show_kf,
    show_ma_dBm, show_ma_dist, t, true_dists,
):
    mo.stop(
        quiz_toggle.value and not s2_unlocked(),
        mo.callout(mo.md(
            "🔒 **Section 2: Kalman Filter** — answer the quiz in Section 1 to unlock."
        ), kind="warn"),
    )

    # ── compute ───────────────────────────────────────────────────────────────
    d_ma_dBm  = rssi_to_distance(moving_average(rssi_obs, ma_window.value), sc_effective)
    d_ma_dist = moving_average(d_raw, ma_window.value)
    kf_result = run_range_kf(d_raw, dt=dt_slider.value,
                             Q_var=Q_slider.value, R_var=R_slider.value, P0=P0_slider.value)

    def _rmse(e, r): return np.sqrt(np.nanmean((e - r) ** 2))
    def _mae(e, r):  return np.nanmean(np.abs(e - r))
    metrics = {
        "Raw RSSI→dist": (_rmse(d_raw, true_dists),              _mae(d_raw, true_dists)),
        "MA (dBm)":      (_rmse(d_ma_dBm, true_dists),           _mae(d_ma_dBm, true_dists)),
        "MA (distance)": (_rmse(d_ma_dist, true_dists),          _mae(d_ma_dist, true_dists)),
        "Kalman filter": (_rmse(kf_result.estimates, true_dists), _mae(kf_result.estimates, true_dists)),
    }

    # ── plot ──────────────────────────────────────────────────────────────────
    fig_kf = go.Figure()
    fig_kf.add_trace(go.Scatter(x=t, y=true_dists, name="True distance",
        line=dict(color="#2ecc71", width=2.5)))
    fig_kf.add_trace(go.Scatter(x=t, y=d_raw, name="Raw", mode="markers",
        marker=dict(color="#e74c3c", size=3, opacity=0.4)))
    if show_ma_dBm.value:
        fig_kf.add_trace(go.Scatter(x=t, y=d_ma_dBm, name="MA (dBm)",
            line=dict(color="#e67e22", width=1.5, dash="dot")))
    if show_ma_dist.value:
        fig_kf.add_trace(go.Scatter(x=t, y=d_ma_dist, name="MA (dist)",
            line=dict(color="#f39c12", width=1.5, dash="dash")))
    if show_kf.value:
        fig_kf.add_trace(go.Scatter(x=t, y=kf_result.estimates, name="Kalman filter",
            line=dict(color="#3498db", width=2)))
    fig_kf.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Distance (m)",
        legend=dict(orientation="h", y=-0.18), margin=dict(t=30, b=10))

    # ── metrics table ─────────────────────────────────────────────────────────
    _tbl_rows = "\n".join(
        f"<tr><td>{k}</td><td>{r:.2f} m</td><td>{m:.2f} m</td></tr>"
        for k, (r, m) in metrics.items()
    )
    _tbl = f"""
    <div style="font-size: 1.05rem;">
      <table>
        <thead>
          <tr><th>Estimator</th><th>RMSE</th><th>MAE</th></tr>
        </thead>
        <tbody>
          {_tbl_rows}
        </tbody>
      </table>
    </div>
    """

    # ── layout ────────────────────────────────────────────────────────────────
    mo.vstack([
        mo.md("---\n## Section 2 — Kalman Filter\n\nConfigure the filter on the left, observe its performance on the right."),
        mo.hstack([
            mo.vstack([
                mo.md("**Filter parameters**"),
                labeled_control_row("Process Q", Q_slider, f"{Q_slider.value:.3f}"),
                labeled_control_row("Measure R", R_slider, f"{R_slider.value:.1f} m²"),
                labeled_control_row("Initial P₀", P0_slider, f"{P0_slider.value:.1f}"),
                labeled_control_row("MA window", ma_window, f"{ma_window.value:d}"),
            ], align="start"),
            mo.vstack([
                fig_kf,
                mo.hstack(
                    [show_ma_dBm, show_ma_dist, show_kf],
                    justify="center",
                    align="center",
                    gap="1.5rem",
                ),
            ], align="start"),
            mo.vstack([
                mo.md("**Results**"),
                mo.md(_tbl),
            ], align="start"),
        ], justify="start", align="start", gap="2rem", widths=[1.0, 2.0, 1.0]),
    ])
    return d_ma_dBm, d_ma_dist, kf_result, metrics


@app.cell
def _(mo, quiz_toggle, s2_unlocked):
    mo.stop(quiz_toggle.value and not s2_unlocked())
    mo.callout(mo.md("""
    **🔬 Guided experiments:**

    1. **Q too small**: set Q = 0.001 with *Maneuvering* motion. The filter lags noticeably at direction changes.
    2. **Q too large**: set Q = 5.0. The filter tracks every measurement noise spike — over-responsive.
    3. **R too large**: set R = 80. The filter barely moves; it ignores measurements almost entirely.
    4. **Domain matters**: compare MA (dBm) vs MA (distance). Which has lower RMSE in Scenario C? Why?
    5. **Best combo**: find Q, R values that give the lowest KF RMSE for *Maneuvering* in Scenario C.
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
                "It becomes more responsive to rapid direction changes",
                "It appears smoother but lags behind rapid turns",
                "It has no effect — Q only controls the uncertainty band",
            ],
            label=(
                "With Scenario A and Maneuvering motion, reduce Q from 1.0 to 0.001. "
                "The Kalman estimate now…"
            ),
        ),
        submit_button_label="Submit answer →",
    )
    return (q2_form,)


@app.cell
def _(mo, q2_form, quiz_toggle, s2_unlocked, set_s3):
    mo.stop((not quiz_toggle.value) or (quiz_toggle.value and not s2_unlocked()))
    _correct = q2_form.value is not None and "lags" in q2_form.value
    _wrong   = q2_form.value is not None and "lags" not in q2_form.value
    if _correct:
        set_s3(True)
    _fb = (
        mo.callout(mo.md("✓ **Correct!** Scroll down — Section 3 is now unlocked."), kind="success")
        if _correct else
        mo.callout(mo.md("❌ Not quite. Try Q = 0.001 with Maneuvering motion and observe the lag."), kind="danger")
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
    mo, motion_select, np, quiz_toggle, run_direct_ekf, run_range_kf, scenario_select,
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
    _first_valid = next((p for p in _ls_positions if not np.isnan(p).any()), traj.positions[0])
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

    fig_geo = go.Figure()
    fig_geo.add_trace(go.Scatter(
        x=traj.positions[:, 0], y=traj.positions[:, 1],
        mode="lines", line=dict(color="#2c3e50", width=3), name="True trajectory",
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

    fig_loc = go.Figure()
    fig_loc.add_trace(go.Scatter(
        x=traj.positions[:, 0], y=traj.positions[:, 1],
        mode="lines", line=dict(color="#2c3e50", width=3), name="True trajectory",
    ))
    if show_ls_loc.value:
        fig_loc.add_trace(go.Scatter(
            x=_ls_positions[:, 0], y=_ls_positions[:, 1],
            mode="lines", line=dict(color="#e67e22", width=2, dash="dot"),
            name="Least squares (raw ranges)",
        ))
    if show_kf_loc.value:
        fig_loc.add_trace(go.Scatter(
            x=_kf_positions[:, 0], y=_kf_positions[:, 1],
            mode="lines", line=dict(color="#3498db", width=2.5),
            name="KF + trilateration",
        ))
    if show_ekf_loc.value:
        fig_loc.add_trace(go.Scatter(
            x=_ekf_positions[:, 0], y=_ekf_positions[:, 1],
            mode="lines", line=dict(color="#8e44ad", width=2.5),
            name="Direct EKF",
        ))
    for name, color, anchor in zip(_anchor_names, _anchor_colors, _anchors):
        fig_loc.add_trace(go.Scatter(
            x=[anchor[0]], y=[anchor[1]],
            mode="markers",
            marker=dict(size=12, color=color, symbol="diamond"),
            name=name,
            showlegend=False,
        ))
    fig_loc.update_layout(
        title="2D location estimate",
        height=390,
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        margin=dict(t=40, b=10),
    )

    def _path_rmse(est_xy: np.ndarray, truth_xy: np.ndarray) -> float:
        valid = ~np.isnan(est_xy).any(axis=1)
        if not np.any(valid):
            return np.nan
        return float(np.sqrt(np.mean(np.sum((est_xy[valid] - truth_xy[valid]) ** 2, axis=1))))

    _common_valid = ~np.isnan(_ls_positions).any(axis=1)
    _ls_rmse = _path_rmse(_ls_positions[_common_valid], traj.positions[_common_valid])
    _kf_rmse = _path_rmse(_kf_positions[_common_valid], traj.positions[_common_valid])
    _ekf_rmse = _path_rmse(_ekf_positions[_common_valid], traj.positions[_common_valid])
    _results_table = f"""
    <div style="font-size: 1.05rem;">
      <table>
        <thead>
          <tr><th>Method</th><th>RMSE</th></tr>
        </thead>
        <tbody>
          <tr><td>Least squares</td><td>{_ls_rmse:.2f} m</td></tr>
          <tr><td>KF + trilateration</td><td>{_kf_rmse:.2f} m</td></tr>
          <tr><td>Direct EKF</td><td>{_ekf_rmse:.2f} m</td></tr>
        </tbody>
      </table>
    </div>
    """
    mo.vstack([
        mo.md(f"---\n## Section 3 — Multi-anchor localization\n\n{_anchor_count} anchors observe the moving target. Compare raw least-squares trilateration against per-anchor range KFs followed by trilateration."),
        mo.hstack([
            mo.vstack([fig_geo], align="start"),
            mo.vstack([fig_rssi], align="start"),
            mo.vstack([fig_loc], align="start"),
        ], justify="start", align="start", gap="1.5rem", widths=[1.0, 1.5, 1.0]),
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
                mo.md("*Q, R are shared with Section 2; their EKF interpretation differs.*"),
                labeled_control_row("Process Q", Q_slider, f"{Q_slider.value:.3f}"),
                labeled_control_row("Measure R", R_slider, f"{R_slider.value:.1f} m²"),
                labeled_control_row("Initial P₀", P0_slider, f"{P0_slider.value:.1f}"),
            ], align="start"),
            mo.vstack([
                mo.md(_results_table),
            ], align="start"),
            mo.vstack([
                mo.md("**Map visibility**"),
                show_ls_loc,
                show_kf_loc,
                show_ekf_loc,
            ], align="start"),
        ], justify="start", align="start", gap="2rem", widths=[1.0, 1.0, 0.8, 0.9]),
    ])
    return fig_geo, fig_loc, fig_rssi


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""---
## Section 4 — Useful Algorithms For Future Projects

This lab is not only about this week's exercises. It also gives you two practical filtering algorithms that are useful in future RSSI localization, tracking, and sensing projects:

1. `filters/range_kf.py` for a 1D Kalman filter on range
2. `filters/direct_ekf.py` for a 2D Extended Kalman Filter (EKF) on position and velocity

These are worth keeping as reference implementations. In future projects, you can reuse them, adapt them, and extend them rather than starting from zero each time.
"""),
        mo.callout(mo.md("""
**What to inspect in the code**

- In `range_kf.py`, inspect:
  - the state vector `[distance, velocity]`
  - the state transition matrix `F`
  - the measurement matrix `H`
  - the covariance matrices `Q`, `R`, and `P`
- In `direct_ekf.py`, inspect:
  - the state vector `[x, y, v_x, v_y]`
  - the constant-velocity transition model
  - the nonlinear measurement function `h(x)` that maps state to anchor ranges
  - the Jacobian matrix used to linearize `h(x)` at the current estimate

Do not treat these as black boxes. Before using them, make sure you can explain what each state variable means and what each matrix is doing.
"""), kind="info"),
        mo.md("""
### How You Can Reuse Them

- Convert real RSSI measurements into range estimates.
- Feed those ranges into the provided filter code.
- Compare raw measurements against filtered outputs.
- Adapt the same code structure to other sensing tasks where measurements are noisy.

### When To Use Each Filter

- Use `range_kf.py` when:
  - you are working with one anchor at a time
  - you want to smooth noisy RSSI-to-range estimates
  - you want a simple baseline before moving to full localization
- Use `direct_ekf.py` when:
  - you have multiple anchors
  - you want to estimate 2D position directly
  - your measurement model is nonlinear because each anchor measures Euclidean distance

### What To Tune

- `Q`:
  - controls how much the filter trusts the motion model
  - increase it if the target changes direction or speed quickly
  - decrease it if the estimate is too noisy or jittery
- `R`:
  - controls how much the filter trusts the measurements
  - increase it if RSSI-derived ranges are very noisy, bursty, or unreliable
  - decrease it if the filter is too sluggish and ignores useful measurements
- `P0`:
  - controls how uncertain the initial state is
  - increase it if your initial position/range guess is poor
  - decrease it if you know the initial condition reasonably well
- `dt`:
  - must match the actual sampling interval of your data
  - if it is wrong, the motion model will be wrong too

### Good Habits For Future Projects

- Start with raw RSSI and raw RSSI-to-range plots before filtering anything.
- Start with the 1D range KF to understand tuning on a simpler problem.
- Then move to the direct EKF when you want full 2D localization.
- When tuning, change one parameter at a time and observe what it does to lag, noise, and stability.
- If your real system has packet drops or burst interference, expect to increase `R` and possibly also `Q`.

### Why These Implementations Matter

If you understand these two filters well, you already have reusable building blocks for:

- single-anchor range smoothing
- multi-anchor RSSI localization
- robot or asset tracking
- sensor fusion projects where the measurement model is noisy or nonlinear

The main takeaway is not just how to run this notebook. It is how to recognize a noisy estimation problem, choose an appropriate filter, and tune it sensibly.
"""),
    ])
    return


if __name__ == "__main__":
    app.run()
