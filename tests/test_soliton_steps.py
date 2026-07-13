"""Solver-free synthetic regression tests for the soliton-staircase pipeline.

All tests use SYNTHETIC data (analytic staircases, breather-like power series,
multi-sech circular fields) so they never run the solver -- the full solver
sweep lives in the driver ``analysis/run_detuning_sweep.py``.  Two tiers live
here:

* The PRE-EXISTING ``detect_power_steps`` / ``hold_window_average`` /
  alignment-helper tests.  These are the standing proof that the step detector
  was never retuned to make the staircase appear -- they pin its default
  behaviour (boundary recovery within one sample, no false positives on
  smooth/noisy branches, k-sensitivity, degenerate inputs, the sigma == 0
  fallback) and must keep passing UNCHANGED.
* The CANONICAL FAILURE-MODE fixture :func:`breathing_modulated_field` and the
  regression tests built on it, which encode the two forensically established
  counting defects of the staircase pipeline (see
  ``analysis/results/staircase_forensics.md``): breathing-phase dropout under
  end-of-hold single-snapshot counting, and sibling-crest relative-threshold
  coupling.  Companion counter-mechanics tests (wrap-around clustering,
  persistence rejection, the physics-anchor arm) live in
  ``tests/test_windowed_counting.py``; ``temporal_peak_positions`` angle
  recovery also lives in ``tests/test_dks_access.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from analysis.dks_access import (
    count_solitons_windowed,
    count_temporal_peaks,
    temporal_peak_positions,
)
from analysis.spectral_metrics import (
    detect_power_steps,
    hold_window_average,
    match_steps_to_transitions,
    plot_soliton_steps,
    single_dks_region,
    soliton_count_transitions,
)


# ---------------------------------------------------------------------------
# synthetic staircase
# ---------------------------------------------------------------------------
def _staircase(levels, widths, *, noise=0.0, seed=0):
    """Piecewise-constant y with plateaus ``levels`` of lengths ``widths``.

    Returns ``(x, y, boundaries)`` where ``boundaries[j]`` is the index of the
    LAST sample of plateau j (so the step edge from plateau j to j+1 sits at
    ``boundaries[j]``, between x[boundaries[j]] and x[boundaries[j]+1]).
    """
    rng = np.random.default_rng(seed)
    y, bounds, idx = [], [], -1
    for lev, w in zip(levels, widths):
        y.extend([lev] * w)
        idx += w
        bounds.append(idx)
    y = np.asarray(y, dtype=np.float64)
    if noise:
        y = y + rng.normal(0.0, noise, y.size)
    x = np.arange(y.size, dtype=np.float64)
    return x, y, bounds[:-1]          # drop the last (trace end, not a step)


# ---------------------------------------------------------------------------
# detect_power_steps: boundary recovery within one sample
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_detect_steps_recovers_boundaries_within_one_sample(seed):
    # 3 plateaus, a clear ~2x drop each; plateau ripple ~2% of the drop.
    x, y, bounds = _staircase([1.0, 0.6, 0.25], [20, 15, 18],
                              noise=0.008, seed=seed)
    res = detect_power_steps(x, y, k=6.0)
    assert res["n_steps"] == 2, (res["n_steps"], res["step_x"])
    for detected, truth in zip(sorted(res["edges"]), bounds):
        assert abs(detected - truth) <= 1, (detected, truth)
    # plateaus partition every sample exactly once, in order
    covered = [i for a, b in res["plateaus"] for i in range(a, b + 1)]
    assert covered == list(range(y.size))


def test_detect_steps_boundaries_map_to_x_midpoints():
    x, y, bounds = _staircase([2.0, 1.0], [10, 10], noise=0.0)
    res = detect_power_steps(x, y, k=6.0)
    assert res["n_steps"] == 1
    e = res["edges"][0]
    assert res["step_x"][0] == pytest.approx(0.5 * (x[e] + x[e + 1]))
    assert res["step_dy"][0] == pytest.approx(y[e + 1] - y[e])


# ---------------------------------------------------------------------------
# detect_power_steps: no false positives on smooth data
# ---------------------------------------------------------------------------
def test_detect_steps_smooth_ramp_has_no_steps():
    x = np.linspace(7.0, 29.0, 45)
    y = 1.0 - 0.01 * (x - 7.0)                 # gentle monotone branch, no jump
    res = detect_power_steps(x, y, k=6.0)
    assert res["n_steps"] == 0
    assert res["plateaus"] == [[0, x.size - 1]]


def test_detect_steps_noisy_smooth_branch_no_false_positive():
    rng = np.random.default_rng(3)
    x = np.linspace(7.0, 29.0, 45)
    y = 1.0 - 0.01 * (x - 7.0) + rng.normal(0.0, 0.003, x.size)
    assert detect_power_steps(x, y, k=6.0)["n_steps"] == 0


@pytest.mark.parametrize("seed", range(5))
def test_detect_steps_gentle_branch_then_collapse_finds_one_step(seed):
    # Mirrors the real DKS trace: a gently declining soliton branch, a sharp
    # collapse, then a low (slightly sloped) post-annihilation plateau, with the
    # per-step averaging noise a real trace always carries.  Exactly one step,
    # at the collapse -- the gentle branch/plateau slopes must NOT register.
    rng = np.random.default_rng(seed)
    x = np.arange(40, dtype=np.float64)
    branch = 1.0 - 0.006 * x[:30]              # slow decline (dy ~ -0.006)
    collapsed = 0.16 - 0.001 * (x[30:] - 30)   # low plateau, slight slope
    y = np.concatenate([branch, collapsed]) + rng.normal(0.0, 0.006, x.size)
    res = detect_power_steps(x, y, k=6.0)
    assert res["n_steps"] == 1, (res["n_steps"], res["step_x"])
    assert abs(res["edges"][0] - 29) <= 1
    assert res["step_dy"][0] < 0               # a drop


def test_detect_steps_k_controls_sensitivity():
    x, y, _ = _staircase([1.0, 0.95, 0.2], [15, 10, 15], noise=0.01, seed=1)
    # small first step (5%) + large second (~0.75): a very high k keeps only the
    # large one; a modest k finds both.
    assert detect_power_steps(x, y, k=3.0)["n_steps"] >= 2
    assert detect_power_steps(x, y, k=30.0)["n_steps"] == 1


# ---------------------------------------------------------------------------
# detect_power_steps: degenerate / edge cases (never raise)
# ---------------------------------------------------------------------------
def test_detect_steps_short_input_returns_no_steps():
    for n in (0, 1, 2):
        x = np.arange(n, dtype=float)
        res = detect_power_steps(x, x, k=6.0)
        assert res["n_steps"] == 0


def test_detect_steps_perfectly_linear_plus_jump_sigma_zero_branch():
    # dy is constant on the ramp (MAD == 0), so the robust scale falls back to a
    # data-range floor; the lone jump must still be flagged.
    x = np.arange(30, dtype=np.float64)
    y = 0.02 * x
    y[15:] += 3.0                              # a clean unit-scale jump at edge 14
    res = detect_power_steps(x, y, k=6.0)
    assert res["n_steps"] >= 1
    assert any(abs(e - 14) <= 1 for e in res["edges"])


def test_detect_steps_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        detect_power_steps(np.arange(5.0), np.arange(4.0))


# ---------------------------------------------------------------------------
# hold_window_average: window selection + linear-power averaging
# ---------------------------------------------------------------------------
def test_hold_window_average_selects_final_fraction():
    s = np.arange(100.0)
    r = hold_window_average(s, avg_frac=0.25)
    assert r["i_start"] == 75 and r["n_window"] == 25 and r["n_total"] == 100
    assert r["mean"] == pytest.approx(np.mean(s[75:]))
    assert r["std"] == pytest.approx(np.std(s[75:]))


@pytest.mark.parametrize("n,frac,i0", [(100, 0.25, 75), (40, 0.5, 20),
                                       (7, 0.25, 5), (1, 0.25, 0), (10, 1.0, 0)])
def test_hold_window_average_window_bounds(n, frac, i0):
    r = hold_window_average(np.ones(n), avg_frac=frac)
    assert r["i_start"] == i0
    assert r["n_window"] == n - i0
    assert r["n_window"] >= 1


def test_hold_window_average_is_linear_power_over_breather_cycles():
    # A breather: mean power + a sinusoidal oscillation over an integer number of
    # periods in the averaging window.  The linear-power average recovers the DC
    # mean; the std reports the breathing amplitude (~A/sqrt(2)).
    n = 4000
    period = 180.0
    t = np.arange(n)
    dc, amp = 3.0e-21, 0.12e-21
    power = dc + amp * np.sin(2 * np.pi * t / period)
    r = hold_window_average(power, avg_frac=0.25)         # last 1000 ~ 5.5 periods
    assert r["mean"] == pytest.approx(dc, rel=2e-3)
    assert r["std"] == pytest.approx(amp / np.sqrt(2.0), rel=0.1)


def test_hold_window_average_rejects_complex_and_bad_input():
    with pytest.raises(ValueError):
        hold_window_average(np.ones(10, dtype=complex))       # complex field
    with pytest.raises(ValueError):
        hold_window_average(np.array([]))                     # empty
    with pytest.raises(ValueError):
        hold_window_average(np.ones(10), avg_frac=0.0)        # frac out of range
    with pytest.raises(ValueError):
        hold_window_average(np.ones(10), avg_frac=1.5)
    with pytest.raises(ValueError):
        hold_window_average(np.array([1.0, np.nan, 2.0]))     # non-finite


def test_hold_window_average_matches_manual_on_random_series():
    rng = np.random.default_rng(7)
    s = rng.uniform(1.0, 2.0, 333)
    r = hold_window_average(s, avg_frac=0.3)
    i0 = int(np.floor(333 * 0.7))
    assert r["i_start"] == i0
    assert r["mean"] == pytest.approx(s[i0:].mean())


# ---------------------------------------------------------------------------
# single_dks_region: existence span + annihilation edge
# ---------------------------------------------------------------------------
def test_single_dks_region_span_and_annihilation_edge():
    # branch is single (True) on the high-detuning side; annihilation is the
    # midpoint between the lowest soliton point and the next-lower non-single one.
    dw = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0])       # ascending
    single = np.array([False, False, False, True, True, True, True])
    lo, hi, annih = single_dks_region(dw, single)
    assert (lo, hi) == (6.0, 8.0)
    assert annih == pytest.approx(5.75)                      # midpoint 5.5 & 6.0


def test_single_dks_region_unsorted_input():
    dw = np.array([8.0, 6.0, 4.5, 7.0, 5.0])                 # scrambled order
    single = np.array([True, True, False, True, False])
    lo, hi, annih = single_dks_region(dw, single)
    assert (lo, hi) == (6.0, 8.0)                            # 6,7,8 contiguous
    assert annih == pytest.approx(5.5)                       # midpoint 5.0 & 6.0


def test_single_dks_region_picks_longest_run():
    dw = np.arange(10.0)
    single = np.array([True, False, True, True, True, False, True, True, False, True])
    lo, hi, _ = single_dks_region(dw, single)
    assert (lo, hi) == (2.0, 4.0)                            # the length-3 run


def test_single_dks_region_none_when_no_soliton():
    dw = np.arange(5.0)
    assert single_dks_region(dw, np.zeros(5, bool)) == (None, None, None)


def test_single_dks_region_branch_touches_lower_edge_has_no_annihilation():
    dw = np.array([4.0, 5.0, 6.0])
    single = np.array([True, True, True])                    # single at the edge
    lo, hi, annih = single_dks_region(dw, single)
    assert (lo, hi) == (4.0, 6.0) and annih is None


# ---------------------------------------------------------------------------
# soliton_count_transitions: state-transition edges of a sweep
# ---------------------------------------------------------------------------
def test_soliton_count_transitions_basic_fields():
    dw = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    c = np.array([0, 1, 1, 2, 3])
    tr = soliton_count_transitions(dw, c)
    assert [t["edge_index"] for t in tr] == [0, 2, 3]
    t0 = tr[0]
    assert t0["dw_mid"] == pytest.approx(5.5)
    assert t0["n_low_side"] == 0 and t0["n_high_side"] == 1
    assert t0["delta_n"] == 1                    # one soliton lost going down
    assert tr[2] == {"edge_index": 3, "dw_mid": pytest.approx(8.5),
                     "n_high_side": 3, "n_low_side": 2, "delta_n": 1}


def test_soliton_count_transitions_sorts_ascending_like_single_dks_region():
    dw = np.array([9.0, 5.0, 7.0, 8.0, 6.0])                 # scrambled order
    c = np.array([3, 0, 1, 2, 1])
    # ascending: (5,0) (6,1) (7,1) (8,2) (9,3) -> edges 0, 2, 3
    assert [t["edge_index"]
            for t in soliton_count_transitions(dw, c)] == [0, 2, 3]


def test_soliton_count_transitions_merged_annihilation_delta_n():
    dw = np.arange(4.0)
    c = np.array([1, 3, 3, 3])                   # two solitons lost at one edge
    (t,) = soliton_count_transitions(dw, c)
    assert t["delta_n"] == 2


def test_soliton_count_transitions_constant_and_short_input():
    dw = np.arange(4.0)
    assert soliton_count_transitions(dw, np.ones(4, dtype=int)) == []
    assert soliton_count_transitions(np.array([1.0]), np.array([2])) == []
    assert soliton_count_transitions(np.array([]), np.array([])) == []


def test_soliton_count_transitions_validation():
    dw = np.arange(4.0)
    with pytest.raises(ValueError):              # shape mismatch
        soliton_count_transitions(dw, np.ones(3, dtype=int))
    with pytest.raises(ValueError):              # non-integral count
        soliton_count_transitions(dw, np.array([0.0, 0.5, 1.0, 1.0]))
    with pytest.raises(ValueError):              # negative count
        soliton_count_transitions(dw, np.array([0, -1, 1, 1]))
    with pytest.raises(ValueError):              # non-finite count
        soliton_count_transitions(dw, np.array([0.0, np.nan, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# match_steps_to_transitions: step <-> state-transition alignment
# ---------------------------------------------------------------------------
def _steps_dict(edges, step_dy=None):
    """Minimal detect_power_steps-shaped dict on an integer x grid."""
    edges = list(edges)
    return {"edges": edges,
            "step_x": [e + 0.5 for e in edges],
            "step_dy": list(step_dy) if step_dy is not None
            else [-1.0] * len(edges)}


def test_match_exact_and_within_tolerance():
    dw = np.arange(10.0)
    c = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    tr = soliton_count_transitions(dw, c)        # edges 1, 4, 6
    res = match_steps_to_transitions(_steps_dict([1, 5]), tr, tol_samples=1)
    assert len(res["matched"]) == 2
    m0 = res["matched"][0]
    assert m0["step_edge_index"] == 1 and m0["transition_edge_index"] == 1
    assert m0["edge_distance"] == 0 and m0["delta_n"] == 1
    # step 5 is one sample from both edge 4 and edge 6 -- it pairs with
    # exactly ONE of them (one-to-one matching), the other stays unmatched
    assert res["matched"][1]["step_edge_index"] == 5
    assert res["matched"][1]["transition_edge_index"] in (4, 6)
    assert len(res["unmatched_transitions"]) == 1
    assert res["unmatched_steps"] == []


def test_match_tol_zero_requires_exact_coincidence():
    dw = np.arange(6.0)
    c = np.array([1, 1, 2, 2, 3, 3])             # transition edges 1, 3
    res = match_steps_to_transitions(_steps_dict([2, 3]),
                                     soliton_count_transitions(dw, c),
                                     tol_samples=0)
    assert [m["step_edge_index"] for m in res["matched"]] == [3]
    assert [s["edge_index"] for s in res["unmatched_steps"]] == [2]


def test_match_unmatched_step_is_mi_rise_not_soliton_step():
    # a power discontinuity where the count never changes (the near-resonance
    # MI/CW rise) must land in unmatched_steps, never in matched
    dw = np.arange(8.0)
    c = np.array([0, 0, 0, 0, 1, 1, 2, 2])       # transitions at 3, 5
    res = match_steps_to_transitions(_steps_dict([0, 3, 5]),
                                     soliton_count_transitions(dw, c))
    assert [m["step_edge_index"] for m in res["matched"]] == [3, 5]
    assert [s["edge_index"] for s in res["unmatched_steps"]] == [0]


def test_match_muted_final_transition_stays_unmatched():
    # the power-muted 1->0 annihilation: a count transition with NO power step
    dw = np.arange(8.0)
    c = np.array([0, 1, 1, 1, 2, 2, 3, 3])       # transitions at 0, 3, 5
    res = match_steps_to_transitions(_steps_dict([3, 5]),
                                     soliton_count_transitions(dw, c))
    assert len(res["matched"]) == 2
    (um,) = res["unmatched_transitions"]
    assert um["edge_index"] == 0
    assert um["n_high_side"] == 1 and um["n_low_side"] == 0


def test_match_one_to_one_no_double_counting():
    # two detected steps adjacent to ONE transition: only one may match it
    dw = np.arange(4.0)
    c = np.array([0, 0, 1, 1])                   # single transition at edge 1
    res = match_steps_to_transitions(_steps_dict([0, 2]),
                                     soliton_count_transitions(dw, c),
                                     tol_samples=1)
    assert len(res["matched"]) == 1
    assert len(res["unmatched_steps"]) == 1


def test_match_prefers_closest_pairing():
    dw = np.arange(8.0)
    c = np.array([0, 0, 0, 1, 1, 1, 1, 1])       # single transition at edge 2
    res = match_steps_to_transitions(_steps_dict([2, 3]),
                                     soliton_count_transitions(dw, c),
                                     tol_samples=1)
    (m,) = res["matched"]
    assert m["step_edge_index"] == 2 and m["edge_distance"] == 0


def test_match_validation():
    tr = soliton_count_transitions(np.arange(4.0), np.array([0, 1, 1, 1]))
    with pytest.raises(ValueError):              # negative tolerance
        match_steps_to_transitions(_steps_dict([1]), tr, tol_samples=-1)
    with pytest.raises(ValueError):              # inconsistent steps dict
        match_steps_to_transitions(
            {"edges": [1, 2], "step_x": [1.5], "step_dy": [-1.0]}, tr)


def test_match_end_to_end_with_real_detector_muted_final_edge():
    # Full pipeline on a synthetic descending-sweep staircase (ascending
    # order here): the 0->1 edge is power-muted (MI comb of comparable
    # energy), the 1->2 and 2->3 edges are honest power steps.
    x = np.arange(30, dtype=float)
    c = np.concatenate([np.zeros(8), np.ones(7), 2 * np.ones(7),
                        3 * np.ones(8)]).astype(int)
    y = np.concatenate([np.full(8, 0.55), np.full(7, 0.55),   # muted 0->1
                        np.full(7, 0.75), np.full(8, 1.0)])
    y = y + 0.001 * np.sin(x)
    steps = detect_power_steps(x, y, k=6.0)
    align = match_steps_to_transitions(steps, soliton_count_transitions(x, c))
    assert len(align["matched"]) == 2
    assert sorted(m["transition_edge_index"] for m in align["matched"]) \
        == [14, 21]
    assert all(m["delta_n"] == 1 for m in align["matched"])
    (muted,) = align["unmatched_transitions"]
    assert muted["edge_index"] == 7              # the 1->0 edge, power-muted
    assert align["unmatched_steps"] == []


# ---------------------------------------------------------------------------
# plot smoke test (writes files, no solver)
# ---------------------------------------------------------------------------
def test_plot_soliton_steps_writes_png_and_pdf(tmp_path):
    x, y, _ = _staircase([1.0, 0.6, 0.18], [15, 12, 13], noise=0.01, seed=0)
    steps = detect_power_steps(x, y, k=6.0)
    out = tmp_path / "soliton_steps.png"
    plot_soliton_steps(x, y, out, power_std=np.full(x.size, 0.01),
                       transmission=1.0 - 0.002 * y / y.max(),
                       soliton_region=(20.0, 39.0), annihilation_kappa=19.5,
                       steps=steps, metadata={"kappa_rad_s": 1.519e8,
                                              "caption": "test"})
    assert out.exists() and out.with_suffix(".pdf").exists()
    assert out.stat().st_size > 0


def test_plot_soliton_steps_single_panel_no_steps(tmp_path):
    x = np.linspace(6.0, 12.0, 25)
    y = 1.0 - 0.05 * (x - 6.0)
    out = tmp_path / "steps_1panel.png"
    plot_soliton_steps(x, y, out, steps=detect_power_steps(x, y),
                       soliton_region=(6.1, 12.0), annihilation_kappa=6.05)
    assert out.exists() and out.with_suffix(".pdf").exists()


def test_plot_soliton_steps_state_counts_twin_axis(tmp_path):
    # Staircase with measured counts: matched steps (state-verified), one
    # unmatched MI-rise discontinuity, one power-muted transition, plus the
    # any-soliton region shading.  Smoke-tests every new annotation path.
    x = np.arange(32, dtype=float)
    c = np.concatenate([np.zeros(8), np.ones(8), 2 * np.ones(8),
                        3 * np.ones(8)]).astype(int)
    y = np.concatenate([np.full(8, 0.55), np.full(8, 0.55),  # muted 0->1 edge
                        np.full(8, 0.75), np.full(8, 1.0)])
    y = y + 0.001 * np.sin(x)
    y[0] = 0.9                                   # MI/CW rise, no state change
    steps = detect_power_steps(x, y, k=6.0)
    out = tmp_path / "steps_state_counts.png"
    plot_soliton_steps(x, y, out, steps=steps, state_counts=c,
                       transmission=1.0 - 0.002 * y,
                       soliton_region=(8.0, 15.0),
                       any_soliton_region=(8.0, 31.0),
                       annihilation_kappa=7.5,
                       metadata={"caption": "state-counts smoke test"})
    assert out.exists() and out.with_suffix(".pdf").exists()
    assert out.stat().st_size > 0


def test_plot_soliton_steps_state_counts_unsorted_input(tmp_path):
    # state_counts must be re-sorted with the detuning axis exactly like the
    # power trace; descending (sweep-order) input must render identically.
    x = np.arange(24, dtype=float)
    c = np.concatenate([np.ones(12), 2 * np.ones(12)]).astype(int)
    y = np.concatenate([np.full(12, 0.6), np.full(12, 1.0)])
    steps = detect_power_steps(x, y, k=6.0)
    out = tmp_path / "steps_desc.png"
    plot_soliton_steps(x[::-1], y[::-1], out, steps=steps,
                       state_counts=c[::-1])
    assert out.exists()


# ===========================================================================
# Canonical failure-mode fixture (solver-free) for the staircase counting
# defects, plus the regression tests built on it.
# ===========================================================================
# Physics scale of the fixture: with these values B2_REF_SYNTH =
# 2*delta_omega/gamma = 1.0, i.e. the analytic single-soliton peak power (the
# physics anchor of count_solitons_windowed's v2 acceptance rule) is exactly
# 1.0 in fixture power units.
DELTA_OMEGA_SYNTH = 1.5e9
GAMMA_SYNTH = 3.0e9
B2_REF_SYNTH = 2.0 * DELTA_OMEGA_SYNTH / GAMMA_SYNTH        # == 1.0
CLUSTER_TOL_SYNTH = 0.05          # rad; well below the fixture pulse spacing


def _wrap_angle(x):
    """Wrap angle(s) to [-pi, pi)."""
    return np.angle(np.exp(1j * np.asarray(x)))


def breathing_modulated_field(n_solitons=5, n_snapshots=8, n_tau=2048,
                              phase_seed=0, mod_depth=0.7,
                              adversarial_lock=False, bg_level=0.05,
                              width_cells=15.0):
    """Canonical synthetic encoding of the two forensically established
    counting defects of the staircase pipeline (see
    ``analysis/results/staircase_forensics.md``):

    1. BREATHING-PHASE DROPOUT under single-snapshot counting (forensics 4-F,
       verdict "counting artifact"): each soliton's peak power is modulated
       sinusoidally in "time" (across the snapshot axis) with per-soliton
       phases drawn from ``phase_seed``, so on unlucky snapshots a trough-phase
       pulse falls below 0.5 of the MOMENTARY maximum (set by whichever sibling
       is at crest) and the end-of-hold 0.5-of-max peak count undercounts --
       even though every pulse ALWAYS stays >= 20x the CW background and its
       position never moves.
    2. SIBLING-CREST RELATIVE-THRESHOLD COUPLING (forensics 5-D Stage B,
       verdict "RELATIVE-THRESHOLD COUPLING CONFIRMED"): with
       ``adversarial_lock=True`` the designated victim (pulse 0) breathes
       around 0.30 * B2_ref exactly ANTI-phase to a designated crest sibling
       (pulse ``n_solitons // 2``) breathing around 2.5 * B2_ref, so the
       victim's modulation trough coincides with the sibling's crest in EVERY
       snapshot (phase-locked worst case) and any momentary-max rule --
       including the counter's deprecated 0.25-of-max relative arm -- rejects
       the victim in every snapshot, while the victim always remains >= 20x
       the background (never dim, only OUT-SHONE).

    Construction: circular multi-sech fields on a CW background of amplitude
    ``bg_level`` (power ``bg_level**2``), ``n_solitons`` pulses of angular
    width ``width_cells`` grid cells at FIXED (pinned) positions across all
    ``n_snapshots`` snapshots.  Peak powers are ``B2_REF_SYNTH * (1 +
    mod_depth * sin(2*pi*k/n_snapshots + phi_j))`` with ``phi_j`` from
    ``np.random.default_rng(phase_seed)``; ``mod_depth = 0.7`` puts troughs at
    0.3 * B2_ref -- below 0.5 of a sibling crest (1.7 * B2_ref) yet ~120x the
    default background power.  The physics-anchor floor ``soliton_frac *
    B2_ref = 0.1`` therefore sits BETWEEN the background floor and the deepest
    trough, so the v2 rule keeps every pulse in every snapshot.

    Returns ``(snapshots, centers)``: complex array of shape ``(n_snapshots,
    n_tau)`` and the pinned pulse angles (rad).
    """
    theta = 2.0 * np.pi * np.arange(n_tau) / n_tau
    w = width_cells * 2.0 * np.pi / n_tau
    centers = (2.0 * np.pi * np.arange(n_solitons) / n_solitons + 0.5) \
        % (2.0 * np.pi)
    phases = np.random.default_rng(phase_seed).uniform(
        0.0, 2.0 * np.pi, n_solitons)
    victim, crest = 0, n_solitons // 2
    snaps = []
    for k in range(n_snapshots):
        f = np.full(n_tau, bg_level, dtype=complex)
        s_lock = np.sin(2.0 * np.pi * k / n_snapshots)
        for j, (c, ph) in enumerate(zip(centers, phases)):
            if adversarial_lock and j == victim:
                p = B2_REF_SYNTH * (0.30 - 0.05 * s_lock)   # in [0.25, 0.35]
            elif adversarial_lock and j == crest:
                p = B2_REF_SYNTH * (2.50 + 0.50 * s_lock)   # in [2.0, 3.0]
            else:
                p = B2_REF_SYNTH * (1.0 + mod_depth * np.sin(
                    2.0 * np.pi * k / n_snapshots + ph))
            f += np.sqrt(p) / np.cosh(_wrap_angle(theta - c) / w)
        snaps.append(f)
    return np.array(snaps), centers


def test_fixture_encodes_the_documented_regime():
    """The fixture's own contract: troughs below 0.5-of-momentary-max at some
    phases, every pulse always >= 20x background, and the physics-anchor floor
    strictly between the background floor and the deepest trough."""
    snaps, centers = breathing_modulated_field()
    n_tau = snaps.shape[1]
    idx = np.round(centers * n_tau / (2.0 * np.pi)).astype(int) % n_tau
    ratios, floors = [], []
    for s in snaps:
        p = np.abs(s) ** 2
        peaks = p[idx]
        ratios.append(peaks.min() / peaks.max())
        assert np.all(peaks >= 20.0 * np.median(p))          # never dim
        floors.append(5.0 * float(np.median(p)))             # bg-floor arm
    assert min(ratios) < 0.5                # some phase drops below 0.5-of-max
    anchor = 0.1 * B2_REF_SYNTH             # soliton_frac * B2_ref
    deepest_trough = B2_REF_SYNTH * (1.0 - 0.7)
    assert max(floors) < anchor < deepest_trough


def test_single_snapshot_count_reproduces_end_of_hold_dropout():
    """(i) count_temporal_peaks on individual unlucky snapshots undercounts --
    the original end-of-hold defect (forensics 4-F: 'counting artifact')."""
    snaps, _ = breathing_modulated_field()
    singles = [count_temporal_peaks(s) for s in snaps]
    assert min(singles) < 5, singles         # unlucky phases undercount ...
    assert max(singles) <= 5                 # ... and nothing is ever invented


def test_windowed_counter_recovers_exact_count_with_persistence():
    """(ii) count_solitons_windowed over >= 8 phase-spread snapshots returns
    exactly n_solitons, every cluster's persistence reported and > 0.5.

    delta_omega/gamma are chosen so soliton_frac * B2_ref = 0.1 sits between
    the background floor (~0.014) and the deepest breathing trough (0.3)."""
    for n_solitons in (3, 5):
        snaps, centers = breathing_modulated_field(n_solitons=n_solitons)
        res = count_solitons_windowed(snaps,
                                      cluster_tol_rad=CLUSTER_TOL_SYNTH,
                                      delta_omega=DELTA_OMEGA_SYNTH,
                                      gamma=GAMMA_SYNTH)
        assert res["count"] == n_solitons
        assert np.allclose(np.sort(res["cluster_angles_rad"]),
                           np.sort(centers), atol=0.02)
        pf = res["persistence_fractions"]
        assert len(pf) == n_solitons         # every cluster reported
        assert all(f > 0.5 for f in pf)


def test_adversarial_lock_deprecated_rule_undercounts_v2_rule_does_not():
    """(iii) Phase-locked worst case: the DEPRECATED relative rule --
    reconstructed LOCALLY here, never resurrected in library code -- accepts a
    candidate iff height >= 0.25 * momentary snapshot max, and drops the
    locked victim in EVERY snapshot (persistence 0 < 0.5), undercounting to
    n_solitons - 1.  The v2 physics-anchored rule counts n_solitons with
    persistence 1.0 for every cluster (forensics 5-D Stage B)."""
    n_solitons = 5
    snaps, centers = breathing_modulated_field(adversarial_lock=True)
    n_tau = snaps.shape[1]

    # deprecated rule, local reconstruction: per-center persistence under
    # "accept iff height >= 0.25 * momentary snapshot max"
    hits = np.zeros(n_solitons)
    for s in snaps:
        p = np.abs(s) ** 2
        peaks, _ = find_peaks(np.concatenate([p, p]), height=0.25 * p.max())
        angs = 2.0 * np.pi * np.unique(peaks % n_tau) / n_tau
        for j, c in enumerate(centers):
            if any(abs(_wrap_angle(a - c)) < CLUSTER_TOL_SYNTH for a in angs):
                hits[j] += 1
    pf_old = hits / snaps.shape[0]
    assert pf_old[0] < 0.5                       # the locked victim drops out
    assert int(np.sum(pf_old >= 0.5)) == n_solitons - 1   # old rule undercounts

    res = count_solitons_windowed(snaps, cluster_tol_rad=CLUSTER_TOL_SYNTH,
                                  delta_omega=DELTA_OMEGA_SYNTH,
                                  gamma=GAMMA_SYNTH)
    assert res["count"] == n_solitons
    assert len(res["persistence_fractions"]) == n_solitons
    assert all(f == 1.0 for f in res["persistence_fractions"])


def test_starvation_arithmetic_regressions_remain_green():
    """(iv) The --starvation forensics arithmetic (falsified-hypothesis
    record): the driver's actual cadence hold_rt // 32 puts EIGHT snapshots in
    the final-avg_frac counting window (not the two of the starvation
    hypothesis' hold_rt // 8), and healthy count_agreement values are
    quantized in eighths.  The full set lives in
    tests/test_windowed_counting.py; this pins the same helpers into the
    staircase fast tier."""
    from analysis.staircase_forensics import (_empirical_denominator,
                                              _n_in_window, _on_grid)
    for hold_rt in (2000, 1600):                 # accepted cfg + variant 2
        assert _n_in_window(hold_rt, 0.25, max(hold_rt // 32, 1)) == 8
    assert _n_in_window(2000, 0.25, max(2000 // 8, 1)) == 2   # the hypothesis
    eighths = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    assert _empirical_denominator(eighths) == 8
    assert _on_grid(eighths, 8) is True
    assert _on_grid(eighths, 2) is False


# ---------------------------------------------------------------------------
# temporal_peak_positions: seeded-angle recovery on 1-, 3-, 5-sech fields
# (companion coverage with solver-adjacent fixtures lives in
# tests/test_dks_access.py; this variant is fully synthetic and pins the
# one-grid-cell accuracy, including a seam-straddling pulse)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angles", [
    [3.1],                                       # 1 sech
    [0.7, 2.9, 5.3],                             # 3 sech
    [0.0, 1.3, 2.6, 3.9, 5.2],                   # 5 sech, one at the 0/2pi seam
])
def test_temporal_peak_positions_recover_seeded_angles(angles):
    n_tau = 2048
    theta = 2.0 * np.pi * np.arange(n_tau) / n_tau
    f = np.full(n_tau, 0.05, dtype=complex)
    for a in angles:
        f += 1.0 / np.cosh(_wrap_angle(theta - a) / (15.0 * 2 * np.pi / n_tau))
    pos = temporal_peak_positions(f)
    assert pos.size == len(angles)
    cell = 2.0 * np.pi / n_tau
    for a in angles:
        d = np.min(np.abs(_wrap_angle(pos - a)))
        assert d <= cell + 1e-12, (a, pos)       # within one grid cell


# ---------------------------------------------------------------------------
# End-to-end synthetic staircase: N stepping 5 -> 1, stock detector at its
# default k, full alignment.  This demonstrates the UNMODIFIED detector
# suffices for quantized steps -- no retuning anywhere.
# ---------------------------------------------------------------------------
def test_end_to_end_synthetic_staircase_5_to_1_stock_detector():
    rng = np.random.default_rng(11)
    per, u_bg, u_sol = 10, 0.40, 0.15
    counts = np.repeat([1, 2, 3, 4, 5], per).astype(int)   # ascending detuning
    x = np.arange(counts.size, dtype=float)
    y = u_bg + counts * u_sol + rng.normal(0.0, 0.004, counts.size)  # ripple
    steps = detect_power_steps(x, y)             # DEFAULT k = 6, untouched
    transitions = soliton_count_transitions(x, counts)
    assert len(transitions) == 4                 # 1->2, 2->3, 3->4, 4->5
    align = match_steps_to_transitions(steps, transitions)
    assert len(align["matched"]) == 4            # every N-step state-verified
    assert align["unmatched_steps"] == []        # ... and nothing else fires
    assert align["unmatched_transitions"] == []
    assert all(m["delta_n"] == 1 for m in align["matched"])
    # each matched height is the one-soliton quantum, not detector noise
    for m in align["matched"]:
        assert abs(m["step_dy"]) == pytest.approx(u_sol, abs=0.02)


# ---------------------------------------------------------------------------
# Deprecation guard: the removed relative arm must warn loudly
# ---------------------------------------------------------------------------
def test_rel_height_candidate_keyword_emits_deprecation_warning():
    snaps = np.ones((2, 64), dtype=complex)      # tiny synthetic input
    with pytest.warns(DeprecationWarning, match="rel_height_candidate"):
        count_solitons_windowed(snaps, cluster_tol_rad=0.1,
                                rel_height_candidate=0.25)
