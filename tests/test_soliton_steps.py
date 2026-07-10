"""Unit tests for the Feature 3 soliton-step helpers in analysis.spectral_metrics.

All tests use SYNTHETIC data (analytic staircases, breather-like power series) so
they never run the solver -- the full solver sweep lives in the driver
``analysis/run_detuning_sweep.py``.  They pin the two reusable pieces the
staircase figure depends on:

* :func:`detect_power_steps` -- recovers known plateau boundaries of a noisy
  staircase within one sample, does not fire on a smooth branch, and finds the
  single drop of a gently-sloped-branch-then-collapse trace (the real DKS case).
* :func:`hold_window_average` -- selects the correct FINAL window and averages in
  LINEAR power (rejecting dB/complex input), which is how each per-detuning value
  is cycle-averaged.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.spectral_metrics import (
    detect_power_steps,
    hold_window_average,
    plot_soliton_steps,
    single_dks_region,
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
