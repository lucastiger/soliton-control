"""Minimal integration test for the detuning-sweep driver (runs the solver).

This is the SLOW-tier counterpart to the pure-synthetic unit tests in
``tests/test_soliton_steps.py`` (which never touch the solver).  It mirrors the
plumbing-only style of ``tests/test_adiabatic_sweeps.py``: a *reduced-length*
warm-continuation MULTI-SOLITON sweep (few steps, short holds, small ``n_tau``,
N = 3 seeds) just deep enough to exercise the driver end-to-end -- deterministic
multi-soliton seeding with the settled-peak-count assertion, continuation,
linear-power hold-window averaging, per-hold comb power / peak positions /
breathing fields, npz round-trip, and the staircase / step helpers -- without
asserting the full production physics.  It writes only to ``tmp_path`` so the
committed ``analysis/results`` artifacts are never clobbered.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.dks_access import attach_dispersion, load_cavity_params
from analysis.run_detuning_sweep import (
    SOLITON_LABELS,
    SweepConfig,
    load_sweep_npz,
    matched_step_contrast,
    run_detuning_sweep,
    save_sweep_npz,
    staircase_transition_edges,
    write_noise_off_config,
)
from analysis.spectral_metrics import detect_power_steps, single_dks_region

# Reduced-length sweep: enough to settle 3 seeded solitons and take a couple of
# warm-continuation steps; small n_tau keeps it CI-cheap.
CFG = SweepConfig(dw_start_kappa=10.0, dw_stop_kappa=8.0, n_steps=3,
                  settle_rt=1500, hold_rt=800, n_tau=1024, n_solitons=3)


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    cav = attach_dispersion(load_cavity_params(), CFG.n_tau)
    noise_cfg = write_noise_off_config()
    try:
        out = run_detuning_sweep(cav, CFG, config_path=noise_cfg)
    finally:
        noise_cfg.unlink(missing_ok=True)
    return out


def test_sweep_arrays_shape_and_finiteness(sweep):
    for key in ("dw_over_kappa", "P_intra", "P_intra_std", "P_trans",
                "U_int", "is_single", "np_label", "n_peaks",
                "P_comb", "P_comb_std", "soliton_count", "contrast",
                "is_breather", "is_stationary", "breathing_relstd"):
        assert key in sweep, key
        assert np.asarray(sweep[key]).shape == (CFG.n_steps,)
    for key in ("dw_over_kappa", "P_intra", "P_trans", "U_int", "P_comb"):
        assert np.all(np.isfinite(sweep[key]))
    # detunings are the requested decreasing grid
    assert np.allclose(sweep["dw_over_kappa"], np.linspace(10.0, 8.0, 3))
    # intracavity power is positive and transmission below the pump power
    assert np.all(sweep["P_intra"] > 0)
    assert np.all(sweep["P_trans"] < CFG.pin_w)


def test_multi_soliton_observables(sweep):
    # comb power is positive and strictly below the total (the pump line is out)
    assert np.all(sweep["P_comb"] > 0)
    assert np.all(sweep["P_comb"] < sweep["P_intra"])
    # the settle assertion guarantees N seeds survived; on this short in-branch
    # sweep the peak count stays positive and bounded by N
    assert np.all(sweep["n_peaks"] >= 1)
    assert np.all(sweep["soliton_count"] >= 0)
    assert np.all(sweep["soliton_count"] <= CFG.n_solitons)
    # peak positions: NaN-padded (n_steps, max_peaks) angles in [0, 2*pi)
    pos = sweep["peak_positions_rad"]
    assert pos.ndim == 2 and pos.shape[0] == CFG.n_steps
    finite = pos[np.isfinite(pos)]
    assert np.all((finite >= 0.0) & (finite < 2.0 * np.pi))
    # per-row finite-position count == n_peaks
    assert np.array_equal(np.isfinite(pos).sum(axis=1), sweep["n_peaks"])


def test_noise_off_config_zeroes_temperature():
    import yaml
    p = write_noise_off_config()
    try:
        cfg = yaml.safe_load(p.read_text())["physical_parameters"]
        assert float(cfg["T_k"]) == 0.0
    finally:
        p.unlink(missing_ok=True)


def test_npz_roundtrip_preserves_arrays_and_config(sweep, tmp_path):
    path = tmp_path / "detuning_sweep.npz"
    save_sweep_npz(path, sweep, CFG)
    loaded, cfg2 = load_sweep_npz(path)
    assert cfg2 == CFG
    assert np.allclose(loaded["P_intra"], sweep["P_intra"])
    assert np.allclose(loaded["P_comb"], sweep["P_comb"])
    assert np.array_equal(loaded["soliton_count"], sweep["soliton_count"])
    assert np.allclose(loaded["peak_positions_rad"],
                       sweep["peak_positions_rad"], equal_nan=True)
    assert loaded["is_single"].dtype == bool
    assert loaded["is_breather"].dtype == bool
    assert loaded["is_stationary"].dtype == bool
    assert float(loaded["kappa_rad_s"]) == pytest.approx(sweep["kappa_rad_s"])


def test_region_and_step_helpers_run_on_real_sweep(sweep):
    dwk = sweep["dw_over_kappa"]
    order = np.argsort(dwk)
    P = sweep["P_intra"][order]
    # helpers must run without error on the real (short) trace
    lo, hi, annih = single_dks_region(dwk[order], sweep["is_single"][order])
    steps = detect_power_steps(dwk[order], P / P.max())
    assert steps["n_steps"] >= 0
    # this reduced sweep sits on the soliton branch (8-10 kappa), so the
    # region helper must at least not crash -- assert the contract, not a
    # specific band.
    assert (lo is None) or (lo <= hi)


# ---------------------------------------------------------------------------
# Pure-synthetic staircase-helper tests (no solver)
# ---------------------------------------------------------------------------
def test_soliton_labels_taxonomy():
    # 4 = multi-soliton, 5 = soliton crystal, 6 = single soliton
    assert SOLITON_LABELS == (4, 5, 6)


def test_staircase_transition_edges_excludes_final_annihilation():
    counts = [0, 0, 1, 1, 3, 3, 5, 5]        # ascending detuning
    all_tr, matched = staircase_transition_edges(counts)
    assert all_tr == [1, 3, 5]               # 0->1, 1->3, 3->5
    assert matched == [3, 5]                 # the ->0 edge (i=1) is excluded


def test_matched_step_contrast_detected_and_matched_step_wins():
    # staircase with two genuine steps on a gently rippling baseline
    x = np.arange(12, dtype=float)
    y = np.concatenate([np.zeros(4), np.full(4, 1.0), np.full(4, 2.2)])
    y = y + 0.001 * np.sin(x)
    counts = np.concatenate([np.ones(4), 2 * np.ones(4), 3 * np.ones(4)])
    steps = detect_power_steps(x, y, k=6.0)
    all_tr, matched = staircase_transition_edges(counts)
    res = matched_step_contrast(y, steps, matched)
    assert res["matched_detected_edges"]      # both steps detected & matched
    assert res["contrast"] > 6.0              # far above the ripple MAD


def test_matched_step_contrast_zero_when_no_matched_detection():
    # smooth ramp: the detector finds nothing, so the contrast must be 0
    x = np.arange(10, dtype=float)
    y = 0.1 * x
    counts = np.concatenate([np.ones(5), 2 * np.ones(5)])
    steps = detect_power_steps(x, y, k=6.0)
    _, matched = staircase_transition_edges(counts)
    res = matched_step_contrast(y, steps, matched)
    assert res["contrast"] == 0.0
    assert res["matched_detected_edges"] == []
