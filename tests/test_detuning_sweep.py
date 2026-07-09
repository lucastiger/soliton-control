"""Minimal integration test for the detuning-sweep driver (runs the solver).

This is the SLOW-tier counterpart to the pure-synthetic unit tests in
``tests/test_soliton_steps.py`` (which never touch the solver).  It mirrors the
plumbing-only style of ``tests/test_adiabatic_sweeps.py``: a *reduced-length*
warm-continuation sweep (few steps, short holds, small ``n_tau``) just deep
enough to exercise the driver end-to-end -- seed, continuation, linear-power
hold-window averaging, npz round-trip, and the single-DKS-region / step helpers
-- without asserting the full production physics.  It writes only to ``tmp_path``
so the committed ``analysis/results`` artifacts are never clobbered.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.dks_access import attach_dispersion, load_cavity_params
from analysis.run_detuning_sweep import (
    SweepConfig,
    load_sweep_npz,
    run_detuning_sweep,
    save_sweep_npz,
    write_noise_off_config,
)
from analysis.spectral_metrics import detect_power_steps, single_dks_region

# Reduced-length sweep: enough to seed a soliton and take a couple of warm-
# continuation steps; small n_tau keeps it CI-cheap.
CFG = SweepConfig(dw_start_kappa=10.0, dw_stop_kappa=8.0, n_steps=3,
                  settle_rt=1500, hold_rt=800, n_tau=1024)


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
                "U_int", "is_single", "np_label", "n_peaks"):
        assert key in sweep, key
        assert np.asarray(sweep[key]).shape == (CFG.n_steps,)
    for key in ("dw_over_kappa", "P_intra", "P_trans", "U_int"):
        assert np.all(np.isfinite(sweep[key]))
    # detunings are the requested decreasing grid
    assert np.allclose(sweep["dw_over_kappa"], np.linspace(10.0, 8.0, 3))
    # intracavity power is positive and transmission below the pump power
    assert np.all(sweep["P_intra"] > 0)
    assert np.all(sweep["P_trans"] < CFG.pin_w)


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
    assert loaded["is_single"].dtype == bool
    assert float(loaded["kappa_rad_s"]) == pytest.approx(sweep["kappa_rad_s"])


def test_region_and_step_helpers_run_on_real_sweep(sweep):
    dwk = sweep["dw_over_kappa"]
    order = np.argsort(dwk)
    P = sweep["P_intra"][order]
    # helpers must run without error on the real (short) trace
    lo, hi, annih = single_dks_region(dwk[order], sweep["is_single"][order])
    steps = detect_power_steps(dwk[order], P / P.max())
    assert steps["n_steps"] >= 0
    # this reduced sweep sits on the single-DKS branch (8-10 kappa), so the
    # region should be found (or, if the small grid degrades the fit, at least
    # not crash) -- assert the contract, not a specific band.
    assert (lo is None) or (lo <= hi)
