"""Fast regression test for the adiabatic detuning-sweep driver.

Runs a *reduced-length* version of analysis.adiabatic_sweeps (short t_slow so it
stays CI-cheap) and asserts the four required validations hold on the mechanics:

  V1  forward sweep ignites MI inside the (0.5..5)*kappa window; the held-CW
      deep-blue control stays label 1.
  V2  forward and reverse label/U_int trajectories differ measurably (hysteresis).
  V3  with thermal ON, delta_omega_eff shifts DOWN as U_int rises, and the steady
      thermal_shift magnitude is a sane fraction of kappa (not tens).
  V4  no NaN/Inf over the sweeps; held-CW control U_int tail rel-std < 5%.

The full science run (t_slow ~ 7*tau_th, see the module docstring) is executed
via `python -m analysis.adiabatic_sweeps`; this test only pins the plumbing and
the qualitative physics so a regression fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analysis.adiabatic_sweeps import (
    _thermo_optic_coeff,
    run_sweep,
    validate,
)
from simulator.lle_solver import (
    _load_config,
    d2_to_beta2_lle,
    resolve_cavity_rates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"

T_SLOW = 30_000          # reduced length: enough to ignite MI + show hysteresis
SNAP_INTERVAL = 200


@pytest.fixture(scope="module")
def sweeps():
    phys = _load_config(CONFIG_PATH)
    kappa_i, kappa_c, kappa = resolve_cavity_rates(CONFIG_PATH)
    beta2 = d2_to_beta2_lle(float(phys["d2_rad_per_s2"]), float(phys["fsr_hz"]))
    to_coeff = _thermo_optic_coeff(CONFIG_PATH)

    fwd_dw = np.linspace(-2.0 * kappa, 5.0 * kappa, T_SLOW, dtype=np.float32)[None, :]
    rev_dw = np.linspace(5.0 * kappa, -2.0 * kappa, T_SLOW, dtype=np.float32)[None, :]
    ctrl_dw = np.full((1, T_SLOW), -4.0 * kappa, dtype=np.float32)

    fwd = run_sweep(fwd_dw, T_SLOW, beta2, kappa, kappa_c, 0, SNAP_INTERVAL)
    rev = run_sweep(rev_dw, T_SLOW, beta2, kappa, kappa_c, 0, SNAP_INTERVAL)
    ctrl = run_sweep(ctrl_dw, T_SLOW, beta2, kappa, kappa_c, 0, SNAP_INTERVAL)
    results = validate(fwd, rev, ctrl, kappa, to_coeff)
    return fwd, rev, ctrl, kappa, results


def test_v1_mi_ignition_and_cw_control(sweeps):
    fwd, rev, ctrl, kappa, results = sweeps
    ok, detail = results["V1_MI_ignition"]
    assert ok, detail
    # held-CW control must never leave CW/off
    assert int(np.max(ctrl["label"])) <= 1, "deep-blue control left CW"


def test_v2_hysteresis(sweeps):
    fwd, rev, ctrl, kappa, results = sweeps
    ok, detail = results["V2_hysteresis"]
    assert ok, detail


def test_v3_thermal_sign(sweeps):
    fwd, rev, ctrl, kappa, results = sweeps
    ok, detail = results["V3_thermal_sign"]
    assert ok, detail
    # thermal_shift must be negative (heating redshifts omega_res -> lowers delta_omega)
    to_coeff = _thermo_optic_coeff(CONFIG_PATH)
    shift = -to_coeff * fwd["DeltaT_full"]
    assert np.all(shift <= 1e-6), "thermal_shift went positive (wrong sign)"


def test_v4_numerical_health(sweeps):
    fwd, rev, ctrl, kappa, results = sweeps
    ok, detail = results["V4_numerical_health"]
    assert ok, detail
    for rec in (fwd, rev, ctrl):
        assert np.all(np.isfinite(rec["U_int_full"]))
        assert np.all(np.isfinite(rec["e_snaps"]))


def test_no_single_soliton_claim(sweeps):
    """Sanity: the bare linear sweep should NOT be producing clean single solitons
    (label 6) — nucleation is follow-up work, not something this sweep achieves."""
    fwd, rev, ctrl, kappa, results = sweeps
    # It is fine if none appear; assert we are honest about it in the max corr.
    assert np.max(fwd["sech2_corr"]) < 0.95, (
        "unexpected sech^2-perfect state — revisit the no-single-soliton claim"
    )
