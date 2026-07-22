"""Regression guards proving the pre-noise physics figures are invariant.

Context: the colored-noise / FSR / metrology branch adds only opt-in noise
channels; a reviewer independently confirmed a default solver run on this
branch is byte-for-byte identical to ``main``. These tests put that invariance
on the record for the two headline pre-noise artifacts — the single-DKS
optical spectrum (with its dispersive-wave wing peaks) and the soliton
staircase — and add a fast default-path golden-hash tripwire that pins any
future accidental drift of the default (noise-off-equivalent) solver path.

Design (why each test is shaped the way it is):

* ``test_dks_spectrum_invariance`` regenerates the single-DKS spectrum at the
  committed production recipe (n_tau = 16384, 12000 settle RT, seed 0,
  production numerics) and asserts <=1e-6 dB/mode agreement with the committed
  ``dks_single_soliton_spectrum.npz`` over the physical window
  |mu| <= n_tau/3, plus the two symmetric dispersive-wave peaks at the
  reference mode indices (+/-2) and levels (+/-0.5 dB). It is a SLOW
  regression harness (~6 min, one long solver run), in the same spirit as
  ``test_dks_spectral_integrity.py``; the DW peaks sit at |mu| ~ 3000-3300 so
  the full 16384 grid is required to resolve them inside |mu| <= n_tau/3.
  Gated behind ``RUN_SLOW_REGRESSION`` so the default CI suite stays fast and
  green; the equivalent comparison was executed for GATE V-A and the fast
  structural + golden-hash tests below run unconditionally.

* ``test_dks_committed_dw_peaks_structural`` is the always-on companion: a
  static check that the committed npz already carries exactly two symmetric
  DW peaks at the provenance-recorded indices/levels (no solver run).

* ``test_staircase_step_locations_invariance`` follows the repository's
  staircase-test convention (``test_soliton_staircase.py`` loads the committed
  ``detuning_sweep.npz`` and re-derives, never re-running the multi-hour
  multi-soliton warm continuation): it re-derives the 68 detected step
  locations and the single-DKS existence region from the committed npz and
  asserts they reproduce the committed ``spectral_metrics.json`` record to
  machine precision, and that the npz still hashes to its recorded provenance
  sha256 (the record is unchanged). The invariance of the underlying noise-off
  solver path is pinned by the golden hash below.

* ``test_default_path_golden_hash`` is the belt-and-suspenders drift guard: a
  short default solve (n_tau = 1024, 1500 RT, production numerics) hashed and
  compared against a golden committed here. The golden was computed on THIS
  branch and MUST equal what ``main`` produces (the reviewer-confirmed
  byte-for-byte default-path identity); any future change that perturbs the
  default path fails this test.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "analysis" / "results"

# Default-path golden hash: short default solve, production numerics, seed 0.
# Computed on the colored-noise branch; MUST equal main (default path is
# byte-for-byte identical to main). If this fails, the default solver path
# drifted — investigate before changing this constant.
_GOLDEN_DEFAULT_HASH = (
    "647a8557bd7e270c314021696a97774dffd32b9547d08499a7dc3ff47926cef8"
)

_RUN_SLOW = os.environ.get("RUN_SLOW_REGRESSION", "0") == "1"


# ---------------------------------------------------------------------------
# 1. DKS spectrum invariance (slow; gated) + DW peaks
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _RUN_SLOW,
    reason="slow (~6 min) production DKS regen; set RUN_SLOW_REGRESSION=1. "
    "Equivalent comparison run for GATE V-A; fast guards below always run.",
)
def test_dks_spectrum_invariance():
    from analysis.dks_access import (
        OPERATING_DW_KAPPA, PIN_W, PRODUCTION_NUMERICS, access_by_seeding,
        attach_dispersion, dispersive_wave_peaks, load_cavity_params,
        stationary_snapshot_spectrum,
    )

    ref = np.load(RESULTS / "dks_single_soliton_spectrum.npz")
    n_tau = int(ref["mu"].size)
    cav = load_cavity_params()
    cav = attach_dispersion(cav, n_tau)
    dw = OPERATING_DW_KAPPA * cav.kappa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = access_by_seeding(dw, cav, t_slow=12_000, seed=0, n_tau=n_tau,
                                pin=PIN_W, **PRODUCTION_NUMERICS)
        sp = stationary_snapshot_spectrum(res, cav, n_check_rt=304, pin=PIN_W,
                                          **PRODUCTION_NUMERICS)

    win = np.abs(ref["mu"]) <= n_tau // 3
    dev = np.abs(np.asarray(sp["power_db"])[win] - ref["power_db"][win])
    assert np.nanmax(dev) <= 1e-6, f"max |ΔdB|/mode = {np.nanmax(dev):.3e}"
    assert np.array_equal(sp["mu"], ref["mu"])

    peaks = dispersive_wave_peaks(
        {"mu": sp["mu"], "wavelength_nm": sp["wavelength_nm"],
         "power_db": sp["power_db"], "power_norm": sp["power_norm"]}, dw)
    ref_dws = json.loads((RESULTS / "dks_artifact_provenance.json").read_text()
                         )["measured"]["dispersive_waves"]
    assert len(peaks) == len(ref_dws) == 2
    pn = sorted(peaks, key=lambda p: p["mu"])
    pr = sorted(ref_dws, key=lambda p: p["mu"])
    assert pn[0]["mu"] < 0 < pn[1]["mu"], "DW peaks must be symmetric"
    for a, b in zip(pn, pr):
        assert abs(a["mu"] - b["mu"]) <= 2, (a["mu"], b["mu"])
        assert abs(a["power_db"] - b["power_db"]) <= 0.5, (a, b)


def test_dks_committed_dw_peaks_structural():
    """Always-on: the committed npz carries two symmetric DW peaks at the
    provenance-recorded indices/levels (static, no solver run)."""
    prov = json.loads(
        (RESULTS / "dks_artifact_provenance.json").read_text())
    dws = prov["measured"]["dispersive_waves"]
    assert len(dws) == 2, dws
    by_mu = sorted(dws, key=lambda d: d["mu"])
    assert by_mu[0]["mu"] < 0 < by_mu[1]["mu"], "DW peaks not symmetric"
    # prominence well above the sech tail (real DWs, not floor bumps)
    for d in dws:
        assert d["prominence_db"] > 50.0, d
    # and the npz actually contains those modes at those levels
    z = np.load(RESULTS / "dks_single_soliton_spectrum.npz")
    mu, pdb = z["mu"], z["power_db"]
    for d in dws:
        j = int(np.where(mu == d["mu"])[0][0])
        assert abs(pdb[j] - d["power_db"]) <= 0.5, (d, pdb[j])


# ---------------------------------------------------------------------------
# 2. Staircase step-location invariance (re-derive from committed npz)
# ---------------------------------------------------------------------------
def test_staircase_step_locations_invariance():
    from analysis.run_detuning_sweep import load_sweep_npz
    from analysis.spectral_metrics import (
        DEFAULT_STEP_K, detect_power_steps, single_dks_region,
    )

    npz = RESULTS / "detuning_sweep.npz"
    sm = json.loads((RESULTS / "spectral_metrics.json").read_text())
    ss = sm["soliton_step"]

    # (a) the committed record is unchanged: npz hashes to its recorded sha256.
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    assert sha == ss["provenance"]["sweep_npz_sha256"], "detuning_sweep.npz drifted"

    sweep, _cfg = load_sweep_npz(npz)
    dwk = np.asarray(sweep["dw_over_kappa"])
    order = np.argsort(dwk)
    dwk_a = dwk[order]
    is_single = np.asarray(sweep["is_single"]).astype(bool)
    pc = np.asarray(sweep["P_comb"])[order]
    pc_norm = pc / float(np.max(pc))

    # (b) detected step locations reproduce the committed record bit-for-bit.
    steps = detect_power_steps(dwk_a, pc_norm, k=DEFAULT_STEP_K)
    rederived = np.array(sorted(float(x) for x in steps["step_x"]))
    committed = np.array(sorted(
        ss["power_trace_discontinuities"]["detected_over_kappa"]))
    assert rederived.shape == committed.shape, (rederived.size, committed.size)
    assert np.array_equal(rederived, committed), (
        f"max |Δ| = {np.max(np.abs(rederived - committed)):.3e} κ")

    # (c) single-DKS existence region reproduces exactly; single soliton is
    # reached at the same detuning.
    lo, hi, _annih = single_dks_region(dwk, is_single)
    com = ss["single_dks_existence_region_over_kappa"]
    assert np.array_equal([lo, hi], com), (lo, hi, com)
    assert bool(is_single.any()), "no single-soliton hold in the committed sweep"


# ---------------------------------------------------------------------------
# 3. Default-path golden hash (fast drift tripwire)
# ---------------------------------------------------------------------------
def test_default_path_golden_hash():
    import jax

    from analysis.dks_access import PRODUCTION_NUMERICS
    from simulator.lle_solver import solve_lle_ssfm_jax

    kappa, kappa_c = 1.519e8, 1.215e8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_lle_ssfm_jax(
            pin=0.214, delta_omega=10.0 * kappa, t_slow=1500,
            beta=[1.578e-18], kappa=kappa, kappa_c=kappa_c,
            rng_key=jax.random.PRNGKey(0), n_tau=1024, snapshot_interval=100,
            **PRODUCTION_NUMERICS)
    h = hashlib.sha256()
    for k in sorted(sol.keys()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(sol[k])).tobytes())
    assert h.hexdigest() == _GOLDEN_DEFAULT_HASH, (
        f"default-path output drifted: {h.hexdigest()} != golden "
        f"{_GOLDEN_DEFAULT_HASH} (this golden MUST equal main)")
