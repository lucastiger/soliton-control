"""Tests for the noise-enabled publication validation campaign.

Two tiers, following the repository convention
(``tests/test_regression_figures.py`` / ``tests/test_noise_metrology.py``):

* ALWAYS-ON assertions read the committed
  ``analysis/results/validation/campaign_report.json`` -- the production run's
  quantitative record -- and encode the campaign's decision rules on it:
  DW peak positions invariant (dispersion-set) and any level/width change
  attributed to the vacuum-floor + pump-jitter budget (not a bug); the
  staircase single-soliton success rate and step bias < jitter; the
  cross-realization β-line curvature a2 > 0 with a Taylor control >=10x
  smaller; and the far-wing vacuum floor within a factor 3 of ħω₀/2.

* CHEAP CI RE-DERIVATIONS run unconditionally and recompute the cheap physics
  from scratch (no committed artifact): the dispersion-set DW crossing modes
  (pure dispersion, no solver), the full-stack sidecar switch state, and a
  fast quantum-vacuum-floor solve at n_tau = 512 that confirms the
  ``|fft(E)_μ|² -> n_tau²·ħω₀/2`` normalization the whole campaign rests on.

* The SLOW full-fidelity re-derivation (a real n_tau = 16384 DW-survival
  ensemble) is gated behind ``RUN_SLOW_VALIDATION=1`` so the default suite
  stays fast and green.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VALIDATION = REPO / "analysis" / "results" / "validation"
REPORT_JSON = VALIDATION / "campaign_report.json"

_RUN_SLOW = os.environ.get("RUN_SLOW_VALIDATION", "0") == "1"


# ---------------------------------------------------------------------------
# Committed-report fixture (always-on assertions)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report():
    if not REPORT_JSON.exists():
        pytest.skip(f"committed campaign report missing: {REPORT_JSON}")
    return json.loads(REPORT_JSON.read_text())


def _need(report, key):
    if key not in report:
        pytest.skip(f"workstream block '{key}' not in committed report")
    return report[key]


def test_report_is_a_production_run(report):
    """The committed report must be the production run (quick == False)."""
    assert report.get("meta", {}).get("quick") is False
    prov = report.get("provenance", {})
    assert prov.get("quick") is False
    assert prov.get("git_commit") and prov.get("physical_parameters_sha256")


# ---- Workstream 1: DW-peak survival ---------------------------------------
def test_w1_dw_positions_invariant(report):
    """DW peak positions are dispersion-set: OFF peaks at their phase-match
    crossings, and any peak still resolvable above the floor stays within ±2."""
    w1 = _need(report, "workstream1_dw_survival")
    assert w1["dw_off_peaks_at_phase_match_crossing"] is True
    assert w1["max_resolvable_dw_position_shift_modes"] <= 2
    assert w1["dw_positions_invariant_within_2_modes"] is True


def test_w1_change_is_physical_not_a_bug(report):
    """The wing-level change is inside the vacuum-floor + pump-jitter budget,
    and every DW peak is accounted for as either surviving-and-invariant or
    submerged below the vacuum floor -- physical, not an anomalous shift."""
    w1 = _need(report, "workstream1_dw_survival")
    assert w1["broadening_budget"]["within_budget"] is True
    n_survive = w1["n_dw_peaks_surviving_above_floor"]
    n_submerged = w1["n_dw_peaks_submerged_below_floor"]
    assert n_survive + n_submerged == len(w1["dw_peaks"])
    for p in w1["dw_peaks"]:
        assert p["off_peak_at_crossing"] is True
        # SNR over the vacuum floor and submersion depth are consistent numbers.
        assert math.isfinite(p["snr_db_off"])
        assert math.isfinite(p["submerged_below_vacuum_floor_db"])


def test_w1_vacuum_floor_consistent(report):
    """The empirical far-wing floor sits within a factor 3 of n_tau²·ħω₀/2."""
    w1 = _need(report, "workstream1_dw_survival")
    mult = w1["empirical_far_wing_floor_multiple_of_vacuum"]
    assert 1.0 / 3.0 <= mult <= 3.0
    assert w1["vacuum_floor_db_rel_pump_off"] < 0.0


# ---- Workstream 2: Monte-Carlo staircase ----------------------------------
def test_w2_single_soliton_access_and_bias(report):
    """Single-soliton state still reached in a majority of realizations, and
    every matched transition's mean offset (bias) is below its jitter."""
    w2 = _need(report, "workstream2_staircase")
    assert w2["single_soliton_success_rate"] >= 0.5
    assert w2["all_bias_lt_jitter"] is True
    # bias must be smaller than 3*jitter for each robust soliton-number boundary.
    assert len(w2["level_crossings"]) >= 1
    for t in w2["level_crossings"]:
        assert abs(t["bias_kappa"]) < 3.0 * max(t["jitter_kappa"], 1e-12)


# ---- Workstream 3: cross-realization linewidth ----------------------------
def test_w3_cross_realization_a2_positive(report):
    """Across independent pump realizations the β-line curvature a2 > 0 with
    high bootstrap significance."""
    w3 = _need(report, "workstream3_linewidth")
    assert w3["a2_cross_realization_mean"] > 0.0
    assert w3["a2_bootstrap_p_positive"] >= 0.9


def test_w3_taylor_control_10x_smaller(report):
    """The Taylor-D2 negative control's curvature is >=10x smaller than the
    measured-D_int result (pure quadratic dispersion -> pure common mode)."""
    w3 = _need(report, "workstream3_linewidth")
    assert w3["flagship_taylor_control_10x_smaller"] is True
    assert w3["flagship_taylor_ratio"] <= 0.1


# ---- Workstream 4: RF-beatnote / coherence --------------------------------
def test_w4_rep_rate_and_limits(report):
    """The rep-rate linewidth and S_rep-vs-limit numbers are finite/physical."""
    w4 = _need(report, "workstream4_beatnote")
    assert math.isfinite(w4["rep_rate_linewidth_hz_mean"])
    assert w4["rep_rate_linewidth_hz_mean"] >= 0.0
    assert math.isfinite(w4["srep_over_trn_limit_band_median_db"])
    assert math.isfinite(w4["srep_quantum_limited_floor_hz2_per_hz"])


# ---- Workstream 5: vacuum-floor + energy budget ---------------------------
def test_w5_vacuum_floor_within_factor_3(report):
    """Far-wing modal energy asymptotes to ħω₀/2 within a factor 3, and the
    vacuum contribution to P_abs is negligible for the thermal ODE."""
    w5 = _need(report, "workstream5_vacuum_budget")
    assert w5["floor_within_factor_3"] is True
    assert 1.0 / 3.0 <= w5["far_wing_floor_multiple_of_half_hbar_omega0"] <= 3.0
    # κ_i·n_tau·ħω₀/2 must be a tiny fraction of the real absorbed power.
    assert w5["p_abs_vacuum_over_real"] < 1.0e-3


# ---------------------------------------------------------------------------
# Cheap CI re-derivations (always-on, no committed artifact)
# ---------------------------------------------------------------------------
def test_dw_crossings_are_dispersion_set():
    """The phase-matched DW crossing modes at the operating point are fixed by
    the measured dispersion alone (no solver): μ ≈ +3268 / −3050."""
    from analysis.dks_access import (
        OPERATING_DW_KAPPA, dispersive_wave_crossings, load_cavity_params,
    )
    cav = load_cavity_params()
    crossings = dispersive_wave_crossings(OPERATING_DW_KAPPA * cav.kappa)
    mus = sorted(int(c["mu"]) for c in crossings)
    assert len(mus) == 2
    assert abs(mus[0] - (-3050)) <= 2
    assert abs(mus[1] - 3268) <= 2


def test_full_stack_sidecar_enables_all_channels():
    """The full-stack sidecar turns on every stochastic channel of the paper."""
    import yaml

    from analysis.noise_validation_campaign import _full_stack_sidecar
    from analysis.dks_access import CONFIG_PATH

    cfg = yaml.safe_load(open(_full_stack_sidecar(CONFIG_PATH), encoding="utf-8"))
    pp = cfg["physical_parameters"]
    assert pp["quantum_noise_enabled"] == 1
    assert pp["pump_noise_enabled"] == 1
    assert pp["pump_freq_noise_h0_hz2_per_hz"] > 0
    assert pp["pump_freq_noise_hm1_hz3_per_hz"] > 0
    assert pp["trn_psd_model"] == "kondratiev_gorodetsky"
    assert pp["fsr_noise_enabled"] == 1


def test_vacuum_floor_normalization_ci():
    """CI re-derivation of the vacuum-floor normalization.

    A short quantum-vacuum-only solve at n_tau = 512, blue-detuned (no comb),
    cold start: the far-from-pump modes fill to the symmetric-ordered vacuum
    occupation. Their mean raw |fft(E)_μ|² must reproduce n_tau²·ħω₀/2 within a
    factor 3 -- the exact normalization every workstream's floor rests on.
    """
    from analysis.dks_access import PIN_W, _run, load_cavity_params
    from analysis.noise_validation_campaign import _hbar_omega0, _sidecar
    from analysis.run_detuning_sweep import write_noise_off_config

    cav = load_cavity_params()
    n_tau = 512
    cfg = _sidecar(write_noise_off_config(), "qonly_ci",
                   quantum_noise_enabled=1, quantum_noise_injection_cadence=1)
    dw = -3.0 * cav.kappa  # blue side: CW background only, far modes = vacuum
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = _run(dw, 500, cav, e0=None, seed=0, n_tau=n_tau, pin=PIN_W,
                   snapshot_interval=1, config_path=cfg,
                   n_substeps=1, dealias_two_thirds=False,
                   edge_absorber=False, dispersion_validity_mask=False)
    snaps = np.asarray(sol["E_snapshots"])[0][-150:]
    power = np.abs(np.fft.fftshift(np.fft.fft(snaps, axis=-1), axes=-1)) ** 2
    mean_power = power.mean(axis=0)
    mu = np.arange(n_tau) - n_tau // 2
    far = (np.abs(mu) >= n_tau // 5) & (np.abs(mu) <= n_tau // 3)
    floor = n_tau ** 2 * _hbar_omega0() / 2.0
    multiple = float(np.median(mean_power[far]) / floor)
    assert 1.0 / 3.0 <= multiple <= 3.0, (
        f"far-mode |fft|² = {multiple:.2f} × n_tau²·ħω₀/2 (expected ~1)")


# ---------------------------------------------------------------------------
# Slow full-fidelity re-derivation (gated)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _RUN_SLOW,
    reason="slow n_tau=16384 DW-survival ensemble; set RUN_SLOW_VALIDATION=1.")
def test_slow_w1_dw_survival_rederive(tmp_path):
    """Re-run Workstream 1 (quick sizes, real n_tau = 16384) and re-assert the
    decision rule: OFF peaks at their crossings, positions invariant, and the
    wing change inside the vacuum + pump-jitter budget."""
    from analysis.noise_validation_campaign import workstream1

    block, _ens = workstream1(tmp_path, seeds=4, quick=True)
    assert block["dw_off_peaks_at_phase_match_crossing"] is True
    assert block["max_resolvable_dw_position_shift_modes"] <= 2
    assert block["broadening_budget"]["within_budget"] is True
