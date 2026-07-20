"""FSR-noise, probe, and metrology tests (categories 7-10 + limiting behavior).

Covers the Q3 test plan:
  (7) FSR term exactness: with a deterministic constant dD1 injected, mode
      mu acquires the predicted extra phase -mu*dD1*t (spectral-shift check,
      machine precision);
  (8) probes: mode_probe_history equals the FFT of E_snapshots at coinciding
      steps (machine precision) and is passive;
  (9) metrology on synthetic data: an artificial comb with known injected
      common-mode + repetition-rate frequency noise; the tape-model fit must
      recover S_c and S_rep within 10% and mu_fix within +-2 modes
      -> STOP-AND-REPORT GATE 3;
  plus the limiting-behavior physics assertions (T_k -> 0 collapses every
  dT-derived channel including FSR noise; trn_psd_model changes conserve the
  Eq. 129 variance within 2%) and unit anchors for the beta-separation
  linewidth (white noise: FWHM = pi*h0) and the peak-trajectory jitter.

Category 10 (full suite green with defaults) is the whole tests/ directory.
"""

from __future__ import annotations

import math
import warnings

import jax
import numpy as np
import pytest

from analysis.noise_metrology import (
    effective_linewidth,
    frequency_noise_psd,
    peak_angle_trajectory,
    rep_rate_phase,
    tape_model_fit,
    timing_jitter,
    unwrapped_phases,
)
from simulator.colored_noise import integrate_psd
from simulator.noise_models import TotalNoise, _load_config as nm_load_cfg

KAPPA = 1.519e8
KAPPA_C = 1.215e8
T_R = 1.0 / 2.46e10


def _noise_off_cfg(tmp_path):
    from analysis.run_detuning_sweep import write_noise_off_config

    return str(write_noise_off_config(out_path=tmp_path / "noiseoff.yaml"))


# ---------------------------------------------------------------------------
# (7) FSR term: exact spectral shift for a deterministic constant dD1
# ---------------------------------------------------------------------------
def test_fsr_constant_dd1_exact_phase(tmp_path):
    """A pure mode mu0 under constant dD1 gains phase -mu0*dD1*t exactly.

    Undriven (pin = 0), noise-off sidecar, single occupied mode: |E(tau)| is
    uniform, so the Kerr phase is uniform and identical between the runs —
    the ONLY difference the override introduces is the per-mode linear
    detuning mu*dD1, i.e. after k+1 round trips the probe phase differs by
    -mu0*dD1*(k+1)*t_r (the spectral shift mu0*dD1/2pi of that comb line).
    """
    from simulator.lle_solver import solve_lle_ssfm_jax

    cfgp = _noise_off_cfg(tmp_path)
    n_tau, t_slow, mu0 = 64, 40, 5
    theta = 2.0 * math.pi * np.arange(n_tau) / n_tau
    e0 = (1e-3 * np.exp(1j * mu0 * theta)).astype(np.complex128)
    dd1 = 1.0e3                                      # rad/s, constant
    common = dict(pin=0.0, delta_omega=2.0 * KAPPA, t_slow=t_slow,
                  beta=[1.578e-18], kappa=KAPPA, kappa_c=KAPPA_C,
                  rng_key=jax.random.PRNGKey(0), n_tau=n_tau,
                  snapshot_interval=10, config_path=cfgp, e0_override=e0,
                  mode_probe_indices=(mu0,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # pin=0 MI-threshold warning
        base = solve_lle_ssfm_jax(**common)
        shifted = solve_lle_ssfm_jax(
            **common, fsr_delta_d1_override=np.full(t_slow, dd1))

    phi_base = np.unwrap(np.angle(base["mode_probe_history"][0, :, 0]))
    phi_shift = np.unwrap(np.angle(shifted["mode_probe_history"][0, :, 0]))
    t = (np.arange(t_slow) + 1) * T_R
    predicted = -mu0 * dd1 * t
    measured = phi_shift - phi_base
    # machine precision: the two runs share every other term bit-for-bit
    assert np.max(np.abs(measured - predicted)) < 1e-9, (
        np.max(np.abs(measured - predicted)))
    # amplitude untouched (pure phase term)
    assert np.allclose(np.abs(shifted["mode_probe_history"]),
                       np.abs(base["mode_probe_history"]), rtol=1e-12)
    # spectral-shift reading: the fitted line slope is the frequency shift
    # (tolerance set by polyfit conditioning on ~ns abscissae, not physics —
    # the pointwise phase check above is the machine-precision statement)
    slope = np.polyfit(t, measured, 1)[0]
    assert slope == pytest.approx(-mu0 * dd1, rel=1e-6)


def test_fsr_tk_zero_channel_identically_zero(tmp_path):
    """fsr_noise_enabled = 1 with T_k = 0 yields dD1 == 0 and an unchanged
    solution (every dT-derived channel collapses with T_k)."""
    from simulator.lle_solver import solve_lle_ssfm_jax

    cfgp = _noise_off_cfg(tmp_path)
    common = dict(pin=1e-3, delta_omega=3.0 * KAPPA, t_slow=30,
                  beta=[1.578e-18], kappa=KAPPA, kappa_c=KAPPA_C,
                  rng_key=jax.random.PRNGKey(1), n_tau=64,
                  snapshot_interval=10, config_path=cfgp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = solve_lle_ssfm_jax(**common)
        fsr = solve_lle_ssfm_jax(**common, fsr_noise_enabled=True)
    assert np.all(fsr["fsr_delta_d1_history"] == 0.0)
    assert np.array_equal(base["U_int_history"], fsr["U_int_history"])
    assert np.array_equal(base["E_snapshots"], fsr["E_snapshots"])


# ---------------------------------------------------------------------------
# (8) Probes: exact vs snapshots, passive, validated
# ---------------------------------------------------------------------------
def test_probe_history_equals_fft_of_snapshots():
    from simulator.lle_solver import solve_lle_ssfm_jax

    mus = (0, 1, -1, 7, -13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_lle_ssfm_jax(
            pin=1e-3, delta_omega=3.0 * KAPPA, t_slow=50, beta=[1.578e-18],
            kappa=KAPPA, kappa_c=KAPPA_C, rng_key=jax.random.PRNGKey(2),
            n_tau=64, snapshot_interval=10, quantum_noise_enabled=True,
            mode_probe_indices=mus)
    probes = sol["mode_probe_history"]
    assert probes.shape == (1, 50, len(mus))
    assert probes.dtype == np.complex128
    bins = [m % 64 for m in mus]
    for si, step in enumerate(range(0, 50, 10)):
        want = np.fft.fft(sol["E_snapshots"][0, si])[bins]
        got = probes[0, step]
        assert np.array_equal(want, got), step   # machine-exact


def test_probes_are_passive_and_validated():
    from simulator.lle_solver import solve_lle_ssfm_jax

    common = dict(pin=1e-3, delta_omega=3.0 * KAPPA, t_slow=30,
                  beta=[1.578e-18], kappa=KAPPA, kappa_c=KAPPA_C,
                  rng_key=jax.random.PRNGKey(3), n_tau=64,
                  snapshot_interval=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = solve_lle_ssfm_jax(**common)
        probed = solve_lle_ssfm_jax(**common, mode_probe_indices=(0, 3))
    for k in base:
        assert np.array_equal(base[k], probed[k]), k
    assert "mode_probe_history" not in base

    with pytest.raises(AssertionError, match="at most 16"):
        solve_lle_ssfm_jax(**common, mode_probe_indices=tuple(range(17)))
    with pytest.raises(AssertionError, match="n_tau/2"):
        solve_lle_ssfm_jax(**common, mode_probe_indices=(40,))


# ---------------------------------------------------------------------------
# (9) Metrology on synthetic data  -> GATE 3
# ---------------------------------------------------------------------------
def _white_freq_phase(rng, n, h, f_s):
    """Unwrapped phase [rad] whose frequency noise is white at h [Hz^2/Hz]."""
    dnu = rng.standard_normal(n) * math.sqrt(h * f_s / 2.0)
    return 2.0 * math.pi * np.cumsum(dnu) / f_s


def test_tape_model_recovers_synthetic_comb_gate3():
    f_s = 1.0 / T_R
    n = 1 << 16
    h_c = 1.0        # Hz^2/Hz  (independent common-mode frequency noise)
    h_r = 1.0e-4     # Hz^2/Hz  (repetition-rate frequency noise)
    mu_fix_true = 30
    mus = (-100, -60, -20, 20, 60, 100)

    rng = np.random.default_rng(20260720)
    psi_c = _white_freq_phase(rng, n, h_c, f_s)
    psi_r = _white_freq_phase(rng, n, h_r, f_s)
    # Elastic tape with a fix point at mu_fix_true: phi_mu = phi_c + mu*phi_r
    # with phi_c = psi_c - mu_fix*psi_r  =>  S_c = h_c + mu_fix^2*h_r,
    # S_cr = -mu_fix*h_r, S_rep = h_r, mu_fix = -S_cr/S_rep.
    phi_c = psi_c - mu_fix_true * psi_r
    probes = np.stack(
        [np.exp(1j * (phi_c + mu * psi_r)) for mu in mus], axis=1)

    phases = unwrapped_phases(probes)
    fit = tape_model_fit(phases, mus, T_R, nperseg=1 << 13)
    band = (fit["f"] > f_s / 2048) & (fit["f"] < f_s / 8)

    s_c_meas = float(np.mean(fit["S_c"][band]))
    s_rep_meas = float(np.mean(fit["S_rep"][band]))
    s_cr_meas = float(np.mean(fit["S_cr"][band]))
    s_c_true = h_c + mu_fix_true**2 * h_r
    mu_fix_band = float(-s_cr_meas / s_rep_meas)
    mu_fix_median = float(np.nanmedian(fit["mu_fix"][band]))

    assert abs(s_c_meas - s_c_true) / s_c_true < 0.10, s_c_meas
    assert abs(s_rep_meas - h_r) / h_r < 0.10, s_rep_meas
    assert abs(mu_fix_band - mu_fix_true) <= 2.0, mu_fix_band
    assert abs(mu_fix_median - mu_fix_true) <= 2.0, mu_fix_median

    # Repetition-rate phase from two probe columns: the common mode cancels
    # exactly, leaving psi_r; its frequency-noise PSD is white at h_r.
    phi_rep = rep_rate_phase(phases, mus, 0, len(mus) - 1)
    f_r, s_r = frequency_noise_psd(phi_rep, T_R, nperseg=1 << 13)
    band_r = (f_r > f_s / 2048) & (f_r < f_s / 8)
    s_rep_direct = float(np.mean(s_r[band_r]))
    assert abs(s_rep_direct - h_r) / h_r < 0.10, s_rep_direct

    print("\n" + "=" * 72)
    print("GATE 3 (after test 9): tape-model recovery on the synthetic comb")
    print(f"{'quantity':<34}{'measured':>16}{'target':>12}{'tol':>8}")
    print(f"{'S_c  [Hz^2/Hz]':<34}{s_c_meas:>16.4e}{s_c_true:>12.4e}"
          f"{'10%':>8}")
    print(f"{'S_rep [Hz^2/Hz]':<34}{s_rep_meas:>16.4e}{h_r:>12.4e}{'10%':>8}")
    print(f"{'S_rep (direct phi_rep)':<34}{s_rep_direct:>16.4e}{h_r:>12.4e}"
          f"{'10%':>8}")
    print(f"{'mu_fix (band average)':<34}{mu_fix_band:>16.2f}"
          f"{mu_fix_true:>12d}{'+-2':>8}")
    print(f"{'mu_fix (per-bin median)':<34}{mu_fix_median:>16.2f}"
          f"{mu_fix_true:>12d}{'+-2':>8}")
    print("=" * 72)


def test_tape_model_requires_five_probes():
    phases = np.zeros((256, 4))
    with pytest.raises(ValueError, match=">= 5"):
        tape_model_fit(phases, (-2, -1, 1, 2), T_R)


def test_effective_linewidth_white_noise_analytic():
    """White frequency noise h0: the beta-line integral gives FWHM = pi*h0."""
    f_s = 1.0 / T_R
    h0 = 1.0e8
    rng = np.random.default_rng(7)
    phi = _white_freq_phase(rng, 1 << 17, h0, f_s)
    f, s = frequency_noise_psd(phi, T_R, nperseg=1 << 13)
    fwhm = effective_linewidth(f, s)
    assert fwhm == pytest.approx(math.pi * h0, rel=0.15), fwhm
    # a quiet PSD never crossing the beta line reports 0
    assert effective_linewidth(f, np.full_like(f, 1e-6)) == 0.0


def test_peak_trajectory_and_timing_jitter_recovery():
    """theta_max(t) recovers a known moving-pulse trajectory sub-cell."""
    n_tau, n_snap = 256, 512
    theta_grid = 2.0 * math.pi * np.arange(n_tau) / n_tau
    drift = np.linspace(0.0, 1.5, n_snap)
    wobble = 0.05 * np.sin(2.0 * math.pi * np.arange(n_snap) / 64.0)
    theta0 = 1.0 + drift + wobble
    snaps = np.stack([
        1.0 / np.cosh((np.angle(np.exp(1j * (theta_grid - t0)))) / 0.05)
        for t0 in theta0
    ]).astype(np.complex128)
    rec = peak_angle_trajectory(snaps)
    err = np.max(np.abs(rec - theta0))
    assert err < 2.0 * math.pi / n_tau, err     # sub-cell accurate

    tj = timing_jitter(snaps, snapshot_interval=10, t_r=T_R)
    # the wobble line dominates the detrended PSD at f_mod = fs_snap/64
    fs_snap = 1.0 / (10 * T_R)
    f_peak = tj["f"][np.argmax(tj["S_theta"])]
    assert f_peak == pytest.approx(fs_snap / 64.0, rel=0.3), f_peak
    assert tj["jitter_rms_s"] > 0.0


def test_timing_jitter_cross_checks_phi_rep():
    """phi_rep(t) = -theta_max(t) + const for a rigidly moving pulse.

    Build the comb of a moving sech pulse (E~_mu = A_mu*exp(-i*mu*theta0)),
    read probe phases, and compare phi_rep against -theta0.
    """
    n_tau, n_t = 128, 400
    mus = (-20, -10, -5, 5, 10, 20)
    rng = np.random.default_rng(3)
    theta0 = 0.5 + np.cumsum(1e-3 + 1e-3 * rng.standard_normal(n_t))
    amp = 1.0 / np.cosh(np.asarray(mus) / 15.0)
    probes = amp[None, :] * np.exp(
        -1j * np.outer(theta0, np.asarray(mus)))
    phases = unwrapped_phases(probes)
    phi_rep = rep_rate_phase(phases, mus, 0, len(mus) - 1)
    d_phi = phi_rep - phi_rep[0]
    d_theta = -(theta0 - theta0[0])
    assert np.max(np.abs(d_phi - d_theta)) < 1e-9


# ---------------------------------------------------------------------------
# Limiting behavior: T_k -> 0 collapse; variance conservation across models
# ---------------------------------------------------------------------------
def test_tk_zero_collapses_all_delta_t_channels():
    base = dict(nm_load_cfg(None))
    base["T_k"] = 0.0
    for model_kw in (
        {"trn_psd_model": "single_pole"},
        {"trn_psd_model": "kondratiev_gorodetsky", "trn_R_m": 100e-6,
         "trn_da_m": 2.0e-6, "trn_db_m": 8.0e-7},
    ):
        cfg = {**base, **model_kw}
        tn = TotalNoise(cfg)
        key = jax.random.PRNGKey(0)
        assert np.all(np.asarray(tn.sample(key, 512)) == 0.0), model_kw
        comb, dt = tn.sample_with_delta_t(key, 512)
        assert np.all(np.asarray(comb) == 0.0) and np.all(
            np.asarray(dt) == 0.0), model_kw
        full, dt_full = tn.sample_full_with_delta_t(key, 512)
        assert np.all(full == 0.0) and np.all(dt_full == 0.0), model_kw


def test_variance_conserved_across_psd_models(tmp_path):
    """integral_0^{f_s/2} S_dT df equals the Eq. 129 variance for every model.

    single_pole holds it analytically (the Lorentzian tail beyond f_s/2
    carries ~1e-6 of the norm at tau_th = 5 us), kondratiev_gorodetsky by
    construction (renormalization), csv by inheriting the tabulated shape.
    """
    base = dict(nm_load_cfg(None))
    f_s = 1.0 / TotalNoise(base).t_r
    var_129 = TotalNoise(base).var_delta_t

    geo = {"trn_R_m": 100e-6, "trn_da_m": 2.0e-6, "trn_db_m": 8.0e-7}
    kg_cfg = {**base, "trn_psd_model": "kondratiev_gorodetsky", **geo}
    kg = TotalNoise(kg_cfg)
    f_tab = np.logspace(1, math.log10(f_s / 2.0), 600)
    csv_file = tmp_path / "kg_tab.csv"
    np.savetxt(csv_file,
               np.column_stack([f_tab, kg.delta_t_psd(f_tab)]), delimiter=",")
    csv_cfg = {**base, "trn_psd_model": "csv",
               "trn_psd_csv_path": str(csv_file),
               "trn_csv_units": "S_delta_T"}

    rows = []
    for name, cfg in (("single_pole", base), ("kondratiev_gorodetsky", kg_cfg),
                      ("csv(K-G tab)", csv_cfg)):
        tn = TotalNoise(cfg)
        var = integrate_psd(tn.delta_t_psd, f_lo=1.0, f_hi=f_s / 2.0)
        rel = abs(var - var_129) / var_129
        rows.append((name, var, rel))
        assert rel < 0.02, (name, var, var_129)
    print("\n[variance conservation] Eq.129 = %.6e K^2" % var_129)
    for name, var, rel in rows:
        print(f"  {name:<24} integral = {var:.6e} K^2  (rel err {rel:.3%})")
