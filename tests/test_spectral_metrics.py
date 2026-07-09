"""Unit tests for analysis.spectral_metrics (Feature 1: 3 dB spectral span).

All tests use SYNTHETIC combs -- analytic sech^2 combs whose 3 dB (half-power)
width is known in closed form -- so they never run the solver.  The synthetic
comb is

    P(mu) = sech^2(mu / w_eff)              (optionally with a strong pump spike
                                             to emulate the DKS CW background),

for which the pump-excluded envelope (reference = strongest non-pump line at
mu = +/-1, power sech^2(1/w_eff)) crosses ``level_db`` below its maximum at

    mu = w_eff * arccosh( cosh(1/w_eff) * 10**(level_db/20) ),

so the closed-form full width is ``FWHM = 2 * mu`` (see ``_sech2_fwhm_expected``).
These cover the mandated robustness requirements: agreement with the analytic
width to < 2 %, stability under doubling the mode count, and stability under
changing the smoothing window between 3 and 5.  A separately built fringed comb
exercises the data-derived auto-smoothing that the real cycle-averaged breather
spectrum needs.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from analysis.spectral_metrics import (
    average_power_spectrum,
    comb_line_powers,
    conversion_efficiency_report,
    intracavity_comb_fraction,
    pump_to_comb_efficiency,
    sech2_core_fwhm,
    spectral_envelope_db,
    three_db_span,
)

FSR_HZ = 24.6e9          # a representative FSR; metrics are FSR-linear in Hz
LEVEL_DB = 3.0


# ---------------------------------------------------------------------------
# synthetic combs + analytic reference
# ---------------------------------------------------------------------------
def _sech2(mu, w_eff):
    return 1.0 / np.cosh(np.asarray(mu, dtype=np.float64) / w_eff) ** 2


def synth_sech2_comb(n, w_eff, *, pump_boost=1.0, center=0):
    """Fftshifted sech^2 comb of ``n`` modes; optional pump spike at mu=center."""
    mu = np.arange(n, dtype=np.int64) - n // 2
    P = _sech2(mu - center, w_eff)
    if pump_boost != 1.0:
        P = P.copy()
        P[mu == center] *= pump_boost
    return mu, P


def _sech2_fwhm_expected(w_eff, level_db=LEVEL_DB):
    """Closed-form ``level_db`` full width of a pump-excluded sech^2 comb.

    Reference is the strongest non-pump line at mu = +/-1 (power sech^2(1/w)):
    sech^2(mu/w) = sech^2(1/w) * 10**(-level_db/10) gives the half-width below.
    """
    half = w_eff * math.acosh(math.cosh(1.0 / w_eff) * 10.0 ** (level_db / 20.0))
    return 2.0 * half


def _meta():
    return {"fsr_hz": FSR_HZ, "pump_mu": 0}


# ---------------------------------------------------------------------------
# comb_line_powers
# ---------------------------------------------------------------------------
def test_comb_line_powers_passthrough_and_sort():
    mu = np.array([2, -1, 0, 1, -2])
    P = np.array([0.3, 0.8, 5.0, 0.9, 0.2])
    m, p = comb_line_powers(P, mu)
    assert np.array_equal(m, np.array([-2, -1, 0, 1, 2]))
    assert np.allclose(p, np.array([0.2, 0.8, 5.0, 0.9, 0.3]))
    assert p.dtype == np.float64


def test_comb_line_powers_default_fftshift_index():
    P = np.arange(8.0)
    m, _ = comb_line_powers(P)                 # default fftshift indexing
    assert np.array_equal(m, np.arange(8) - 4)


def test_comb_line_powers_rejects_bad_values():
    with pytest.raises(ValueError):
        comb_line_powers(np.array([1.0, -0.1, 2.0]), np.array([-1, 0, 1]))
    with pytest.raises(ValueError):
        comb_line_powers(np.array([1.0, np.nan, 2.0]), np.array([-1, 0, 1]))


def test_comb_line_powers_argmax_fallback_warns_when_no_pump():
    mu = np.array([3, 4, 5])                    # no mu == 0
    P = np.array([1.0, 9.0, 2.0])
    with pytest.warns(RuntimeWarning):
        comb_line_powers(P, mu)                 # pump_mu=0 absent -> fallback


# ---------------------------------------------------------------------------
# analytic-width agreement (< 2 %)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w_eff", [80.0, 120.0, 180.0])
@pytest.mark.parametrize("smooth", [3, 5, "auto"])
def test_three_db_span_matches_analytic_sech2(w_eff, smooth):
    # strong pump spike (100x) present: a correct metric must EXCLUDE it.
    mu, P = synth_sech2_comb(8192, w_eff, pump_boost=100.0)
    res = three_db_span(mu, P, _meta(), smooth_modes=smooth, level_db=LEVEL_DB)
    expected = _sech2_fwhm_expected(w_eff)
    rel = abs(res["span_modes"] - expected) / expected
    assert rel < 0.02, (f"span {res['span_modes']:.2f} vs analytic "
                        f"{expected:.2f} modes ({rel:.2%}) for w={w_eff}, "
                        f"smooth={smooth}")
    # the band must straddle the pump and be symmetric about it
    assert res["left_crossing_mu"] < 0 < res["right_crossing_mu"]
    assert abs(res["left_crossing_mu"] + res["right_crossing_mu"]) < 0.02 * expected
    # Hz conversion is exactly span_modes * FSR
    assert res["span_hz"] == pytest.approx(res["span_modes"] * FSR_HZ, rel=1e-12)


def test_pump_exclusion_changes_result():
    """Including the pump spike would wreck the metric; exclusion must be real."""
    mu, P = synth_sech2_comb(8192, 120.0, pump_boost=100.0)
    good = three_db_span(mu, P, _meta(), smooth_modes=5)
    # If the pump line were kept, its 100x spike (a single mode) would dominate
    # the envelope reference; the exclude_pump=False envelope peak sits at mu=0.
    env_mu, env_db = spectral_envelope_db(mu, P, exclude_pump=False,
                                          smooth_modes=1)
    assert abs(env_mu[np.argmax(env_db)]) < 1.0         # peak pinned at pump
    # with exclusion the peak is off the pump, near the sech^2 core
    assert np.isfinite(good["span_modes"])


# ---------------------------------------------------------------------------
# robustness (i): doubling the mode count / FFT resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smooth", [5, "auto"])
def test_robustness_mode_count_doubling(smooth):
    mu1, P1 = synth_sech2_comb(4096, 120.0, pump_boost=100.0)
    mu2, P2 = synth_sech2_comb(8192, 120.0, pump_boost=100.0)   # 2x modes, same w
    s1 = three_db_span(mu1, P1, _meta(), smooth_modes=smooth)["span_modes"]
    s2 = three_db_span(mu2, P2, _meta(), smooth_modes=smooth)["span_modes"]
    assert abs(s1 - s2) / s1 < 0.02, f"{s1:.2f} vs {s2:.2f} modes under 2x modes"


# ---------------------------------------------------------------------------
# robustness (ii): smoothing window 3 vs 5
# ---------------------------------------------------------------------------
def test_robustness_smoothing_window_3_vs_5():
    mu, P = synth_sech2_comb(8192, 120.0, pump_boost=100.0)
    s3 = three_db_span(mu, P, _meta(), smooth_modes=3)["span_modes"]
    s5 = three_db_span(mu, P, _meta(), smooth_modes=5)["span_modes"]
    assert abs(s3 - s5) / s3 < 0.02, f"window 3 vs 5: {s3:.2f} vs {s5:.2f} modes"


# ---------------------------------------------------------------------------
# fringed comb: data-derived auto-smoothing (the real cycle-averaged case)
# ---------------------------------------------------------------------------
def _fringed_comb(n, w_eff, *, fringe_db=3.5, period=18.0, pump_boost=100.0):
    """sech^2 comb with a quasi-periodic dB interference modulation deeper than
    the 3 dB level, plus two modest asymmetric spikes (like the real breather
    comb's tallest interference maxima) that jump the envelope reference between
    smoothing windows -- the pathology the data-derived auto-window handles."""
    mu = np.arange(n, dtype=np.int64) - n // 2
    core_db = 10.0 * np.log10(_sech2(mu, w_eff))
    db = core_db + fringe_db * np.cos(2.0 * np.pi * mu / period)
    for loc, amp in ((17, 2.2), (54, 2.6)):
        db = db + amp * np.exp(-0.5 * ((mu - loc) / 1.2) ** 2)
    P = 10.0 ** (db / 10.0)
    P[mu == 0] = _sech2(np.array([0]), w_eff)[0] * pump_boost
    return mu, P


def test_fringed_comb_light_window_is_fringe_dominated_auto_is_clean():
    mu, P = _fringed_comb(8192, 120.0)
    # a light median cannot tame a >3 dB-deep modulation: the -3 dB level slices
    # through the fringes, giving many crossings (a fringe-dominated envelope).
    for w in (3, 5):
        r = three_db_span(mu, P, _meta(), smooth_modes=w)
        assert r["n_crossings"] > 2, f"window {w} unexpectedly clean"
    # auto grows the window until the envelope is a single clean lobe.
    auto = three_db_span(mu, P, _meta(), smooth_modes="auto")
    assert auto["n_crossings"] == 2
    assert auto["params"]["smooth_modes_used"] > 5
    assert not auto["warnings"], auto["warnings"]


def test_fringed_comb_auto_stable_under_mode_count_and_recovers_width():
    # For a DETERMINISTIC cycle-averaged spectrum the operative robustness axis
    # is FFT resolution / mode count (the fringes are fixed in mu), not phase.
    base = three_db_span(*_fringed_comb(8192, 120.0),
                         metadata=_meta(), smooth_modes="auto")["span_modes"]
    doubled = three_db_span(*_fringed_comb(16384, 120.0),
                            metadata=_meta(), smooth_modes="auto")["span_modes"]
    assert abs(base - doubled) / base < 0.02, f"{base:.1f} vs {doubled:.1f} (2x)"
    # the fringe-averaged auto span recovers the underlying sech^2 FWHM
    expected = _sech2_fwhm_expected(120.0)
    assert abs(base - expected) / expected < 0.15


# ---------------------------------------------------------------------------
# edge cases (return NaN, never raise)
# ---------------------------------------------------------------------------
def test_edge_case_no_crossing_flat_comb():
    mu = np.arange(1024, dtype=np.int64) - 512
    P = np.ones(mu.size)                        # flat: 0 dB everywhere
    P[mu == 0] = 50.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = three_db_span(mu, P, _meta(), smooth_modes=3)
    assert math.isnan(res["span_modes"])
    assert any("dynamic range" in w or "never" in w for w in res["warnings"])


def test_edge_case_one_sided_truncated_band():
    # sech^2 whose left side is truncated well inside the -3 dB half-width, so
    # the left crossing is never bracketed.
    w = 120.0
    mu = np.arange(-30, 3001, dtype=np.int64)
    P = _sech2(mu, w)
    P[mu == 0] *= 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = three_db_span(mu, P, _meta(), smooth_modes=5)
    assert res["one_sided"] is True
    assert math.isnan(res["left_crossing_mu"])
    assert math.isfinite(res["right_crossing_mu"])
    assert math.isnan(res["span_modes"])


def test_edge_case_missing_fsr_gives_nan_hz():
    mu, P = synth_sech2_comb(4096, 120.0, pump_boost=100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = three_db_span(mu, P, {"pump_mu": 0}, smooth_modes=5)  # no fsr_hz
    assert math.isfinite(res["span_modes"])
    assert math.isnan(res["span_hz"])


def test_dispersive_wave_dominated_warns():
    # weak narrow core + a strong far "dispersive wave" peak that dominates the
    # envelope; retain it with a deep floor.  The 3 dB band then does not
    # straddle mu=0 -> a warning naming the shoulder is emitted.
    mu = np.arange(4096, dtype=np.int64) - 2048
    core = _sech2(mu, 25.0)
    dw = 6.0 * np.exp(-0.5 * ((mu - 700) / 12.0) ** 2)
    P = core + dw
    P[mu == 0] *= 100.0
    with pytest.warns(RuntimeWarning):
        res = three_db_span(mu, P, _meta(), smooth_modes=5, floor_db=200.0)
    assert any("does not straddle" in w for w in res["warnings"])
    assert res["envelope_peak_mu"] > 400


# ---------------------------------------------------------------------------
# result schema / units / definition
# ---------------------------------------------------------------------------
def test_result_carries_units_definition_and_params():
    mu, P = synth_sech2_comb(8192, 120.0, pump_boost=100.0)
    res = three_db_span(mu, P, _meta(), smooth_modes="auto")
    for key in ("metric_definition", "units", "params", "span_modes",
                "span_hz", "span_thz", "span_ghz", "reference_level_db"):
        assert key in res
    assert res["units"]["span_hz"] == "Hz"
    assert res["params"]["level_db"] == LEVEL_DB
    assert "FWHM" not in res["metric_definition"] or "half-power" in \
        res["metric_definition"]


# ---------------------------------------------------------------------------
# sech^2 cross-check
# ---------------------------------------------------------------------------
def test_sech2_core_fwhm_crosscheck_recovers_width():
    w = 120.0
    mu, P = synth_sech2_comb(8192, w, pump_boost=100.0)
    cc = sech2_core_fwhm(mu, P, core_mu=300, fsr_hz=FSR_HZ)
    assert cc["fit_rms_db"] < 0.5
    assert abs(cc["width_w_modes"] - w) / w < 0.02
    expected = _sech2_fwhm_expected(w)
    # cross-check FWHM (referenced to the fit's own peak at mu=0) is close to the
    # pump-excluded analytic width for a broad comb
    assert abs(cc["fwhm_modes"] - expected) / expected < 0.03


# ---------------------------------------------------------------------------
# convenience wrapper: LINEAR-power averaging (never dB, never complex)
# ---------------------------------------------------------------------------
def test_average_power_spectrum_is_linear_power_not_complex():
    rng = np.random.default_rng(0)
    n_tau, n_avg = 512, 64
    mu = np.arange(n_tau) - n_tau // 2
    amp = np.sqrt(_sech2(mu, 40.0))            # fixed per-mode amplitude
    # snapshots share the SAME power spectrum but random per-mode phases (like a
    # breather sampled at different phases): power average must recover amp**2,
    # complex-field averaging would cancel the wings toward zero.
    fields = np.empty((n_avg, n_tau), dtype=np.complex128)
    for r in range(n_avg):
        modes_shift = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, n_tau))
        modes = np.fft.ifftshift(modes_shift)
        fields[r] = np.fft.ifft(modes)
    m, P = average_power_spectrum(fields)
    assert np.array_equal(m, mu)
    assert np.allclose(P, amp ** 2, atol=1e-6 * amp.max() ** 2)
    # complex-averaged-then-power is far smaller in the wings (phase cancellation)
    complex_avg = np.mean(fields, axis=0)
    wrong = np.abs(np.fft.fftshift(np.fft.fft(complex_avg))) ** 2
    wing = np.abs(mu) > 120
    assert wrong[wing].mean() < 0.05 * P[wing].mean()


def test_average_power_spectrum_is_power_passthrough():
    powers = np.stack([np.array([1.0, 2.0, 3.0, 4.0]),
                       np.array([3.0, 2.0, 1.0, 0.0])])
    m, P = average_power_spectrum(powers, is_power=True)
    assert np.allclose(P, np.array([2.0, 2.0, 2.0, 2.0]))


# ===========================================================================
# Feature 2: conversion efficiency
# ===========================================================================
KAPPA_C = 1.215e8        # config kappa_c_rad_per_s (external out-coupling rate)
PIN_W = 0.214            # config on-chip pump power


def _abs_field_spectrum(n=512, w=40.0, pump=5.0, seed=2):
    """A synthetic ABSOLUTE spectrum P = |fftshift(fft(E))|^2 and its field E.

    A sech^2 comb with a strong pump line and random per-mode phases, so
    <|E|^2> = mean(|E|^2) is a genuine (un-normalised) intracavity power scale.
    """
    rng = np.random.default_rng(seed)
    mu = np.arange(n) - n // 2
    amp = np.sqrt(_sech2(mu, w))
    amp[mu == 0] = pump
    modes = amp * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, n))
    e_field = np.fft.ifft(np.fft.ifftshift(modes))
    P = np.abs(np.fft.fftshift(np.fft.fft(e_field))) ** 2
    return mu, P, e_field


# ----- eta_intra: exact known-power identity -------------------------------
def test_intracavity_comb_fraction_exact_known_powers():
    # Known pump power P0 and known total comb power Pc: eta_intra == Pc/(P0+Pc)
    # EXACTLY (all values are exactly representable in float64).
    mu = np.array([-2, -1, 0, 1, 2])
    P0, comb = 7.0, np.array([0.5, 1.5, 0.0, 2.0, 1.0])
    Pc = comb.sum()
    P = comb.copy()
    P[mu == 0] = P0
    assert intracavity_comb_fraction(mu, P) == Pc / (P0 + Pc)   # exact


def test_intracavity_comb_fraction_pump_is_mu0_not_argmax():
    # Pump-suppressed comb: a NON-pump line is the strongest. eta_intra must
    # still treat mu=0 as the pump, never the max line.
    mu = np.array([-1, 0, 1, 2])
    P = np.array([2.0, 1.0, 9.0, 3.0])          # argmax at mu=+1
    assert intracavity_comb_fraction(mu, P) == (15.0 - 1.0) / 15.0


def test_intracavity_comb_fraction_in_unit_interval():
    mu, P = synth_sech2_comb(2048, 60.0, pump_boost=80.0)
    eta = intracavity_comb_fraction(mu, P)
    assert 0.0 <= eta < 1.0


# ----- eta_intra: scale / normalisation invariance -------------------------
def test_intracavity_comb_fraction_scale_invariant():
    mu, P = synth_sech2_comb(2048, 60.0, pump_boost=80.0)
    base = intracavity_comb_fraction(mu, P)
    assert intracavity_comb_fraction(mu, P / P.max()) == pytest.approx(base, rel=1e-12)
    assert intracavity_comb_fraction(mu, 1234.5 * P) == pytest.approx(base, rel=1e-12)


def test_intracavity_comb_fraction_invariant_under_fft_zero_padding():
    # Mandated invariance: zero-padding the temporal field by 2x interpolates the
    # spectrum; the PHYSICAL cavity modes are the even bins of the padded FFT and
    # their per-mode powers are unchanged (fft(E_pad)[2k] == fft(E)[k]), so
    # eta_intra over the physical modes is identical.
    n = 256
    mu = np.arange(n) - n // 2
    rng = np.random.default_rng(1)
    amp = np.sqrt(_sech2(mu, 30.0))
    amp[mu == 0] += 6.0
    modes = amp * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, n))
    e_field = np.fft.ifft(np.fft.ifftshift(modes))

    P1 = np.abs(np.fft.fftshift(np.fft.fft(e_field))) ** 2
    e1 = intracavity_comb_fraction(mu, P1)

    e_pad = np.concatenate([e_field, np.zeros_like(e_field)])   # 2x temporal pad
    phys_bins = np.fft.fft(e_pad)[0::2]                          # == fft(e_field)
    P2 = np.abs(np.fft.fftshift(phys_bins)) ** 2
    assert np.allclose(P2, P1, rtol=0, atol=1e-9 * P1.max())     # powers unchanged
    assert intracavity_comb_fraction(mu, P2) == pytest.approx(e1, rel=1e-9)


# ----- eta_intra: ratio of averages, not average of ratios -----------------
def test_eta_intra_is_ratio_of_averages_not_average_of_ratios():
    # A breather sampled at many phases: averaging |a_mu|^2 in LINEAR power and
    # THEN taking the ratio (what the metric does) is NOT the mean of the
    # per-snapshot ratios (the ratio is nonlinear in the comb amplitude).
    n, n_avg = 256, 40
    mu = np.arange(n) - n // 2
    base = np.sqrt(_sech2(mu, 25.0))
    snaps = np.empty((n_avg, n))
    per_snap_ratio = np.empty(n_avg)
    for r in range(n_avg):
        g = 1.0 + 0.6 * np.sin(2.0 * np.pi * r / n_avg)     # breathing modulation
        amp = base.copy()
        amp[mu == 0] = 5.0
        amp[mu != 0] *= g
        Pr = amp ** 2
        snaps[r] = Pr
        per_snap_ratio[r] = intracavity_comb_fraction(mu, Pr)

    eta_roa = intracavity_comb_fraction(mu, snaps.mean(axis=0))   # ratio of averages
    eta_aor = float(per_snap_ratio.mean())                        # average of ratios
    assert abs(eta_roa - eta_aor) > 1e-3                          # genuinely differ
    # the metric's value equals eta_intra of the LINEAR-power-averaged spectrum
    m, p_avg = average_power_spectrum(snaps, is_power=True)
    assert eta_roa == pytest.approx(intracavity_comb_fraction(m, p_avg), rel=1e-12)


# ----- eta (pump -> comb): the intracavity->bus derivation -----------------
def test_pump_to_comb_efficiency_matches_derivation():
    mu, P, e_field = _abs_field_spectrum()
    n = mu.size
    eta_intra = intracavity_comb_fraction(mu, P)
    mean_e2 = float(np.mean(np.abs(e_field) ** 2))
    # Parseval: <|E|^2> == sum_mu P_mu / n_tau^2 (the repo FFT normalization).
    assert mean_e2 == pytest.approx(P.sum() / n ** 2, rel=1e-9)
    expected = KAPPA_C * eta_intra * mean_e2 / PIN_W
    eta = pump_to_comb_efficiency(mu, P, {"kappa_c_rad_per_s": KAPPA_C, "pin_w": PIN_W})
    assert eta is not None
    assert eta == pytest.approx(expected, rel=1e-9)
    assert eta > 0.0


def test_pump_to_comb_efficiency_energy_anchor_overrides_normalisation():
    # Given an explicit <|E|^2> anchor, eta is computable even from a
    # pump-normalised spectrum (eta_intra is scale-invariant).
    mu, P, e_field = _abs_field_spectrum()
    eta_intra = intracavity_comb_fraction(mu, P)
    mean_e2 = float(np.mean(np.abs(e_field) ** 2))
    config = {"kappa_c_rad_per_s": KAPPA_C, "pin_w": PIN_W,
              "spectrum_pump_normalized": True,
              "mean_intracavity_energy_j": mean_e2}
    eta = pump_to_comb_efficiency(mu, P / P.max(), config)
    assert eta == pytest.approx(KAPPA_C * eta_intra * mean_e2 / PIN_W, rel=1e-9)


def test_kappa_c_derived_from_coupling_q():
    # kappa_c may come from coupling_q + pump_wavelength_m (omega0 / Q_c), the
    # same fallback as simulator.lle_solver.resolve_cavity_rates.
    mu, P, e_field = _abs_field_spectrum()
    lam, q_c = 1.55e-6, 1.0e7
    kappa_c = 2.0 * np.pi * 299_792_458.0 / lam / q_c
    config = {"coupling_q": q_c, "pump_wavelength_m": lam, "pin_w": PIN_W}
    expected = kappa_c * intracavity_comb_fraction(mu, P) * np.mean(np.abs(e_field) ** 2) / PIN_W
    assert pump_to_comb_efficiency(mu, P, config) == pytest.approx(expected, rel=1e-9)


# ----- eta: the None gates (never guess) -----------------------------------
def test_pump_to_comb_efficiency_none_when_pump_normalised_without_anchor():
    mu, P, _ = _abs_field_spectrum()
    config = {"kappa_c_rad_per_s": KAPPA_C, "pin_w": PIN_W,
              "spectrum_pump_normalized": True}
    assert pump_to_comb_efficiency(mu, P / P.max(), config) is None
    d = conversion_efficiency_report(mu, P / P.max(), config)
    assert d["eta"] is None
    assert "normalis" in d["eta_reason"]              # explains the missing scale
    assert np.isfinite(d["eta_intra"])                # eta_intra still reported


def test_pump_to_comb_efficiency_none_when_config_missing_params():
    mu, P, _ = _abs_field_spectrum()
    assert pump_to_comb_efficiency(mu, P, {"pin_w": PIN_W}) is None            # no kappa_c
    assert pump_to_comb_efficiency(mu, P, {"kappa_c_rad_per_s": KAPPA_C}) is None  # no pin
    d = conversion_efficiency_report(mu, P, {"pin_w": PIN_W})
    assert d["eta"] is None
    assert "kappa_c" in d["eta_reason"]
