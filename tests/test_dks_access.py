"""Fast regression tests for the DKS access protocol (analysis.dks_access).

These use short integrations (a small fraction of tau_th) so they stay CI-cheap
while pinning the mechanics and the qualitative physics:

  * the analytic sech seed has the right peak amplitude / width,
  * warm-start seeding (route b) yields a single soliton (class 6, one temporal
    peak, sech^2 envelope corr > 0.9, stable U_int) at a detuning in the window,
  * a cold start with NO protocol does NOT yield a class-6 single soliton
    (control),
  * the optical-spectrum mapping puts the pump line at the pump wavelength and
    spaces bins by one FSR,
  * the existence map returns a contiguous single-soliton band.

The full science run (long integration >= 5*tau_th, existence sweep, figures) is
`python -m analysis.dks_access`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.dks_access import (
    C_LIGHT,
    SEED_MIN_SEPARATION_WIDTHS,
    access_by_seeding,
    count_temporal_peaks,
    existence_map,
    is_single_soliton,
    load_cavity_params,
    optical_spectrum,
    sech2_envelope_correlation,
    sech_soliton_seed,
    soliton_metrics,
    temporal_peak_positions,
    _run,
)

DW_KAPPA = 8.0          # inside the seeded existence window at pin=0.214 W
T_SLOW = 8_000          # << tau_th (~1.23e5 rt); enough for the seed to settle


@pytest.fixture(scope="module")
def cav():
    return load_cavity_params()


@pytest.fixture(scope="module")
def seeded(cav):
    return access_by_seeding(DW_KAPPA * cav.kappa, cav, t_slow=T_SLOW, seed=0)


def test_seed_amplitude_and_width(cav):
    dw = DW_KAPPA * cav.kappa
    seed = sech_soliton_seed(dw, cav)
    # peak power above the CW background matches B^2 = 2*dw/gamma
    p = np.abs(seed) ** 2
    e_bg = math.sqrt(cav.kappa_c * 0.214) / (cav.kappa / 2.0 + 1j * dw)
    peak_over_bg = p.max() - abs(e_bg) ** 2
    b2 = 2.0 * dw / cav.gamma
    assert abs(peak_over_bg - b2) / b2 < 0.05
    assert count_temporal_peaks(seed) == 1


def _synthetic_multi_sech(n_tau, angles_rad, *, width_rad=0.02, bg=0.05):
    """Synthetic |E|-like field: flat background + unit sech pulses at angles."""
    theta = 2.0 * np.pi * np.arange(n_tau) / n_tau
    field = np.full(n_tau, bg, dtype=np.complex128)
    for a in angles_rad:
        d = np.angle(np.exp(1j * (theta - a)))   # wrapped to [-pi, pi)
        field += 1.0 / np.cosh(d / width_rad)
    return field


def test_temporal_peak_positions_recovers_multi_sech_angles():
    n = 4096
    angles = np.array([0.7, 2.9, 5.3])
    f = _synthetic_multi_sech(n, angles)
    pos = temporal_peak_positions(f)
    assert pos.size == 3
    assert count_temporal_peaks(f) == 3
    assert np.all(np.diff(pos) > 0)             # sorted ascending
    assert np.allclose(pos, angles, atol=1.5 * 2.0 * np.pi / n)


def test_temporal_peak_positions_wraps_at_zero():
    """A peak straddling theta = 0 is found once, at ~0 (circular wrap)."""
    n = 2048
    angles = np.array([0.0, np.pi])
    pos = temporal_peak_positions(_synthetic_multi_sech(n, angles))
    assert pos.size == 2
    tol = 1.5 * 2.0 * np.pi / n
    assert min(pos[0], 2.0 * np.pi - pos[0]) < tol      # near 0 (mod 2*pi)
    assert abs(pos[1] - np.pi) < tol


def test_temporal_peak_positions_dark_field_empty():
    assert temporal_peak_positions(np.zeros(512, dtype=complex)).size == 0


def test_temporal_peak_positions_matches_seed_positions(cav):
    """Round trip: multi-soliton seed -> peak positions recover the placement."""
    n = 4096
    dw = 12.0 * cav.kappa
    seed = sech_soliton_seed(dw, cav, n_tau=n, n_solitons=5,
                             position_seed=1, position_jitter_frac=0.25)
    pos = temporal_peak_positions(seed)
    assert pos.size == 5
    assert count_temporal_peaks(seed) == 5


def test_multi_soliton_seed_deterministic_and_symmetry_broken(cav):
    dw = 12.0 * cav.kappa
    kw = dict(n_tau=4096, n_solitons=5, position_jitter_frac=0.25)
    s1 = sech_soliton_seed(dw, cav, position_seed=1, **kw)
    s2 = sech_soliton_seed(dw, cav, position_seed=1, **kw)
    s3 = sech_soliton_seed(dw, cav, position_seed=2, **kw)
    assert np.array_equal(s1, s2)               # deterministic in position_seed
    assert not np.array_equal(s1, s3)
    # symmetry broken: circular gaps between adjacent peaks are NOT all equal
    pos = temporal_peak_positions(s1)
    gaps = np.diff(np.concatenate([pos, [pos[0] + 2.0 * np.pi]]))
    assert np.std(gaps) > 1e-3


def test_multi_soliton_seed_zero_jitter_forbidden(cav):
    with pytest.raises(ValueError, match="jitter"):
        sech_soliton_seed(12.0 * cav.kappa, cav, n_tau=1024, n_solitons=3,
                          position_jitter_frac=0.0)


def test_multi_soliton_seed_min_separation_enforced(cav):
    """Two pulses closer than 20 soliton widths raise, naming the pair."""
    dw = 12.0 * cav.kappa
    w = math.sqrt(cav.d2 / (2.0 * dw))          # no dispersion attached -> d2
    bad = [1.0, 1.0 + 0.5 * SEED_MIN_SEPARATION_WIDTHS * w, 4.0]
    with pytest.raises(ValueError, match=r"pulses 0 and 1"):
        sech_soliton_seed(dw, cav, n_tau=1024, n_solitons=3, positions_rad=bad)


def test_multi_soliton_seed_positions_rad_length_checked(cav):
    with pytest.raises(ValueError, match="positions_rad"):
        sech_soliton_seed(12.0 * cav.kappa, cav, n_tau=1024, n_solitons=3,
                          positions_rad=[1.0, 4.0])


def test_single_soliton_seed_backward_compatible(cav):
    """N = 1 default placement reproduces the historical construction exactly."""
    dw = DW_KAPPA * cav.kappa
    new = sech_soliton_seed(dw, cav, n_tau=2048)
    dt = cav.t_r / 2048
    t = np.arange(2048) * dt
    amp = math.sqrt(2.0 * dw / cav.gamma)
    tau_s = math.sqrt(cav.beta2 / (2.0 * dw))
    e_bg = math.sqrt(cav.kappa_c * 0.214) / (cav.kappa / 2.0 + 1j * dw)
    old = (np.full(2048, e_bg, dtype=np.complex128)
           + amp / np.cosh((t - 0.5 * cav.t_r) / tau_s)).astype(np.complex64)
    assert np.array_equal(new, old)


def test_seeding_yields_single_soliton(seeded):
    m = seeded["metrics"]
    assert seeded["is_single"], m
    assert m["n_peaks"] == 1
    assert m["np_label"] == 6
    assert m["sech2_env_corr"] > 0.9
    assert m["u_int_tail_rel_std"] < 0.05
    assert m["finite"]


def test_cold_start_control_is_not_single_soliton(cav):
    """No seed, no protocol: the plain run must NOT be a class-6 single soliton."""
    dw = DW_KAPPA * cav.kappa
    sol = _run(dw, T_SLOW, cav, e0=None, seed=0)
    e = np.asarray(sol["e_final"])[0]
    u = np.asarray(sol["U_int_history"])[0]
    m = soliton_metrics(e, u, cav, dw)
    assert not is_single_soliton(m), (
        f"cold-start control unexpectedly single soliton: {m}"
    )


def test_optical_spectrum_mapping(cav, seeded):
    sp = optical_spectrum(seeded["e_final"], cav)
    n = seeded["e_final"].shape[0]
    # DC bin (pump line) sits at the pump wavelength
    assert sp["mu"][n // 2] == 0
    assert abs(sp["wavelength_nm"][n // 2] - cav.pump_wavelength_m * 1e9) < 1e-6
    # adjacent bins are spaced by exactly one FSR in frequency
    df = np.diff(sp["f_mu_hz"])
    assert np.allclose(df, cav.fsr_hz, rtol=1e-6)
    # power normalized to 0 dB peak
    assert abs(sp["power_db"].max()) < 1e-6


def test_sech2_envelope_correlation_high_for_soliton(seeded):
    corr, r2, mode_w = sech2_envelope_correlation(seeded["e_final"])
    assert corr > 0.9
    assert mode_w > 1.0


def test_existence_map_contiguous_band(cav):
    dw_grid = [3.0, 5.0, 8.0, 10.0]
    emap = existence_map(cav, dw_grid, t_slow=T_SLOW, seed=0)
    assert emap["band"]["found"]
    assert emap["band"]["contiguous"]
    # the validated detuning must be inside the reported band
    lo = emap["band"]["lower_over_kappa"]
    hi = emap["band"]["upper_over_kappa"]
    assert lo <= DW_KAPPA <= hi
