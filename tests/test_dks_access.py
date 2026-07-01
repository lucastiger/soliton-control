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
    access_by_seeding,
    count_temporal_peaks,
    existence_map,
    is_single_soliton,
    load_cavity_params,
    optical_spectrum,
    sech2_envelope_correlation,
    sech_soliton_seed,
    soliton_metrics,
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
