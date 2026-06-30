"""Battery validating the physically-scaled OFF threshold and JAX/NumPy parity.

The OFF floor is derived from physical config via ``make_threshold_params``
(f·U_cw,min, U_cw,min = κ_c·pin/((κ/2)²+δω_max²)) rather than a magic constant.
Every synthetic field below is asserted to receive its expected label in BOTH
labelers, and ``assert_labelers_consistent`` is run over the whole battery with
``atol=0`` (zero disagreements).
"""
import numpy as np
import jax.numpy as jnp
import pytest

from simulator.state_labeler import (
    make_state_labeler,
    label_soliton_state,
    assert_labelers_consistent,
    make_threshold_params,
    physical_off_floor,
)

N = 512

# Physical config (SI / rad·s⁻¹). Joule-scale fields, NO normalization.
KAPPA, KAPPA_C, PIN = 1.0e9, 5.0e8, 0.5
DW_MAX = 5.0 * KAPPA
PARAMS = make_threshold_params(KAPPA, KAPPA_C, PIN, DW_MAX, off_fraction=1e-3)

# Labels: 0 off, 1 cw, 2 mi, 3 chaotic, 4 multi, 5 crystal, 6 single
OFF, CW, MI, CHAOTIC, MULTI, CRYSTAL, SINGLE = range(7)


def _U_cw(dw: float) -> float:
    return KAPPA_C * PIN / ((KAPPA / 2.0) ** 2 + dw ** 2)


def _sech2(center, width, amp, n=N):
    x = np.arange(n, dtype=float)
    return (amp / np.cosh((x - center) / width)).astype(np.complex64)


def _build_battery():
    rng = np.random.default_rng(0)
    x = np.arange(N, dtype=float)
    fields = {}

    # empty: noise only, mean|E|² ~ 1e-16 J (far below the ~1e-14 J floor)
    a = np.sqrt(1e-16)
    fields["empty"] = (
        a * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    ).astype(np.complex64)

    # CW at δω = 0: flat field at the brightest CW energy
    A0 = np.sqrt(_U_cw(0.0))
    fields["cw_dw0"] = (
        A0 * np.ones(N)
        + 1e-4 * A0 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    ).astype(np.complex64)

    # CW at δω = 5κ: flat field at the DIMMEST CW energy on the sweep
    A5 = np.sqrt(_U_cw(DW_MAX))
    fields["cw_dw5k"] = (
        A5 * np.ones(N)
        + 1e-4 * A5 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    ).astype(np.complex64)

    # MI: amplitude-modulated multi-bump field, contrast in [2, 8)
    Ami = np.sqrt(_U_cw(0.0))
    mod = 1.0 + 0.85 * np.cos(2 * np.pi * 8 * x / N) + 0.35 * np.cos(2 * np.pi * 16 * x / N)
    fields["mi"] = (Ami * mod).astype(np.complex64)

    # single sech² soliton
    fields["single"] = _sech2(N // 2, 15.0, 1e-3)

    # multi-soliton: two well-separated solitons
    fields["multi"] = _sech2(N // 4, 12.0, 1e-3) + _sech2(3 * N // 4, 12.0, 1e-3)

    # soliton crystal: 8 evenly spaced identical solitons (narrow -> high contrast)
    cryst = np.zeros(N, dtype=np.complex64)
    for k in range(8):
        cryst = cryst + _sech2((k + 0.5) * N / 8, 3.0, 1e-3)
    fields["crystal"] = cryst

    return fields


EXPECTED = {
    "empty": OFF,
    "cw_dw0": CW,
    "cw_dw5k": CW,
    "mi": MI,
    "single": SINGLE,
    "multi": MULTI,
    "crystal": CRYSTAL,
}

_BATTERY = _build_battery()
_JAX_LABELER = make_state_labeler(PARAMS)


def test_physical_off_floor_formula():
    """Floor is f·U_cw,min, U_cw,min taken at the largest |δω| in the sweep."""
    f = 1e-3
    expected = f * KAPPA_C * PIN / ((KAPPA / 2.0) ** 2 + DW_MAX ** 2)
    assert physical_off_floor(KAPPA, KAPPA_C, PIN, DW_MAX, f) == pytest.approx(expected)
    assert PARAMS["power_floor"] == pytest.approx(expected)
    # floor sits between the empty-cavity numerical floor and the dimmest CW state
    assert 1e-16 < PARAMS["power_floor"] < _U_cw(DW_MAX)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_jax_label_matches_expected(name):
    E = _BATTERY[name]
    label = int(_JAX_LABELER(jnp.array(E, dtype=jnp.complex64)))
    assert label == EXPECTED[name], f"JAX labeled {name} as {label}, expected {EXPECTED[name]}"


@pytest.mark.parametrize("name", list(EXPECTED))
def test_numpy_label_matches_expected(name):
    E = _BATTERY[name]
    label = label_soliton_state(E, threshold_params=PARAMS)
    assert label == EXPECTED[name], f"NumPy labeled {name} as {label}, expected {EXPECTED[name]}"


@pytest.mark.parametrize("name", list(EXPECTED))
def test_labelers_consistent_atol0(name):
    # zero disagreements between JAX and NumPy across the whole battery
    assert_labelers_consistent(_BATTERY[name], atol=0.0, threshold_params=PARAMS)


def test_normal_cw_field_is_cw_not_off():
    """A normal CW field in physical Joules (no normalization) must be CW (1)."""
    for name in ("cw_dw0", "cw_dw5k"):
        E = _BATTERY[name]
        p_mean = float(np.mean(np.abs(E) ** 2))
        assert p_mean >= PARAMS["power_floor"]  # not below the OFF floor
        assert int(_JAX_LABELER(jnp.array(E, dtype=jnp.complex64))) == CW
        assert label_soliton_state(E, threshold_params=PARAMS) == CW


def test_empty_cavity_is_off():
    E = _BATTERY["empty"]
    assert int(_JAX_LABELER(jnp.array(E, dtype=jnp.complex64))) == OFF
    assert label_soliton_state(E, threshold_params=PARAMS) == OFF
