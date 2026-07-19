"""Labeler robustness to the quantum-vacuum floor (Workstream B).

With the quantum Langevin drive on, every spectral mode carries the
symmetric-ordered vacuum occupation <|E_mu|^2> = n_tau^2*hbar*omega0/2 whose
single-snapshot power is EXPONENTIALLY distributed: log10-power std =
(pi/sqrt(6))/ln 10 ~ 0.557 decades ~ 5.6 dB per mode. The historical
single-DKS envelope gate (mono_frac of per-mode log10 steps against a 0.05
tolerance) read those fluctuations as non-monotonic structure and routed an
energetically verified single soliton to class 3.

The fix (simulator/state_labeler.py): for the envelope gate ONLY, the linear
spectrum is (i) circularly moving-averaged over ``envelope_smooth_modes``
(odd-adjusted; residual ~5.6/sqrt(w) dB) and (ii) clipped at the absolute,
state-independent ``vacuum_floor_level`` = margin * n_tau^2*hbar*omega0/2
BEFORE peak normalization — clipped wings form an exactly flat plateau
("envelope terminated", trivially monotone). Additionally ``power_floor`` is
lifted to max(off_fraction*U_cw_min, margin*n_tau*hbar*omega0/2) so a pure
vacuum-filled cavity labels OFF, never CW. All three knobs are inactive by
default and the inactive path is a build-time Python branch tracing the exact
historical arithmetic — pinned here against a golden battery recorded from
the PRE-change labeler.
"""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from analysis.dks_access import (
    PIN_W,
    PRODUCTION_NUMERICS,
    _run,
    attach_dispersion,
    load_cavity_params,
    sech_soliton_seed,
)
from analysis.run_detuning_sweep import write_noise_off_config
from simulator.lle_solver import _load_config, hbar_omega0_from_config, solve_lle_ssfm_jax
from simulator.state_labeler import (
    label_soliton_state,
    make_state_labeler,
    make_threshold_params,
)

KAPPA = 1.519e8
KAPPA_C = 1.215e8
HBW = hbar_omega0_from_config(_load_config())


def _active_params(n_tau: int, margin: float = 10.0, w: int = 8) -> dict:
    """Threshold params with the vacuum-floor features ON (solver derivation)."""
    return make_threshold_params(
        KAPPA,
        KAPPA_C,
        0.214,
        11.0 * KAPPA,
        vacuum_floor_level=margin * (n_tau**2) * HBW / 2.0,
        envelope_smooth_modes=w,
        vacuum_off_floor=margin * n_tau * HBW / 2.0,
    )


def _vacuum_field(n_tau: int, rng: np.random.Generator) -> np.ndarray:
    """Time-domain vacuum draw: per-sample complex variance n_tau*hbar*w0/2."""
    sq = math.sqrt(n_tau * HBW / 4.0)  # per-quadrature std
    return (
        sq * rng.normal(size=n_tau) + 1j * sq * rng.normal(size=n_tau)
    ).astype(np.complex128)


# ---------------------------------------------------------------------------
# 1. Bit-identity of the inactive path (golden battery from the PRE-change code)
# ---------------------------------------------------------------------------
def _battery() -> dict[str, np.ndarray]:
    n = 512
    rng = np.random.default_rng(1234)
    x = np.arange(n, dtype=float)
    sech = lambda c, w, a: a / np.cosh((x - c) / w)  # noqa: E731
    fields = {
        "flat_low": np.ones(n, dtype=np.complex128) * 1e-9,
        "cw_bright": np.full(n, 3e-6 + 1j * 1e-6),
        "single_soliton": (sech(n // 2, 12.0, 8e-5) + 2e-6).astype(np.complex128),
        "two_soliton": (
            sech(n // 4, 10.0, 8e-5) + sech(3 * n // 4, 10.0, 8e-5) + 2e-6
        ).astype(np.complex128),
        "crystal5": sum(
            sech((i + 0.5) * n / 5.0, 6.0, 8e-5) for i in range(5)
        ).astype(np.complex128)
        + 2e-6,
        "chaotic": (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(
            np.complex128
        )
        * 3e-6,
        "mi_rolls": (2e-6 * (1.0 + 0.9 * np.cos(2 * np.pi * 8 * x / n))).astype(
            np.complex128
        ),
    }
    spike = np.full(n, 3e-6, dtype=np.complex128)
    spike[100] += 4e-5
    fields["cw_spike"] = spike
    for s in range(3):
        r = np.random.default_rng(100 + s)
        fields[f"noisy_cw_{s}"] = (
            3e-6 * (1 + 0.02 * r.normal(size=n))
            + 1j * 3e-6 * 0.02 * r.normal(size=n)
        ).astype(np.complex128)
    return fields


# Labels recorded from the PRE-change labeler (commit aa5889e) on the exact
# battery above; JAX == NumPy for every case, under both the default and the
# physically-scaled threshold sets.
_GOLDEN = {
    "flat_low": 0,
    "cw_bright": 1,
    "single_soliton": 6,
    "two_soliton": 4,
    "crystal5": 5,
    "chaotic": 3,
    "mi_rolls": 2,
    "cw_spike": 1,
    "noisy_cw_0": 1,
    "noisy_cw_1": 1,
    "noisy_cw_2": 1,
}


@pytest.mark.parametrize("param_set", ["defaults", "physical"])
def test_inactive_path_bit_identical_to_golden_battery(param_set):
    params = (
        None
        if param_set == "defaults"
        else make_threshold_params(KAPPA, KAPPA_C, 0.214, 12 * KAPPA)
    )
    lab = make_state_labeler(params)
    for name, field in _battery().items():
        assert int(lab(jnp.asarray(field))) == _GOLDEN[name], (param_set, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sech2 curve_fit cosh overflow
            assert int(label_soliton_state(field, params)) == _GOLDEN[name], (
                param_set,
                name,
            )


# ---------------------------------------------------------------------------
# 2. The fix + JAX/NumPy parity on synthetic noise-on fields
# ---------------------------------------------------------------------------
def _synthetic_dks_with_vacuum(n_tau: int, seed: int) -> np.ndarray:
    # Pump-dominant proportions matching the real 11*kappa state (bright CW
    # background + narrow sech), so the entropy/contrast gates behave as they
    # do on solver fields; the wings are pure vacuum.
    x = np.arange(n_tau, dtype=float)
    field = (8e-5 / np.cosh((x - n_tau / 2) / 6.0) + 2e-5).astype(np.complex128)
    return field + _vacuum_field(n_tau, np.random.default_rng(seed))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vacuum_floor_fixes_single_dks_and_paths_agree(seed):
    """Active params: 6 in BOTH paths; inactive JAX: 3 (the caveat, pinned)."""
    n_tau = 1024
    field = _synthetic_dks_with_vacuum(n_tau, seed)
    active = _active_params(n_tau)
    assert int(make_state_labeler(active)(jnp.asarray(field))) == 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert int(label_soliton_state(field, active)) == 6
    # the historical gate reads the vacuum wings as non-monotonic -> class 3
    inactive = make_threshold_params(KAPPA, KAPPA_C, 0.214, 11.0 * KAPPA)
    assert int(make_state_labeler(inactive)(jnp.asarray(field))) == 3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multi_soliton_with_vacuum_labels_4(seed):
    n_tau = 1024
    x = np.arange(n_tau, dtype=float)
    field = (
        8e-5 / np.cosh((x - n_tau / 4) / 6.0)
        + 8e-5 / np.cosh((x - 3 * n_tau / 4 - 17) / 6.0)
        + 2e-5
    ).astype(np.complex128)
    field = field + _vacuum_field(n_tau, np.random.default_rng(1000 + seed))
    active = _active_params(n_tau)
    assert int(make_state_labeler(active)(jnp.asarray(field))) == 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert int(label_soliton_state(field, active)) == 4


# ---------------------------------------------------------------------------
# 3. Vacuum background must label OFF (never CW/MI)
# ---------------------------------------------------------------------------
def test_pure_vacuum_field_labels_off_with_lifted_floor():
    n_tau = 1024
    field = _vacuum_field(n_tau, np.random.default_rng(7))
    active = _active_params(n_tau)
    assert int(make_state_labeler(active)(jnp.asarray(field))) == 0
    assert int(label_soliton_state(field, active)) == 0
    # without the lift (pin=0 => CW-derived floor 0) the vacuum is promoted
    no_lift = make_threshold_params(KAPPA, KAPPA_C, 0.0, 0.0)
    assert int(make_state_labeler(no_lift)(jnp.asarray(field))) != 0


def test_solver_pin_zero_noise_on_labels_off():
    """End-to-end: pin=0 with the Langevin drive on labels 0 throughout."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_lle_ssfm_jax(
            pin=0.0,
            delta_omega=0.0,
            t_slow=400,
            beta=[1.578e-18],
            kappa=KAPPA,
            kappa_c=KAPPA_C,
            rng_key=jax.random.PRNGKey(3),
            n_tau=256,
            snapshot_interval=50,
            quantum_noise_enabled=True,
        )
    labels = np.asarray(sol["label_history"])[0]
    assert np.all(labels == 0), labels.tolist()


def test_off_floor_lift_arithmetic():
    p = make_threshold_params(KAPPA, KAPPA_C, 0.214, 11 * KAPPA, vacuum_off_floor=1e-3)
    assert p["power_floor"] == 1e-3
    p2 = make_threshold_params(KAPPA, KAPPA_C, 0.214, 11 * KAPPA, vacuum_off_floor=0.0)
    from simulator.state_labeler import physical_off_floor

    assert p2["power_floor"] == physical_off_floor(KAPPA, KAPPA_C, 0.214, 11 * KAPPA)


# ---------------------------------------------------------------------------
# 4. Real-solver acceptance: single DKS labels 6 with noise ON
# ---------------------------------------------------------------------------
def test_real_soliton_run_labels_6_noise_on_and_off():
    """Warm-start 11*kappa single DKS (Taylor path, n_tau=8192), 300 RT.

    The scan labeler must produce 6 on every snapshot with the quantum drive
    OFF (legacy behavior) and ON (the vacuum-floor-aware labeler the solver
    now builds when the channel is enabled).
    """
    cfg = str(write_noise_off_config())
    cav = load_cavity_params()
    n_tau = 8192
    seed_f = sech_soliton_seed(11.0 * cav.kappa, cav, n_tau=n_tau, pin=PIN_W)
    for qn in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = _run(
                11.0 * cav.kappa, 300, cav, e0=seed_f, seed=0, n_tau=n_tau,
                pin=PIN_W, snapshot_interval=50, config_path=cfg,
                quantum_noise_enabled=qn,
            )
        labels = np.asarray(sol["label_history"])[0]
        assert np.all(labels == 6), (qn, labels.tolist())


# ---------------------------------------------------------------------------
# 5. No collateral damage: the 8*kappa deterministic breather keeps its class
# ---------------------------------------------------------------------------
def test_breather_labels_preserved_at_8kappa():
    """Noise-ON label sequence at the 8*kappa breather == noise-OFF reference.

    The noise-OFF sequence is measured in this test and pinned as the
    reference (per-snapshot the labeler calls the breathing soliton class 6;
    the breathing itself shows in U_int with sigma/mu ~ 4%); noise ON must
    reproduce it, and the breathing amplitude must be unchanged (<10% rel).
    """
    cfg = str(write_noise_off_config())
    n_tau = 8192
    cav = attach_dispersion(load_cavity_params(), n_tau)
    seed_f = sech_soliton_seed(8.0 * cav.kappa, cav, n_tau=n_tau, pin=PIN_W)
    results = {}
    for qn in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            settle = _run(
                8.0 * cav.kappa, 1200, cav, e0=seed_f, seed=0, n_tau=n_tau,
                pin=PIN_W, snapshot_interval=1200, config_path=cfg,
                quantum_noise_enabled=qn, **PRODUCTION_NUMERICS,
            )
            sol = _run(
                8.0 * cav.kappa, 300, cav,
                e0=np.asarray(settle["e_final"])[0],
                delta_t0=settle["delta_t_final"], seed=1, n_tau=n_tau,
                pin=PIN_W, snapshot_interval=25, config_path=cfg,
                quantum_noise_enabled=qn, **PRODUCTION_NUMERICS,
            )
        u = np.asarray(sol["U_int_history"])[0]
        results[qn] = {
            "labels": np.asarray(sol["label_history"])[0],
            "relstd": float(np.std(u) / np.mean(u)),
        }
    ref = results[False]
    assert np.array_equal(results[True]["labels"], ref["labels"]), (
        ref["labels"].tolist(),
        results[True]["labels"].tolist(),
    )
    # deterministic breather: sigma/mu ~ 4%; quantum noise must not change it
    assert ref["relstd"] > 0.01, ref["relstd"]
    assert abs(results[True]["relstd"] / ref["relstd"] - 1.0) < 0.10


# ---------------------------------------------------------------------------
# 6. Statistical margins of the design
# ---------------------------------------------------------------------------
def test_smoothing_residual_matches_exponential_statistics():
    """Vacuum-wing log-power std: ~5.57 dB raw; ~5.57/sqrt(w_eff) smoothed."""
    n_tau = 4096
    rng = np.random.default_rng(11)
    spec = np.abs(np.fft.fftshift(np.fft.fft(_vacuum_field(n_tau, rng)))) ** 2
    raw_db_std = float(np.std(10 * np.log10(spec)))
    assert abs(raw_db_std / 5.57 - 1.0) < 0.10, raw_db_std

    w_eff = 9  # odd-adjusted envelope_smooth_modes = 8
    sm = sum(np.roll(spec, k) for k in range(-(w_eff // 2), w_eff // 2 + 1)) / w_eff
    sm_db_std = float(np.std(10 * np.log10(sm)))
    assert abs(sm_db_std / (5.57 / math.sqrt(w_eff)) - 1.0) < 0.30, sm_db_std
