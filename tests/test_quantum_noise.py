"""Tests for the physically normalized quantum vacuum noise channel.

Reference: Herr, Tikan & Kippenberg, "Frequency combs and coherent dissipative
structures in nonlinear optical microresonators", arXiv:2604.05897, Sec. V.B.2
(Eqs. 17 and 126). The solver implements the truncated-Wigner (symmetric-
ordering) c-number limit: additive complex Gaussian noise with
<xi_mu(t) xi_mu'*(t')> = (1/2) delta(t-t') delta_mumu', whose undriven steady
state is the symmetric-ordered vacuum occupation of 1/2 photon per mode.

Conventions verified here (all anchored to the repo's operational identities,
NOT to unit comments): |E_j|^2 carries intracavity energy in Joules, the true
intracavity energy is U_true = (1/n_tau) * sum_j |E_j|^2, and with the
NumPy/JAX FFT convention the photon number of mode mu is
n_mu = |Etilde_mu|^2 / (n_tau^2 * hbar*omega0).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from simulator.lle_solver import (
    _PER_TRAJ,
    _STATE_LABELER,
    _load_config,
    _qnoise_increment,
    _single_trajectory_solver,
    _thermal_params,
    d2_to_beta2_lle,
    hbar_omega0_from_config,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

_PHYS = _load_config()
_KAPPA_I, _KAPPA_C, _KAPPA = resolve_cavity_rates()
_T_R = 1.0 / float(_PHYS["fsr_hz"])
_BETA2 = float(d2_to_beta2_lle(_PHYS["d2_rad_per_s2"], _PHYS["fsr_hz"]))
_HBW = hbar_omega0_from_config(_PHYS)


def _solve_quiet(**kw):
    """solve_lle_ssfm_jax with the expected-noise warnings suppressed.

    Vacuum-equilibrium runs use pin=0 (below the MI threshold by construction)
    and sit above the pin-scaled labeler OFF floor; both warnings are expected
    and irrelevant to the physics under test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_lle_ssfm_jax(**kw)


def _write_sidecar(tmp_path: Path, name: str, updates: dict) -> Path:
    """Write a config sidecar = base config + flat physical_parameters updates.

    A key mapped to None in ``updates`` is REMOVED (for the missing-key
    default tests).
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    pp = cfg.setdefault("physical_parameters", {})
    for k, v in updates.items():
        if v is None:
            pp.pop(k, None)
        else:
            pp[k] = v
    out = tmp_path / name
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


# ---------------------------------------------------------------------------
# 1. Unit — variance of the Langevin injection
# ---------------------------------------------------------------------------
def test_injection_variance_matches_prescription():
    """Per-quadrature injection variance = hbar*omega0*kappa*n_tau*dt_fine/4.

    Draws >= 1e5 increments through the exact in-solver code path
    (_qnoise_increment on a fold_in key ladder) and checks the per-quadrature
    sample variance to 2%.
    """
    n_tau = 8192
    dt_fine = _T_R  # fine_cadence_M = 1
    var_target = _HBW * _KAPPA * n_tau * dt_fine / 4.0
    scale = math.sqrt(var_target)
    key = jax.random.PRNGKey(7)
    n_draws = 13  # 13 * 8192 = 106496 >= 1e5 samples
    z = np.concatenate(
        [np.asarray(_qnoise_increment(key, i, n_tau, scale)) for i in range(n_draws)]
    )
    assert z.size >= 100_000
    for quad in (z.real, z.imag):
        var = float(np.var(quad))
        assert abs(var / var_target - 1.0) < 0.02, (var, var_target)
        # zero-mean to ~5 standard errors
        assert abs(float(np.mean(quad))) < 5.0 * scale / math.sqrt(z.size)
    # total per-sample variance = 2x per-quadrature
    assert abs(float(np.mean(np.abs(z) ** 2)) / (2.0 * var_target) - 1.0) < 0.02


# ---------------------------------------------------------------------------
# 2. Unit — vacuum seed occupancy (half a photon per mode at t=0)
# ---------------------------------------------------------------------------
def test_vacuum_seed_occupancy_half_photon():
    """Enabled cold start, tiny pin: mean modal photon number ~ 0.5.

    The initial field is CW + vacuum draw; after ONE round trip the modal
    occupation has only decayed by e^{-kappa*t_r} ~ 0.6% (and the Langevin
    drive partly refills it), far inside the 5% tolerance. The uniform CW
    component (and the discrete-map CW mismatch) lives entirely in mu=0, which
    is removed exactly by subtracting the field's fast-time mean.
    """
    n_tau = 8192
    sol = _solve_quiet(
        pin=1e-6,
        delta_omega=0.0,
        t_slow=1,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(11),
        n_tau=n_tau,
        snapshot_interval=1,
        quantum_noise_enabled=True,
    )
    e1 = np.asarray(sol["E_snapshots"])[0, 0]
    dev = e1 - np.mean(e1)
    n_mu = np.abs(np.fft.fft(dev)) ** 2 / (n_tau**2 * _HBW)
    mean_n = float(np.mean(np.delete(n_mu, 0)))  # mu=0 bin is exactly 0 by construction
    assert abs(mean_n / 0.5 - 1.0) < 0.05, mean_n


# ---------------------------------------------------------------------------
# 3+4. Integration — vacuum equilibrium and cavity linewidth (shared solve)
# ---------------------------------------------------------------------------
N_TAU_EQ = 1024
T_SLOW_EQ = 6000
SNAP_INT_EQ = 2  # snapshot every 2 RT; windows/lags stated in RT below


@pytest.fixture(scope="module")
def vacuum_equilibrium(tmp_path_factory):
    """pin=0, T_k=0 sidecar, masks off, M=1: 4 trajectories of pure vacuum.

    Photon lifetime 1/kappa ~ 162 RT; t_slow = 6000 RT (~37 lifetimes), the
    last 3000 RT (~18 lifetimes past start) are the measurement window.
    """
    tmp = tmp_path_factory.mktemp("qn_sidecar")
    sidecar = _write_sidecar(tmp, "sin_params_tk0.yaml", {"T_k": 0.0})
    sol = _solve_quiet(
        pin=0.0,
        delta_omega=np.zeros((4, T_SLOW_EQ)),
        t_slow=T_SLOW_EQ,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(21),
        n_tau=N_TAU_EQ,
        config_path=str(sidecar),
        snapshot_interval=SNAP_INT_EQ,
        n_substeps=1,
        dealias_two_thirds=False,
        edge_absorber=False,
        dispersion_validity_mask=False,
        fine_cadence_M=1,
        quantum_noise_enabled=True,
    )
    snaps = np.asarray(sol["E_snapshots"])  # (4, 3000, N_TAU_EQ)
    assert snaps.shape == (4, T_SLOW_EQ // SNAP_INT_EQ, N_TAU_EQ)
    return sol, snaps


def test_vacuum_equilibrium_occupation(vacuum_equilibrium):
    """<n_mu> = 0.5 +- 10% (grand mean tighter: [0.45, 0.55]); flat in mu."""
    _, snaps = vacuum_equilibrium
    window = snaps[:, snaps.shape[1] // 2 :, :]  # last 3000 RT
    spec = np.abs(np.fft.fft(window, axis=-1)) ** 2
    n_mu = spec.mean(axis=(0, 1)) / (N_TAU_EQ**2 * _HBW)  # (n_tau,), per-mode mean

    grand_mean = float(np.mean(n_mu))  # all modes (masks off: all thermalize)
    assert 0.45 < grand_mean < 0.55, grand_mean

    # 8 bins in mu across |mu| <= n_tau/3, each within 15% of 0.5
    mu = np.fft.fftfreq(N_TAU_EQ) * N_TAU_EQ
    keep = np.abs(mu) <= N_TAU_EQ / 3.0
    mu_k, n_k = mu[keep], n_mu[keep]
    edges = np.linspace(mu_k.min(), mu_k.max(), 9)
    for i in range(8):
        sel = (mu_k >= edges[i]) & (
            mu_k <= edges[i + 1] if i == 7 else mu_k < edges[i + 1]
        )
        assert sel.any()
        bin_mean = float(np.mean(n_k[sel]))
        assert abs(bin_mean / 0.5 - 1.0) < 0.15, (i, bin_mean)


def test_vacuum_autocorrelation_linewidth(vacuum_equilibrium):
    """|<a_0(t+tau) a_0*(t)>| decays at kappa/2 within 10% (0.2-2 lifetimes)."""
    _, snaps = vacuum_equilibrium
    a0 = snaps.sum(axis=-1)  # Etilde_0(t) = sum_j E_j, shape (4, n_snap)
    a = a0[:, a0.shape[1] // 2 :]  # last 3000 RT
    lifetime_rt = 1.0 / (_KAPPA * _T_R)  # ~162 RT
    lags_snap = np.arange(
        int(round(0.2 * lifetime_rt / SNAP_INT_EQ)),
        int(round(2.0 * lifetime_rt / SNAP_INT_EQ)) + 1,
    )
    corr = np.empty(lags_snap.size)
    for i, lag in enumerate(lags_snap):
        c = np.mean(a[:, lag:] * np.conj(a[:, : a.shape[1] - lag]))
        corr[i] = np.abs(c)
    tau_s = lags_snap * SNAP_INT_EQ * _T_R
    slope = np.polyfit(tau_s, np.log(corr), 1)[0]
    rate = -slope
    assert abs(rate / (_KAPPA / 2.0) - 1.0) < 0.10, (rate, _KAPPA / 2.0)


# ---------------------------------------------------------------------------
# 5. Physics — MI sideband selection from vacuum (paper Eq. 62)
# ---------------------------------------------------------------------------
def test_mi_sideband_selection_from_vacuum(tmp_path):
    """The fastest-growing MI sideband matches paper Eq. 62 within +-15%.

    mu_th = sqrt[(kappa/D2)(1 + sqrt(P_in/P_th - 1))], P_th = kappa^3/(8 gamma
    kappa_c) (~3.5 mW here => mu_th ~ 1.88e2). Pump 0.214 W, Taylor dispersion
    D2 = 2pi*6 kHz, fixed blue (MI-side) detuning -2.5*kappa, quantum noise as
    the ONLY seed: the enabled cold start is CW + vacuum draw, which bypasses
    the legacy 1e-3 seed path by construction. The T_k=0 / tiny-Gamma_th
    sidecar removes stochastic and thermo-optic detuning drift so the operating
    detuning stays put. At -2.5*kappa the steady CW is gamma*U = 0.719*kappa
    (single real root), whose fastest-growing index sqrt(2(2*gamma*U -
    delta)/D2) = 178 sits within 5% of Eq. 62's threshold-scan value 187.8 —
    both well inside the +-15% band.
    """
    gamma = float(_PHYS["gamma_LLE_per_J_per_s"])
    d2 = float(_PHYS["d2_rad_per_s2"])
    p_in = 0.214
    p_th = _KAPPA**3 / (8.0 * gamma * _KAPPA_C)
    mu_th = math.sqrt((_KAPPA / d2) * (1.0 + math.sqrt(p_in / p_th - 1.0)))

    sidecar = _write_sidecar(
        tmp_path, "mi_quiet.yaml", {"T_k": 0.0, "Gamma_th": 1e-4}
    )
    n_tau, t_slow, snap = 1024, 6000, 25
    sol = _solve_quiet(
        pin=p_in,
        delta_omega=-2.5 * _KAPPA,
        t_slow=t_slow,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(29),
        n_tau=n_tau,
        config_path=str(sidecar),
        snapshot_interval=snap,
        quantum_noise_enabled=True,
    )
    snaps = np.asarray(sol["E_snapshots"])[0]              # (240, n_tau)
    n_mode = np.abs(np.fft.fft(snaps, axis=-1)) ** 2 / (n_tau**2 * _HBW)
    mu = np.fft.fftfreq(n_tau) * n_tau
    band = (np.abs(mu) >= 10) & (np.abs(mu) <= n_tau / 3.0)  # exclude pump region

    # First snapshot where the strongest sideband passes 1e5 photons: far above
    # the 0.5-photon vacuum, still ~1e-4 of the ~7e9-photon pump => the
    # dynamics are still in the LINEAR MI-growth stage at detection.
    peak_per_snap = n_mode[:, band].max(axis=-1)
    idx = np.argmax(peak_per_snap > 1e5)
    assert peak_per_snap[idx] > 1e5, "MI sidebands never reached detection level"
    mu_star = abs(float(mu[band][np.argmax(n_mode[idx][band])]))
    assert abs(mu_star / mu_th - 1.0) < 0.15, (mu_star, mu_th)


# ---------------------------------------------------------------------------
# 6. Determinism under fixed seed
# ---------------------------------------------------------------------------
def test_determinism_same_key_bitidentical_different_key_not():
    kw = dict(
        pin=0.214,
        delta_omega=4.0 * _KAPPA,
        t_slow=30,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        n_tau=256,
        snapshot_interval=5,
        quantum_noise_enabled=True,
    )
    a = _solve_quiet(rng_key=jax.random.PRNGKey(3), **kw)
    b = _solve_quiet(rng_key=jax.random.PRNGKey(3), **kw)
    c = _solve_quiet(rng_key=jax.random.PRNGKey(4), **kw)
    assert np.array_equal(a["E_snapshots"], b["E_snapshots"])
    assert not np.array_equal(a["E_snapshots"], c["E_snapshots"])


def test_vmapped_trajectories_use_independent_noise():
    """Two identical-detuning trajectories in one batch must decorrelate."""
    sol = _solve_quiet(
        pin=0.0,
        delta_omega=np.zeros((2, 200)),
        t_slow=200,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(5),
        n_tau=128,
        snapshot_interval=50,
        quantum_noise_enabled=True,
    )
    snaps = np.asarray(sol["E_snapshots"])
    assert not np.array_equal(snaps[0], snaps[1])


# ---------------------------------------------------------------------------
# 7. Backward compatibility — legacy cold-start seed statistics with flag off
# ---------------------------------------------------------------------------
def test_flag_off_legacy_seed_statistics():
    """Flag off: initial deviation from CW has per-quadrature std 1e-3*|e_cw|.

    After ONE round trip the seed noise has decayed by only e^{-kappa*t_r/2}
    (~0.3%); the uniform CW (and its discrete-map mismatch) is removed by
    subtracting the fast-time mean. Tolerance 5%.
    """
    n_tau = 4096
    dw = 4.0 * _KAPPA
    pin = 0.214
    sol = _solve_quiet(
        pin=pin,
        delta_omega=dw,
        t_slow=1,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(13),
        n_tau=n_tau,
        snapshot_interval=1,
    )
    e1 = np.asarray(sol["E_snapshots"])[0, 0]
    dev = e1 - np.mean(e1)
    std_q = math.sqrt(float(np.mean(np.abs(dev) ** 2)) / 2.0)
    e_cw_abs = math.sqrt(_KAPPA_C * pin) / math.hypot(_KAPPA / 2.0, dw)
    target = 1e-3 * e_cw_abs
    assert abs(std_q / target - 1.0) < 0.05, (std_q, target)


# ---------------------------------------------------------------------------
# 8. Config handling
# ---------------------------------------------------------------------------
def _short_solve(config_path=None, **kw):
    return _solve_quiet(
        pin=0.05,
        delta_omega=3.0 * _KAPPA,
        t_slow=20,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(17),
        n_tau=128,
        snapshot_interval=5,
        config_path=None if config_path is None else str(config_path),
        **kw,
    )


def test_config_missing_keys_default_off(tmp_path):
    """A config without the quantum keys behaves exactly like keys = 0."""
    sidecar_missing = _write_sidecar(
        tmp_path,
        "no_qn_keys.yaml",
        {
            "quantum_noise_enabled": None,
            "quantum_noise_seed_vacuum_init": None,
            "hbar_omega0_j": None,
        },
    )
    a = _short_solve(config_path=sidecar_missing)
    b = _short_solve()  # committed config: keys present, = 0
    c = _short_solve(quantum_noise_enabled=False)
    assert np.array_equal(a["E_snapshots"], b["E_snapshots"])
    assert np.array_equal(a["E_snapshots"], c["E_snapshots"])


def test_config_enabled_via_sidecar_key(tmp_path):
    """quantum_noise_enabled = 1 in the config turns the channel on."""
    sidecar_on = _write_sidecar(tmp_path, "qn_on.yaml", {"quantum_noise_enabled": 1})
    a = _short_solve(config_path=sidecar_on)
    b = _short_solve()
    assert not np.array_equal(a["E_snapshots"], b["E_snapshots"])


def test_config_rejects_non_boolean_flag():
    with pytest.raises(AssertionError, match="quantum_noise_enabled"):
        _short_solve(quantum_noise_enabled=0.5)


def test_labeler_floor_warning_fires_when_constructed_to():
    """U_vac >= 0.5x the OFF floor (tiny pin) must warn, not fail."""
    # pytest.warns tolerates the co-fired below-MI-threshold pin warning; it
    # only requires at least one match.
    with pytest.warns(UserWarning, match="OFF floor"):
        solve_lle_ssfm_jax(
            pin=1e-6,
            delta_omega=0.0,
            t_slow=1,
            beta=[_BETA2],
            kappa=_KAPPA,
            kappa_c=_KAPPA_C,
            rng_key=jax.random.PRNGKey(19),
            n_tau=1024,
            snapshot_interval=1,
            quantum_noise_enabled=True,
        )


def test_labeler_floor_warning_absent_at_operating_pin():
    """At pin = 0.214 W the OFF floor is ~1e4x above U_vac: no warning."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _short_solve(quantum_noise_enabled=True)
    assert not [w for w in rec if "OFF floor" in str(w.message)]


# ---------------------------------------------------------------------------
# _PER_TRAJ smoke + structural guarantees for the disabled path
# ---------------------------------------------------------------------------
def _per_traj_args(n_traj=2, n_tau=64, t_slow=3, qnoise_enabled=False):
    thermal = _thermal_params()
    thermal["Gamma_th"] = float(_PHYS.get("Gamma_th", thermal["Gamma_th"]))
    thermal["kappa_i"] = float(_PHYS.get("kappa_i_rad_per_s", _KAPPA - _KAPPA_C))
    thermal = {k: jnp.array(v, dtype=jnp.float64) for k, v in thermal.items()}
    qnoise_scale = (
        float(math.sqrt(_HBW * _KAPPA * n_tau * _T_R / 4.0)) if qnoise_enabled else 0.0
    )
    return (
        jnp.full((n_traj, t_slow), 2.0 * _KAPPA, dtype=jnp.float64),  # delta_omega
        0.05,                     # pin
        int(t_slow),              # t_slow (static)
        (_BETA2,),                # beta (static)
        float(_PHYS["gamma_LLE_per_J_per_s"]),  # gamma
        float(_KAPPA),            # kappa
        float(_KAPPA_C),          # kappa_c
        int(n_tau),               # n_tau (static)
        float(_T_R),              # t_r
        1.0,                      # l_eff
        1,                        # snapshot_interval (static)
        jax.random.split(jax.random.PRNGKey(0), n_traj),   # rng_key
        thermal,                  # thermal
        _STATE_LABELER,           # state_labeler (static)
        jnp.zeros((n_traj, t_slow), dtype=jnp.float64),    # noise_sequence
        jnp.zeros((n_traj, n_tau), dtype=jnp.complex128),  # e0_override
        jnp.zeros((n_traj,), dtype=jnp.float64),           # delta_t0_override
        None,                     # d_int_grid
        1,                        # n_substeps (static)
        False,                    # dealias_two_thirds (static)
        False,                    # edge_absorber (static)
        0.12,                     # edge_absorber_frac
        False,                    # dispersion_validity_mask (static)
        float(jnp.pi),            # validity_phase_threshold
        1,                        # fine_cadence_M (static)
        jax.random.split(jax.random.PRNGKey(1), n_traj),   # qnoise_key
        qnoise_scale,             # qnoise_scale
        bool(qnoise_enabled),     # qnoise_enabled (static)
        False,                    # qnoise_roundtrip (static; fine cadence)
        None,                     # pump_scale_sequence (RIN disabled)
        None,                     # fsr_noise_sequence (FSR noise disabled)
        (),                       # mode_probe_indices (static; no probes)
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_per_traj_traces_with_new_signature(enabled):
    out = _PER_TRAJ(*_per_traj_args(qnoise_enabled=enabled))
    assert np.asarray(out["E_snapshots"]).shape == (2, 3, 64)
    assert np.isfinite(np.asarray(out["U_int_history"])).all()


def _collect_primitives(jaxpr, acc):
    for eqn in jaxpr.eqns:
        acc.append(eqn.primitive.name)
        for v in eqn.params.values():
            vals = v if isinstance(v, (list, tuple)) else [v]
            for item in vals:
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    _collect_primitives(inner, acc)
    return acc


def _scan_body_primitives(qnoise_enabled: bool) -> list[str]:
    """Primitive names inside the solver's lax.scan body jaxpr."""
    args = _per_traj_args(n_traj=1, qnoise_enabled=qnoise_enabled)

    def _one_traj(delta_omega, rng_key, noise_seq, e0, dt0, qnoise_key):
        return _single_trajectory_solver(
            delta_omega[0], args[1], args[2], args[3], args[4], args[5],
            args[6], args[7], args[8], args[9], args[10], rng_key,
            args[12], args[13], noise_seq[0], e0[0], dt0[0], args[17],
            args[18], args[19], args[20], args[21], args[22], args[23],
            args[24], qnoise_key, args[26], qnoise_enabled, False,
            None, None, (),
        )

    jpr = jax.make_jaxpr(_one_traj)(
        args[0], args[11][0], args[14], args[15], args[16], args[25][0]
    )
    for eqn in jpr.jaxpr.eqns:
        if eqn.primitive.name == "scan":
            return _collect_primitives(eqn.params["jaxpr"].jaxpr, [])
    raise AssertionError("no scan primitive found in the solver jaxpr")


def test_spectral_floor_wing_band_inside_dealias_window():
    """The report's floor-measurement band must lie inside |mu| <= n_tau/3.

    With dealias_two_thirds ON (the production stack used for the -81.3 dB
    wing measurement) the modes beyond n_tau/3 are zeroed every kick and are
    deliberately under-occupied; measuring the vacuum floor there would read
    low. Pins the band for the committed measurement (n_tau = 8192:
    |mu| in [1998, 2596], boundary 2730).
    """
    from analysis.quantum_noise_report import wing_band

    for n_tau in (1024, 4096, 8192, 16384):
        lo, hi = wing_band(n_tau)
        assert 0 < lo < hi <= n_tau / 3.0, (n_tau, lo, hi)


# ---------------------------------------------------------------------------
# Injection cadence (quantum_noise_injection_cadence: 0 = fine, 1 = roundtrip)
# ---------------------------------------------------------------------------
def test_cadence_bit_identical_at_M1():
    """With fine_cadence_M = 1 the two cadences must be bit-identical.

    Same fold_in index (step*1 + 0) and the same scale (sqrt(M) = 1), so the
    traces coincide exactly.
    """
    kw = dict(
        pin=0.05,
        delta_omega=3.0 * _KAPPA,
        t_slow=40,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(9),
        n_tau=256,
        snapshot_interval=10,
        fine_cadence_M=1,
        quantum_noise_enabled=True,
    )
    fine = _solve_quiet(quantum_noise_injection_cadence=0, **kw)
    rt = _solve_quiet(quantum_noise_injection_cadence=1, **kw)
    for key in fine:
        assert np.array_equal(fine[key], rt[key]), key


def test_cadence_rejects_invalid_value():
    with pytest.raises(AssertionError, match="quantum_noise_injection_cadence"):
        _short_solve(quantum_noise_enabled=True, quantum_noise_injection_cadence=2)


def test_roundtrip_cadence_vacuum_equilibrium_M4(tmp_path):
    """M = 4 with roundtrip cadence: <n_mu> = 0.5 +- 10% and kappa/2 decay.

    One injection per round trip with variance kappa*t_r/2 per mode against a
    per-round-trip decay e^{-kappa*t_r}: the same discrete map as the M = 1
    fine cadence, so the steady occupation (0.5015) and the linewidth are
    unchanged; this pins that the sqrt(M) rescaling and the m = 0 gating are
    right (an un-rescaled roundtrip injection at M = 4 would give 0.125).
    """
    sidecar = _write_sidecar(tmp_path, "tk0_cadence.yaml", {"T_k": 0.0})
    n_tau, t_slow, snap = 512, 4000, 2
    sol = _solve_quiet(
        pin=0.0,
        delta_omega=np.zeros((4, t_slow)),
        t_slow=t_slow,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(23),
        n_tau=n_tau,
        config_path=str(sidecar),
        snapshot_interval=snap,
        fine_cadence_M=4,
        quantum_noise_enabled=True,
        quantum_noise_injection_cadence=1,
    )
    snaps = np.asarray(sol["E_snapshots"])
    win = snaps[:, snaps.shape[1] // 2 :, :]
    spec = np.abs(np.fft.fft(win, axis=-1)) ** 2
    n_mu = spec.mean(axis=(0, 1)) / (n_tau**2 * _HBW)
    grand = float(np.mean(n_mu))
    assert abs(grand / 0.5 - 1.0) < 0.10, grand

    # Phase-corrected all-modes decay estimator (every mode decays at kappa/2
    # under the KNOWN linear phase exp(-i*D_int(mu)*tau); rotating it out lets
    # C_mu(tau) average coherently over modes without the |.|-estimator's
    # noise-floor bias, which dominates a single-mode fit at this reduced
    # 512-mode / 2000-RT statistics budget).
    from simulator.lle_solver import _build_omega_grid, build_dispersion

    modes = np.fft.fft(snaps, axis=-1)[:, snaps.shape[1] // 2 :, :]
    mu_full = np.fft.fftfreq(n_tau) * n_tau
    keep = np.abs(mu_full) <= n_tau / 3.0
    wk = modes[:, :, keep]
    disp = np.asarray(build_dispersion(_build_omega_grid(n_tau, _T_R), (_BETA2,)))[keep]
    lifetime_rt = 1.0 / (_KAPPA * _T_R)
    lags = np.arange(
        int(round(0.2 * lifetime_rt / snap)),
        int(round(2.0 * lifetime_rt / snap)) + 1,
        4,
    )
    dphi = disp * snap * _T_R
    corr = np.array(
        [
            float(
                np.mean(
                    np.real(
                        np.mean(
                            wk[:, l:, :] * np.conj(wk[:, : wk.shape[1] - l, :]),
                            axis=(0, 1),
                        )
                        * np.exp(1j * dphi * l)
                    )
                )
            )
            for l in lags
        ]
    )
    rate = -np.polyfit(lags * snap * _T_R, np.log(corr), 1)[0]
    assert abs(rate / (_KAPPA / 2.0) - 1.0) < 0.10, rate / (_KAPPA / 2.0)


def test_roundtrip_cadence_mi_sideband(tmp_path):
    """MI sideband selection also holds under M = 4 + roundtrip cadence."""
    gamma = float(_PHYS["gamma_LLE_per_J_per_s"])
    d2 = float(_PHYS["d2_rad_per_s2"])
    p_in = 0.214
    p_th = _KAPPA**3 / (8.0 * gamma * _KAPPA_C)
    mu_th = math.sqrt((_KAPPA / d2) * (1.0 + math.sqrt(p_in / p_th - 1.0)))
    sidecar = _write_sidecar(
        tmp_path, "mi_quiet_cadence.yaml", {"T_k": 0.0, "Gamma_th": 1e-4}
    )
    n_tau, t_slow, snap = 1024, 6000, 25
    sol = _solve_quiet(
        pin=p_in,
        delta_omega=-2.5 * _KAPPA,
        t_slow=t_slow,
        beta=[_BETA2],
        kappa=_KAPPA,
        kappa_c=_KAPPA_C,
        rng_key=jax.random.PRNGKey(31),
        n_tau=n_tau,
        config_path=str(sidecar),
        snapshot_interval=snap,
        fine_cadence_M=4,
        quantum_noise_enabled=True,
        quantum_noise_injection_cadence=1,
    )
    snaps = np.asarray(sol["E_snapshots"])[0]
    n_mode = np.abs(np.fft.fft(snaps, axis=-1)) ** 2 / (n_tau**2 * _HBW)
    mu = np.fft.fftfreq(n_tau) * n_tau
    band = (np.abs(mu) >= 10) & (np.abs(mu) <= n_tau / 3.0)
    peak = n_mode[:, band].max(axis=-1)
    idx = np.argmax(peak > 1e5)
    assert peak[idx] > 1e5, "MI sidebands never reached detection level"
    mu_star = abs(float(mu[band][np.argmax(n_mode[idx][band])]))
    assert abs(mu_star / mu_th - 1.0) < 0.15, (mu_star, mu_th)


def test_disabled_path_adds_no_rng_to_scan_body():
    """Structural regression: flag off => ZERO RNG primitives in the scan body.

    The legacy cold-start seed RNG lives OUTSIDE the scan, so any RNG
    primitive inside the scan body would mean the disabled branch traced the
    Langevin draw. Combined with the full-suite pass and the (out-of-repo)
    bit-identity check against the pre-change solver, this pins the disabled
    path to the legacy computation.
    """
    prims_off = _scan_body_primitives(False)
    rng_off = [p for p in prims_off if ("threefry" in p or "random" in p or "prng" in p)]
    assert rng_off == [], rng_off

    prims_on = _scan_body_primitives(True)
    rng_on = [p for p in prims_on if ("threefry" in p or "random" in p or "prng" in p)]
    assert rng_on, "enabled path must draw RNG inside the scan body"
    # and the enabled body strictly grows: the noise ops exist only when on
    assert len(prims_on) > len(prims_off)
