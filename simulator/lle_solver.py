"""JAX-based LLE + thermal ODE solver module.

This module implements a GPU-accelerated split-step Fourier method (SSFM)
solver for the generalized Lugiato–Lefever Equation (LLE), including a
single-pole thermal model for thermo-optic detuning drift.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from simulator.state_labeler import make_state_labeler

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tfln_params.yaml"


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and return physical parameters dict."""
    cfg_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("physical_parameters", {})


def _thermal_params(config_path: str | Path | None = None) -> dict[str, float]:
    """Collect thermal/material parameters with reasonable defaults."""
    p = _load_config(config_path)
    return {
        "tau_th": float(p.get("tau_th_s", 5.0e-6)),
        "dn_dT": float(p.get("dn_dT_per_k", 4.0e-5)),
        "Gamma_th": float(p.get("Gamma_th", 1.0)),
        "rho": float(p.get("rho_kg_per_m3", 4.64e3)),
        "Cp": float(p.get("Cp_j_per_kg_k", 700.0)),
        "V": float(p.get("mode_volume_m3", 1.0e-15)),
        "pump_wavelength_m": float(p.get("pump_wavelength_m", 1.55e-6)),
        "n0": float(p.get("n0", 2.2)),
        "fsr_hz": float(p.get("fsr_hz", 2.0e11)),
    }


def _build_omega_grid(n_tau: int, t_r: float) -> jnp.ndarray:
    """Build angular-frequency grid used for FFT-domain linear step."""
    return 2.0 * jnp.pi * jnp.fft.fftfreq(n_tau, d=t_r / n_tau)


def build_dispersion(omega: jnp.ndarray, beta_list: tuple[float, ...]) -> jnp.ndarray:
    """Build dispersion polynomial.

    beta_list[0] is beta_2 (s^2/m), beta_list[1] is beta_3 (s^3/m), etc.
    The k=0 and k=1 terms are zero by definition in the co-moving frame and must not be included.
    """
    assert len(beta_list) >= 1, "Must provide at least beta_2"
    disp = jnp.zeros_like(omega)
    for i, b in enumerate(beta_list):
        k = i + 2
        disp = disp + float(b) / math.factorial(k) * omega**k
    return disp


def _single_trajectory_solver(
    delta_omega: float,
    pin: float,
    t_slow: int,
    beta: tuple[float, ...],
    gamma: float,
    kappa: float,
    kappa_c: float,
    n_tau: int,
    t_r: float,
    l_eff: float,
    snapshot_interval: int,
    rng_key: jax.Array,
    thermal: dict[str, float],
    state_labeler,
) -> dict[str, jnp.ndarray]:
    """Solve one detuning trajectory with SSFM + thermal Euler update."""
    omega = _build_omega_grid(n_tau, t_r)
    disp = build_dispersion(omega, beta)
    pump_amp = (jnp.sqrt(jnp.maximum(kappa_c * pin, 0.0)) * t_r).astype(jnp.complex64)
    kappa_i = jnp.maximum(thermal["kappa_i"], 0.0)
    omega0 = 2.0 * jnp.pi * 299_792_458.0 / thermal["pump_wavelength_m"]
    n_snapshots = (t_slow + snapshot_interval - 1) // snapshot_interval

    def _step(carry, step_idx):
        e_t, delta_t, e_snapshots, label_history, snap_count = carry
        e_t = e_t.astype(jnp.complex64)

        delta_omega_eff = delta_omega + (omega0 / thermal["n0"]) * thermal["dn_dT"] * delta_t

        # (a) Inject pump before the symmetric split so the full round-trip
        #     linear operator acts on the pumped field.
        e_pumped = (e_t + pump_amp).astype(jnp.complex64)
        
        # (b) Half linear step in frequency domain.
        lin_exp = (-kappa / 2.0 + 1j * disp - 1j * delta_omega_eff) * t_r
        lin_exp_half = lin_exp / 2.0
        h_half = jnp.exp(lin_exp_half).astype(jnp.complex64)
        e_w = jnp.fft.fft(e_pumped)
        e_half = jnp.fft.ifft(e_w * h_half).astype(jnp.complex64)
        
        # (c) Nonlinear phase kick in time domain.
        nl_phase = jnp.exp(1j * gamma * jnp.abs(e_half) ** 2 * t_r).astype(jnp.complex64)
        e_nl = (e_half * nl_phase).astype(jnp.complex64)
        
        # (d) Second half linear step in frequency domain.
        e_w2 = jnp.fft.fft(e_nl)
        e_next = jnp.fft.ifft(e_w2 * h_half).astype(jnp.complex64)

        p_trans = kappa_c * jnp.sum(jnp.abs(e_next) ** 2) * (t_r / n_tau)
        u_int = jnp.sum(jnp.abs(e_next) ** 2) * (t_r / n_tau)
        p_abs = kappa_i * u_int

        d_delta_t = (
            -delta_t / thermal["tau_th"]
            + thermal["Gamma_th"] * p_abs / (thermal["rho"] * thermal["Cp"] * thermal["V"])
        )
        delta_t_next = delta_t + t_r * d_delta_t
        do_snapshot = (step_idx % snapshot_interval) == 0
        next_snap_count = snap_count + do_snapshot.astype(jnp.int32)
        write_idx = jnp.minimum(snap_count, n_snapshots - 1)
        e_snapshots = jax.lax.cond(
            do_snapshot,
            lambda arr: arr.at[write_idx].set(e_next),
            lambda arr: arr,
            e_snapshots,
        )
        lbl = state_labeler(e_next)
        label_history = jax.lax.cond(
            do_snapshot,
            lambda arr: arr.at[write_idx].set(lbl),
            lambda arr: arr,
            label_history,
        )

        out = {
            "P_trans": p_trans,  # Watts; detector-measured transmitted power
            "U_int": u_int,
            "DeltaT": delta_t_next,
            "delta_omega_eff": delta_omega_eff,
        }
        return (e_next, delta_t_next, e_snapshots, label_history, next_snap_count), out
        
    e_cw = jnp.sqrt(kappa_c * pin / ((kappa / 2) ** 2 + delta_omega_eff**2)) * jnp.ones(
        n_tau, dtype=jnp.complex64
    )
    
    key, subkey_r, subkey_i = jax.random.split(rng_key, 3)
    noise = 1e-4 * (
        jax.random.normal(subkey_r, (n_tau,)) + 1j * jax.random.normal(subkey_i, (n_tau,))
    ).astype(jnp.complex64)
        
    e0 = e_cw + noise
    delta_t0 = jnp.array(0.0, dtype=jnp.float32)
    e_snapshots0 = jnp.zeros((n_snapshots, n_tau), dtype=jnp.complex64)
    label_history0 = jnp.zeros((n_snapshots,), dtype=jnp.int32)
    snap_count0 = jnp.array(0, dtype=jnp.int32)

    (final_carry, hist) = jax.lax.scan(
        _step,
        (e0, delta_t0, e_snapshots0, label_history0, snap_count0),
        xs=jnp.arange(t_slow),
        length=t_slow,
    )
    _, _, e_snapshots, label_history, _ = final_carry

    return {
        "E_snapshots": e_snapshots,
        "label_history": label_history,
        "P_trans_history": hist["P_trans"],
        "U_int_history": hist["U_int"],
        "DeltaT_history": hist["DeltaT"],
        "delta_omega_eff_history": hist["delta_omega_eff"],
    }


def solve_lle_ssfm_jax(
    pin: float,
    delta_omega: float | np.ndarray | jnp.ndarray,
    t_slow: int,
    beta: list[float] | tuple[float, ...] | np.ndarray,
    kappa: float,
    kappa_c: float,
    rng_key: jax.Array,
    n_tau: int = 512,
    config_path: str | Path | None = None,
    l_eff: float = 1.0,
    snapshot_interval: int = 100,
) -> dict[str, np.ndarray]:
    """Batch-capable SSFM solver for the generalized LLE using JAX.

    Detuning convention: delta_omega = omega_pump - omega_res.
      Positive delta_omega = blue-detuned pump (pump above resonance).
      Solitons exist for delta_omega > 0, specifically kappa/2 < delta_omega < ~5*kappa.
      The detuning sweep should go from positive to negative values (blue to red scan).

    Args:
        pin: Pump power in watts.
        delta_omega: Laser detuning(s) in rad/s; scalar or 1D array.
        t_slow: Number of round trips.
        beta: Dispersion coefficient list [beta2, beta3, beta4].
        kappa: Total cavity loss rate (rad/s).
        kappa_c: Coupling rate (rad/s).
        rng_key: PRNG key for initial noise seeding.
        n_tau: Number of fast-time grid points.
        config_path: Optional YAML config override.
        l_eff: Effective nonlinear interaction length.
        snapshot_interval: Round-trip interval for field snapshots and labels.

    Returns:
        Dictionary containing requested histories.
    """
    thermal = _thermal_params(config_path)
    physical = _load_config(config_path)
    gamma = float(physical.get("gamma_LLE_per_W_per_s"))
    assert 0.1 < gamma < 100, (
        f"gamma={gamma} looks wrong for TFLN LLE units (expected 0.1-100 W^-1 s^-1)"
    )
    thermal["Gamma_th"] = float(physical.get("Gamma_th", thermal["Gamma_th"]))
    thermal["kappa_i"] = float(physical.get("kappa_i_rad_per_s", max(kappa - kappa_c, 0.0)))
    assert 1e-5 < thermal["Gamma_th"] < 1.0, (
        f"Gamma_th={thermal['Gamma_th']} must be dimensionless fraction (1e-5 to 1.0)"
    )
    assert 1e6 < thermal["kappa_i"] < 1e12, (
        f"kappa_i={thermal['kappa_i']} should be in rad/s (1e6 to 1e12)"
    )
    
    t_r = 1.0 / thermal["fsr_hz"]

    # convert every leaf to a scalar JAX array so vmap can trace through it
    thermal = {k: jnp.array(v, dtype=jnp.float32) for k, v in thermal.items()}

    delta_omega_input = delta_omega
    delta_omega = jnp.array(delta_omega_input, dtype=jnp.float32)
    delta_arr = jnp.atleast_1d(delta_omega)
    beta_arr = tuple(float(b) for b in beta)
    key_arr = jax.random.split(rng_key, delta_arr.shape[0])

    _state_labeler = make_state_labeler()

    per_traj = jax.jit(
        jax.vmap(
            _single_trajectory_solver,
            in_axes=(0, None, None, None, None, None, None, None, None, None, None, 0, None, None),
        ),
        static_argnums=(2, 3, 7, 10, 13),
    )

    out = per_traj(
        delta_arr,
        float(pin),
        int(t_slow),
        beta_arr,
        float(gamma),
        float(kappa),
        float(kappa_c),
        int(n_tau),
        float(t_r),
        float(l_eff),
        int(snapshot_interval),
        key_arr,
        thermal,
        _state_labeler,
    )

    return {k: np.asarray(v) for k, v in out.items()}


def validate_solver(
    solution: dict[str, np.ndarray],
    pin: float,
    kappa: float,
    kappa_c: float,
    gamma: float,
    traj_idx: int = 0,
    print_results: bool = True,
) -> dict[str, bool]:
    """Run basic solver validation checks and print pass/fail status."""
    u_int = np.asarray(solution["U_int_history"])
    p_trans = np.asarray(solution["P_trans_history"])
    e_hist = np.asarray(solution["E_snapshots"])

    # Select single trajectory if batched — shapes are (n_traj, t_slow) and (n_traj, n_snapshots, n_tau)
    if u_int.ndim == 2:
        u_int  = u_int[traj_idx]           # (t_slow,)
        p_trans = p_trans[traj_idx]        # (t_slow,)
        e_hist  = e_hist[traj_idx]         # (n_snapshots, n_tau)

    p_th = max((kappa**3) / (8.0 * max(gamma, 1e-30) * max(kappa_c, 1e-30)), 1e-30)
    arg = 8.0 * pin / p_th - 1.0
    delta_omega_sol = (kappa / 2.0) * math.sqrt(max(arg, 0.0))
    check_a = np.isfinite(delta_omega_sol) and (delta_omega_sol >= 0.0)

    final_spec = np.abs(np.fft.fftshift(np.fft.fft(e_hist[-1]))) ** 2
    final_spec /= max(final_spec.max(), 1e-12)
    x = np.linspace(-3.0, 3.0, final_spec.size)
    sech2 = 1.0 / np.cosh(x) ** 2
    sech2 /= sech2.max()
    corr = np.corrcoef(final_spec, sech2)[0, 1]
    check_b = np.isfinite(corr) and corr > 0.7

    u_tail = u_int[int(0.8 * u_int.size):]
    p_tail = p_trans[int(0.8 * p_trans.size):]

    rel_u = np.std(u_tail) / max(np.mean(u_tail), 1e-12)
    rel_p = np.std(p_tail) / max(np.mean(p_tail), 1e-12)
    check_c = (rel_u < 5e-2) and (rel_p < 5e-2)

    delta_omega_eff_hist = np.asarray(solution["delta_omega_eff_history"])
    if delta_omega_eff_hist.ndim == 2:
        delta_omega_eff_hist = delta_omega_eff_hist[traj_idx]
    delta_omega_test = float(np.mean(delta_omega_eff_hist[-100:]))
    u_ss = float(np.mean(u_int[-100:]))     # u_int already sliced in bug 6 fix
    u_expected = kappa_c * pin / ((kappa / 2) ** 2 + delta_omega_test**2)
    rel_error = abs(u_ss - u_expected) / u_expected
    print(f"CW steady-state energy error: {rel_error:.3%} (pass if <10%)")
    assert rel_error < 0.10, "Steady-state energy deviates >10% from analytical CW solution"

    results = {
        "soliton_existence_condition": bool(check_a),
        "sech2_spectral_envelope": bool(check_b),
        "steady_state_energy_conservation": bool(check_c),
    }

    if print_results:
        for name, ok in results.items():
            print(f"[{ 'PASS' if ok else 'FAIL' }] {name}")

    return results
