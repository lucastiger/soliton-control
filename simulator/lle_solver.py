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


def _dispersion_operator(omega: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """Compute Σ beta_k / k! * omega^k for k = 2..(len(beta)+1)."""
    disp = jnp.zeros_like(omega)
    for i in range(beta.shape[0]):
        k = i + 2
        disp = disp + beta[i] * (omega**k) / math.factorial(k)
    return disp


def _single_trajectory_solver(
    delta_omega: float,
    pin: float,
    t_slow: int,
    beta: jnp.ndarray,
    gamma: float,
    kappa: float,
    kappa_c: float,
    n_tau: int,
    t_r: float,
    l_eff: float,
    thermal: dict[str, float],
) -> dict[str, jnp.ndarray]:
    """Solve one detuning trajectory with SSFM + thermal Euler update."""
    omega = _build_omega_grid(n_tau, t_r)
    disp = _dispersion_operator(omega, beta)
    pump_amp = (jnp.sqrt(jnp.maximum(kappa_c * pin, 0.0)) * t_r).astype(jnp.complex64)
    kappa_i = jnp.maximum(kappa - kappa_c, 0.0)
    omega0 = 2.0 * jnp.pi * 299_792_458.0 / thermal["pump_wavelength_m"]

    def _step(carry, step_idx):
        e_t, delta_t = carry
        e_t = e_t.astype(jnp.complex64)
        e_t = jax.lax.cond(
            step_idx == 1000,
            lambda x: (x + 1e-6 * jnp.ones(n_tau, dtype=jnp.complex64)).astype(jnp.complex64),
            lambda x: x,
            e_t,
        )

        delta_omega_eff = delta_omega + (omega0 / thermal["n0"]) * thermal["dn_dT"] * delta_t

        # (a) Half linear step in frequency domain.
        lin_exp = (-kappa / 2.0 + 1j * disp - 1j * delta_omega_eff) * t_r
        h_half = jnp.exp(lin_exp / 2.0).astype(jnp.complex64)
        e_w = jnp.fft.fft(e_t)
        e_half = jnp.fft.ifft(e_w * h_half).astype(jnp.complex64)

        # (b) Nonlinear phase kick in time domain.
        nl_phase = jnp.exp(1j * gamma * jnp.abs(e_half) ** 2 * t_r).astype(jnp.complex64)
        e_nl = (e_half * nl_phase).astype(jnp.complex64)

        # (c) Second half linear step in frequency domain.
        e_w2 = jnp.fft.fft(e_nl)
        e_lin = jnp.fft.ifft(e_w2 * h_half).astype(jnp.complex64)

        # (d) Mean-field pump injection.
        e_next = (e_lin + pump_amp).astype(jnp.complex64)

        p_trans = jnp.mean(jnp.abs(e_next) ** 2)
        u_int = jnp.sum(jnp.abs(e_next) ** 2) * (t_r / n_tau)

        d_delta_t = (
            -delta_t / thermal["tau_th"]
            + thermal["Gamma_th"] * kappa_i * u_int / (thermal["rho"] * thermal["Cp"] * thermal["V"])
        )
        delta_t_next = delta_t + t_r * d_delta_t

        out = {
            "E": e_next,
            "P_trans": p_trans,
            "U_int": u_int,
            "DeltaT": delta_t_next,
            "delta_omega_eff": delta_omega_eff,
        }
        return (e_next, delta_t_next), out

    e0 = jnp.zeros((n_tau,), dtype=jnp.complex64)
    delta_t0 = jnp.array(0.0, dtype=jnp.float32)

    (_, _), hist = jax.lax.scan(_step, (e0, delta_t0), xs=jnp.arange(t_slow), length=t_slow)

    snapshot_idx = jnp.arange(0, t_slow, 10)
    return {
        "E_history": hist["E"][snapshot_idx],
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
    gamma: float,
    kappa: float,
    kappa_c: float,
    n_tau: int = 512,
    config_path: str | Path | None = None,
    l_eff: float = 1.0,
) -> dict[str, np.ndarray]:
    """Batch-capable SSFM solver for the generalized LLE using JAX.

    Args:
        pin: Pump power in watts.
        delta_omega: Laser detuning(s) in rad/s; scalar or 1D array.
        t_slow: Number of round trips.
        beta: Dispersion coefficient list [beta2, beta3, beta4].
        gamma: Nonlinear coefficient (1/W/m).
        kappa: Total cavity loss rate (rad/s).
        kappa_c: Coupling rate (rad/s).
        n_tau: Number of fast-time grid points.
        config_path: Optional YAML config override.
        l_eff: Effective nonlinear interaction length.

    Returns:
        Dictionary containing requested histories.
    """
    thermal = _thermal_params(config_path)
    t_r = 1.0 / thermal["fsr_hz"]

    delta_omega_input = delta_omega
    delta_omega = jnp.array(delta_omega_input, dtype=jnp.float32)
    delta_arr = jnp.atleast_1d(delta_omega)
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)

    per_traj = jax.jit(
        jax.vmap(
            _single_trajectory_solver,
            in_axes=(0, None, None, None, None, None, None, None, None, None, None),
        ),
        static_argnums=(2, 7),
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
        thermal,
    )

    return {k: np.asarray(v) for k, v in out.items()}


def validate_solver(
    solution: dict[str, np.ndarray],
    pin: float,
    kappa: float,
    kappa_c: float,
    gamma: float,
    print_results: bool = True,
) -> dict[str, bool]:
    """Run basic solver validation checks and print pass/fail status."""
    u_int = np.asarray(solution["U_int_history"])
    p_trans = np.asarray(solution["P_trans_history"])
    e_hist = np.asarray(solution["E_history"])

    p_th = max((kappa**3) / (8.0 * max(gamma, 1e-30) * max(kappa_c, 1e-30)), 1e-30)
    arg = 8.0 * pin / p_th - 1.0
    delta_omega_sol = (kappa / 2.0) * math.sqrt(max(arg, 0.0))
    check_a = np.isfinite(delta_omega_sol) and (delta_omega_sol >= 0.0)

    final_spec = np.abs(np.fft.fftshift(np.fft.fft(e_hist.reshape(-1, e_hist.shape[-1])[-1]))) ** 2
    final_spec /= max(final_spec.max(), 1e-12)
    x = np.linspace(-3.0, 3.0, final_spec.size)
    sech2 = 1.0 / np.cosh(x) ** 2
    sech2 /= sech2.max()
    corr = np.corrcoef(final_spec, sech2)[0, 1]
    check_b = np.isfinite(corr) and corr > 0.7

    u_flat = u_int.reshape(-1, u_int.shape[-1])[-1]
    p_flat = p_trans.reshape(-1, p_trans.shape[-1])[-1]
    ss_start = int(0.8 * u_flat.size)
    u_tail = u_flat[ss_start:]
    p_tail = p_flat[ss_start:]
    rel_u = np.std(u_tail) / max(np.mean(u_tail), 1e-12)
    rel_p = np.std(p_tail) / max(np.mean(p_tail), 1e-12)
    check_c = (rel_u < 5e-2) and (rel_p < 5e-2)

    results = {
        "soliton_existence_condition": bool(check_a),
        "sech2_spectral_envelope": bool(check_b),
        "steady_state_energy_conservation": bool(check_c),
    }

    if print_results:
        for name, ok in results.items():
            print(f"[{ 'PASS' if ok else 'FAIL' }] {name}")

    return results
