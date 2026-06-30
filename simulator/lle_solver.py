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

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"


def gamma_nlse_to_lle(gamma_nlse_per_w_per_m: float, fsr_hz: float, n_eff: float = 2.2) -> float:
    """Convert γ_NLSE [W⁻¹m⁻¹] to γ_LLE [J⁻¹s⁻¹].

    Derivation: equating NLSE and LLE nonlinear phases,
        γ_NLSE · P · L_RT  =  γ_LLE · U_int · t_r
        γ_NLSE · (U/t_r) · v_g·t_r  =  γ_LLE · U · t_r
        γ_LLE  =  γ_NLSE · v_g / t_r  =  γ_NLSE · v_g · FSR

    Units check: [W⁻¹m⁻¹] · [m/s] · [1/s] = W⁻¹s⁻²
                 = (J/s)⁻¹ · s⁻¹ = J⁻¹s⁻¹  ✓
    """
    c   = 299_792_458.0
    v_g = c / n_eff
    return gamma_nlse_per_w_per_m * v_g * fsr_hz   # J⁻¹s⁻¹

def d2_to_beta2_lle(d2_rad_per_s2: float, fsr_hz: float) -> float:
    """Convert integrated dispersion D2 [rad/s²] to LLE beta_2 [s].

    In the microresonator LLE the dispersion polynomial is parameterised by the
    integrated dispersion coefficients Dₖ (rad/s^k).  The mapping to the LLE
    β coefficients (units: s^(k-1)) is:

        β₂ = D₂ / D₁²     (s)
        β₃ = D₃ / D₁³     (s²)

    where D₁ = 2π·FSR.

    Sign convention: D₂ > 0  →  β₂ > 0  →  anomalous dispersion.

    Example (TFLN, 200 GHz FSR, D₂ = 2π × 2 MHz):
        d2_to_beta2_lle(1.2566e7, 2e11) ≈ 7.9e-18  s
    """
    d1 = 2.0 * math.pi * fsr_hz          # rad/s
    return d2_rad_per_s2 / d1 ** 2


def d3_to_beta3_lle(d3_rad_per_s3: float, fsr_hz: float) -> float:
    """Convert D3 [rad/s³] to LLE beta_3 [s²].  See d2_to_beta2_lle."""
    d1 = 2.0 * math.pi * fsr_hz
    return d3_rad_per_s3 / d1 ** 3


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and return physical parameters dict."""
    cfg_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("physical_parameters", {})


def resolve_cavity_rates(config_path=None):
    """Resolve (kappa_i, kappa_c, kappa_total) in rad/s from config — single source of truth.

    kappa_i: prefer explicit `kappa_i_rad_per_s`; else omega0 / intrinsic_q.
    kappa_c: prefer explicit `kappa_c_rad_per_s`; else omega0 / coupling_q;
             else fall back to kappa_i (critical coupling) and warn.
    kappa_total = kappa_i + kappa_c.
    omega0 = 2*pi*c / pump_wavelength_m.
    Mirror the existing kappa_i/Q_i consistency check (lle_solver.py ~l.320-337):
    if an explicit kappa_c_rad_per_s is present AND coupling_q is present, warn when they
    disagree by >15%. Same for kappa_i vs intrinsic_q. Returns floats.
    """
    import warnings as _warnings

    physical = _load_config(config_path)
    _lam = float(physical.get("pump_wavelength_m", 1.55e-6))
    omega0 = 2.0 * math.pi * 299_792_458.0 / _lam

    # --- kappa_i: prefer explicit kappa_i_rad_per_s, else omega0 / intrinsic_q ---
    _q_i = float(physical.get("intrinsic_q", 0) or 0)
    if physical.get("kappa_i_rad_per_s") is not None:
        kappa_i = float(physical["kappa_i_rad_per_s"])
        if _q_i > 0:
            _kappa_i_from_q = omega0 / _q_i
            _rel_diff = abs(_kappa_i_from_q - kappa_i) / max(kappa_i, 1e-30)
            if _rel_diff > 0.15:
                _warnings.warn(
                    f"κ_i from Q_i ({_kappa_i_from_q:.3e} rad/s) differs from "
                    f"kappa_i_rad_per_s ({kappa_i:.3e} rad/s) by {_rel_diff:.1%}. "
                    f"Reconcile config: either remove intrinsic_q or update kappa_i_rad_per_s "
                    f"to {_kappa_i_from_q:.3e}.",
                    stacklevel=2,
                )
    elif _q_i > 0:
        kappa_i = omega0 / _q_i
    else:
        raise ValueError(
            "Cannot resolve kappa_i: config has neither kappa_i_rad_per_s nor intrinsic_q."
        )

    # --- kappa_c: prefer explicit kappa_c_rad_per_s, else omega0 / coupling_q,
    #     else fall back to kappa_i (critical coupling) and warn ---
    _q_c = float(physical.get("coupling_q", 0) or 0)
    if physical.get("kappa_c_rad_per_s") is not None:
        kappa_c = float(physical["kappa_c_rad_per_s"])
        if _q_c > 0:
            _kappa_c_from_q = omega0 / _q_c
            _rel_diff_c = abs(_kappa_c_from_q - kappa_c) / max(kappa_c, 1e-30)
            if _rel_diff_c > 0.15:
                _warnings.warn(
                    f"κ_c from Q_c ({_kappa_c_from_q:.3e} rad/s) differs from "
                    f"kappa_c_rad_per_s ({kappa_c:.3e} rad/s) by {_rel_diff_c:.1%}. "
                    f"Reconcile config: either remove coupling_q or update kappa_c_rad_per_s "
                    f"to {_kappa_c_from_q:.3e}.",
                    stacklevel=2,
                )
    elif _q_c > 0:
        kappa_c = omega0 / _q_c
    else:
        kappa_c = kappa_i
        _warnings.warn(
            f"No kappa_c_rad_per_s or coupling_q in config; assuming critical coupling "
            f"κ_c = κ_i = {kappa_i:.3e} rad/s.",
            stacklevel=2,
        )

    kappa_total = kappa_i + kappa_c
    return float(kappa_i), float(kappa_c), float(kappa_total)


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
        sign = (-1) ** k   # +1 for even, -1 for odd
        disp = disp + sign * float(b) / math.factorial(k) * omega ** k
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
    noise_sequence: jnp.ndarray,   # shape (t_slow,), rad/s, AR(1) pre-generated
    e0_override: jnp.ndarray,     # warm-start field; pass jnp.zeros((n_tau,), complex64) for cold start
    delta_t0_override: jnp.ndarray,  # warm-start thermal state (scalar); pass jnp.zeros(()) for cold start
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

        # Deterministic thermal detuning shift
        thermal_shift = (omega0 / thermal["n0"]) * thermal["dn_dT"] * delta_t
        # Stochastic TCCR/TRN/PyroEO detuning noise at this round trip
        freq_noise = noise_sequence[step_idx]
        delta_omega_eff = delta_omega[step_idx] + thermal_shift + freq_noise   # <-- CHANGED

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

        # Through-port power via energy balance (exact for all-pass ring, any state).
        # P_trans = P_in - κ_i * U_int / t_r = P_in - κ_i * mean(|E|²)
        # This is independent of the coupling geometry and holds for CW, MI,
        # single-soliton, and multi-soliton states identically.
        u_int = jnp.sum(jnp.abs(e_next) ** 2) * (t_r / n_tau)   # J
        
        #clip to [0, pin]. Physical P_trans is non-negative.
        # The energy-balance formula underestimates P_trans during transients when
        # dU_int/dt > 0 (cavity filling). Clip to remove unphysical artifacts.
        p_trans = jnp.clip(pin - kappa_i * u_int / t_r, 0.0, pin)            #W

        # u_int is mean(|E|²)·t_r (J·s); absorbed *power* (W) needs the /t_r,
        # matching the energy-balance form used for p_trans above.
        p_abs = kappa_i * u_int / t_r

        # --- Single-pole thermal model (units of every factor) -----------------
        #   dΔT/dt = -ΔT/τ_th + Γ_th·P_abs/(ρ·Cp·V)
        #   [K/s]  = [K]/[s]  + [·]·[W] / ([kg/m³]·[J/(kg·K)]·[m³])
        #          = [K/s]    +        [J/s] / [J/K]   = [K/s]            ✓
        # Γ_th is DIMENSIONLESS. It is NOT a bare absorbed-power fraction: that is
        # already carried by κ_i (P_abs = κ_i·⟨|E|²⟩). Γ_th is the lumped
        # absorbed-power → temperature-rise coupling
        #     Γ_th = η_abs · (V_mode / V_thermal),
        # i.e. the material-absorption fraction of κ_i loss (η_abs, vs scattering)
        # times the correction for using the OPTICAL mode volume V here while the
        # heat actually fills the larger thermal-diffusion volume V_thermal =
        # L_cav·π·α·τ_th. Equivalently R_th = τ_th·Γ_th/(ρ·Cp·V) = η_abs/(π·k_th·L_cav)
        # (τ_th cancels; the steady state is set by the spreading resistance only).
        # See config/sin_params.yaml for the SiN numbers: Γ_th = 4.72e-3, R_th ≈
        # 0.545 K/W, giving a steady thermal_shift of O(0.1–few)×κ. A solve-time
        # pre-flight assertion (solve_lle_ssfm_jax) flags a mis-set Γ_th loudly.
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
            "P_trans": p_trans,        # through-port power (detector-matched)
            "U_int": u_int,
            "DeltaT": delta_t_next,
            "delta_omega_eff": delta_omega_eff,
        }
        return (e_next, delta_t_next, e_snapshots, label_history, next_snap_count), out
        
    # CW steady state: e_ss = sqrt(κ_c·P_in) / (κ/2 + i·δω)
    # = sqrt(κ_c·P_in) · (κ/2 - i·δω) / ((κ/2)² + δω²)
    # Starting with only the real part (old code) produces an 83° phase error
    # at δω = +4κ, causing ~50 ns of spurious Rabi oscillations.
    _amp = jnp.sqrt(jnp.maximum(kappa_c * pin, 0.0))
    _d2  = (kappa / 2.0) ** 2 + delta_omega[0] ** 2
    e_cw = (_amp * (kappa / 2.0) / _d2 + 1j * (-_amp * delta_omega[0] / _d2)) * jnp.ones(
        n_tau, dtype=jnp.complex64
    )

    key, subkey_r, subkey_i = jax.random.split(rng_key, 3)
    # Scale noise to 0.1% of the CW amplitude at the starting detuning.
    # The former absolute level 1e-4 is ~8× larger than |e_cw| at δω = +4κ
    # (|e_cw| ≈ 1.24e-5), so U_int is noise-dominated at t=0, clipping
    # P_trans to zero for the first ~566 round trips of every trajectory.
    noise = 1e-3 * jnp.abs(e_cw[0]) * (
        jax.random.normal(subkey_r, (n_tau,)) + 1j * jax.random.normal(subkey_i, (n_tau,))
    ).astype(jnp.complex64)

    # Select warm-start vs cold-start without a scalar jnp.where on complex arrays.
    # jnp.where with a scalar condition is unsafe for complex dtypes under jit/vmap:
    # type promotion can silently drop the imaginary part.
    # Instead, use a float mask broadcast elementwise over the n_tau dimension.
    _is_cold = jnp.all(e0_override == 0.0).astype(jnp.float32)   # 1.0 = cold, 0.0 = warm
    _cold = (e_cw + noise).astype(jnp.complex64)
    _warm = e0_override.astype(jnp.complex64)
    e0 = (_is_cold * _cold + (1.0 - _is_cold) * _warm).astype(jnp.complex64)
    
    delta_t0 = delta_t0_override.astype(jnp.float32)
    e_snapshots0 = jnp.zeros((n_snapshots, n_tau), dtype=jnp.complex64)
    label_history0 = jnp.zeros((n_snapshots,), dtype=jnp.int32)
    snap_count0 = jnp.array(0, dtype=jnp.int32)

    (final_carry, hist) = jax.lax.scan(
        _step,
        (e0, delta_t0, e_snapshots0, label_history0, snap_count0),
        xs=jnp.arange(t_slow),
        length=t_slow,
    )
    e_final, delta_t_final, e_snapshots, label_history, _ = final_carry

    return {
        "E_snapshots": e_snapshots,
        "label_history": label_history,
        "P_trans_history": hist["P_trans"],
        "U_int_history": hist["U_int"],
        "DeltaT_history": hist["DeltaT"],
        "delta_omega_eff_history": hist["delta_omega_eff"],
        "delta_t_final": delta_t_final,
        "e_final": e_final,           # ← exact field at step t_slow-1
    }


_STATE_LABELER = make_state_labeler()
_PER_TRAJ = jax.jit(
    jax.vmap(
        _single_trajectory_solver,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, 0, None, None, 0, 0, 0),
    ),
    static_argnums=(2, 3, 7, 10, 13),
)


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
    snapshot_interval: int = 10,
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
    gamma = float(physical.get("gamma_LLE_per_J_per_s"))
    assert 1e15 < gamma < 1e25, (
        f"gamma_LLE = {gamma:.3e} J⁻¹s⁻¹ outside expected range at 200 GHz FSR. "
        f"Use gamma_nlse_to_lle() to compute from γ_NLSE."
    )
    thermal["Gamma_th"] = float(physical.get("Gamma_th", thermal["Gamma_th"]))
    thermal["kappa_i"] = float(physical.get("kappa_i_rad_per_s", max(kappa - kappa_c, 0.0)))

    # --- κ_i / Q consistency check ---
    import warnings as _warnings
    physical = _load_config(config_path)
    _lam = float(physical.get("pump_wavelength_m", 1.55e-6))
    _omega0_est = 2.0 * math.pi * 299_792_458.0 / _lam
    _q_i = float(physical.get("intrinsic_q", 0))
    if _q_i > 0:
        _kappa_i_from_q = _omega0_est / _q_i
        _kappa_i_direct = thermal["kappa_i"]
        _rel_diff = abs(_kappa_i_from_q - _kappa_i_direct) / max(_kappa_i_direct, 1e-30)
        if _rel_diff > 0.15:
            _warnings.warn(
                f"κ_i from Q_i ({_kappa_i_from_q:.3e} rad/s) differs from "
                f"kappa_i_rad_per_s ({_kappa_i_direct:.3e} rad/s) by {_rel_diff:.1%}. "
                f"Reconcile config: either remove intrinsic_q or update kappa_i_rad_per_s "
                f"to {_kappa_i_from_q:.3e}.",
                stacklevel=2,
            )
    
    assert 1e-5 < thermal["Gamma_th"] < 1.0, (
        f"Gamma_th={thermal['Gamma_th']} must be dimensionless fraction (1e-5 to 1.0)"
    )
    assert 1e6 < thermal["kappa_i"] < 1e12, (
        f"kappa_i={thermal['kappa_i']} should be in rad/s (1e6 to 1e12)"
    )
    
    t_r = 1.0 / thermal["fsr_hz"]

    # --- Pre-flight: verify pin is in a physically meaningful regime ---
    # MI onset (δω→0): γ·U_cw = κ/2 with U_cw = κ_c·pin/(κ/2)^2  ⇒
    #   P_th = κ^3 / (8·γ_LLE·κ_c)   [W].  (No stray 1/t_r: |E|²∈J already.)    
    # The simulation only produces interesting states (MI, multi-soliton, single-soliton)
    # for pin > P_th. Warn if pin is more than 10x below threshold so the caller
    # knows their dataset will be all-CW.
    _p_th = (kappa / 2.0) ** 2 * kappa / (2.0 * gamma * kappa_c)
    if pin < 0.1 * _p_th:
        import warnings as _w
        _w.warn(
            f"pin={pin*1e3:.1f} mW is {_p_th/pin:.0f}x below the MI threshold "
            f"P_th={_p_th*1e3:.1f} mW. All trajectories will be CW (label 1). "
            f"Increase pin or adjust Q_i / A_eff in config.",
            stacklevel=2,
        )

    # --- Pre-flight: steady-state thermo-optic shift sanity check ---
    # A mis-set Γ_th silently destabilises the SSFM loop: the deterministic
    # thermal_shift = (ω0/n0)·dn_dT·ΔT enters the linear operator every round
    # trip, so a shift of tens of κ drives the thermal feedback past a
    # bistability edge and blows the field up to NaN. We catch it here from the
    # ANALYTIC steady state (no clipping of the actual run-time shift):
    #   ⟨|E|²⟩_cw = κ_c·P_in / ((κ/2)² + Δω²)          (J, CW intracavity energy)
    #   P_abs     = κ_i·⟨|E|²⟩_cw                       (W)
    #   ΔT_ss     = τ_th·Γ_th·P_abs / (ρ·Cp·V)          (K)
    #   shift_ss  = (ω0/n0)·dn_dT·ΔT_ss                 (rad/s)
    # The assertion is evaluated at the WORST CASE Δω=0 (on resonance, where
    # ⟨|E|²⟩ — and hence P_abs and the shift — is maximal). That makes the
    # tripwire a property of (Γ_th, P_in, κ, thermal params) ALONE, independent
    # of the requested detuning/scan, so a grossly mis-set Γ_th fails loudly even
    # for a far-detuned CW run. We also report the shift at the actual operating
    # detuning(s) for context.
    _r_th = thermal["tau_th"] * thermal["Gamma_th"] / (
        thermal["rho"] * thermal["Cp"] * thermal["V"]
    )                                                                       # K/W
    _omega0_pf = 2.0 * math.pi * 299_792_458.0 / thermal["pump_wavelength_m"]
    _to_coeff = (_omega0_pf / thermal["n0"]) * thermal["dn_dT"]             # rad/s/K

    def _steady_shift(_dw):                       # rad/s, at detuning _dw (rad/s)
        _e2 = kappa_c * float(pin) / ((kappa / 2.0) ** 2 + _dw ** 2)        # J
        return _to_coeff * _r_th * (thermal["kappa_i"] * _e2)              # rad/s

    _worst_shift = float(_steady_shift(0.0))                                # Δω=0
    _dw_op = np.atleast_1d(np.asarray(delta_omega, dtype=float))
    _op_shift = float(np.max(np.abs(_steady_shift(_dw_op))))                # operating pt
    _shift_max_over_kappa = float(physical.get("thermal_shift_max_over_kappa", 15.0))
    # Euler thermal-step stability: ΔT_{n+1} = (1 - t_r/τ_th)·ΔT_n + drive.
    # |1 - t_r/τ_th| < 1 ⟺ stable; t_r/τ_th ≪ 1 here so it is far from the limit.
    _euler_eig = 1.0 - t_r / thermal["tau_th"]
    print(
        f"[thermal pre-flight] t_r/tau_th = {t_r / thermal['tau_th']:.3e}, "
        f"Euler eigenvalue = {_euler_eig:.9f} (stable if |.|<1), "
        f"R_th = {_r_th:.3e} K/W, worst-case (Δω=0) thermal_shift = "
        f"{_worst_shift / kappa:.2f}×κ, operating-point thermal_shift = "
        f"{_op_shift / kappa:.2f}×κ (limit {_shift_max_over_kappa:.1f}×κ)"
    )
    assert _worst_shift < _shift_max_over_kappa * kappa, (
        f"Worst-case steady thermal_shift {_worst_shift:.3e} rad/s = "
        f"{_worst_shift / kappa:.1f}×κ exceeds {_shift_max_over_kappa:.1f}×κ. "
        f"Gamma_th={thermal['Gamma_th']:.3e} is likely mis-set (it is a lumped "
        f"η_abs·V_mode/V_thermal coupling — see the Γ_th derivation in "
        f"config/sin_params.yaml), or P_in is too high for this thermal model. "
        f"Reduce Gamma_th, or raise thermal_shift_max_over_kappa if this "
        f"thermal-triangle regime is intended."
    )

    # convert every leaf to a scalar JAX array so vmap can trace through it
    thermal = {k: jnp.array(v, dtype=jnp.float32) for k, v in thermal.items()}

    delta_omega_input = delta_omega
    delta_omega = jnp.array(delta_omega_input, dtype=jnp.float32)
    delta_arr = jnp.atleast_1d(delta_omega)
    beta_arr = tuple(float(b) for b in beta)

    # Guard: catch accidental use of fiber-optics β₂ units (s²/m).
    # LLE β₂ = D₂/D₁² ≈ 1e-18–1e-16 s for typical microresonators.
    if len(beta_arr) >= 1 and beta_arr[0] != 0.0:
        b2_mag = abs(beta_arr[0])
        if not (1e-20 < b2_mag < 1e-12):
            raise ValueError(
                f"beta[0] (β₂) = {beta_arr[0]:.3e} is outside the expected LLE range "
                f"[1e-20, 1e-12] s.  Fiber-optics β₂ (s²/m) must be converted first: "
                f"use d2_to_beta2_lle(d2_rad_per_s2, fsr_hz)."
            )
    
    # Split into three independent groups from a single chain
    key, key_field, key_noise = jax.random.split(rng_key, 3)
    key_arr    = jax.random.split(key_field, delta_arr.shape[0])   # for e0
    noise_keys = jax.random.split(key_noise, delta_arr.shape[0])   # for AR(1)

    # --- Generate per-trajectory noise sequences ---
    from simulator.noise_models import TotalNoise, _load_config as _nm_load_cfg
    _nm_cfg = _nm_load_cfg(config_path)
    _noise_model = TotalNoise(_nm_cfg)

    # shape: (n_traj, t_slow)  — one AR(1) sequence per trajectory
    def _gen_noise(key):
        return _noise_model.sample(key, int(t_slow))

    noise_sequences = jax.vmap(_gen_noise)(noise_keys)   # (n_traj, t_slow)
    
    n_traj = delta_arr.shape[0]
    e0_cold = jnp.zeros((n_traj, n_tau), dtype=jnp.complex64)
    delta_t0_cold = jnp.zeros((n_traj,), dtype=jnp.float32)
    out = _PER_TRAJ(
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
        _STATE_LABELER,
        noise_sequences,
        e0_cold,
        delta_t0_cold,
    )

    return {k: np.asarray(v) for k, v in out.items()}


def validate_solver(
    solution: dict[str, np.ndarray],
    pin: float,
    kappa: float,
    kappa_c: float,
    gamma: float,
    t_r: float,
    traj_idx: int = 0,
    print_results: bool = True,
    config_path=None,
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

    # MI/soliton threshold in energy-normalized LLE (|E|^2 ~ J), δω→0:
    # P_th = κ^3 / (8·gamma_LLE·kappa_c)
    p_th = max(
        (kappa / 2.0) ** 2 * kappa / (2.0 * max(gamma, 1e-30) * max(kappa_c, 1e-30)),
        1e-30
    )
    arg = pin / p_th - 1.0          # >0 means above MI threshold
    check_a = np.isfinite(p_th) and (p_th > 0) and (arg > 0)

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
    
    # --- CW steady-state energy balance ---
    # In the round-trip field normalisation |E|² ~ J, the analytic CW energy is:
    #   U_cw = κ_c · P_in · t_r / ((κ/2)² + Δω²)
    # The t_r factor converts from power-normalised (W) to energy-normalised (J).
    thermal_p = _thermal_params(config_path)   # need t_r — pass config_path through
    t_r_val = 1.0 / thermal_p["fsr_hz"]

    delta_omega_test = float(np.mean(delta_omega_eff_hist[-100:]))
    u_ss = float(np.mean(u_int[-100:]))
    u_expected = kappa_c * pin * t_r_val / ((kappa / 2) ** 2 + delta_omega_test ** 2)
    rel_error = abs(u_ss - u_expected) / max(u_expected, 1e-30)
    print(f"CW steady-state energy error: {rel_error:.3%} (pass if <10%)")
    assert rel_error < 0.10, (
        f"Steady-state energy deviates {rel_error:.1%} from analytical CW solution. "
        f"u_ss={u_ss:.3e} J, u_expected={u_expected:.3e} J, "
        f"delta_omega_eff={delta_omega_test:.3e} rad/s"
    )

    results = {
        "soliton_existence_condition": bool(check_a),
        "sech2_spectral_envelope": bool(check_b),
        "steady_state_energy_conservation": bool(check_c),
    }

    if print_results:
        for name, ok in results.items():
            print(f"[{ 'PASS' if ok else 'FAIL' }] {name}")

    return results
