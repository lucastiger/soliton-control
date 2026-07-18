"""JAX-based LLE + thermal ODE solver module.

This module implements a GPU-accelerated split-step Fourier method (SSFM)
solver for the generalized Lugiato–Lefever Equation (LLE), including a
https://github.com/lucastiger/soliton-control/edit/main/simulator/lle_solver.pysingle-pole thermal model for thermo-optic detuning drift.
"""

from __future__ import annotations

import functools
import math
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import jax

# Enable 64-bit precision (float64/complex128) process-wide. JAX bakes the x64
# flag in at ARRAY-CREATION time, so this MUST run before any jax.numpy array is
# built — hence it sits here, immediately after `import jax` and before both
# `import jax.numpy` and the state_labeler import (which may create arrays). The
# solver runs the SSFM loop for hundreds of thousands of round trips; in
# complex64 the accumulated roundoff pins the spectral floor at ~-70 dB and
# buries sub--70 dB structure (e.g. dispersive waves). float64 pushes that floor
# far lower. Guarded so it is set exactly once (idempotent across re-imports).
if not jax.config.read("jax_enable_x64"):
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import yaml

from simulator.state_labeler import (
    make_state_labeler,
    make_threshold_params,
    physical_off_floor,
)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

# Reduced Planck constant [J·s] (CODATA). Used only by the quantum-vacuum-noise
# channel to convert intracavity energy |E|² [J] to photon number via ħω₀.
_HBAR_J_S = 1.054571817e-34


def hbar_omega0_from_config(physical: dict[str, Any]) -> float:
    """ħω₀ [J] at the pump: config override ``hbar_omega0_j`` if > 0, else ħ·2πc/λ_p.

    ω₀ = 2πc/pump_wavelength_m (1.21526e15 rad/s at λ_p = 1.55 µm, so
    ħω₀ = 1.2816e-19 J). Using the PUMP-mode ħω₀ for every comb mode
    over-/under-counts the photon energy of mode μ by |μ|·FSR/f₀ — <1% across
    the comb span here — which is the documented approximation of the
    quantum-noise normalization. ``hbar_omega0_j`` <= 0 (the config encodes
    "auto" as 0 because physical_parameters leaves must stay numeric) or a
    missing key both mean "compute from the pump wavelength".
    """
    override = float(physical.get("hbar_omega0_j", 0.0) or 0.0)
    if override > 0.0:
        return override
    lam = float(physical.get("pump_wavelength_m", 1.55e-6))
    return _HBAR_J_S * 2.0 * math.pi * 299_792_458.0 / lam


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

    beta_list[0] is beta_2 (s), beta_list[1] is beta_3 (s^2), etc.
    (LLE convention beta_k = D_k / D_1^k; NOT fiber GVD s^2/m.)
    The k=0 and k=1 terms are zero by definition in the co-moving frame and must not be included.
    """
    assert len(beta_list) >= 1, "Must provide at least beta_2"
    disp = jnp.zeros_like(omega)
    for i, b in enumerate(beta_list):
        k = i + 2
        # All orders enter with +beta_k/k!: D_int = ½D₂μ² + ⅙D₃μ³ + …
        disp = disp + float(b) / math.factorial(k) * omega ** k
    return disp


_DEFAULT_DINT_CSV = (
    Path(__file__).resolve().parents[1] / "config" / "pyLLE_dispersion_w4400_h800.csv"
)

# Module-level cache keyed on (n_tau, resolved_csv_path) so repeated calls do not
# re-read/re-interpolate the (multi-thousand-row) dispersion CSV.
_DINT_GRID_CACHE: dict[tuple[int, str], "DintGrid"] = {}


class DintGrid(NamedTuple):
    """Measured integrated-dispersion grid returned by :func:`load_dint_grid`.

    Attributes:
        grid: D_int(mu) [rad/s], shape (n_tau,), in FFT-bin order (NOT fftshifted),
            jnp float64 — ready to drop into the linear operator as ``disp``.
        d1:  Measured D1 = 2π·FSR [rad/s] from a smooth-trend fit, excluding the
            5 < |mu| region around a known pump-neighborhood defect (a plain
            central difference at mu=0 is biased +2π·3.35 MHz by that defect),
            so callers can reconcile the FSR / round-trip time.

    A ``NamedTuple`` so it is importable, picklable, and supports both attribute
    access (``res.grid``) and tuple unpacking (``grid, d1 = res``).
    """

    grid: jnp.ndarray
    d1: float


def load_dint_grid(n_tau, csv_path=None, config_path=None) -> "DintGrid":
    """Return D_int(mu) [rad/s], shape (n_tau,), in FFT-bin order (NOT fftshifted),
    ready to drop into the linear operator as ``disp``.

    - Default csv_path = config/pyLLE_dispersion_w4400_h800.csv.
    - Load (mu, f_hz). Compute omega = 2*pi*f. Locate mu==0 -> omega0.
    - D1 = smooth-trend fit of omega, excluding the 5 < |mu| region around a
      known pump-neighborhood defect (rad/s). Returned too (as the ``d1``
      attribute of the DintGrid result) so callers can reconcile FSR.
    - D_int(mu) = omega - omega0 - D1*mu.
    - Build the integer FFT mode grid: k = np.round(np.fft.fftfreq(n_tau,
      d=t_r/n_tau) / fsr).astype(int), where fsr = D1/(2*pi) and t_r = 1/fsr.
      (This yields the same bin ordering as _build_omega_grid, so disp aligns
      with the existing FFT convention.)
    - D_int_grid = np.interp(k, mu_csv, D_int_csv). Assert the mu=0 bin is 0.
    - D_int_grid is returned as a jnp float64 array (complex-compatible for the
      -1j*disp linear-operator term).

    Args:
        n_tau: Number of fast-time / FFT grid points.
        csv_path: Optional path to the (mu, f_hz) CSV. Defaults to the pyLLE grid.
        config_path: Accepted for API symmetry with the rest of the solver; the
            dispersion source is the CSV, so this argument is currently unused.

    Returns:
        DintGrid(grid, d1) — see :class:`DintGrid`.
    """
    del config_path  # reserved for API symmetry; dispersion comes from the CSV
    csv_path = Path(csv_path) if csv_path is not None else _DEFAULT_DINT_CSV

    cache_key = (int(n_tau), str(csv_path.resolve()))
    cached = _DINT_GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Columns: mode number mu (int), resonance frequency f [Hz]; no header.
    data = np.loadtxt(csv_path, delimiter=",")
    mu_csv = data[:, 0].astype(np.int64)
    f_hz = data[:, 1].astype(np.float64)

    # Work in float64: omega ~ 1.2e15 rad/s while D_int ~ 1e8–1e14 rad/s, so
    # D_int = omega - omega0 - D1*mu is a catastrophic-cancellation subtraction
    # that must not be done in float32.
    omega = 2.0 * np.pi * f_hz
    i0 = int(np.where(mu_csv == 0)[0][0])
    omega0 = omega[i0]

    # Central difference of omega at mu==0 (relies on mu=±1 flanking mu=0 in the
    # contiguous integer mode list).
    # 3-point central difference is biased +2π·3.35 MHz by a localized
    # pump-neighborhood defect (|Δf| up to ~27 MHz for |mu|<=4), tilting
    # D_int by (Δd1)·mu: provably harmless for mode powers (pure drift),
    # but it corrupts every crossing/DW readout. Fit the smooth trend,
    # excluding the defect region. omega0 stays measured: D_int(0) == 0.
    _sel = (np.abs(mu_csv) <= 600) & (np.abs(mu_csv) > 5)
    _pf = np.polynomial.Polynomial.fit(mu_csv[_sel].astype(float), omega[_sel], 7)
    d1 = float(_pf.deriv()(0.0))                        # rad/s
    d_int_csv = omega - omega0 - d1 * mu_csv            # rad/s

    fsr = d1 / (2.0 * np.pi)                            # Hz
    t_r = 1.0 / fsr                                     # s
    # Same bin ordering as _build_omega_grid: fftfreq/fsr gives the integer mode
    # index per FFT bin ([0,1,..,n/2-1,-n/2,..,-1]).
    k = np.round(np.fft.fftfreq(int(n_tau), d=t_r / int(n_tau)) / fsr).astype(int)

    d_int_grid = np.interp(k, mu_csv, d_int_csv)
    # np.interp holds D_int FLAT beyond the CSV span (here the red edge: the CSV
    # stops at mu=-3261 while an n_tau>=8192 FFT reaches mu=-4096..). A flat clamp
    # injects a slope discontinuity at the CSV edge — a spurious kink that seeds
    # aliasing. Replace the out-of-range modes with a LINEAR extension that
    # continues the boundary slope (C1-continuous), so the extrapolated edge is
    # smooth; the solver's edge absorber then damps those modes cleanly.
    lo_mu, hi_mu = int(mu_csv[0]), int(mu_csv[-1])
    slope_lo = float(d_int_csv[1] - d_int_csv[0])       # per unit mu at low edge
    slope_hi = float(d_int_csv[-1] - d_int_csv[-2])     # per unit mu at high edge
    below = k < lo_mu
    above = k > hi_mu
    d_int_grid[below] = d_int_csv[0] + slope_lo * (k[below] - lo_mu)
    d_int_grid[above] = d_int_csv[-1] + slope_hi * (k[above] - hi_mu)
    assert d_int_grid[0] == 0.0, (
        f"mu=0 FFT bin (k={k[0]}) must have D_int == 0, got {d_int_grid[0]!r}."
    )

    result = DintGrid(grid=jnp.asarray(d_int_grid, dtype=jnp.float64), d1=float(d1))
    _DINT_GRID_CACHE[cache_key] = result
    return result


# Super-Gaussian edge-absorber shape: A(mu) = exp(-STRENGTH * s**POWER) with
# s = 0 at the interior onset ramping to 1 at the Nyquist edge. POWER=8 keeps
# A~1 across most of the ramp then rolls off sharply; STRENGTH=40 makes the very
# edge ~e^-40 (effectively zero), so energy reaching the extrapolated grid edge
# is damped rather than aliased/folded back into the interior.
_ABSORBER_POWER = 8.0
_ABSORBER_STRENGTH = 40.0

# Dispersion-validity mask shape (opt-in guard; see solve_lle_ssfm_jax). The
# linear half-step applies the EXACT exponential exp(-i*(D_int+delta)*dt), which
# is correct at ANY phase, so |D_int*t_r| by itself is NOT an error metric — an
# earlier mask keyed to |D_int*t_r| > 1 amputated real comb structure (soliton
# tail and dispersive waves, ~|mu| > 1000 on the measured grid). The genuine
# discrete-map artifact is spurious four-wave mixing that phase-matches when the
# linear-phase MISMATCH accrued across one nonlinear kick,
# |D_int - delta_omega|*dt_sub, approaches 2*pi. The mask therefore keys to that
# per-sub-step mismatch phase, with the default threshold pi safely below the
# 2*pi onset. validity = exp(-STRENGTH*over^POWER) with
# over = max(phase_sub/threshold - 1, 0): exactly 1 inside the window, rolling
# to ~0 within ~half the threshold above it.
_VALIDITY_POWER = 2.0
_VALIDITY_STRENGTH = 10.0


def _qnoise_increment(
    qnoise_key: jax.Array, fine_step_index, n_tau: int, scale
) -> jnp.ndarray:
    """One quantum-vacuum Langevin increment (n_tau,) for one fine step.

    Truncated-Wigner (symmetric-ordering) c-number limit of the vacuum input
    noise √κ·ξ̂_μ(t), ⟨ξ̂_μ(t)ξ̂†_μ′(t′)⟩ = δ(t−t′)δ_μμ′ (arXiv:2604.05897
    Eq. 126): i.i.d. complex Gaussian per fast-time sample with per-quadrature
    std ``scale`` = √(ħω₀·κ·n_tau·dt_fine/4), i.e. total per-sample variance
    ħω₀·κ·n_tau·dt_fine/2 — equivalent (by Parseval, Ẽ_μ = a_μ·n_tau·√(ħω₀))
    to every mode μ receiving an independent photon-amplitude increment of
    total variance (κ/2)·dt_fine, whose steady state in the undriven linear
    cavity is the symmetric-ordered vacuum occupation of ½ photon per mode.
    Time-domain injection deliberately avoids extra FFTs.

    The per-step key is ``fold_in(qnoise_key, fine_step_index)`` with
    ``fine_step_index = step_idx·fine_cadence_M + m`` — deterministic given the
    trajectory key, independent of n_substeps, and generated in-scan (never a
    pre-materialized (t_slow, n_tau) array).
    """
    k = jax.random.fold_in(qnoise_key, fine_step_index)
    k_re, k_im = jax.random.split(k)
    draw = (
        jax.random.normal(k_re, (n_tau,), dtype=jnp.float64)
        + 1j * jax.random.normal(k_im, (n_tau,), dtype=jnp.float64)
    )
    return (scale * draw).astype(jnp.complex128)


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
    e0_override: jnp.ndarray,     # warm-start field; pass jnp.zeros((n_tau,), complex128) for cold start
    delta_t0_override: jnp.ndarray,  # warm-start thermal state (scalar); pass jnp.zeros(()) for cold start
    d_int_grid: jnp.ndarray | None,  # per-mode measured D_int(mu) (FFT-bin order); None = Taylor path
    n_substeps: int,              # Strang sub-steps per round trip (static); 1 = legacy single step
    dealias_two_thirds: bool,     # static; zero |mu|>n_tau/3 after each nonlinear kick (2/3 rule)
    edge_absorber: bool,          # static; super-Gaussian edge damping once per round trip
    edge_absorber_frac: float,    # outer fraction of |mu| (each side) over which the absorber ramps
    dispersion_validity_mask: bool,  # static; opt-in damping of modes whose per-sub-step mismatch phase exceeds the threshold
    validity_phase_threshold: float,  # |D_int - delta_omega|*dt_sub threshold (rad) for the validity mask
    fine_cadence_M: int,          # static; advance thermal/detuning/energy at dt=t_r/M (1 = per-round-trip)
    qnoise_key: jax.Array,        # per-trajectory PRNG key for the quantum Langevin drive
    qnoise_scale: float,          # per-quadrature injection std √(ħω₀·κ·n_tau·dt_fine/4); 0.0 = disabled
    qnoise_enabled: bool,         # static; False traces ZERO extra ops (bit-identical legacy path)
) -> dict[str, jnp.ndarray]:
    """Solve one detuning trajectory with SSFM + thermal Euler update.

    The round trip is integrated with ``n_substeps`` Strang sub-steps, each over
    dt = t_r / n_substeps, so the per-sub-step linear phase |D_int·dt| shrinks
    with n_substeps. This suppresses the split-step spurious-sideband instability
    that appears once |D_int·t_r| approaches π at large |mu| (an n_tau-independent
    artifact). With n_substeps=1 the arithmetic reduces exactly to the legacy
    single Strang step (bit-identical regression guard). The detuning
    (thermal_shift + noise), thermal Euler update, energy balance, snapshots, and
    labels are all evaluated ONCE per round trip on the end-of-round-trip field.

    Two anti-aliasing toggles (both default OFF at the public API, giving
    bit-identical legacy behaviour): ``dealias_two_thirds`` zeros the Fourier
    modes with |mu| > n_tau/3 after each nonlinear kick (the standard 2/3 rule
    that removes cubic-nonlinearity aliasing), and ``edge_absorber`` multiplies
    the field once per round trip by a super-Gaussian in mode space that is ~1
    across the interior and ramps to strong attenuation over the outer
    ``edge_absorber_frac`` of |mu| on each side, damping energy that reaches the
    (partly extrapolated) grid edges instead of letting it fold back inward.

    ``fine_cadence_M`` (default 1) advances the WHOLE evolution -- field, thermal
    ODE, detuning, pump, energy balance, and the anti-aliasing masks -- at the
    fine cadence dt = t_r / M rather than refreshing the thermal/detuning/energy
    once per round trip. This removes the residual t_r-periodic modulation of the
    once-per-round-trip mean-field map (which can drive parametric round-trip-map
    resonances); the total physical time is unchanged (M fine steps per round
    trip). M=1 is bit-identical to the per-round-trip integrator. Histories are
    still recorded once per round trip on the end-of-round-trip field.

    ``qnoise_enabled`` (static Python bool) turns on the quantum-vacuum Langevin
    drive (arXiv:2604.05897 Eq. 126): once per FINE step (never per sub-step, so
    the injected variance is independent of n_substeps) every fast-time sample
    receives an i.i.d. complex Gaussian increment of per-quadrature std
    ``qnoise_scale`` = √(ħω₀·κ·n_tau·dt_fine/4), injected AFTER the sub-step
    loop and BEFORE the edge_absorber / dispersion_validity_mask applications so
    the numerical masks damp rather than re-populate edge modes (with
    dealias_two_thirds ON the modes |mu| > n_tau/3 are therefore under-occupied
    by construction; validation statistics must restrict to |mu| <= n_tau/3).
    When False, this function traces exactly the legacy computation — no RNG
    calls and no arithmetic are added to the scan body.
    """
    omega = _build_omega_grid(n_tau, t_r)
    # Per-mode measured dispersion when supplied, else the Taylor beta polynomial.
    # Cast to omega's real dtype and rely on the (n_tau,) shape guaranteed upstream.
    if d_int_grid is not None:
        disp = jnp.asarray(d_int_grid).astype(omega.dtype)
    else:
        disp = build_dispersion(omega, beta)
    # Time-step hierarchy: each round trip is fine_cadence_M fine steps of
    # dt_fine = t_r/M (the cadence of the thermal ODE / detuning / energy), and
    # each fine step is n_substeps Strang sub-steps of dt_sub = dt_fine/n_substeps
    # (the field split-step). The pump kick sqrt(max(κ_c·pin,0))·dt_sub is
    # injected per field sub-step, so the drive summed over a round trip is
    # M·n_substeps·(F·dt_sub) = F·t_r (unchanged). With M=1 and n_substeps=1,
    # dt_fine=dt_sub=t_r exactly, so the legacy path stays bit-identical.
    dt_fine = t_r / fine_cadence_M
    dt_sub = dt_fine / n_substeps
    pump_kick = (jnp.sqrt(jnp.maximum(kappa_c * pin, 0.0)) * dt_sub).astype(jnp.complex128)
    kappa_i = jnp.maximum(thermal["kappa_i"], 0.0)
    omega0 = 2.0 * jnp.pi * 299_792_458.0 / thermal["pump_wavelength_m"]
    n_snapshots = (t_slow + snapshot_interval - 1) // snapshot_interval

    # Anti-aliasing masks in FFT-bin mode order (|mu| per bin). Built once and
    # only when the corresponding toggle is on, so the OFF path is untouched.
    if dealias_two_thirds or edge_absorber:
        abs_mu = jnp.abs(jnp.fft.fftfreq(n_tau) * n_tau)   # |mu| per FFT bin
    if dealias_two_thirds:
        # 2/3 rule: keep |mu| <= n_tau/3, zero the rest.
        mask_23 = (abs_mu <= (n_tau / 3.0)).astype(jnp.float64)
    if edge_absorber:
        mu_max = n_tau / 2.0
        onset = (1.0 - edge_absorber_frac) * mu_max        # interior edge of ramp
        s = jnp.clip((abs_mu - onset) / jnp.maximum(mu_max - onset, 1.0), 0.0, 1.0)
        absorber = jnp.exp(-_ABSORBER_STRENGTH * s ** _ABSORBER_POWER)
    if dispersion_validity_mask:
        # The exact linear exponential is valid at ANY phase; the genuine
        # discrete-map artifact is spurious FWM phase-matching when the linear
        # MISMATCH phase per nonlinear kick nears 2*pi. Key the mask to the
        # sub-step actually taken and to the detuned dispersion. disp is
        # D_int(mu) [rad/s] in FFT-bin order; delta_omega is the (t_slow,)
        # programmed-detuning schedule and the mask is built once, so use its
        # first value (sweeps move by ~kappa, negligible against the ~1e3*kappa
        # phase scale where the mask engages).
        phase_sub = jnp.abs(disp - delta_omega[0]) * dt_sub
        over = jnp.maximum(phase_sub / validity_phase_threshold - 1.0, 0.0)
        validity = jnp.exp(-_VALIDITY_STRENGTH * over ** _VALIDITY_POWER)

    def _fine_step(e_cur, delta_t_cur, dw_step, freq_noise, step_idx, m):
        """Advance the field and thermal state by ONE fine step dt_fine = t_r/M.

        The thermal detuning shift is recomputed from the CURRENT ΔT (so the
        thermo-optic feedback runs at the fine cadence, not once per round trip),
        the field takes n_substeps Strang sub-steps over dt_fine, the quantum
        Langevin increment is added (only when qnoise_enabled; keyed on the
        global fine-step index step_idx*fine_cadence_M + m, with m the static
        fine-step index within the round trip), the masks are applied, then the
        single-pole thermal ODE is Euler-stepped by dt_fine.
        Returns (e_next, delta_t_next, u_int, delta_omega_eff).
        """
        # Deterministic thermal detuning shift. δω = ω_res − ω_pump, so heating
        # (dn/dT>0 → n↑ → ω_res↓) LOWERS δω: the shift enters with a minus sign.
        thermal_shift = -(omega0 / thermal["n0"]) * thermal["dn_dT"] * delta_t_cur
        delta_omega_eff = dw_step + thermal_shift + freq_noise

        # Half-linear operator over dt_sub/2. Dispersion enters as -i·D_int, the
        # SAME sign as -i·delta_omega_eff (they share one detuning axis); for
        # anomalous D₂>0 this supports MI/soliton formation.
        lin_exp = (-kappa / 2.0 - 1j * disp - 1j * delta_omega_eff) * dt_sub
        h_half = jnp.exp(lin_exp / 2.0).astype(jnp.complex128)

        # n_substeps Strang sub-steps over dt_sub: pump kick, L·N·L (unrolled).
        e_sub = e_cur
        for _ in range(n_substeps):
            e_pumped = (e_sub + pump_kick).astype(jnp.complex128)
            e_w = jnp.fft.fft(e_pumped)
            e_half = jnp.fft.ifft(e_w * h_half).astype(jnp.complex128)
            nl_phase = jnp.exp(1j * gamma * jnp.abs(e_half) ** 2 * dt_sub).astype(jnp.complex128)
            e_nl = (e_half * nl_phase).astype(jnp.complex128)
            # 2/3 de-alias right after the cubic kick (frequency domain).
            e_w2 = jnp.fft.fft(e_nl)
            if dealias_two_thirds:
                e_w2 = e_w2 * mask_23
            e_sub = jnp.fft.ifft(e_w2 * h_half).astype(jnp.complex128)
        e_next = e_sub

        # Quantum-vacuum Langevin injection: after the sub-step loop, BEFORE the
        # numerical masks (so they damp, never re-populate, the edge modes).
        # Static Python branch — the disabled path traces zero extra ops.
        if qnoise_enabled:
            e_next = e_next + _qnoise_increment(
                qnoise_key, step_idx * fine_cadence_M + m, n_tau, qnoise_scale
            )

        # Edge absorber then dispersion-validity mask (per fine step; M=1 -> once
        # per round trip, as before).
        if edge_absorber:
            e_next = jnp.fft.ifft(jnp.fft.fft(e_next) * absorber).astype(jnp.complex128)
        if dispersion_validity_mask:
            e_next = jnp.fft.ifft(jnp.fft.fft(e_next) * validity).astype(jnp.complex128)

        # Energy balance (exact for an all-pass ring, any state): P_abs = κ_i·⟨|E|²⟩,
        # independent of dt. u_int is the physical intracavity energy (uses t_r).
        u_int = jnp.sum(jnp.abs(e_next) ** 2) * (t_r / n_tau)   # J
        p_abs = kappa_i * u_int / t_r

        # Single-pole thermal ODE, Euler-stepped by dt_fine:
        #   dΔT/dt = -ΔT/τ_th + Γ_th·P_abs/(ρ·Cp·V)   (Γ_th dimensionless; see
        #   config/sin_params.yaml for the SiN numbers and the pre-flight guard).
        d_delta_t = (
            -delta_t_cur / thermal["tau_th"]
            + thermal["Gamma_th"] * p_abs / (thermal["rho"] * thermal["Cp"] * thermal["V"])
        )
        delta_t_next = delta_t_cur + dt_fine * d_delta_t
        return e_next, delta_t_next, u_int, delta_omega_eff

    def _step(carry, step_idx):
        e_t, delta_t, e_snapshots, label_history, snap_count = carry
        e_t = e_t.astype(jnp.complex128)
        dw_step = delta_omega[step_idx]
        # Stochastic TCCR/TRN/PyroEO detuning noise for this round trip (held
        # across the M fine steps; constant detuning has no t_r-periodicity).
        freq_noise = noise_sequence[step_idx]

        # fine_cadence_M fine steps of dt_fine per round trip (static -> unrolled).
        e_next = e_t
        delta_t_next = delta_t
        u_int = jnp.sum(jnp.abs(e_t) ** 2) * (t_r / n_tau)
        delta_omega_eff = dw_step
        for m in range(fine_cadence_M):
            e_next, delta_t_next, u_int, delta_omega_eff = _fine_step(
                e_next, delta_t_next, dw_step, freq_noise, step_idx, m
            )

        # Through-port power via energy balance, on the end-of-round-trip field.
        # Clip to [0, pin] (the balance underestimates P_trans during filling).
        p_trans = jnp.clip(pin - kappa_i * u_int / t_r, 0.0, pin)            #W
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
        n_tau, dtype=jnp.complex128
    )

    key, subkey_r, subkey_i = jax.random.split(rng_key, 3)
    # Scale noise to 0.1% of the CW amplitude at the starting detuning.
    # The former absolute level 1e-4 is ~8× larger than |e_cw| at δω = +4κ
    # (|e_cw| ≈ 1.24e-5), so U_int is noise-dominated at t=0, clipping
    # P_trans to zero for the first ~566 round trips of every trajectory.
    noise = 1e-3 * jnp.abs(e_cw[0]) * (
        jax.random.normal(subkey_r, (n_tau,), dtype=jnp.float64)
        + 1j * jax.random.normal(subkey_i, (n_tau,), dtype=jnp.float64)
    ).astype(jnp.complex128)

    # Select warm-start vs cold-start without a scalar jnp.where on complex arrays.
    # jnp.where with a scalar condition is unsafe for complex dtypes under jit/vmap:
    # type promotion can silently drop the imaginary part.
    # Instead, use a float mask broadcast elementwise over the n_tau dimension.
    _is_cold = jnp.all(e0_override == 0.0).astype(jnp.float64)   # 1.0 = cold, 0.0 = warm
    _cold = (e_cw + noise).astype(jnp.complex128)
    _warm = e0_override.astype(jnp.complex128)
    e0 = (_is_cold * _cold + (1.0 - _is_cold) * _warm).astype(jnp.complex128)

    delta_t0 = delta_t0_override.astype(jnp.float64)
    e_snapshots0 = jnp.zeros((n_snapshots, n_tau), dtype=jnp.complex128)
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


# Fallback labeler (conservative literal floor); the solver builds a
# physically-scaled labeler per config via _physical_state_labeler() below.
_STATE_LABELER = make_state_labeler()


@functools.lru_cache(maxsize=None)
def _physical_state_labeler(
    kappa: float,
    kappa_c: float,
    pin: float,
    delta_omega_max: float,
    off_fraction: float = 1e-3,
):
    """Build (and cache) a state labeler whose OFF floor = f·U_cw,min from config.

    Cached on the physical params so repeated solver calls with the same config
    reuse one labeler object — JAX treats it as a static arg, so a stable object
    identity avoids needless recompilation. A genuinely different config produces
    a new labeler (and a correct recompile, since the OFF floor changed).
    """
    params = make_threshold_params(
        kappa, kappa_c, pin, delta_omega_max, off_fraction=off_fraction
    )
    return make_state_labeler(params)


# Argument order mirrors _single_trajectory_solver. Vmapped (axis 0):
# delta_omega(0), rng_key(11), noise_sequence(14), e0_override(15),
# delta_t0_override(16), qnoise_key(25). Static: t_slow(2), beta(3), n_tau(7),
# snapshot_interval(10), state_labeler(13), n_substeps(18),
# dealias_two_thirds(19), edge_absorber(20), dispersion_validity_mask(22),
# fine_cadence_M(24), qnoise_enabled(27).
_PER_TRAJ = jax.jit(
    jax.vmap(
        _single_trajectory_solver,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, 0, None, None, 0, 0, 0, None, None, None, None, None, None, None, None, 0, None, None),
    ),
    static_argnums=(2, 3, 7, 10, 13, 18, 19, 20, 22, 24, 27),
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
    e0_override: np.ndarray | jnp.ndarray | None = None,
    delta_t0_override: np.ndarray | jnp.ndarray | float | None = None,
    d_int_grid: np.ndarray | jnp.ndarray | None = None,
    n_substeps: int = 1,
    dealias_two_thirds: bool = False,
    edge_absorber: bool = False,
    edge_absorber_frac: float = 0.12,
    dispersion_validity_mask: bool = False,
    validity_phase_threshold: float = float(jnp.pi),
    fine_cadence_M: int = 1,
    quantum_noise_enabled: bool | None = None,
    quantum_noise_seed_vacuum_init: bool | None = None,
) -> dict[str, np.ndarray]:
    """Batch-capable SSFM solver for the generalized LLE using JAX.

    Detuning convention: delta_omega = omega_res - omega_pump  (cavity minus pump);
      this matches the implemented dynamical term  -1j * delta_omega * E.
      Positive delta_omega = red-detuned pump (pump below resonance) = soliton side.
      Solitons exist for kappa/2 < delta_omega < ~5*kappa.
      For adiabatic soliton access, sweep delta_omega from negative to positive
      (blue-to-red pump scan).

    Quantum noise: with ``quantum_noise_enabled`` (config key or kwarg; OFF by
      default) the solver adds the physically normalized quantum-vacuum Langevin
      drive of Herr, Tikan & Kippenberg, arXiv:2604.05897, Sec. V.B.2, Eq. 126:
      each mode receives sqrt(kappa)*xi_mu(t) with
      <xi_mu(t) xi_mu'^dagger(t')> = delta(t-t') delta_mumu', both loss baths
      (kappa_0, kappa_ex) combined since both are vacuum/coherent (no squeezed
      baths). In the classical truncated-Wigner (symmetric-ordering) c-number
      limit this is additive complex Gaussian noise with
      <xi_mu xi_mu'*> = 1/2 delta(t-t') delta_mumu', whose undriven steady state
      is the symmetric-ordered vacuum occupation of 1/2 photon per mode. It is
      injected in the TIME domain once per fine step (dt_fine = t_r/M): every
      fast-time sample gets an i.i.d. complex Gaussian of per-quadrature std
      sqrt(hbar*omega0*kappa*n_tau*dt_fine/4), which by Parseval
      (Emode_mu = a_mu*n_tau*sqrt(hbar*omega0), photon number
      n_mu = |Emode_mu|^2/(n_tau^2*hbar*omega0)) equals an independent
      photon-amplitude increment of variance (kappa/2)*dt_fine per mode.
      hbar*omega0 is evaluated at the pump for ALL modes (<1% error over the
      comb span). When additionally ``quantum_noise_seed_vacuum_init`` is on
      (default when enabled), a COLD start seeds the analytic CW state with one
      complex Gaussian draw of per-sample variance hbar*omega0*n_tau/2
      (<n_mu> = 1/2 at t=0) instead of the legacy 1e-3*|e_cw| noise; the legacy
      seed path itself is untouched (bypassed via the warm-start branch). With
      the flag OFF the solver is bit-identical to the legacy solver: no RNG
      calls or arithmetic are added to the scan body and the RNG key chain of
      the legacy paths is unchanged.

    Args:
        pin: Pump power in watts.
        delta_omega: Detuning sweep (omega_res - omega_pump) in rad/s, shape
            (n_traj, t_slow). A scalar or 1-D (t_slow,) input is broadcast to a
            single trajectory. delta_omega[traj, step] is the detuning per round trip.
        t_slow: Number of round trips.
        beta: Dispersion coefficient list [beta2, beta3, beta4].
        kappa: Total cavity loss rate (rad/s).
        kappa_c: Coupling rate (rad/s).
        rng_key: PRNG key for initial noise seeding.
        n_tau: Number of fast-time grid points.
        config_path: Optional YAML config override.
        l_eff: Effective nonlinear interaction length.
        snapshot_interval: Round-trip interval for field snapshots and labels.
        e0_override: Optional warm-start intracavity field. ``None`` (default) =
            cold start from the analytic CW state + seeding noise (the historical
            behaviour). If given, it is the exact initial field E(tau) and NO
            seeding noise is added, so it can inject an analytic soliton ansatz for
            deterministic DKS access. Shape (n_tau,) is broadcast to every
            trajectory; shape (n_traj, n_tau) supplies a per-trajectory warm start.
        delta_t0_override: Optional warm-start thermal state DeltaT (K). ``None``
            (default) = start cold (DeltaT=0). Scalar broadcasts to all
            trajectories; shape (n_traj,) is per-trajectory.
        d_int_grid: Optional per-mode integrated dispersion D_int(mu) [rad/s] in
            FFT-bin order, shape (n_tau,). ``None`` (default) = build dispersion
            from the Taylor ``beta`` list (historical behavior). When given it is
            used verbatim as the linear-operator dispersion (broadcast across all
            trajectories) and the Taylor path is bypassed; use
            :func:`load_dint_grid` to build it from a measured dispersion CSV.
        n_substeps: Number of Strang split-step sub-steps per round trip
            (positive int, default 1). Each sub-step integrates over
            dt = t_r / n_substeps, shrinking the per-sub-step linear phase
            |D_int·dt| so the split-step spurious-sideband instability (which
            appears once |D_int·t_r| nears π at large |mu|) is pushed out of the
            physical band. n_substeps=1 reproduces the legacy single-step solver
            bit-for-bit. The pump drive is distributed across sub-steps so the
            per-round-trip total is unchanged; detuning, thermal update, energy
            balance, snapshots, and labels are computed once per round trip.
        dealias_two_thirds: If True, zero the Fourier modes with |mu| > n_tau/3
            after each nonlinear kick (standard 2/3 de-aliasing of the cubic
            nonlinearity). Default False (bit-identical legacy behaviour).
        edge_absorber: If True, multiply the field once per round trip by a
            super-Gaussian in mode space that is ~1 across the interior and ramps
            to strong attenuation over the outer ``edge_absorber_frac`` of |mu| on
            each side, damping energy at the (partly extrapolated) grid edges.
            Default False. Both toggles OFF reproduce the current solver exactly;
            production physical runs should set both ON.
        edge_absorber_frac: Outer fraction of |mu| (each side) over which the
            edge absorber ramps (default 0.12). At n_tau=8192 the onset is
            |mu| ~ 3604 and at 16384 ~ 7209, both beyond the physical window
            (|mu|<=2744) so the absorber never touches it.
        dispersion_validity_mask: If True, smoothly damp (once per fine step)
            the modes whose per-sub-step linear-phase MISMATCH
            |D_int - delta_omega|*dt_sub exceeds ``validity_phase_threshold``.
            The linear step applies the exact exponential, which is valid at any
            phase, so a large |D_int*t_r| does NOT by itself invalidate the map
            (an earlier |D_int*t_r|-keyed mask amputated real soliton-tail and
            dispersive-wave spectrum). The only genuine discrete-map artifact is
            spurious four-wave mixing that phase-matches when the mismatch phase
            per nonlinear kick nears 2*pi; this mask is an opt-in guard for
            coarse n_substeps=1 runs where that onset can fall inside the
            resolved band. With n_substeps>=4 the onset sits far outside the
            physical window and the mask should stay OFF. Default False
            (bit-identical).
        validity_phase_threshold: per-sub-step mismatch-phase threshold (rad)
            for the validity mask (default pi, below the ~2*pi spurious-FWM
            onset).
        fine_cadence_M: Advance the whole evolution (field, thermal ODE, detuning,
            pump, energy balance, masks) at the fine cadence dt = t_r / M instead
            of refreshing the thermal/detuning/energy once per round trip. This
            removes the residual t_r-periodic modulation of the mean-field map (a
            driver of parametric round-trip-map resonances) at fixed total
            physical time. Default 1 = bit-identical per-round-trip integrator.
        quantum_noise_enabled: Master switch for the quantum-vacuum Langevin
            drive (see the "Quantum noise" section above). ``None`` (default) =
            read the ``quantum_noise_enabled`` config key (itself defaulting to
            off); an explicit bool overrides the config. OFF is bit-identical to
            the legacy solver.
        quantum_noise_seed_vacuum_init: When the master switch is on, seed a
            COLD start with the half-photon-per-mode vacuum draw instead of the
            legacy 1e-3*|e_cw| noise. ``None`` (default) = read the
            ``quantum_noise_seed_vacuum_init`` config key (default on). Ignored
            when the master switch is off or when ``e0_override`` is given.

    Returns:
        Dictionary containing requested histories.
    """

    if int(n_substeps) < 1:
        raise ValueError(f"n_substeps must be a positive integer, got {n_substeps}.")
    if int(fine_cadence_M) < 1:
        raise ValueError(f"fine_cadence_M must be a positive integer, got {fine_cadence_M}.")
    n_substeps = int(n_substeps)
    fine_cadence_M = int(fine_cadence_M)
    dealias_two_thirds = bool(dealias_two_thirds)
    edge_absorber = bool(edge_absorber)
    dispersion_validity_mask = bool(dispersion_validity_mask)
    validity_phase_threshold = float(validity_phase_threshold)
    edge_absorber_frac = float(edge_absorber_frac)

    thermal = _thermal_params(config_path)
    physical = _load_config(config_path)
    gamma = float(physical.get("gamma_LLE_per_J_per_s"))
    assert 1e15 < gamma < 1e25, (
        f"gamma_LLE = {gamma:.3e} J⁻¹s⁻¹ outside expected range (config FSR ≈ 24.6 GHz). "
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
        # max() guards the pin=0 (undriven cavity, e.g. vacuum-equilibrium
        # runs) case against ZeroDivisionError in the message formatting.
        _w.warn(
            f"pin={pin*1e3:.1f} mW is {_p_th/max(pin, 1e-300):.0f}x below the MI threshold "
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
    thermal = {k: jnp.array(v, dtype=jnp.float64) for k, v in thermal.items()}

    # delta_omega is a per-trajectory, per-round-trip sweep of shape (n_traj, t_slow)
    # (it is vmapped over axis 0 and indexed as delta_omega[step] inside the solver).
    # Accept convenience inputs and normalize to 2-D:
    #   scalar         -> (1, t_slow) constant detuning
    #   1-D (t_slow,)  -> (1, t_slow) single-trajectory sweep
    #   2-D            -> used as-is, must be (n_traj, t_slow)
    delta_omega_arr = jnp.asarray(delta_omega, dtype=jnp.float64)
    if delta_omega_arr.ndim == 0:
        delta_arr = jnp.broadcast_to(delta_omega_arr, (1, int(t_slow)))
    elif delta_omega_arr.ndim == 1:
        if delta_omega_arr.shape[0] != int(t_slow):
            raise ValueError(
                f"1-D delta_omega must have length t_slow={t_slow}, "
                f"got {delta_omega_arr.shape[0]}."
            )
        delta_arr = delta_omega_arr[None, :]
    elif delta_omega_arr.ndim == 2:
        if delta_omega_arr.shape[1] != int(t_slow):
            raise ValueError(
                f"2-D delta_omega must be (n_traj, t_slow={t_slow}), "
                f"got {tuple(delta_omega_arr.shape)}."
            )
        delta_arr = delta_omega_arr
    else:
        raise ValueError(f"delta_omega must be 0/1/2-D, got ndim={delta_omega_arr.ndim}.")
    beta_arr = tuple(float(b) for b in beta)

    # Guard: catch accidental use of fiber-optics β₂ units (s²/m).
    # LLE β₂ = D₂/D₁² ≈ 1e-18–1e-16 s for typical microresonators.
    # Only applies to the Taylor beta path; when a measured d_int_grid drives the
    # dispersion, beta is unused so this range check is skipped.
    if d_int_grid is None and len(beta_arr) >= 1 and beta_arr[0] != 0.0:
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
    # One more subkey for the quantum vacuum noise, split from the LEFTOVER
    # chain `key` — NOT by widening the 3-way split above, which would change
    # key_field/key_noise and break the bit-identity of the flag-off RNG stream.
    key, key_qnoise = jax.random.split(key, 2)

    # --- Quantum vacuum noise (arXiv:2604.05897 Sec. V.B.2, Eq. 126) ---------
    # Resolve the flags: explicit kwarg wins, else the flat config keys. The
    # config encodes the booleans as 0/1 (physical_parameters leaves must stay
    # numeric — see tests/test_config.py); accept bool or 0/1 and nothing else.
    def _as_flag(name: str, value) -> bool:
        assert isinstance(value, (bool, int, np.integer)) and int(value) in (0, 1), (
            f"{name} must be boolean-valued (bool or 0/1), got {value!r}."
        )
        return bool(value)

    if quantum_noise_enabled is None:
        quantum_noise_enabled = physical.get("quantum_noise_enabled", 0)
    qn_enabled = _as_flag("quantum_noise_enabled", quantum_noise_enabled)
    if quantum_noise_seed_vacuum_init is None:
        quantum_noise_seed_vacuum_init = physical.get(
            "quantum_noise_seed_vacuum_init", 1
        )
    qn_seed_vacuum = _as_flag(
        "quantum_noise_seed_vacuum_init", quantum_noise_seed_vacuum_init
    )

    hbar_omega0 = hbar_omega0_from_config(physical)
    dt_fine = t_r / fine_cadence_M
    if qn_enabled:
        # Per-quadrature time-domain injection std per fine step (see the
        # "Quantum noise" docstring section for the Parseval derivation).
        qnoise_scale = float(math.sqrt(hbar_omega0 * kappa * n_tau * dt_fine / 4.0))
        # Perturbation sanity numbers, logged once per solve. The XPM
        # cross-check 2*gamma*U_vac == gamma*hbar_omega0*n_tau holds identically
        # (U_vac = n_tau*hbar_omega0/2), so both are printed from one formula.
        u_vac = n_tau * hbar_omega0 / 2.0                       # J
        kerr_vac = gamma * u_vac                                # rad/s (SPM)
        inj_rt = math.sqrt(hbar_omega0 * kappa * n_tau * t_r / 4.0)
        print(
            f"[quantum noise] hbar*omega0 = {hbar_omega0:.4e} J, "
            f"U_vac = n_tau*hbar*omega0/2 = {u_vac:.3e} J, "
            f"Kerr shift gamma*U_vac = {kerr_vac:.3e} rad/s "
            f"= {kerr_vac / kappa:.2e}*kappa (XPM cross-check 2*gamma*U_vac = "
            f"{2.0 * kerr_vac:.3e} rad/s), per-round-trip per-quadrature "
            f"injection std = {inj_rt:.3e} (field units), per-fine-step "
            f"(dt_fine = t_r/{fine_cadence_M}) std = {qnoise_scale:.3e}"
        )
        # The steady vacuum background must sit well below the labeler's OFF
        # floor, or OFF/CW labels can flip once the background thermalizes.
        _dw_max_qn = float(np.max(np.abs(np.asarray(delta_arr))))
        _off_floor = physical_off_floor(
            float(kappa), float(kappa_c), float(pin), _dw_max_qn
        )
        if not (u_vac < 0.5 * _off_floor):
            import warnings as _w
            _w.warn(
                f"Vacuum background U_vac = n_tau*hbar*omega0/2 = {u_vac:.3e} J "
                f"is not < 0.5x the labeler OFF floor "
                f"physical_off_floor(...) = {_off_floor:.3e} J: the thermalized "
                f"vacuum background may flip OFF/CW labels. Consider a larger "
                f"off_fraction for the state labeler (or a smaller n_tau).",
                stacklevel=2,
            )
    else:
        qnoise_scale = 0.0

    # Per-trajectory quantum-noise keys (independent of the legacy key chains).
    # Built unconditionally so the _PER_TRAJ signature is uniform; when the
    # flag is off the traced argument is unused and dead-code-eliminated.
    key_qnoise_inj, key_qnoise_seed = jax.random.split(key_qnoise, 2)
    qnoise_keys = jax.random.split(key_qnoise_inj, delta_arr.shape[0])

    # --- Generate per-trajectory noise sequences ---
    from simulator.noise_models import TotalNoise, _load_config as _nm_load_cfg
    _nm_cfg = _nm_load_cfg(config_path)
    _noise_model = TotalNoise(_nm_cfg)

    # shape: (n_traj, t_slow)  — one AR(1) sequence per trajectory
    def _gen_noise(key):
        return _noise_model.sample(key, int(t_slow))

    # Noise models emit float32; upcast the detuning-noise sequences to float64
    # so the delta_omega_eff axis (and thus the linear-operator phase) is fully
    # double precision inside the SSFM loop.
    noise_sequences = jax.vmap(_gen_noise)(noise_keys).astype(jnp.float64)  # (n_traj, t_slow)
    
    n_traj = delta_arr.shape[0]

    # Warm-start handling. Cold start (override None) uses an all-zero e0, which the
    # low-level solver detects (via jnp.all(e0 == 0)) to build the analytic CW state
    # plus seeding noise. A non-zero e0_override is passed through verbatim as the
    # exact initial field (no seeding noise), enabling deterministic soliton seeding.
    if e0_override is None:
        if qn_enabled and qn_seed_vacuum:
            # Vacuum-scale cold-start seed (half a photon per mode): analytic CW
            # state + one complex Gaussian draw of per-sample variance
            # sigma0^2 = hbar*omega0*n_tau/2 (per-quadrature std sigma0_q =
            # sqrt(hbar*omega0*n_tau/4)), so by Parseval <n_mu> = 1/2 at t=0.
            # Built HERE (host-side, Python branch) and handed to the solver as
            # a verbatim warm start, so the legacy 1e-3*|e_cw| traced seed path
            # is bypassed WITHOUT being restructured — the flag-off path stays
            # bit-identical.
            _amp0 = math.sqrt(max(float(kappa_c) * float(pin), 0.0))
            _dw0 = np.asarray(delta_arr[:, 0], dtype=np.float64)     # (n_traj,)
            _den0 = (float(kappa) / 2.0) ** 2 + _dw0 ** 2
            e_cw_traj = jnp.asarray(
                _amp0 * (float(kappa) / 2.0) / _den0
                - 1j * _amp0 * _dw0 / _den0,
                dtype=jnp.complex128,
            )                                                        # (n_traj,)
            sigma0_q = math.sqrt(hbar_omega0 * n_tau / 4.0)

            def _vac_draw(k):
                k_re, k_im = jax.random.split(k)
                return sigma0_q * (
                    jax.random.normal(k_re, (n_tau,), dtype=jnp.float64)
                    + 1j * jax.random.normal(k_im, (n_tau,), dtype=jnp.float64)
                )

            vac = jax.vmap(_vac_draw)(jax.random.split(key_qnoise_seed, n_traj))
            e0_init = (e_cw_traj[:, None] + vac).astype(jnp.complex128)
        else:
            e0_init = jnp.zeros((n_traj, n_tau), dtype=jnp.complex128)
    else:
        # Accept complex64 (or real) warm-start fields but upcast to complex128.
        e0_arr = jnp.asarray(e0_override, dtype=jnp.complex128)
        if e0_arr.ndim == 1:
            if e0_arr.shape[0] != n_tau:
                raise ValueError(
                    f"e0_override 1-D length must be n_tau={n_tau}, got {e0_arr.shape[0]}."
                )
            e0_init = jnp.broadcast_to(e0_arr, (n_traj, n_tau))
        elif e0_arr.ndim == 2:
            if e0_arr.shape != (n_traj, n_tau):
                raise ValueError(
                    f"e0_override 2-D shape must be (n_traj={n_traj}, n_tau={n_tau}), "
                    f"got {tuple(e0_arr.shape)}."
                )
            e0_init = e0_arr
        else:
            raise ValueError(f"e0_override must be 1/2-D, got ndim={e0_arr.ndim}.")

    if delta_t0_override is None:
        delta_t0_init = jnp.zeros((n_traj,), dtype=jnp.float64)
    else:
        dt0_arr = jnp.asarray(delta_t0_override, dtype=jnp.float64)
        if dt0_arr.ndim == 0:
            delta_t0_init = jnp.broadcast_to(dt0_arr, (n_traj,))
        elif dt0_arr.ndim == 1 and dt0_arr.shape[0] == n_traj:
            delta_t0_init = dt0_arr
        else:
            raise ValueError(
                f"delta_t0_override must be scalar or shape (n_traj={n_traj},), "
                f"got shape {tuple(dt0_arr.shape)}."
            )

    # Physically-scaled OFF floor: f·U_cw,min with U_cw,min at the largest |δω|
    # in the sweep. Built from the concrete config so OFF tracks the energy scale
    # rather than a magic constant. Cached on the params so identity is stable.
    delta_omega_max = float(np.max(np.abs(np.asarray(delta_arr))))
    state_labeler = _physical_state_labeler(
        float(kappa), float(kappa_c), float(pin), delta_omega_max
    )

    # Optional measured per-mode dispersion: cast to the solver dtype and pin the
    # shape to (n_tau,). Passed with a None (broadcast) vmap axis so it is shared
    # across all trajectories rather than mapped per trajectory.
    if d_int_grid is None:
        d_int_grid_arr = None
    else:
        d_int_grid_arr = jnp.asarray(d_int_grid, dtype=jnp.float64)
        if d_int_grid_arr.shape != (int(n_tau),):
            raise ValueError(
                f"d_int_grid must have shape (n_tau={n_tau},), "
                f"got {tuple(d_int_grid_arr.shape)}."
            )

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
        state_labeler,
        noise_sequences,
        e0_init,
        delta_t0_init,
        d_int_grid_arr,
        n_substeps,
        dealias_two_thirds,
        edge_absorber,
        edge_absorber_frac,
        dispersion_validity_mask,
        validity_phase_threshold,
        fine_cadence_M,
        qnoise_keys,
        qnoise_scale,
        qn_enabled,
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
