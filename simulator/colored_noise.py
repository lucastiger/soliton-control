"""General PSD-specified colored-noise engine and target PSD models.

This module generalizes the repository's AR(1)-only noise machinery: any
stationary Gaussian noise channel can now be synthesized from an arbitrary
target one-sided power spectral density S(f), supplied as

* a Python callable ``S(f) -> array`` (one-sided, units X**2/Hz for a
  process of units X, e.g. (rad/s)**2/Hz for a detuning noise),
* a named analytic model built by the factories below
  (:func:`single_pole_psd`, :func:`kondratiev_gorodetsky_psd`), or
* a two-column CSV (f [Hz], S) loaded with :func:`csv_psd`
  (log-log interpolated, clamped flat outside the tabulated span).

Everything here is HOST-SIDE numpy in float64: FFT synthesis of long
sequences is cheap on the host and must not depend on the JAX x64 flag.
Determinism is anchored to JAX PRNG keys via :func:`np_generator_from_key`
(the full key data is folded into a ``numpy.random.SeedSequence``), so
distinct JAX keys give independent, fully reproducible numpy streams.

No matplotlib import (this module is loaded by the solver hot path).

Synthesis recipe (exact; see :func:`synthesize_from_psd`)
---------------------------------------------------------
For a length-``N`` real sequence at sample rate ``f_s``: on the rfft bins
``k = 0..N/2`` with ``f_k = k*f_s/N``, draw ``zeta_k`` as a standard complex
normal (E|zeta_k|**2 = 1) for ``0 < k < N/2`` and a standard real normal at
``k = 0`` and (even ``N``) ``k = N/2``; set

    c_k = zeta_k * sqrt(S(f_k) * f_s * N / 2)

and ``x = irfft(c, n=N)``. Then ``Var(x) = sum_k S(f_k)*Delta_f ~
integral_0^{f_s/2} S(f) df`` (the one-sided convention) and the Welch PSD of
``x`` reproduces ``S(f_k)`` bin by bin. The DC bin is clamped,
``S(0) := S(f_1)``, so 1/f-type spectra cannot inject a divergent DC power;
DC and Nyquist each carry ~1/N of the variance.

Aliasing note: the channels built from this engine are sampled once per
round trip at ``f_s = 1/t_r ~ 24.6 GHz``, far above any thermal band
(the thermorefractive spectrum has decayed by many orders of magnitude at
f_s/2), so synthesis-band truncation/aliasing of the physical spectra is
negligible.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_K_B = 1.380649e-23  # Boltzmann constant [J/K]


# ---------------------------------------------------------------------------
# Determinism: JAX key -> numpy Generator
# ---------------------------------------------------------------------------
def np_generator_from_key(key) -> np.random.Generator:
    """Deterministic host-side numpy ``Generator`` derived from a JAX PRNG key.

    The full key data is folded into a ``SeedSequence`` (little-endian byte
    concatenation of the uint32 key words), so distinct JAX keys give
    independent, reproducible numpy streams regardless of the JAX backend or
    x64 flag. This is the SINGLE seeding convention used by every host-side
    noise synthesis in the repository (pump noise adopted it first; the
    colored TRN/segment-continuity paths reuse it), and it is pinned by a
    determinism test in tests/test_colored_noise.py.
    """
    import jax  # local import: keep this module importable without tracing

    data = np.asarray(jax.random.key_data(key), dtype=np.uint32).ravel()
    entropy = int.from_bytes(data.tobytes(), "little")
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def synthesize_from_psd(rng: np.random.Generator, n: int, psd, f_s: float,
                        clamp_dc: bool = True) -> np.ndarray:
    """Length-``n`` float64 real sequence whose Welch PSD matches ``psd``.

    Implements the module-docstring recipe exactly: rfft bins
    ``f_k = k*f_s/n``; ``zeta_k`` standard complex normal for ``0 < k < n/2``,
    standard real normal at ``k = 0`` and (even n) ``k = n/2``;
    ``c_k = zeta_k*sqrt(S(f_k)*f_s*n/2)``; ``x = irfft(c, n=n)``. Then
    ``Var(x) = integral_0^{f_s/2} S df`` and the sequence is stationary from
    sample 0 (no AR(1)-style start-up transient — this is what the
    segment-continuity fix relies on).

    Args:
        rng: numpy Generator (see :func:`np_generator_from_key`). The draw
            order (all real parts, then all imaginary parts) is part of the
            determinism contract.
        n: Sequence length. ``n < 2`` returns zeros.
        psd: Callable ``S(f) -> array`` one-sided PSD [X**2/Hz]; evaluated on
            the non-negative rfft frequency grid. Negative values are
            clipped to 0.
        f_s: Sample rate [Hz].
        clamp_dc: If True (default) clamp ``S(0) := S(f_1)`` so 1/f-type
            spectra stay finite at DC. ``False`` preserves whatever the
            callable returns at f = 0 (legacy pump-noise behaviour, where
            the callables clamp their own DC bin).

    Returns:
        (n,) float64 array with ``Var(x) ~ integral_0^{f_s/2} S(f) df``.
    """
    n = int(n)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    f = np.fft.rfftfreq(n, d=1.0 / float(f_s))               # (n//2 + 1,)
    s = np.asarray(psd(f), dtype=np.float64)
    if s.shape != f.shape:
        s = np.broadcast_to(s, f.shape).astype(np.float64)
    if clamp_dc:
        s = s.copy()
        s[0] = s[1]                                          # S(0) := S(f_1)
    amp = np.sqrt(np.maximum(s, 0.0) * float(f_s) * n / 2.0)
    zr = rng.standard_normal(f.size)
    zi = rng.standard_normal(f.size)
    z = (zr + 1j * zi) / math.sqrt(2.0)                      # E|z|^2 = 1
    z[0] = zr[0]                                             # DC bin real
    if n % 2 == 0:
        z[-1] = zi[-1]                                       # Nyquist bin real
    return np.fft.irfft(z * amp, n=n).astype(np.float64)


def integrate_psd(psd, f_lo: float, f_hi: float, n_grid: int = 20_000) -> float:
    """``integral_{f_lo}^{f_hi} S(f) df`` on a log-spaced grid (trapezoid).

    Used to (a) renormalize the Kondratiev–Gorodetsky asymptotic PSD to the
    thermodynamic variance and (b) provide the variance target of the engine
    unit tests. ``f_lo`` must be > 0 (log grid); pick it low enough that the
    omitted band [0, f_lo] is negligible for the model at hand (the K-G
    spectrum behaves as f**-1/2 at low f, so its [0, f_lo] contribution
    scales as sqrt(f_lo) and vanishes as f_lo -> 0).
    """
    if not (f_lo > 0.0 and f_hi > f_lo):
        raise ValueError(f"need 0 < f_lo < f_hi, got ({f_lo!r}, {f_hi!r}).")
    f = np.logspace(math.log10(f_lo), math.log10(f_hi), int(n_grid))
    s = np.asarray(psd(f), dtype=np.float64)
    return float(np.trapezoid(s, f))


# ---------------------------------------------------------------------------
# Named PSD models
# ---------------------------------------------------------------------------
def single_pole_psd(variance: float, tau: float):
    """One-sided single-pole (Lorentzian) PSD with total variance ``variance``.

        S(f) = 4*variance*tau / (1 + (2*pi*f*tau)**2)   [X**2/Hz]

    ``integral_0^inf S df = variance`` — the spectral twin of the repository's
    AR(1) generator (an AR(1) with correlation time tau sampled at
    dt << tau has exactly this spectrum). Units of S follow the units of
    ``variance`` (K**2 -> K**2/Hz, etc.).
    """
    variance = float(variance)
    tau = float(tau)

    def _psd(f):
        f = np.asarray(f, dtype=np.float64)
        return (4.0 * variance * tau) / (1.0 + (2.0 * np.pi * f * tau) ** 2)

    return _psd


def kondratiev_gorodetsky_psd(
    T_k: float,
    kappa_th: float,
    rho: float,
    cp: float,
    R: float,
    d_a: float,
    d_b: float,
    mode_volume: float,
    f_max: float,
    f_lo: float = 1.0,
):
    """Analytic WGM thermorefractive temperature PSD, arXiv:2604.05897 Eq. 130.

    One-sided S_dT(omega) of the mode-averaged temperature fluctuation of a
    whispering-gallery/ring mode of radius ``R`` and Gaussian mode
    half-dimensions ``d_a`` (major) and ``d_b`` (minor):

        S_dT(omega) = [k_B*T**2 / sqrt(pi**3 * kappa_th * rho * C * omega)]
                      * [R * sqrt(d_a**2 - d_b**2)]**-1
                      * [1 + (omega*tau_d)**(3/4)]**-2,
        tau_d = (pi/4)**(1/3) * (rho*C/kappa_th) * d_b**2,

    with k_B the Boltzmann constant, T the temperature [K], kappa_th the
    thermal conductivity [W/(m K)], rho the density [kg/m^3], C the specific
    heat [J/(kg K)]. The formula DEGENERATES for d_a ~ d_b (the prefactor
    diverges); the paper notes a rescaling for that limit, so this factory
    asserts ``d_a >= 1.2*d_b`` and refers the user to the paper's remark
    otherwise.

    Renormalization (documented behaviour): the analytic form is an
    ASYMPTOTIC matching of the low- and high-frequency limits, so its
    absolute integral does not exactly reproduce the thermodynamic variance.
    The returned PSD is therefore rescaled by a single constant so that

        integral_0^{f_max} S_dT(f) df  ==  k_B*T**2/(rho*C*V)      (Eq. 129)

    with V = ``mode_volume`` — the total variance is pinned to
    thermodynamics and the analytic curve only supplies the SHAPE.
    ``f_max`` should be the synthesis Nyquist (f_s/2). Returns
    ``(psd_callable, var_eq129)`` where the callable maps f [Hz] to
    S_dT [K**2/Hz]; at ``T_k = 0`` both are identically zero.
    """
    T_k = float(T_k)
    if not (d_a > 0.0 and d_b > 0.0 and R > 0.0):
        raise ValueError(
            f"Kondratiev-Gorodetsky geometry must be positive: "
            f"R={R!r} m, d_a={d_a!r} m, d_b={d_b!r} m."
        )
    if not (d_a >= 1.2 * d_b):
        raise ValueError(
            f"Kondratiev-Gorodetsky PSD requires d_a >= 1.2*d_b "
            f"(got d_a={d_a:.3e} m, d_b={d_b:.3e} m): the Eq. 130 prefactor "
            f"[R*sqrt(d_a^2-d_b^2)]^-1 degenerates for d_a ~ d_b. See the "
            f"paper's rescaling remark for near-circular modes "
            f"(arXiv:2604.05897, discussion around Eq. 130)."
        )
    var_eq129 = _K_B * T_k**2 / (float(rho) * float(cp) * float(mode_volume))
    if var_eq129 == 0.0:                    # T_k = 0: noise-off convention
        return (lambda f: np.zeros_like(np.asarray(f, dtype=np.float64)),
                0.0)

    tau_d = (math.pi / 4.0) ** (1.0 / 3.0) * (
        float(rho) * float(cp) / float(kappa_th)
    ) * float(d_b) ** 2
    pref = (_K_B * T_k**2) / (
        float(R) * math.sqrt(float(d_a) ** 2 - float(d_b) ** 2)
    )

    def _shape(f):
        f = np.asarray(f, dtype=np.float64)
        omega = 2.0 * np.pi * np.maximum(f, 1e-30)
        s = (
            pref
            / np.sqrt(math.pi**3 * float(kappa_th) * float(rho) * float(cp)
                      * omega)
            / (1.0 + (omega * tau_d) ** 0.75) ** 2
        )
        return s

    # Pin integral_0^{f_max} S df to the Eq. 129 thermodynamic variance. The
    # shape ~ f**-1/2 below 1/tau_d, so the omitted [0, f_lo] band carries
    # O(sqrt(f_lo)) of the norm — negligible at the default f_lo = 1 Hz
    # against f_max ~ 1e10 Hz.
    norm = integrate_psd(_shape, f_lo=float(f_lo), f_hi=float(f_max))
    scale = var_eq129 / norm

    def _psd(f):
        return scale * _shape(f)

    return _psd, var_eq129


def csv_psd(csv_path: str | Path):
    """One-sided PSD from a two-column CSV ``f [Hz], S`` (e.g. measured/FEM).

    Follows the Huang et al. 2019 style of tabulated thermorefractive
    spectra: values are interpolated LINEARLY IN LOG-LOG space (power laws
    become straight lines) and clamped FLAT outside the tabulated span
    (S(f < f_min) = S(f_min), S(f > f_max) = S(f_max)). f = 0 is guarded by
    the low-side clamp (the engine additionally clamps its DC bin). Rows
    with non-positive f or S are dropped (log space); at least two valid
    rows are required. Units of S are whatever the file tabulates — the
    caller selects the interpretation (S_dT [K^2/Hz] vs S_domega
    [(rad/s)^2/Hz]) via the ``trn_csv_units`` config key.
    """
    path = Path(csv_path)
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"{path}: expected two columns (f_hz, S), got shape "
                         f"{data.shape}.")
    f_tab, s_tab = data[:, 0], data[:, 1]
    keep = (f_tab > 0.0) & (s_tab > 0.0) & np.isfinite(f_tab) & np.isfinite(s_tab)
    f_tab, s_tab = f_tab[keep], s_tab[keep]
    if f_tab.size < 2:
        raise ValueError(
            f"{path}: need >= 2 rows with f > 0 and S > 0 for log-log "
            f"interpolation, got {f_tab.size}."
        )
    order = np.argsort(f_tab)
    log_f, log_s = np.log10(f_tab[order]), np.log10(s_tab[order])

    def _psd(f):
        f = np.asarray(f, dtype=np.float64)
        # Guard f <= 0 before the log; those bins land on the flat low clamp.
        lf = np.log10(np.maximum(f, 10.0 ** log_f[0]))
        return 10.0 ** np.interp(lf, log_f, log_s)   # np.interp clamps flat

    return _psd
