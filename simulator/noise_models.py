"""Noise model implementations for TFLN simulation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import welch


_DEF_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and return physical parameters dict."""
    cfg_path = Path(config_path) if config_path is not None else _DEF_CFG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("physical_parameters", {})


def _ar1_samples(key, N, tau_corr, sigma_physical, t_r):
    """Generate AR(1) samples with target stationary variance."""
    alpha = jnp.exp(-t_r / tau_corr)
    sigma_step = sigma_physical * jnp.sqrt(1 - alpha**2)
    xi = jax.random.normal(key, shape=(N,), dtype=jnp.float32)

    def scan_fn(x_prev, xi_n):
        x_next = alpha * x_prev + sigma_step * xi_n
        return x_next, x_next

    _, samples = jax.lax.scan(scan_fn, jnp.zeros((), dtype=jnp.float32), xi)
    return samples


class TRNoise:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t_r = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.omega_0 = 2.0 * math.pi * 299_792_458.0 / float(cfg.get("pump_wavelength_m", 1.55e-6))
        self.n0 = float(cfg.get("n0", 2.2))
        self.dn_dT = float(cfg.get("dn_dT_per_k", 4.0e-5))
        self.tau_th = float(cfg.get("tau_th_s", 5.0e-6))
        self.rho = float(cfg.get("rho_kg_per_m3", 4.64e3))
        self.cp = float(cfg.get("Cp_j_per_kg_k", 700.0))
        self.v = float(cfg.get("mode_volume_m3", 1.0e-15))
        self.kappa_th = float(cfg.get("kappa_th_w_per_m_k", 4.6))
        self.T_k = float(cfg.get("T_k", 300.0))
        self.k_b = 1.380649e-23
        self.var_delta_t = self.k_b * self.T_k**2 / (self.rho * self.cp * self.v)
        self.sigma_trn = (self.omega_0 / self.n0) * self.dn_dT * math.sqrt(self.var_delta_t)

    def sample(self, key, N) -> jnp.ndarray:
        return _ar1_samples(key, N, self.tau_th, self.sigma_trn, self.t_r)

    def psd(self, f) -> jnp.ndarray:
        s_delta_t = (
            (4.0 * self.k_b * self.T_k**2 * self.tau_th) / (self.rho * self.cp * self.v)
        ) / (1.0 + (2.0 * jnp.pi * f * self.tau_th) ** 2)
        return ((self.omega_0 / self.n0) * self.dn_dT) ** 2 * s_delta_t


class PyroEONoise:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t_r = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.omega_0 = 2.0 * math.pi * 299_792_458.0 / float(cfg.get("pump_wavelength_m", 1.55e-6))
        self.n0 = float(cfg.get("n0", 2.2))
        self.r33 = float(cfg.get("eo_r33_m_per_v", 3.1e-11))
        self.p = float(cfg.get("pyroelectric_coeff_c_per_m2_k", 9.6e-2))
        self.tau_th = float(cfg.get("tau_th_s", 5.0e-6))
        self.rho = float(cfg.get("rho_kg_per_m3", 4.64e3))
        self.cp = float(cfg.get("Cp_j_per_kg_k", 700.0))
        self.v = float(cfg.get("mode_volume_m3", 1.0e-15))
        self.kappa_th = float(cfg.get("kappa_th_w_per_m_k", 4.6))
        self.T_k = float(cfg.get("T_k", 300.0))
        self.eps0 = 8.8541878128e-12
        self.k_b = 1.380649e-23
        self.var_delta_t = self.k_b * self.T_k**2 / (self.rho * self.cp * self.v)

        self.eps_r_z = float(cfg.get("eps_r_z", 28.0))
        # Geometric screening from dielectric boundary conditions in thin-film stack.
        # 1D approximation: E_LN = P / (ε₀ * ε_r_eff) where ε_r_eff accounts for
        # the fraction of field extending into cladding layers.
        _t_ln  = float(cfg.get("t_ln_m", 4.0e-7))
        _t_top = float(cfg.get("t_clad_top_m", 1.0e-6))
        _t_bot = float(cfg.get("t_clad_bot_m", 2.0e-6))
        _er_top = float(cfg.get("eps_r_clad_top", 1.0))
        _er_bot = float(cfg.get("eps_r_clad_bot", 3.9))
        self.eps_r_eff = (
            self.eps_r_z
            + _er_top * (_t_top / _t_ln)
            + _er_bot * (_t_bot / _t_ln)
        )
        # η_geom = eps_r_z / eps_r_eff  (implicitly encoded by using eps_r_eff)
        self.sigma_pyroeo = (
            self.omega_0 * self.n0**2 * self.r33 * self.p
            / (2.0 * self.eps0 * self.eps_r_eff)
        ) * math.sqrt(self.var_delta_t)

    def sample(self, key, N) -> jnp.ndarray:
        return _ar1_samples(key, N, self.tau_th, self.sigma_pyroeo, self.t_r)

    def psd(self, f) -> jnp.ndarray:
        s_delta_t = (
            (4.0 * self.k_b * self.T_k**2 * self.tau_th) / (self.rho * self.cp * self.v)
        ) / (1.0 + (2.0 * jnp.pi * f * self.tau_th) ** 2)
        scale = (self.omega_0 * self.n0**2 * self.r33 * self.p / (2.0 * self.eps0 * self.eps_r_eff)) ** 2
        return scale * s_delta_t


class TCCRNoise:
    def __init__(self, cfg):
        self.t_r         = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.omega_0     = 2.0 * math.pi * 299_792_458.0 / float(cfg.get("pump_wavelength_m", 1.55e-6))
        self.tau_carrier = float(cfg.get("tau_carrier_s", 1.0e-7))
        self.k_b         = 1.380649e-23
        self.T_k         = float(cfg.get("T_k", 300.0))

        # Physical path: surface carrier shot noise → EO frequency shift
        n_s       = float(cfg.get("surface_state_density_per_m2", 1.0e16))   # m⁻²
        r33       = float(cfg.get("eo_r33_m_per_v",  3.1e-11))               # m/V
        n0        = float(cfg.get("n0", 2.2))
        eps0      = 8.8541878128e-12
        eps_r_eff = float(cfg.get("eps_r_z", 28.0))   # simplified; use PyroEO value for full model
        A_eff     = float(cfg.get("effective_mode_area_m2", 1.0e-12))         # m²
        t_ln      = float(cfg.get("t_ln_m", 4.0e-7))                         # m
        e_charge  = 1.602176634e-19                                            # C

        # Equilibrium surface carrier number within mode footprint
        N_s_eq = n_s * A_eff                                                   # dimensionless

        # EO frequency shift per carrier [rad/s per carrier]
        # Derivation: delta_n = -n0^3 * r33 * E / 2  =>  delta_omega = omega_0 * delta_n / n0
        #                     = -omega_0 * n0 ^ 2 * r33 * E / 2
        # E_per_carrier is already in V/m; t_ln does NOT appear here.
        E_per_carrier = e_charge / (eps0 * eps_r_eff * A_eff)   # V/m per carrier
        dw_dNs = -self.omega_0 * n0**2 * r33 * E_per_carrier / 2.0   # rad/s per carrier  ← NO t_ln

        # Two-sided TCCR PSD at f=0: S0 = (dω/dNs)² · N_s_eq · 2·τ_carrier
        self.s0_tccr    = dw_dNs**2 * N_s_eq * 2.0 * self.tau_carrier        # (rad/s)²/Hz ✓
        self.var_tccr   = self.s0_tccr / (2.0 * self.tau_carrier)             # stationary variance
        self.sigma_tccr = math.sqrt(max(self.var_tccr, 0.0))

        # Sanity: for chi2 platforms (e.g. TFLN) sigma_tccr ~ [1e4, 1e11] rad/s.
        # sigma_tccr == 0 is the expected SiN case (r33 = 0): skip the warning.
        if self.sigma_tccr > 0.0 and not (1e4 < self.sigma_tccr < 1e11):
            import warnings
            warnings.warn(
                f"TCCRNoise.sigma_tccr = {self.sigma_tccr:.2e} rad/s is outside the "
                f"expected physical range [1e4, 1e11] rad/s. "
                f"Check surface_state_density_per_m2 and eo_r33_m_per_v in config.",
                stacklevel=2,
            )

        kappa_estimate = 2.0 * 2.0 * math.pi * 299_792_458.0 / (
            float(cfg.get("pump_wavelength_m", 1.55e-6)) * float(cfg.get("intrinsic_q", 2e6))
        )
        if self.sigma_tccr > kappa_estimate:
            import warnings
            warnings.warn(
                f"sigma_tccr ({self.sigma_tccr:.2e} rad/s) > kappa ({kappa_estimate:.2e} rad/s). "
                f"TCCR noise is non-perturbative and will destabilize all solitons. "
                f"Reduce surface_state_density_per_m2 (currently {n_s:.1e} m^-2) or calibrate "
                f"against the Yu lab's measured noise floor before generating the training dataset.",
                stacklevel=2,
            )


    def sample(self, key, N) -> jnp.ndarray:
        return _ar1_samples(key, N, self.tau_carrier, self.sigma_tccr, self.t_r)

    def psd(self, f) -> jnp.ndarray:
        return self.s0_tccr / (1.0 + (2.0 * jnp.pi * f * self.tau_carrier) ** 2)


class TotalNoise:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trn = TRNoise(cfg)
        self.pyroeo = PyroEONoise(cfg)
        self.tccr = TCCRNoise(cfg)
        self.t_r = self.trn.t_r
        self.omega_0 = self.trn.omega_0
        self.n0 = self.trn.n0
        self.dn_dT = self.trn.dn_dT
        self.r33 = self.pyroeo.r33
        self.p = self.pyroeo.p
        
        self.eps0 = self.pyroeo.eps0
        self.eps_r_z = self.pyroeo.eps_r_z
        self.eps_r_eff = self.pyroeo.eps_r_eff
        
        self.tau_th = self.trn.tau_th
        self.var_delta_t = self.trn.var_delta_t
        self.tau_carrier = self.tccr.tau_carrier

    def sample(self, key, N) -> jnp.ndarray:
        key_thermal, key_tccr = jax.random.split(key, 2)
        temp_noise = _ar1_samples(key_thermal, N, self.tau_th, math.sqrt(self.var_delta_t), self.t_r)
        trn_noise = (self.omega_0 / self.n0 * self.dn_dT) * temp_noise
        pyroeo_noise = (self.omega_0 * self.n0**2 * self.r33 * self.p / (2.0 * self.eps0 * self.eps_r_eff)) * temp_noise
        tccr_noise = self.tccr.sample(key_tccr, N)
        
        # Sign convention: PyroEO *partially cancels* TRN for z-cut TFLN with
        # air top-cladding (Yu lab geometry).  For SiO₂-clad or flipped substrate,
        # the sign of pyroeo_noise may need to flip.  Verify against Fig. 2 of the
        # TCCR paper (DOI to be added) before generating the training dataset.
        return (trn_noise - pyroeo_noise + tccr_noise).astype(jnp.float32)


def _np_generator_from_key(key) -> np.random.Generator:
    """Deterministic host-side numpy Generator derived from a JAX PRNG key.

    Pump-noise synthesis is done on the HOST in float64 (FFT synthesis of long
    sequences is cheap in numpy and must not depend on the jax x64 flag, which
    may be unset when this module is imported standalone). The full key data is
    folded into a SeedSequence, so distinct JAX keys give independent, fully
    reproducible numpy streams.
    """
    data = np.asarray(jax.random.key_data(key), dtype=np.uint32).ravel()
    entropy = int.from_bytes(data.tobytes(), "little")
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _synthesize_from_onesided_psd(rng: np.random.Generator, n: int, psd_fn,
                                  f_s: float) -> np.ndarray:
    """Real sequence x (n,) float64 with one-sided target PSD ``psd_fn(f)``.

    FFT synthesis at sample rate ``f_s``: on the rfft grid f_k = k·f_s/n draw
    Hermitian-symmetric unit-variance complex Gaussians ζ_k (E|ζ_k|² = 1) and
    set c_k = ζ_k·sqrt(S(f_k)·f_s·n/2); x = irfft(c). Then
    Var(x) = (2/n²)·Σ_k E|c_k|² = Σ_k S(f_k)·Δf ≈ ∫₀^{f_s/2} S df, i.e. the
    one-sided convention, and the Welch PSD of x reproduces S(f_k) bin by bin.
    The DC and (even-n) Nyquist bins must be real for a real x: they get real
    unit-variance draws with the same amplitude scaling. Whatever value
    ``psd_fn`` returns at f = 0 sets the DC-bin variance, so 1/f-type inputs
    must clamp their own DC bin (see PumpNoise: S(f₀) := S(f₁)); those two
    single bins carry ~1/n of the total variance and are irrelevant to any
    band-averaged PSD check.
    """
    if n < 2:
        return np.zeros(int(n), dtype=np.float64)
    f = np.fft.rfftfreq(int(n), d=1.0 / float(f_s))          # (n//2 + 1,)
    s = np.asarray(psd_fn(f), dtype=np.float64)
    amp = np.sqrt(np.maximum(s, 0.0) * float(f_s) * int(n) / 2.0)
    zr = rng.standard_normal(f.size)
    zi = rng.standard_normal(f.size)
    z = (zr + 1j * zi) / math.sqrt(2.0)                      # E|z|² = 1
    z[0] = zr[0]                                             # DC bin real
    if int(n) % 2 == 0:
        z[-1] = zi[-1]                                       # Nyquist bin real
    return np.fft.irfft(z * amp, n=int(n)).astype(np.float64)


class PumpNoise:
    """Pump-laser frequency noise and RIN (arXiv:2604.05897 Secs. V.B.4–V.B.5).

    Two channels, both sampled once per round trip at f_s = 1/t_r and both
    synthesized HOST-SIDE in float64 (deterministic per JAX key, independent
    of the jax x64 flag):

    Frequency noise (Sec. V.B.4)
        One-sided PSD of the instantaneous laser-frequency deviation δν_p(t):
            S_δν(f) = h₀ + h₋₁/f   [Hz²/Hz]   on f ∈ [1/(N·t_r), 1/(2·t_r)].
        The white plateau h₀ carries the intrinsic Lorentzian linewidth via
        the standard identity Δν_L = π·h₀ (exposed as
        ``lorentzian_linewidth_hz``); h₋₁ is the flicker (1/f) coefficient
        [Hz³/Hz]. Generation: the white part is i.i.d. Gaussian per round
        trip with variance h₀·f_s/2 (one-sided convention:
        var = ∫₀^{f_s/2} S df); the flicker part is FFT-synthesized with
        S_flicker(f_k) = h₋₁ / max(f_k, f₁) — the DC bin is clamped to the
        first bin f₁ = f_s/N, so the (single-bin) DC variance is
        h₋₁/f₁·Δf = h₋₁/N·f_s/N·(N/f_s) = h₋₁ instead of diverging.
        ``sample_freq`` returns 2π·δν_p(t) in rad/s; the SOLVER subtracts it
        from the detuning (δω ≡ ω_res − ω_p, so a positive laser-frequency
        excursion reduces δω).

    RIN (Sec. V.B.5)
        P_in(t) = P̄_in·(1 + ε(t)) with one-sided PSD
            S_ε(f) = 10^(floor_dBc/10) + 10^(excess_dBc/10)·(f_c/f)  (f < f_c)
                   = 10^(floor_dBc/10)                               (f ≥ f_c)
        [1/Hz]. The floor is i.i.d. Gaussian per round trip (variance
        floor·f_s/2) and the excess is FFT-synthesized exactly like the
        flicker part (same DC clamp, zero above the corner). ε is clipped so
        1 + ε ≥ 0; if more than 0.01% of samples clip, a warning reports the
        clipped fraction.

    ``pump_noise_enabled`` = 0/False forces BOTH channels inert regardless of
    the numeric values (samples are exactly zero, PSDs return zero); the value
    ranges are validated only when enabled. Representative values —
    ECDL: h₀ ≈ 3e3 Hz²/Hz (Δν_L ≈ 10 kHz), h₋₁ ≈ 1e10 Hz³/Hz;
    fiber laser: h₀ ≈ 30 Hz²/Hz (Δν_L ≈ 100 Hz).
    """

    def __init__(self, cfg, enabled: bool | None = None):
        self.cfg = cfg
        self.t_r = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.f_s = 1.0 / self.t_r
        if enabled is None:
            enabled = cfg.get("pump_noise_enabled", 0)
        if not (isinstance(enabled, (bool, int, np.integer)) and int(enabled) in (0, 1)):
            raise ValueError(
                f"pump_noise_enabled must be boolean-valued (bool or 0/1), got {enabled!r}."
            )
        self.enabled = bool(int(enabled))

        self.h0 = float(cfg.get("pump_freq_noise_h0_hz2_per_hz", 0.0))
        self.hm1 = float(cfg.get("pump_freq_noise_hm1_hz3_per_hz", 0.0))
        self.rin_floor_dbc = float(cfg.get("pump_rin_floor_dbc_per_hz", -300.0))
        self.rin_excess_dbc = float(cfg.get("pump_rin_excess_dbc_per_hz", -300.0))
        self.rin_corner_hz = float(cfg.get("pump_rin_corner_hz", 1.0e4))

        if self.enabled:
            if self.h0 < 0.0 or self.hm1 < 0.0:
                raise ValueError(
                    f"pump frequency-noise coefficients must be >= 0: "
                    f"h0 = {self.h0!r} Hz²/Hz, h-1 = {self.hm1!r} Hz³/Hz."
                )
            for name, val in (
                ("pump_rin_floor_dbc_per_hz", self.rin_floor_dbc),
                ("pump_rin_excess_dbc_per_hz", self.rin_excess_dbc),
            ):
                if val > -80.0:
                    raise ValueError(
                        f"{name} = {val!r} exceeds -80 dBc/Hz. RIN levels are "
                        f"dB quantities; a value this large is almost "
                        f"certainly a LINEAR spectral density entered where "
                        f"dBc/Hz is expected (physical lasers sit below "
                        f"-80 dBc/Hz)."
                    )
            if self.rin_corner_hz <= 0.0:
                raise ValueError(
                    f"pump_rin_corner_hz must be > 0, got {self.rin_corner_hz!r}."
                )

        # Effective (inert-when-disabled) parameters used by sample_*/psd_*.
        _on = 1.0 if self.enabled else 0.0
        self._h0 = self.h0 * _on
        self._hm1 = self.hm1 * _on
        self._rin_floor_lin = 10.0 ** (self.rin_floor_dbc / 10.0) * _on   # 1/Hz
        self._rin_excess_lin = 10.0 ** (self.rin_excess_dbc / 10.0) * _on  # 1/Hz

        # Intrinsic Lorentzian linewidth from the white plateau: Δν_L = π·h₀.
        self.lorentzian_linewidth_hz = math.pi * self._h0

    # -- closed-form one-sided PSDs (validation targets) ---------------------
    def psd_freq(self, f) -> np.ndarray:
        """One-sided S_δν(f) [Hz²/Hz] of the laser-frequency deviation δν_p."""
        f = np.asarray(f, dtype=np.float64)
        return self._h0 + self._hm1 / np.maximum(f, np.finfo(np.float64).tiny)

    def psd_rin(self, f) -> np.ndarray:
        """One-sided S_ε(f) [1/Hz] of the relative intensity fluctuation ε."""
        f = np.asarray(f, dtype=np.float64)
        excess = np.where(
            f < self.rin_corner_hz,
            self._rin_excess_lin
            * self.rin_corner_hz
            / np.maximum(f, np.finfo(np.float64).tiny),
            0.0,
        )
        return self._rin_floor_lin + excess

    # -- samplers ------------------------------------------------------------
    def sample_freq(self, key, N: int) -> np.ndarray:
        """2π·δν_p(t) [rad/s], shape (N,), float64, one sample per round trip.

        The caller (solver) applies the sign: δω-noise contribution is
        −2π·δν_p because δω ≡ ω_res − ω_p.
        """
        n = int(N)
        if not self.enabled or (self._h0 == 0.0 and self._hm1 == 0.0):
            return np.zeros(n, dtype=np.float64)
        rng = _np_generator_from_key(key)
        dnu = np.zeros(n, dtype=np.float64)
        if self._h0 > 0.0:  # white: var = h0*f_s/2 (one-sided convention)
            dnu += rng.standard_normal(n) * math.sqrt(self._h0 * self.f_s / 2.0)
        if self._hm1 > 0.0 and n >= 2:  # flicker via FFT synthesis
            f1 = self.f_s / n
            dnu += _synthesize_from_onesided_psd(
                rng, n, lambda f: self._hm1 / np.maximum(f, f1), self.f_s
            )
        return 2.0 * math.pi * dnu

    def sample_rin(self, key, N: int) -> np.ndarray:
        """ε(t) (dimensionless), shape (N,), float64, one sample per round trip.

        Clipped so 1 + ε ≥ 0; warns if more than 0.01% of samples clip.
        """
        n = int(N)
        if not self.enabled:
            return np.zeros(n, dtype=np.float64)
        rng = _np_generator_from_key(key)
        eps = np.zeros(n, dtype=np.float64)
        if self._rin_floor_lin > 0.0:
            eps += rng.standard_normal(n) * math.sqrt(
                self._rin_floor_lin * self.f_s / 2.0
            )
        if self._rin_excess_lin > 0.0 and n >= 2:
            f1 = self.f_s / n
            f_c = self.rin_corner_hz
            eps += _synthesize_from_onesided_psd(
                rng,
                n,
                lambda f: np.where(
                    f < f_c,
                    self._rin_excess_lin * f_c / np.maximum(f, f1),
                    0.0,
                ),
                self.f_s,
            )
        n_clip = int(np.count_nonzero(eps < -1.0))
        if n_clip > 1e-4 * n:
            import warnings

            warnings.warn(
                f"PumpNoise.sample_rin: {n_clip}/{n} samples "
                f"({100.0 * n_clip / n:.3f}%) clipped at ε = -1 (P_in >= 0). "
                f"The configured RIN is so large that the Gaussian model is "
                f"physically strained; the clipped sequence is returned.",
                stacklevel=2,
            )
        return np.maximum(eps, -1.0)


def plot_noise_psd() -> None:
    cfg = _load_config()
    total = TotalNoise(cfg)
    trn = total.trn
    pyro = total.pyroeo
    tccr = total.tccr

    N = 100_000
    key = jax.random.PRNGKey(0)
    samples = np.asarray(total.sample(key, N), dtype=np.float32)
    f_emp, p_emp = welch(samples, fs=1.0 / total.t_r, nperseg=1024)

    f = np.logspace(3, 9, 2000)
    s_trn = np.asarray(trn.psd(f))
    s_pyro = np.asarray(pyro.psd(f))
    s_tccr = np.asarray(tccr.psd(f))

    k_b = 1.380649e-23
    c = 299_792_458.0
    eps0 = 8.8541878128e-12
    n0_si3n4 = 2.0
    dn_dt_si3n4 = 2.45e-5
    rho_si3n4 = 3.17e3
    cp_si3n4 = 700.0
    kappa_si3n4 = 3.0
    v_si3n4 = 1e-15
    tau_si3n4 = 5e-6
    t_k = float(cfg.get("T_k", 300.0))
    omega_0 = 2.0 * math.pi * c / float(cfg.get("pump_wavelength_m", 1.55e-6))
    s_delta_t_si3n4 = (
        (4.0 * k_b * t_k**2 * tau_si3n4) / (rho_si3n4 * cp_si3n4 * v_si3n4)
    ) / (1.0 + (2.0 * np.pi * f * tau_si3n4) ** 2)
    s_si3n4 = ((omega_0 / n0_si3n4) * dn_dt_si3n4) ** 2 * s_delta_t_si3n4

    out = Path("analysis/figures")
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.loglog(f, s_trn, label="TFLN TRN", lw=1.8)
    plt.loglog(f, s_pyro, label="TFLN Pyro-EO", lw=1.8)
    plt.loglog(f, s_tccr, label="TFLN TCCR", lw=1.8)
    plt.loglog(f_emp[1:], p_emp[1:], "k:", lw=2.2, label="Empirical total (Welch)")
    plt.loglog(f, s_si3n4, color="gray", lw=1.6, label="Si₃N₄ TRN reference")
    plt.xlim(1e3, 1e9)
    plt.ylim(None, None)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("S_δω(f)  [(rad/s)²/Hz]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "noise_psd_comparison.pdf")
    plt.close()


def validate_noise_models() -> None:
    cfg = _load_config()
    total = TotalNoise(cfg)
    key = jax.random.PRNGKey(0)

    s10k = total.sample(key, 10_000)
    assert s10k.shape == (10_000,)
    assert s10k.dtype == jnp.float32

    s100k = total.sample(key, 100_000)
    std_total = float(jnp.std(s100k))
    sigma_thermal_combined = abs(total.trn.sigma_trn - total.pyroeo.sigma_pyroeo)  # correlated, opposite sign
    expected_sigma = math.sqrt(sigma_thermal_combined**2 + total.tccr.var_tccr)
    assert 0.1 * expected_sigma < std_total < 10.0 * expected_sigma, (
        f"Total noise std {std_total:.3e} outside expected range "
        f"[{0.1*expected_sigma:.3e}, {10.0*expected_sigma:.3e}]"
    )

    trn_std = float(jnp.std(total.trn.sample(jax.random.PRNGKey(1), 100_000)))
    tccr_std = float(jnp.std(total.tccr.sample(jax.random.PRNGKey(2), 100_000)))
    
    if 0.0 < tccr_std <= trn_std:   # SiN has no TCCR (tccr_std == 0); not a warning condition
        import warnings
        warnings.warn(
            f"TRN ({trn_std:.3e}) >= TCCR ({tccr_std:.3e}) for current config. "
            f"TCCR is not the dominant noise source. For TFLN devices where TCCR should "
            f"dominate, increase surface_state_density_per_m2 or verify A_eff.",
            stacklevel=2,
        )

    tccr_samples = np.asarray(total.tccr.sample(jax.random.PRNGKey(3), 200_000), dtype=np.float64)
    if float(np.std(tccr_samples)) > 0.0:   # only when TCCR is active (chi2 platforms); SiN -> skip
        r1 = np.corrcoef(tccr_samples[:-1], tccr_samples[1:])[0, 1]
        tau_est = -total.t_r / np.log(r1)
        tau_target = float(cfg.get("tau_carrier_s", 1.0e-7))
        assert abs(tau_est - tau_target) / tau_target < 0.5


if __name__ == "__main__":
    validate_noise_models()
    plot_noise_psd()
