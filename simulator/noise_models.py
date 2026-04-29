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


_DEF_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "tfln_params.yaml"


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
        self.sigma_pyroeo = (self.omega_0 * self.n0**2 * self.r33 * self.p / (2.0 * self.eps0)) * math.sqrt(self.var_delta_t)

    def sample(self, key, N) -> jnp.ndarray:
        return _ar1_samples(key, N, self.tau_th, self.sigma_pyroeo, self.t_r)

    def psd(self, f) -> jnp.ndarray:
        s_delta_t = (
            (4.0 * self.k_b * self.T_k**2 * self.tau_th) / (self.rho * self.cp * self.v)
        ) / (1.0 + (2.0 * jnp.pi * f * self.tau_th) ** 2)
        scale = (self.omega_0 * self.n0**2 * self.r33 * self.p / (2.0 * self.eps0)) ** 2
        return scale * s_delta_t


class TCCRNoise:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t_r = 1.0 / float(cfg.get("fsr_hz", 2.0e11))
        self.omega_0 = 2.0 * math.pi * 299_792_458.0 / float(cfg.get("pump_wavelength_m", 1.55e-6))
        self.n0 = float(cfg.get("n0", 2.2))
        self.r33 = float(cfg.get("eo_r33_m_per_v", 3.1e-11))
        self.T_k = float(cfg.get("T_k", 300.0))
        self.v_eff = float(cfg.get("mode_volume_m3", 1.0e-15))
        self.rho_surface = float(cfg.get("surface_state_density_per_m2", 1.0e16))
        self.tau_carrier = float(cfg.get("tau_carrier_s", 1.0e-7))
        self.eps0 = 8.8541878128e-12
        self.k_b = 1.380649e-23
        
        # alpha_tccr is a dimensionless coupling factor encoding EO + surface-state
        # contributions; loaded from config so it can be tuned to match experiment.
        self.alpha_tccr = float(cfg.get("alpha_tccr", 1e-3))
        # S_TCCR(0) = alpha_tccr * (k_B T / V_eff) * omega_0^2 * 2*tau  (one-sided)
        self.s0_tccr = (
            self.alpha_tccr
            * (self.k_b * self.T_k / self.v_eff)
            * self.omega_0 ** 2
        )
        self.var_tccr = self.s0_tccr / (4.0 * self.tau_carrier)
        self.sigma_tccr = math.sqrt(self.var_tccr)

    
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
        self.tau_th = self.trn.tau_th
        self.var_delta_t = self.trn.var_delta_t
        self.tau_carrier = self.tccr.tau_carrier

    def sample(self, key, N) -> jnp.ndarray:
        key_thermal, key_tccr = jax.random.split(key, 2)
        temp_noise = _ar1_samples(key_thermal, N, self.tau_th, math.sqrt(self.var_delta_t), self.t_r)
        trn_noise = (self.omega_0 / self.n0 * self.dn_dT) * temp_noise
        pyroeo_noise = (self.omega_0 * self.n0**2 * self.r33 * self.p / (2.0 * self.eps0)) * temp_noise
        tccr_noise = self.tccr.sample(key_tccr, N)
        return (trn_noise - pyroeo_noise + tccr_noise).astype(jnp.float32)


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
        (4.0 * k_b * t_k**2 * kappa_si3n4) / (rho_si3n4**2 * cp_si3n4**2 * v_si3n4)
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
    expected_sigma = math.sqrt(total.tccr.var_tccr + total.trn.sigma_trn**2)
    assert 0.1 * expected_sigma < std_total < 10.0 * expected_sigma, (
        f"Total noise std {std_total:.3e} outside expected range "
        f"[{0.1*expected_sigma:.3e}, {10.0*expected_sigma:.3e}]"
    )

    trn_std = float(jnp.std(total.trn.sample(jax.random.PRNGKey(1), 100_000)))
    tccr_std = float(jnp.std(total.tccr.sample(jax.random.PRNGKey(2), 100_000)))
    assert tccr_std > trn_std, (
        f"TCCR ({tccr_std:.3e}) should dominate TRN ({trn_std:.3e}) in TFLN"
    )

    tccr_samples = np.asarray(total.tccr.sample(jax.random.PRNGKey(3), 200_000), dtype=np.float64)
    r1 = np.corrcoef(tccr_samples[:-1], tccr_samples[1:])[0, 1]
    tau_est = -total.t_r / np.log(r1)
    tau_target = float(cfg.get("tau_carrier_s", 1.0e-7))
    assert abs(tau_est - tau_target) / tau_target < 0.5


if __name__ == "__main__":
    validate_noise_models()
    plot_noise_psd()
