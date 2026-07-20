#!/usr/bin/env python
"""Pump-laser-noise validation report (arXiv:2604.05897 Secs. V.B.4-V.B.5).

Seeded CLI driver producing the pump-noise deliverables:

(1) Generated frequency-noise S_dnu(f) and RIN S_eps(f) vs the closed-form
    targets (Welch of long host-side sequences).
(2) Single-soliton optical spectrum with pump noise OFF vs ON (cycle-averaged),
    warm-started at delta_omega = 11*kappa with the measured D_int grid.
(3) Time series of the delta_omega_eff decomposition
    (programmed / thermal / TRN / pump).
(4) Repetition-rate f_rep(t) and its Welch PSD, Taylor-D2 vs measured-D_int,
    pump-noise OFF vs ON: the DW-recoil contrast. With Taylor-only dispersion
    pump frequency noise is predominantly common-mode (weak f_rep transduction);
    the measured D_int grid (which carries the dispersive-wave phase matching)
    enhances f_rep transduction. The transduction ratio is reported for both.
(5) Intracavity-energy comparison OFF vs ON.
Plus a RIN -> Delta_T transduction check: a slow eps modulation produces the
analytically predicted Delta_T excursion via R_th (the 0.545 K/W line-source
path documented in config/sin_params.yaml), within 20%.

Outputs
-------
* analysis/figures/pump_noise_*.png            (150 dpi, publication style)
* analysis/results/pump_noise_report.json      (every measured number)

The OFF runs use the write_noise_off_config sidecar (T_k = 0 + quantum off +
pump off) so they are fully deterministic; the ON runs re-enable ONLY the pump
channel on a T_k = 0 sidecar, so they contain ONLY pump-laser noise. All RNG
flows from the single --seed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from analysis.dks_access import (  # noqa: E402
    PIN_W,
    PRODUCTION_NUMERICS,
    _run,
    attach_dispersion,
    load_cavity_params,
    sech_soliton_seed,
    temporal_peak_positions,
)
from analysis.plot_utils import apply_pub_style  # noqa: E402
from analysis.run_detuning_sweep import write_noise_off_config  # noqa: E402
from analysis.spectral_metrics import average_power_spectrum  # noqa: E402
from simulator.lle_solver import _load_config  # noqa: E402
from simulator.noise_models import PumpNoise  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_JSON = Path(__file__).resolve().parent / "results" / "pump_noise_report.json"

SOLITON_DW_KAPPA = 11.0

# ECDL preset (see config comments): white h0 ~ 3e3 Hz^2/Hz (Delta_nu_L ~ 10 kHz),
# flicker h-1 ~ 1e10 Hz^3/Hz.
ECDL = dict(
    pump_noise_enabled=1,
    pump_freq_noise_h0_hz2_per_hz=3.0e3,
    pump_freq_noise_hm1_hz3_per_hz=1.0e10,
    pump_rin_floor_dbc_per_hz=-300.0,      # RIN off for the freq-noise study
    pump_rin_excess_dbc_per_hz=-300.0,
    pump_rin_corner_hz=1.0e4,
)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _pump_on_sidecar(base_off_cfg: str, out: Path, **pump) -> str:
    """T_k=0 (deterministic) sidecar with ONLY the pump channel enabled."""
    with open(base_off_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("physical_parameters", {}).update(pump)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out)


# ---------------------------------------------------------------------------
# (1) generated PSDs vs targets
# ---------------------------------------------------------------------------
def psd_figure(cfg_on: str, seed: int) -> dict:
    phys = _load_config(cfg_on)
    pm = PumpNoise(phys)
    n = 2 ** 20
    dnu = np.asarray(pm.sample_freq(jax.random.PRNGKey(seed), n)) / (2.0 * math.pi)
    # a representative RIN preset for the spectral panel
    pm_rin = PumpNoise(
        dict(phys, pump_noise_enabled=1, pump_rin_floor_dbc_per_hz=-150.0,
             pump_rin_excess_dbc_per_hz=-120.0, pump_rin_corner_hz=1e9)
    )
    eps = np.asarray(pm_rin.sample_rin(jax.random.PRNGKey(seed + 1), n))

    f_nu, p_nu = welch(dnu, fs=pm.f_s, nperseg=2 ** 16)
    f_e, p_e = welch(eps, fs=pm.f_s, nperseg=2 ** 16)

    apply_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.loglog(f_nu[1:], p_nu[1:], color="tab:blue", alpha=0.5, label="Welch (generated)")
    ax1.loglog(f_nu[1:], pm.psd_freq(f_nu[1:]), "k--", lw=1.6, label=r"target $h_0+h_{-1}/f$")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(r"$S_{\delta\nu}(f)$  [Hz$^2$/Hz]")
    ax1.set_title(rf"Frequency noise ($\Delta\nu_L=\pi h_0={pm.lorentzian_linewidth_hz:.0f}$ Hz)")
    ax1.legend()
    ax2.loglog(f_e[1:], p_e[1:], color="tab:red", alpha=0.5, label="Welch (generated)")
    ax2.loglog(f_e[1:], pm_rin.psd_rin(f_e[1:]), "k--", lw=1.6, label=r"target floor + excess/$f$")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel(r"$S_\varepsilon(f)$  [1/Hz]")
    ax2.set_title("Relative intensity noise (RIN)")
    ax2.legend()
    fig.savefig(FIG_DIR / "pump_noise_psd.png")
    plt.close(fig)

    def _octave_err(f, p, target):
        """Worst per-OCTAVE-band mean deviation [dB] over [1e6, 1e9] (3 decades).

        Per-octave averaging (not a per-bin max) matches the 3 dB/octave
        acceptance metric; a single Welch bin carries several dB of chi-square
        variance that is not a PSD-shape error.
        """
        edges = 1e6 * 2.0 ** np.arange(0, math.ceil(math.log2(1e9 / 1e6)) + 1)
        worst = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (f >= lo) & (f < hi)
            if m.sum() < 4:
                continue
            worst = max(worst, abs(10 * np.log10(p[m].mean() / np.asarray(target(f[m])).mean())))
        return float(worst)

    return {
        "lorentzian_linewidth_hz": float(pm.lorentzian_linewidth_hz),
        "freq_worst_octave_dB": _octave_err(f_nu[1:], p_nu[1:], pm.psd_freq),
        "rin_worst_octave_dB": _octave_err(f_e[1:], p_e[1:], pm_rin.psd_rin),
    }


# ---------------------------------------------------------------------------
# f_rep and common-mode extraction
# ---------------------------------------------------------------------------
def _dominant_peak_angle(field: np.ndarray) -> float:
    """Sub-grid azimuthal angle [rad] of the strongest |E|^2 peak.

    A whole-bin peak index quantizes f_rep to the FFT-grid resolution
    (2*pi/n_tau), which floors any wander below one cell to exactly zero
    (masking the weak Taylor-dispersion f_rep transduction). A 3-point
    parabolic fit around the max recovers the continuous peak position.
    """
    p = np.abs(field) ** 2
    n = int(field.shape[0])
    if p.max() <= 0:
        return float("nan")
    i = int(np.argmax(p))
    yl, y0, yr = p[(i - 1) % n], p[i], p[(i + 1) % n]
    denom = yl - 2.0 * y0 + yr
    delta = 0.5 * (yl - yr) / denom if denom != 0 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    return (2.0 * math.pi * ((i + delta) % n)) / n


def _frep_series(snaps: np.ndarray, snap_int: int, t_r: float, f_rep0: float):
    """f_rep deviation [Hz] per snapshot from the soliton peak-angle drift.

    A peak-angle drift d(theta)/dn per round trip means the soliton round-trip
    time deviates by -(d theta/dn)/(2 pi)*t_r, so
    delta_f_rep = f_rep0 * (d theta/dn)/(2 pi).
    """
    ang = np.array([_dominant_peak_angle(s) for s in snaps])
    ang = np.unwrap(ang)
    dtheta_dn = np.gradient(ang) / snap_int                 # rad per round trip
    return f_rep0 * dtheta_dn / (2.0 * math.pi)             # Hz


def _common_mode_freq(snaps: np.ndarray, snap_int: int, t_r: float):
    """Common-mode optical-frequency wander [Hz] from the mean-field phase rate."""
    phi = np.unwrap(np.angle(snaps.mean(axis=1)))
    dphi_dt = np.gradient(phi) / (snap_int * t_r)           # rad/s
    return dphi_dt / (2.0 * math.pi)                        # Hz


# ---------------------------------------------------------------------------
# (2-5) soliton OFF vs ON + DW-recoil contrast
# ---------------------------------------------------------------------------
def soliton_study(cav_taylor, cav_meas, cfg_off, cfg_on, seed, n_tau,
                  settle_rt, run_rt, snap_int, numerics=PRODUCTION_NUMERICS) -> dict:
    dw = SOLITON_DW_KAPPA * cav_meas.kappa
    seed_field = sech_soliton_seed(dw, cav_meas, n_tau=n_tau, pin=PIN_W)
    out = {}
    series = {}
    for disp_name, cav in (("taylor", cav_taylor), ("measured", cav_meas)):
        d_grid = None if disp_name == "taylor" else cav.d_int_grid
        for label, cfg in (("off", cfg_off), ("on", cfg_on)):
            settle = _quiet(_run, dw, settle_rt, cav, e0=seed_field, seed=seed,
                            n_tau=n_tau, pin=PIN_W, snapshot_interval=settle_rt,
                            config_path=cfg, **numerics)
            sol = _quiet(_run, dw, run_rt, cav,
                         e0=np.asarray(settle["e_final"])[0],
                         delta_t0=settle["delta_t_final"], seed=seed + 1,
                         n_tau=n_tau, pin=PIN_W, snapshot_interval=snap_int,
                         config_path=cfg, **numerics)
            snaps = np.asarray(sol["E_snapshots"])[0]
            frep = _frep_series(snaps, snap_int, cav.t_r, cav.fsr_hz)
            cmf = _common_mode_freq(snaps, snap_int, cav.t_r)
            series[(disp_name, label)] = dict(
                snaps=snaps, sol=sol, frep=frep, cmf=cmf,
                U=np.asarray(sol["U_int_history"])[0],
            )

    # --- Figure 2: optical spectrum OFF vs ON (measured dispersion) ----------
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, c in (("off", "0.4"), ("on", "tab:red")):
        s = series[("measured", label)]
        w = s["snaps"][3 * s["snaps"].shape[0] // 4:]
        mu, p_mu = average_power_spectrum(w)
        p_db = 10 * np.log10(np.maximum(p_mu / p_mu.max(), 1e-12))
        ax.plot(mu, p_db, color=c, lw=1.0, label=f"pump noise {label.upper()}")
    ax.set_xlabel(r"Mode number $\mu$")
    ax.set_ylabel("Cycle-averaged power [dB]")
    ax.set_title(rf"Single-soliton spectrum at $\delta\omega={SOLITON_DW_KAPPA:.0f}\kappa$ "
                 f"(measured $D_{{int}}$)")
    ax.set_ylim(-80, 2)
    ax.legend()
    fig.savefig(FIG_DIR / "pump_noise_soliton_spectrum.png")
    plt.close(fig)

    # --- Figure 3: delta_omega_eff decomposition (measured, ON) --------------
    s = series[("measured", "on")]["sol"]
    dwe = np.asarray(s["delta_omega_eff_history"])[0]
    dT = np.asarray(s["DeltaT_history"])[0]
    pump = np.asarray(s["pump_freq_noise_history"])[0]
    phys_on = _load_config(cfg_on)
    omega0 = 2.0 * math.pi * 299_792_458.0 / float(phys_on.get("pump_wavelength_m", 1.55e-6))
    thermal_shift = -(omega0 / float(phys_on["n0"])) * float(phys_on["dn_dT_per_k"]) * dT
    programmed = dw
    trn = dwe - programmed - thermal_shift - pump          # residual = TRN (0 here, T_k=0)
    rt = np.arange(dwe.size)
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    k = cav_meas.kappa
    ax.plot(rt, (dwe - programmed) / k, color="k", lw=1.0, label=r"total $\delta\omega_{eff}-$prog")
    ax.plot(rt, thermal_shift / k, color="tab:orange", lw=1.0, label="thermal")
    ax.plot(rt, pump / k, color="tab:red", lw=1.0, label=r"pump $-2\pi\delta\nu$")
    ax.plot(rt, trn / k, color="tab:blue", lw=0.8, alpha=0.7, label="TRN (residual)")
    ax.set_xlabel("Round trip")
    ax.set_ylabel(r"$\delta\omega$ contribution [$\kappa$]")
    ax.set_title(r"Detuning decomposition (measured $D_{int}$, pump noise ON)")
    ax.legend(ncol=2)
    fig.savefig(FIG_DIR / "pump_noise_detuning_decomp.png")
    plt.close(fig)

    # --- Figure 4: f_rep(t) + Welch, Taylor vs measured, OFF vs ON -----------
    apply_pub_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.5))
    fs_snap = 1.0 / (snap_int * cav_meas.t_r)
    for disp_name, style in (("taylor", "-"), ("measured", "-")):
        for label, alpha in (("off", 0.5), ("on", 1.0)):
            fr = series[(disp_name, label)]["frep"]
            t_ms = np.arange(fr.size) * snap_int * cav_meas.t_r * 1e6
            axa.plot(t_ms, fr, style, alpha=alpha, lw=0.9,
                     label=f"{disp_name} {label.upper()}")
            if fr.size >= 64:
                ff, pf = welch(fr - np.mean(fr), fs=fs_snap, nperseg=min(256, fr.size))
                axb.loglog(ff[1:], pf[1:], style, alpha=alpha, lw=0.9,
                           label=f"{disp_name} {label.upper()}")
    axa.set_xlabel(r"Time [$\mu$s]")
    axa.set_ylabel(r"$\delta f_{rep}$ [Hz]")
    axa.set_title("Repetition-rate wander")
    axa.legend(fontsize=7, ncol=2)
    axb.set_xlabel("Frequency [Hz]")
    axb.set_ylabel(r"$S_{\delta f_{rep}}$ [Hz$^2$/Hz]")
    axb.set_title("f_rep noise PSD")
    if axb.get_legend_handles_labels()[1]:
        axb.legend(fontsize=7, ncol=2)
    fig.savefig(FIG_DIR / "pump_noise_frep.png")
    plt.close(fig)

    # --- Figure 5: intracavity energy OFF vs ON (measured) -------------------
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(8, 4.0))
    for label, c in (("off", "0.4"), ("on", "tab:red")):
        U = series[("measured", label)]["U"]
        ax.plot(np.arange(U.size) * snap_int, U * 1e15, color=c, lw=0.9,
                label=f"pump noise {label.upper()}")
    ax.set_xlabel("Round trip")
    ax.set_ylabel(r"$U_{int}$ [fJ]")
    ax.set_title(r"Intracavity energy (measured $D_{int}$)")
    ax.legend()
    fig.savefig(FIG_DIR / "pump_noise_energy.png")
    plt.close(fig)

    # --- DW-recoil transduction ratio ---------------------------------------
    def _ratio(disp_name):
        on = series[(disp_name, "on")]
        off = series[(disp_name, "off")]
        # subtract the (deterministic) OFF baseline drift so we isolate the
        # pump-noise-induced wander
        frep_w = float(np.std(on["frep"] - off["frep"]))
        cmf_w = float(np.std(on["cmf"] - off["cmf"]))
        return frep_w, cmf_w, (frep_w / cmf_w if cmf_w > 0 else float("nan"))

    ft, ct, rt_ = _ratio("taylor")
    fm, cm, rm = _ratio("measured")
    out["dw_recoil"] = {
        "taylor": {"frep_wander_hz": ft, "common_mode_wander_hz": ct, "ratio": rt_},
        "measured": {"frep_wander_hz": fm, "common_mode_wander_hz": cm, "ratio": rm},
        "enhancement_measured_over_taylor": (rm / rt_ if rt_ > 0 else float("nan")),
    }
    return out


# ---------------------------------------------------------------------------
# RIN -> Delta_T transduction
# ---------------------------------------------------------------------------
def rin_thermal_check(cav, cfg_off, seed, n_tau, n_periods=4, t_slow_cap=None) -> dict:
    """Slow eps modulation -> Delta_T excursion vs R_th*P_abs*eps*H(f) (within 20%).

    The single-pole thermal ODE dDelta_T/dt = -Delta_T/tau_th + Gamma_th*P_abs/
    (rho*Cp*V) linearizes (P_abs = kappa_i*U/t_r ~ P_abs0*(1+eps) at fixed
    detuning, cavity cutoff kappa/2 >> f_mod so U tracks the launched power) to
        Delta_T(t) = R_th * P_abs0 * (1 + eps(t) * H(f)),
        R_th = tau_th*Gamma_th/(rho*Cp*V)  (~0.545 K/W, the line-source path),
        H(f) = 1 / sqrt(1 + (2*pi*f*tau_th)^2)  (thermal low-pass).
    The thermal timescale is tau_th ~ 1.2e5 round trips, so f_mod is placed near
    the cutoff 1/(2*pi*tau_th) to keep the run bounded; H(f) makes the
    prediction exact regardless.
    """
    phys = _load_config(cfg_off)
    tau_th = float(phys["tau_th_s"])
    gamma_th = float(phys["Gamma_th"])
    rho = float(phys["rho_kg_per_m3"]); cp = float(phys["Cp_j_per_kg_k"])
    vol = float(phys["mode_volume_m3"])
    R_th = tau_th * gamma_th / (rho * cp * vol)             # K/W (~0.545)

    dw0 = 3.0 * cav.kappa
    pin = 1.0e-3                                            # below MI threshold: CW only
    f_mod = 1.0 / (2.0 * math.pi * tau_th)                 # thermal cutoff
    H = 1.0 / math.sqrt(1.0 + (2.0 * math.pi * f_mod * tau_th) ** 2)  # = 1/sqrt(2)
    eps_amp = 0.1
    period_rt = (1.0 / f_mod) / cav.t_r
    settle_periods = 2
    t_slow = int((settle_periods + n_periods) * period_rt)
    if t_slow_cap is not None:
        t_slow = min(t_slow, int(t_slow_cap))
    t = np.arange(t_slow) * cav.t_r
    eps = eps_amp * np.sin(2.0 * math.pi * f_mod * t)

    a = cav.kappa / 2.0
    e_cw = (math.sqrt(cav.kappa_c * pin) / (a + 1j * dw0)) * np.ones(n_tau, np.complex128)
    sol = _quiet(_run, dw0, t_slow, cav, e0=e_cw, seed=seed, n_tau=n_tau,
                 pin=pin, snapshot_interval=max(t_slow // 500, 1),
                 config_path=cfg_off, pump_rin_epsilon_override=eps)
    dT = np.asarray(sol["DeltaT_history"])[0]
    U = np.asarray(sol["U_int_history"])[0]
    p_abs0 = cav.kappa_i * float(np.mean(U)) / cav.t_r
    # fit dT modulation amplitude at f_mod over the settled window
    tt = np.arange(dT.size) * cav.t_r
    m = tt >= settle_periods * period_rt * cav.t_r
    if m.sum() < 16:  # capped/too-short run (e.g. --quick): fit not meaningful
        dT_meas = float("nan")
    else:
        basis = np.column_stack([np.cos(2 * math.pi * f_mod * tt[m]),
                                 np.sin(2 * math.pi * f_mod * tt[m]),
                                 np.ones(m.sum()), tt[m] - tt[m].mean()])
        c, *_ = np.linalg.lstsq(basis, dT[m], rcond=None)
        dT_meas = float(np.hypot(c[0], c[1]))
    dT_pred = R_th * p_abs0 * eps_amp * H
    return {
        "R_th_K_per_W": R_th,
        "P_abs0_W": float(p_abs0),
        "eps_amp": eps_amp,
        "f_mod_Hz": float(f_mod),
        "thermal_lowpass_H": float(H),
        "deltaT_measured_K": dT_meas,
        "deltaT_predicted_K": float(dT_pred),
        "deltaT_dc_R_th_P_abs_eps_K": float(R_th * p_abs0 * eps_amp),
        "rel_error": float(abs(dT_meas / dT_pred - 1.0)),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-tau", type=int, default=8192)
    ap.add_argument("--settle-rt", type=int, default=2000)
    ap.add_argument("--run-rt", type=int, default=6000)
    ap.add_argument("--snap-int", type=int, default=5)
    ap.add_argument("--quick", action="store_true",
                    help="tiny run for a smoke test of the whole pipeline")
    args = ap.parse_args()

    # Fast numerics for the smoke test (n_substeps=1 -> no unrolled scan body,
    # so compilation is quick); the production report uses PRODUCTION_NUMERICS.
    numerics = PRODUCTION_NUMERICS
    if args.quick:
        args.n_tau, args.settle_rt, args.run_rt, args.snap_int = 256, 60, 200, 4
        numerics = dict(n_substeps=1, dealias_two_thirds=False,
                        edge_absorber=False, dispersion_validity_mask=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

    import tempfile

    cfg_off = str(write_noise_off_config())
    _fd, _pump_on_path = tempfile.mkstemp(prefix="pump_on_", suffix=".yaml")
    import os

    os.close(_fd)
    cfg_on = _pump_on_sidecar(cfg_off, Path(_pump_on_path), **ECDL)

    cav = load_cavity_params(str(cfg_off))
    cav_meas = attach_dispersion(cav, args.n_tau)
    cav_taylor = cav                                        # Taylor beta path (d_int_grid=None)

    metrics = {"seed": args.seed, "n_tau": args.n_tau,
               "soliton_dw_kappa": SOLITON_DW_KAPPA, "ecdl_preset": ECDL}
    print("[1/3] PSD fidelity ...")
    metrics["psd"] = psd_figure(cfg_on, args.seed)
    print("[2/3] soliton OFF vs ON + DW-recoil contrast ...")
    metrics.update(
        soliton_study(cav_taylor, cav_meas, cfg_off, cfg_on, args.seed,
                      args.n_tau, args.settle_rt, args.run_rt, args.snap_int,
                      numerics=numerics)
    )
    print("[3/3] RIN -> Delta_T transduction ...")
    metrics["rin_thermal"] = rin_thermal_check(
        cav, cfg_off, args.seed, 8,
        t_slow_cap=20000 if args.quick else None,
    )

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")
    dr = metrics["dw_recoil"]
    print(f"DW-recoil f_rep/common-mode ratio: taylor={dr['taylor']['ratio']:.3e}, "
          f"measured={dr['measured']['ratio']:.3e}, "
          f"enhancement={dr['enhancement_measured_over_taylor']:.2f}x")
    rt = metrics["rin_thermal"]
    print(f"RIN->Delta_T: measured={rt['deltaT_measured_K']:.3e} K, "
          f"predicted={rt['deltaT_predicted_K']:.3e} K, "
          f"rel_error={rt['rel_error']*100:.1f}%")


if __name__ == "__main__":
    main()
