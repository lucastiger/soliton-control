#!/usr/bin/env python
"""Colored-noise / FSR-noise / metrology comparison report (Q3 deliverable).

Produces the eight figures of the colored-noise upgrade (150 dpi, shared
style via analysis.plot_utils.apply_pub_style) plus a JSON metrics block per
figure in ``analysis/results/noise_comparison_report.json``:

1. ``noise_psd_models.png``       — S_domega(f): single-pole vs
   Kondratiev-Gorodetsky (Eq. 130, variance pinned to Eq. 129) vs CSV
   round-trip vs empirical Welch of generated samples, with the Si3N4
   single-pole reference curve retained from ``plot_noise_psd``.
2. ``soliton_spectrum_off_on.png`` — single-soliton optical spectrum,
   all noise OFF vs the full stack ON (quantum + pump + TRN(K-G) + FSR),
   cycle-averaged over the final quarter of the run.
3. ``soliton_waveform_off_on.png`` — temporal waveform comparison.
4. ``energy_detuning_series.png``  — intracavity energy and the
   delta_omega_eff decomposition (programmed + thermal + noise) time series.
5. ``s_rep_trn_limit.png``         — S_dnu,rep(f) from the tape-model fit
   with the TRN-limit overlay (D1/omega0)^2 * C_pull^2 * S_dT(f)/(2pi)^2:
   the prescribed quantum+TRN+FSR stack AND a quantum-off variant in which
   the TRN-limited repetition-rate noise is cleanly resolved (the quantum
   per-line phase floor of this device sits above the TRN S_rep, which the
   figure shows honestly).
6. ``linewidth_vs_mu.png``         — per-line effective linewidth
   (beta-separation-line integral) vs mode index, WHITE-PUMP NULL BASELINE:
   quantum + white pump frequency noise at the deep operating point
   transduces almost pure common mode, so the curvature is not significant.
   The DW-recoil positive result is figure 6b below.
6b. ``linewidth_vs_mu_dwrecoil.png`` — the DW-recoil positive demonstration
   (Lei et al. 2021 phenomenology, paper V.B.1): measured-D_int (dispersive-
   wave recoil transduces pump frequency noise into a repetition-rate shift)
   gives a resolvable, significant linewidth parabola with fix point
   mu_fix = -S_cr/S_rep, overlaid against the Taylor-D2 negative control
   (pure quadratic dispersion -> pure common mode -> S_rep -> 0 -> flat). The
   pump is flicker-dominated (the beta-line linewidth is a low-frequency
   probe, where 1/f laser noise and DW recoil both live and the soliton
   stays healthy). mu_fix from the parabola vertex is cross-checked against
   the direct -Re S_cr/S_rep estimator.
7. ``quiet_point_sweep.png``       — S_rep at a fixed offset vs detuning,
   measured D_int vs Taylor D2 (paper Sec. V.B.5 quiet-point signature; the
   minimum may be shallow for this device — whatever the physics gives).
8. ``rf_beatnote.png``             — RF-domain repetition-rate beatnote
   proxy: PSD of the photodetected pulse-train proxy Sum_j |E_j|^2(t) (the
   per-round-trip intracavity energy record). A DC photodetector sees the
   pulse train's slow envelope; its fluctuation spectrum is the standard
   noise diagnostic the paper describes — the true ~24.6 GHz carrier and its
   harmonics lie above this record's Nyquist (f_s/2 = 1/(2 t_r)) by
   construction, so the plotted spectrum is the baseband noise pedestal.

Also reports the performance overheads (probes on/off, FSR on/off at
n_tau = 8192) and the zero-new-ops property of the disabled paths.

Run ``--quick`` for a minutes-scale smoke of the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402

from analysis.dks_access import (  # noqa: E402
    CONFIG_PATH,
    PIN_W,
    PRODUCTION_NUMERICS,
    RESULTS_DIR,
    _run,
    attach_dispersion,
    load_cavity_params,
    sech_soliton_seed,
)
from analysis.noise_metrology import (  # noqa: E402
    cross_frequency_psd,
    effective_linewidth,
    frequency_noise_psd,
    psd_at_offset,
    quiet_point_sweep,
    rep_rate_phase,
    tape_model_fit,
    timing_jitter,
    unwrapped_phases,
)
from analysis.plot_utils import apply_pub_style  # noqa: E402
from analysis.run_detuning_sweep import write_noise_off_config  # noqa: E402
from simulator.colored_noise import (  # noqa: E402
    np_generator_from_key,
    synthesize_from_psd,
)
from simulator.noise_models import TotalNoise, TRNoise, _load_config  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
RESULTS_JSON = RESULTS_DIR / "noise_comparison_report.json"

# Kondratiev-Gorodetsky geometry for THIS SiN device: ring radius from the
# round-trip length (R = L_cav/2pi, L_cav = 5.842 mm), Gaussian mode
# half-dimensions from the 4.4 x 0.8 um waveguide core.
KG_GEOMETRY = dict(trn_R_m=9.298e-4, trn_da_m=2.2e-6, trn_db_m=4.0e-7)
SOLITON_DW_KAPPA = 10.0
ECDL_H0 = 3.0e3          # Hz^2/Hz (figure-2 full stack: a realistic ECDL)
LINEWIDTH_H0 = 3.0e6     # Hz^2/Hz (figure-6: strong plateau so the
#                          beta-line crossing falls inside the resolved band)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _sidecar(base_cfg: str | Path, tag: str, **overrides) -> str:
    cfg = yaml.safe_load(open(base_cfg, encoding="utf-8"))
    cfg["physical_parameters"].update(overrides)
    fd, name = tempfile.mkstemp(prefix=f"sin_params_{tag}_", suffix=".yaml")
    import os

    os.close(fd)
    with open(name, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return name


# ---------------------------------------------------------------------------
# Figure 1: PSD model overlay
# ---------------------------------------------------------------------------
def fig1_psd_models(tmp: Path, seed: int, n_samp: int) -> dict:
    base = _load_config(None)
    f_s = 1.0 / TotalNoise(base).t_r
    trn_sp = TRNoise(base)

    kg_cfg = {**base, "trn_psd_model": "kondratiev_gorodetsky", **KG_GEOMETRY}
    trn_kg = TRNoise(kg_cfg)

    f_tab = np.logspace(1, math.log10(f_s / 2.0), 500)
    csv_path = tmp / "kg_roundtrip.csv"
    np.savetxt(csv_path, np.column_stack(
        [f_tab, trn_kg.delta_t_psd(f_tab)]), delimiter=",")
    csv_cfg = {**base, "trn_psd_model": "csv",
               "trn_psd_csv_path": str(csv_path),
               "trn_csv_units": "S_delta_T"}
    trn_csv = TRNoise(csv_cfg)

    f = np.logspace(3, math.log10(f_s / 2.0), 1200)
    s_sp = np.asarray(trn_sp.psd(f))
    s_kg = np.asarray(trn_kg.psd(f))
    s_csv = np.asarray(trn_csv.psd(f))

    # Empirical Welch of generated K-G samples (delta-omega units).
    from scipy.signal import welch

    x = np.stack([
        trn_kg.c_pull * synthesize_from_psd(
            np_generator_from_key(jax.random.PRNGKey(seed + i)),
            n_samp, trn_kg.delta_t_psd, f_s)
        for i in range(6)
    ])
    f_e, s_e = welch(x, fs=f_s, nperseg=min(1 << 13, n_samp // 4), axis=-1)
    s_e = s_e.mean(axis=0)[1:]
    f_e = f_e[1:]

    # Si3N4 single-pole reference retained from plot_noise_psd.
    k_b = 1.380649e-23
    t_k = float(base.get("T_k", 300.0))
    omega_0 = trn_sp.omega_0
    s_ref = (((omega_0 / 2.0) * 2.45e-5) ** 2
             * (4.0 * k_b * t_k**2 * 5e-6) / (3.17e3 * 700.0 * 1e-15)
             / (1.0 + (2.0 * np.pi * f * 5e-6) ** 2))

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(f, s_sp, label="single_pole (AR(1) twin)", lw=1.6)
    ax.loglog(f, s_kg, label="Kondratiev–Gorodetsky (Eq. 130, "
                             "var pinned to Eq. 129)", lw=1.6)
    ax.loglog(f, s_csv, "--", label="CSV round trip of K–G", lw=1.4)
    ax.loglog(f_e, s_e, ":", color="k", lw=1.8,
              label="empirical Welch of K–G samples")
    ax.loglog(f, s_ref, color="gray", lw=1.2, label="Si$_3$N$_4$ TRN "
              "reference (plot_noise_psd)")
    ax.set_xlabel("Frequency f [Hz]")
    ax.set_ylabel(r"$S_{\delta\omega}(f)$  [(rad/s)$^2$/Hz]")
    ax.set_title("TRN spectral models: analytic, tabulated, and generated")
    ax.legend(loc="lower left", fontsize=7)
    fig.savefig(FIG_DIR / "noise_psd_models.png")
    plt.close(fig)

    # Metrics: octave-band Welch fidelity of the empirical curve vs K-G.
    worst_db = 0.0
    lo = f_e[4]
    while lo * 2 <= f_s / 4:
        m = (f_e >= lo) & (f_e < 2 * lo)
        if m.sum() >= 2:
            tgt = float(np.mean(trn_kg.c_pull**2
                                * np.asarray(trn_kg.delta_t_psd(f_e[m]))))
            worst_db = max(worst_db, abs(10 * math.log10(
                float(np.mean(s_e[m])) / tgt)))
        lo *= 2
    return {
        "kg_tau_d_s": float((math.pi / 4) ** (1 / 3)
                            * (3.17e3 * 700.0 / 30.0)
                            * KG_GEOMETRY["trn_db_m"] ** 2),
        "welch_vs_kg_worst_octave_db": worst_db,
        "var_eq129_K2": trn_kg.var_delta_t,
        "kg_geometry": KG_GEOMETRY,
    }


# ---------------------------------------------------------------------------
# Figures 2-4 + 8: single soliton OFF vs full stack ON
# ---------------------------------------------------------------------------
def figs2348_soliton_off_on(cfg_off: str, cfg_full: str, seed: int,
                            n_tau: int, settle_rt: int, run_rt: int,
                            numerics: dict) -> dict:
    cav = load_cavity_params(CONFIG_PATH)
    cav = attach_dispersion(cav, n_tau)
    dw = SOLITON_DW_KAPPA * cav.kappa
    seed_field = sech_soliton_seed(dw, cav, n_tau=n_tau, pin=PIN_W)
    snap_int = max(run_rt // 64, 1)

    runs = {}
    for name, cfgp in (("off", cfg_off), ("on", cfg_full)):
        s0 = _quiet(_run, dw, settle_rt, cav, e0=seed_field, seed=seed,
                    n_tau=n_tau, pin=PIN_W, snapshot_interval=settle_rt,
                    config_path=cfgp, **numerics)
        sol = _quiet(_run, dw, run_rt, cav,
                     e0=np.asarray(s0["e_final"])[0],
                     delta_t0=float(np.asarray(s0["delta_t_final"]).ravel()[0]),
                     seed=seed, n_tau=n_tau, pin=PIN_W,
                     snapshot_interval=snap_int, config_path=cfgp, **numerics)
        runs[name] = sol

    t_r = cav.t_r
    mu = np.fft.fftshift(np.fft.fftfreq(n_tau) * n_tau)
    theta = 2.0 * np.pi * np.arange(n_tau) / n_tau

    # Cycle-averaged spectrum over the final quarter of the snapshots.
    def _avg_spec(sol):
        snaps = np.asarray(sol["E_snapshots"])[0]
        w = snaps[3 * snaps.shape[0] // 4:]
        sp = np.mean(np.abs(np.fft.fftshift(
            np.fft.fft(w, axis=-1), axes=-1)) ** 2, axis=0)
        return 10.0 * np.log10(np.maximum(sp / sp.max(), 1e-30))

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(mu, _avg_spec(runs["off"]), lw=0.7, label="all noise OFF")
    ax.plot(mu, _avg_spec(runs["on"]), lw=0.7, alpha=0.75,
            label="full stack ON (quantum+pump+TRN(K-G)+FSR)")
    ax.set_xlabel(r"mode index $\mu$")
    ax.set_ylabel("relative power [dB]")
    ax.set_ylim(-160, 5)
    ax.set_title(f"Single-soliton optical spectrum, cycle-averaged "
                 f"(final quarter), $\\delta\\omega = "
                 f"{SOLITON_DW_KAPPA:.0f}\\kappa$")
    ax.legend()
    fig.savefig(FIG_DIR / "soliton_spectrum_off_on.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, style in (("off", "-"), ("on", "--")):
        e_fin = np.asarray(runs[name]["e_final"])[0]
        ax.semilogy(theta, np.abs(e_fin) ** 2, style, lw=1.0,
                    label=f"noise {name.upper()}")
    ax.set_xlabel(r"fast-time angle $\theta$ [rad]")
    ax.set_ylabel(r"$|E(\theta)|^2$  [J]")
    ax.set_title("Temporal waveform: end-of-run field")
    ax.legend()
    fig.savefig(FIG_DIR / "soliton_waveform_off_on.png")
    plt.close(fig)

    # Fig 4: energy + delta_omega_eff decomposition (ON run).
    on = runs["on"]
    u = np.asarray(on["U_int_history"])[0]
    dweff = np.asarray(on["delta_omega_eff_history"])[0]
    d_t = np.asarray(on["DeltaT_history"])[0]
    omega0 = 2.0 * math.pi * 299_792_458.0 / cav.pump_wavelength_m
    phys = _load_config(cfg_full)
    thermal_shift = -(omega0 / float(phys["n0"])) \
        * float(phys["dn_dT_per_k"]) * d_t
    noise_part = dweff - dw - thermal_shift
    t_ms = np.arange(u.size) * t_r * 1e6
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    axs[0].plot(t_ms, u * 1e12, lw=0.6)
    axs[0].set_ylabel(r"$U_{int}$ [pJ]")
    axs[0].set_title("Full-stack run: intracavity energy and "
                     r"$\delta\omega_{eff}$ decomposition")
    axs[1].plot(t_ms, (dweff - dw) / cav.kappa, lw=0.6,
                label=r"$\delta\omega_{eff}-\delta\omega_{prog}$")
    axs[1].plot(t_ms, thermal_shift / cav.kappa, lw=0.9,
                label="thermal shift")
    axs[1].plot(t_ms, noise_part / cav.kappa, lw=0.5, alpha=0.7,
                label="stochastic (TRN/pyro + pump)")
    axs[1].set_xlabel(r"slow time [$\mu$s]")
    axs[1].set_ylabel(r"shift [$\kappa$]")
    axs[1].legend(fontsize=7)
    fig.savefig(FIG_DIR / "energy_detuning_series.png")
    plt.close(fig)

    # Fig 8: RF-domain beatnote proxy — PSD of the photodetected pulse-train
    # proxy Sum_j |E_j|^2(t) = (n_tau/t_r)*U_int(t), one sample per round
    # trip (the DC-photodetector record the paper uses for noise
    # diagnostics; the 24.6 GHz carrier lies above this record's Nyquist).
    from scipy.signal import welch

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    met_rf = {}
    for name in ("off", "on"):
        u_run = np.asarray(runs[name]["U_int_history"])[0]
        p_det = u_run * (n_tau / t_r)          # Sum_j |E_j|^2 proxy [J/s-ish]
        f_rf, s_rf = welch(p_det - p_det.mean(), fs=1.0 / t_r,
                           nperseg=min(1 << 12, u_run.size // 4))
        ax.loglog(f_rf[1:], s_rf[1:], lw=0.8,
                  label=f"noise {name.upper()}")
        met_rf[name] = {"rms_rel": float(np.std(p_det) / np.mean(p_det))}
    ax.set_xlabel("offset frequency f [Hz]")
    ax.set_ylabel(r"PSD of $\Sigma_j |E_j|^2$  [a.u.$^2$/Hz]")
    ax.set_title("RF beatnote proxy: DC-photodetected pulse-train "
                 "power fluctuations")
    ax.legend()
    fig.savefig(FIG_DIR / "rf_beatnote.png")
    plt.close(fig)

    # timing jitter cross-check on the ON run snapshots
    tj = timing_jitter(np.asarray(on["E_snapshots"])[0], snap_int, t_r)
    u_off = np.asarray(runs["off"]["U_int_history"])[0]
    return {
        "U_off_mean_J": float(np.mean(u_off[-run_rt // 4:])),
        "U_on_mean_J": float(np.mean(u[-run_rt // 4:])),
        "U_on_relstd": float(np.std(u[-run_rt // 4:])
                             / np.mean(u[-run_rt // 4:])),
        "thermal_shift_kappa_mean": float(np.mean(thermal_shift) / cav.kappa),
        "noise_part_kappa_rms": float(np.std(noise_part) / cav.kappa),
        "timing_jitter_rms_s": tj["jitter_rms_s"],
        "rf_beatnote": met_rf,
    }


# ---------------------------------------------------------------------------
# Figure 5: S_rep(f) with the TRN-limit overlay (validation 1)
# ---------------------------------------------------------------------------
def fig5_trn_limited_frep(cfg_trn_q: str, cfg_trn_only: str, seed: int,
                          n_tau: int, t_slow: int, probe_mus: tuple,
                          numerics: dict) -> dict:
    cav = load_cavity_params(CONFIG_PATH)   # Taylor D2 (FSR channel is
    #                                         dispersion-independent)
    dw = SOLITON_DW_KAPPA * cav.kappa
    seed_field = sech_soliton_seed(dw, cav, n_tau=n_tau, pin=PIN_W)

    trn_kg = TRNoise({**_load_config(None),
                      "trn_psd_model": "kondratiev_gorodetsky",
                      **KG_GEOMETRY})
    d1_over_omega0 = (2.0 * math.pi * cav.fsr_hz) / trn_kg.omega_0

    results = {}
    for name, cfgp in (("quantum+TRN+FSR", cfg_trn_q),
                       ("TRN+FSR only", cfg_trn_only)):
        s0 = _quiet(_run, dw, 2000, cav, e0=seed_field, seed=seed,
                    n_tau=n_tau, pin=PIN_W, snapshot_interval=2000,
                    config_path=cfgp, **numerics)
        sol = _quiet(_run, dw, t_slow, cav,
                     e0=np.asarray(s0["e_final"])[0],
                     delta_t0=float(np.asarray(s0["delta_t_final"]).ravel()[0]),
                     seed=seed + 1, n_tau=n_tau, pin=PIN_W,
                     snapshot_interval=max(t_slow // 16, 1),
                     config_path=cfgp, mode_probe_indices=probe_mus,
                     **numerics)
        probes = np.asarray(sol["mode_probe_history"])[0]
        phases = unwrapped_phases(probes[t_slow // 4:])
        nseg = min(1 << 13, phases.shape[0] // 4)
        # DIRECT estimators from the extreme probe pair. The tape-model
        # per-bin fit cannot resolve S_rep here: TRN is fully correlated
        # between the common and rep components with common/rep power ratio
        # (omega0/D1)^2 ~ 6e7, far beyond the fit's multiplicative-noise
        # dynamic range (see tape_model_fit docstring). The half-difference
        # cancels the common mode DETERMINISTICALLY (exact for the elastic
        # tape); the half-sum is the common mode.
        phi_rep = rep_rate_phase(phases, probe_mus, 0, len(probe_mus) - 1)
        phi_c = 0.5 * (phases[:, 0] + phases[:, -1])
        f_r, s_rep = frequency_noise_psd(phi_rep, cav.t_r, nperseg=nseg)
        _, s_c = frequency_noise_psd(phi_c, cav.t_r, nperseg=nseg)
        results[name] = {"f": f_r, "S_rep": s_rep, "S_c": s_c}

    # TRN limit: S_dnu,rep(f) = (D1/omega0)^2 * C_pull^2 * S_dT(f) / (2pi)^2.
    f = results["TRN+FSR only"]["f"]
    s_rep_pred = (d1_over_omega0 ** 2 * trn_kg.c_pull ** 2
                  * np.asarray(trn_kg.delta_t_psd(f)) / (2 * math.pi) ** 2)
    s_c_pred = (trn_kg.c_pull ** 2
                * np.asarray(trn_kg.delta_t_psd(f)) / (2 * math.pi) ** 2)

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in ("quantum+TRN+FSR", "TRN+FSR only"):
        r = results[name]
        ax.loglog(r["f"], r["S_rep"], lw=0.9,
                  label=rf"$S_{{\delta\nu,rep}}$ ({name})")
    ax.loglog(results["TRN+FSR only"]["f"], results["TRN+FSR only"]["S_c"],
              lw=0.9, alpha=0.6, label=r"$S_{\delta\nu,c}$ (TRN+FSR only)")
    ax.loglog(f, s_rep_pred, "k--", lw=1.6,
              label=r"TRN limit $(D_1/\omega_0)^2 C_{pull}^2 "
                    r"S_{\delta T}(f)/(2\pi)^2$")
    ax.loglog(f, s_c_pred, "k:", lw=1.2,
              label=r"common TRN $C_{pull}^2 S_{\delta T}(f)/(2\pi)^2$")
    ax.set_xlabel("offset frequency f [Hz]")
    ax.set_ylabel(r"$S_{\delta\nu}(f)$  [Hz$^2$/Hz]")
    ax.set_title("Repetition-rate frequency noise vs the TRN(K-G) limit\n"
                 "(pairwise-difference estimator; quantum per-line floor "
                 "shown by the full-stack curve)")
    ax.legend(fontsize=7)
    fig.savefig(FIG_DIR / "s_rep_trn_limit.png")
    plt.close(fig)

    # Band metrics: ratio of measured to predicted over the low-offset
    # decade (TRN-dominated band), and the inferred fix point
    # mu_fix = -sqrt(S_c/S_rep) (fully correlated, positive correlation:
    # phi_rep = (D1/omega0)*phi_c => mu_fix = -omega0/D1).
    r0 = results["TRN+FSR only"]
    band = (f >= f[2]) & (f <= f[2] * 30)
    ratio_db = 10.0 * np.log10(r0["S_rep"][band] / s_rep_pred[band])
    ratio_c_db = 10.0 * np.log10(r0["S_c"][band] / s_c_pred[band])
    mu_fix_inferred = -math.sqrt(
        float(np.median(r0["S_c"][band]))
        / float(np.median(r0["S_rep"][band])))
    rq = results["quantum+TRN+FSR"]
    return {
        "d1_over_omega0": d1_over_omega0,
        "c_pull_rad_s_K": trn_kg.c_pull,
        "trn_only_srep_band_median_ratio_db": float(np.median(ratio_db)),
        "trn_only_srep_band_max_abs_ratio_db": float(np.max(np.abs(ratio_db))),
        "trn_only_sc_band_median_ratio_db": float(np.median(ratio_c_db)),
        "quantum_stack_srep_band_median_ratio_db": float(np.median(
            10 * np.log10(rq["S_rep"][band] / s_rep_pred[band]))),
        "mu_fix_inferred_from_sc_over_srep": mu_fix_inferred,
        "mu_fix_predicted_minus_omega0_over_d1": float(-1.0 / d1_over_omega0),
    }


# ---------------------------------------------------------------------------
# Figure 6: linewidth vs mu parabola (validation 2)
# ---------------------------------------------------------------------------
def fig6_linewidth_parabola(cfg_lw: str, seed: int, n_tau: int, t_slow: int,
                            probe_mus: tuple, numerics: dict) -> dict:
    cav = load_cavity_params(CONFIG_PATH)
    cav = attach_dispersion(cav, n_tau)      # DW recoil needs measured D_int
    dw = SOLITON_DW_KAPPA * cav.kappa
    seed_field = sech_soliton_seed(dw, cav, n_tau=n_tau, pin=PIN_W)
    s0 = _quiet(_run, dw, 2000, cav, e0=seed_field, seed=seed, n_tau=n_tau,
                pin=PIN_W, snapshot_interval=2000, config_path=cfg_lw,
                **numerics)
    sol = _quiet(_run, dw, t_slow, cav, e0=np.asarray(s0["e_final"])[0],
                 delta_t0=float(np.asarray(s0["delta_t_final"]).ravel()[0]),
                 seed=seed + 1, n_tau=n_tau, pin=PIN_W,
                 snapshot_interval=max(t_slow // 16, 1), config_path=cfg_lw,
                 mode_probe_indices=probe_mus, **numerics)
    probes = np.asarray(sol["mode_probe_history"])[0]
    cut = t_slow // 4
    phases = unwrapped_phases(probes[cut:])
    # LAB-FRAME line phases: the solver frame co-rotates with the pump, so
    # the probes record only the frame-relative (transduced) part. The lab
    # comb line is nu_mu = nu_pump + Delta_mu; add the pump phase
    # phi_pump = 2*pi*Int delta_nu_p dt = -cumsum(pump_freq_noise_history)*
    # t_r (the history stores the -2*pi*delta_nu_p detuning contribution).
    pump_hist = np.asarray(sol["pump_freq_noise_history"])[0]
    phi_pump = -np.cumsum(pump_hist) * cav.t_r
    phases_lab = phases + phi_pump[cut:, None]
    nseg = min(1 << 13, phases.shape[0] // 4)

    linewidths = []
    for j in range(len(probe_mus)):
        f, s = frequency_noise_psd(phases_lab[:, j], cav.t_r, nperseg=nseg)
        linewidths.append(effective_linewidth(f, s))
    linewidths = np.asarray(linewidths)
    mus = np.asarray(probe_mus, dtype=float)

    # Parabola fit of the SQUARED linewidth (the beta-line area is the
    # quadratic-in-mu object: A_mu = A_c + 2 mu A_cr + mu^2 A_rep).
    coef = np.polyfit(mus, linewidths**2, 2)
    mu_fix = float(-coef[1] / (2.0 * coef[0])) if coef[0] != 0 else float("nan")
    mu_grid = np.linspace(mus.min() * 1.1, mus.max() * 1.1, 400)
    par = np.polyval(coef, mu_grid)

    # Direct fix-point estimate, immune to the beta-integral's sensitivity
    # to isolated spectral features: mu_fix(f) = -Re S_cr(f)/S_rep(f) with
    # S_rep from the extreme-pair half-difference and S_cr from the CSD of
    # the pair half-sum (common) with the half-difference (rep). Band-median
    # over the resolved pump band.
    phi_rep = rep_rate_phase(phases_lab, probe_mus, 0, len(probe_mus) - 1)
    phi_c = 0.5 * (phases_lab[:, 0] + phases_lab[:, -1])
    f_r, s_rep_d = frequency_noise_psd(phi_rep, cav.t_r, nperseg=nseg)
    f_x, s_cr = cross_frequency_psd(phi_c, phi_rep, cav.t_r, nperseg=nseg)
    band = (f_r > 1e6) & (f_r < 3e7)
    mu_fix_direct = float(np.median(
        -np.real(s_cr[band]) / np.maximum(s_rep_d[band], 1e-300)))

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(mus, linewidths / 1e6, "o", ms=5, label="per-line effective "
            "linewidth (beta-separation integral, lab frame)")
    ax.plot(mu_grid, np.sqrt(np.maximum(par, 0.0)) / 1e6, "-", lw=1.4,
            label="quadratic fit of FWHM$^2(\\mu)$")
    if np.isfinite(mu_fix):
        ax.axvline(mu_fix, color="gray", ls=":",
                   label=f"parabola vertex $\\mu_{{fix}} = {mu_fix:.0f}$")
    ax.axvline(mu_fix_direct, color="C3", ls="--", lw=1.0,
               label=f"CSD fix point $\\mu_{{fix}} = {mu_fix_direct:.0f}$")
    ax.set_xlabel(r"mode index $\mu$")
    ax.set_ylabel("effective linewidth [MHz]")
    ax.set_title("Comb-line linewidth vs mode index — WHITE-PUMP NULL BASELINE\n"
                 "(quantum + white pump freq noise, measured $D_{int}$; the "
                 "DW-recoil positive result is linewidth_vs_mu_dwrecoil)")
    ax.legend(fontsize=7)
    fig.savefig(FIG_DIR / "linewidth_vs_mu.png")
    plt.close(fig)

    curvature_significant = bool(
        coef[0] > 0 and (linewidths.max() - linewidths.min())
        > 0.05 * linewidths.mean())
    return {
        "role": "white-pump null baseline: the white-plateau pump preset at "
                "the deep operating point transduces almost pure common mode, "
                "so the β-line curvature is dominated by estimator noise and "
                "is NOT significant. The DW-recoil positive demonstration "
                "(flicker pump, low-f-resolved linewidths) is in "
                "fig6_linewidth_dwrecoil / linewidth_vs_mu_dwrecoil.png.",
        "probe_mus": list(map(int, probe_mus)),
        "linewidths_hz": [float(x) for x in linewidths],
        "parabola_coef_fwhm2": [float(c) for c in coef],
        "mu_fix_parabola_vertex": mu_fix,
        "mu_fix_direct_csd": mu_fix_direct,
        "curvature_positive_and_significant": curvature_significant,
        "pump_h0_hz2_per_hz": LINEWIDTH_H0,
    }


# ---------------------------------------------------------------------------
# Figure 6b: DW-recoil linewidth-vs-mu parabola (measured D_int) + Taylor
# negative control (Q3 closing validation B)
# ---------------------------------------------------------------------------
# Pump preset for the DW-recoil linewidth study: an ECDL white plateau plus a
# STRONG flicker (1/f) term. The β-separation-line linewidth is dominated by
# LOW-frequency frequency noise, which is exactly where (a) 1/f laser noise
# lives and (b) the measured dispersion's dispersive-wave recoil transduces
# pump frequency noise into a group-velocity / repetition-rate shift. Flicker
# also keeps the soliton healthy (its high-frequency power — which would
# destabilize the soliton — stays low, unlike a strong white plateau, which
# destroys the soliton at the levels needed for a comparable recoil signal).
DW_RECOIL_H0 = 3.0e3     # Hz^2/Hz (ECDL white plateau, Δν_L = π·h0 ≈ 9.4 kHz)
DW_RECOIL_HM1 = 1.0e13   # Hz^3/Hz (strong flicker: the low-f DW-recoil driver)
# Symmetric probe modes kept INSIDE the good-SNR core of the comb (|E_μ| well
# above the vacuum floor at both dispersions), so the Taylor control's far
# lines are not spuriously broadened by low-SNR phase noise.
DW_RECOIL_PROBE_MUS = (-750, -600, -450, -300, -150, 150, 300, 450, 600, 750)


def _line_linewidths(phases, t_r, probe_mus, nperseg):
    """Per-line β-separation-line effective linewidths [Hz], detrend OFF.

    ``detrend=False`` in the Welch estimate is essential: the default
    per-segment constant detrend removes the low-frequency flicker / DW-recoil
    content that the β-separation-line linewidth integral fundamentally
    depends on (with it on, near-fix-point lines floor at 0 and the parabola
    cannot be located). No metrology-numerics change — the ``detrend`` knob is
    an existing parameter of ``frequency_noise_psd``.
    """
    out = []
    for j in range(len(probe_mus)):
        f, s = frequency_noise_psd(phases[:, j], t_r, nperseg=nperseg,
                                   detrend=False)
        out.append(effective_linewidth(f, s))
    return np.asarray(out)


def _dw_recoil_run(cfg, measured, seed, n_tau, t_slow, probe_mus, numerics,
                   n_boot):
    cav = load_cavity_params(CONFIG_PATH)
    if measured:
        cav = attach_dispersion(cav, n_tau)
    dw = SOLITON_DW_KAPPA * cav.kappa
    seed_field = sech_soliton_seed(dw, cav, n_tau=n_tau, pin=PIN_W)
    s0 = _quiet(_run, dw, 2000, cav, e0=seed_field, seed=seed, n_tau=n_tau,
                pin=PIN_W, snapshot_interval=2000, config_path=cfg, **numerics)
    sol = _quiet(_run, dw, t_slow, cav, e0=np.asarray(s0["e_final"])[0],
                 delta_t0=float(np.asarray(s0["delta_t_final"]).ravel()[0]),
                 seed=seed + 1, n_tau=n_tau, pin=PIN_W,
                 snapshot_interval=max(t_slow // 16, 1), config_path=cfg,
                 mode_probe_indices=probe_mus, **numerics)
    probes = np.asarray(sol["mode_probe_history"])[0][t_slow // 4:]
    e_final = np.asarray(sol["e_final"])[0]
    contrast = float((np.abs(e_final) ** 2).max()
                     / max((np.abs(e_final) ** 2).mean(), 1e-300))
    phases = unwrapped_phases(probes)
    mus = np.asarray(probe_mus, dtype=float)
    nseg = min(1 << 12, phases.shape[0] // 8)

    # Per-line β-separation-line linewidths AND their significance from ONE
    # consistent estimator: the per-segment frequency-noise periodograms
    # (Welch segments via scipy.spectrogram, Hann window, no overlap,
    # detrend OFF so the low-frequency 1/f + DW-recoil content the β-line
    # integral needs is preserved). The POINT linewidth of each line is the
    # β-integral of the mean-over-segments PSD; the significance is a
    # SEGMENT-RESAMPLING bootstrap (resample the intact, continuous segments
    # with replacement, re-average, refit) — this preserves each segment's
    # low-f content, unlike a moving-block time bootstrap whose concatenation
    # discontinuities corrupt it (and unlike the raw parabola-fit covariance,
    # whose near-zero residual on this deterministic-transduction parabola
    # overstates significance). No metrology-numerics change: the
    # β-separation-line rule and the frequency-noise definition are the same
    # as analysis/noise_metrology.effective_linewidth / frequency_noise_psd.
    from scipy.signal import spectrogram

    dnu = np.diff(phases, axis=0) / (2.0 * math.pi * cav.t_r)   # (N-1, n_probe)
    f_sg, _t_sg, sxx = spectrogram(
        dnu, fs=1.0 / cav.t_r, nperseg=int(nseg), noverlap=0, window="hann",
        detrend=False, scaling="density", mode="psd", axis=0)   # (nf, n_probe, nseg)
    n_seg = sxx.shape[-1]
    keep = f_sg > 0
    f_pos = f_sg[keep]
    sxx = sxx[keep]                                             # (nf, n_probe, nseg)
    beta = 8.0 * math.log(2.0) * f_pos / math.pi ** 2          # β-separation line

    def _lw_from_psd(psd_lines):                                # (nf, n_probe) -> (n_probe,)
        area = np.trapezoid(np.where(psd_lines > beta[:, None], psd_lines, 0.0),
                            f_pos, axis=0)
        return np.sqrt(8.0 * math.log(2.0) * np.maximum(area, 0.0))

    lw = _lw_from_psd(sxx.mean(axis=2))                        # point estimate
    coef, cov = np.polyfit(mus, lw ** 2, 2, cov=True)
    a2 = float(coef[0])
    mu_fix_vertex = float(-coef[1] / (2.0 * coef[0])) if coef[0] != 0 else float("nan")

    rng = np.random.default_rng(0)
    a2_boot = []
    for _ in range(max(int(n_boot), 100)):
        pick = rng.integers(0, n_seg, n_seg)
        lwb = _lw_from_psd(sxx[:, :, pick].mean(axis=2))
        a2_boot.append(np.polyfit(mus, lwb ** 2, 2)[0])
    a2_boot = np.asarray(a2_boot)
    a2_med = float(np.median(a2_boot))
    a2_std = float(np.std(a2_boot, ddof=1))
    # A flat control (all linewidths at the β-floor -> a2 == 0, std == 0) has
    # ZERO significant curvature, not infinite (0/0). Only a genuinely
    # nonzero, perfectly reproduced a2 would be infinitely significant.
    if a2_std > 0:
        sig = float(a2 / a2_std)
    elif a2 == 0.0:
        sig = 0.0
    else:
        sig = float("inf")

    # Direct fix point −Re S_cr(f)/S_rep(f) over the low band where the
    # flicker-driven DW-recoil dominates; S_rep from the extreme-pair
    # half-difference, S_cr from its CSD with the half-sum common mode.
    phi_rep = rep_rate_phase(phases, probe_mus, 0, len(probe_mus) - 1)
    phi_c = 0.5 * (phases[:, 0] + phases[:, -1])
    f_r, s_rep = frequency_noise_psd(phi_rep, cav.t_r, nperseg=nseg,
                                     detrend=False)
    f_x, s_cr = cross_frequency_psd(phi_c, phi_rep, cav.t_r, nperseg=nseg)
    band = (f_r > 1e5) & (f_r < 1e7)
    mu_fix_direct = float(np.median(
        -np.real(s_cr[band]) / np.maximum(np.abs(s_rep[band]), 1e-300)))
    s_rep_band = float(np.median(s_rep[band]))

    return {
        "linewidths_hz": lw,
        "mus": mus,
        "coef": coef,
        "a2": a2,
        "a2_bootstrap_median": a2_med,
        "a2_bootstrap_std": a2_std,
        "curvature_significance_sigma": sig,
        "mu_fix_vertex": mu_fix_vertex,
        "mu_fix_direct": mu_fix_direct,
        "s_rep_band_hz2_per_hz": s_rep_band,
        "soliton_contrast": contrast,
    }


def fig6_linewidth_dwrecoil(cfg_dw: str, seed: int, n_tau: int, t_slow: int,
                            numerics: dict, *, n_boot: int = 200,
                            probe_mus=DW_RECOIL_PROBE_MUS) -> dict:
    """DW-recoil comb-line linewidth parabola (measured D_int) vs Taylor control.

    Physics (paper V.B.1, Lei et al. 2021): the comb-line frequency-noise PSD
    decomposes as S_μ(f) = S_c(f) + 2μ S_cr(f) + μ² S_rep(f); the β-line
    effective linewidth is therefore a parabola in μ with vertex at the fix
    point μ_fix = −S_cr/S_rep. With PURE quadratic dispersion (Taylor) pump
    frequency noise is transferred equally to every line (pure common mode ⇒
    S_rep → 0 ⇒ no curvature). With the MEASURED d_int_grid, dispersive-wave
    recoil converts pump frequency noise into a repetition-rate shift ⇒
    S_rep > 0 ⇒ a resolvable, significant parabola. This function runs both,
    fits the parabola, bootstraps the curvature-coefficient significance, and
    cross-checks μ_fix against the direct −Re S_cr/S_rep estimator.
    """
    res = {}
    for name, measured in (("measured D_int", True), ("Taylor D2", False)):
        res[name] = _dw_recoil_run(cfg_dw, measured, seed, n_tau, t_slow,
                                   probe_mus, numerics, n_boot)

    m, t = res["measured D_int"], res["Taylor D2"]
    a2_ratio = (m["a2"] / t["a2"]) if t["a2"] not in (0.0,) else float("inf")
    srep_ratio = (m["s_rep_band_hz2_per_hz"]
                  / max(abs(t["s_rep_band_hz2_per_hz"]), 1e-300))

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    mus = np.asarray(probe_mus, dtype=float)
    mu_grid = np.linspace(mus.min() * 1.08, mus.max() * 1.08, 400)
    colors = {"measured D_int": "C0", "Taylor D2": "C1"}
    for name in ("measured D_int", "Taylor D2"):
        r = res[name]
        ax.plot(mus, r["linewidths_hz"] / 1e6, "o", ms=5, color=colors[name],
                label=f"{name} (β-line linewidth)")
        par = np.polyval(r["coef"], mu_grid)
        ax.plot(mu_grid, np.sqrt(np.maximum(par, 0.0)) / 1e6, "-", lw=1.3,
                color=colors[name])
    mf = m["mu_fix_vertex"]
    if np.isfinite(mf):
        ax.axvline(mf, color="C3", ls=":", lw=1.1,
                   label=f"measured $\\mu_{{fix}}$ = {mf:.0f} "
                         f"(direct {m['mu_fix_direct']:.0f})")
    ax.set_xlabel(r"mode index $\mu$")
    ax.set_ylabel("effective linewidth [MHz]")
    ax.set_title(
        "Comb-line linewidth vs mode index: DW-recoil parabola "
        "(measured $D_{int}$) vs Taylor control\n"
        f"(quantum + flicker pump noise; measured curvature "
        f"{m['curvature_significance_sigma']:.1f}σ, "
        f"$a_2$ ratio {a2_ratio:.0f}×, $S_{{rep}}$ ratio {srep_ratio:.1e}×)")
    ax.legend(fontsize=7, loc="upper center")
    fig.savefig(FIG_DIR / "linewidth_vs_mu_dwrecoil.png")
    plt.close(fig)

    def _pack(r):
        return {
            "probe_mus": [int(x) for x in probe_mus],
            "linewidths_hz": [float(x) for x in r["linewidths_hz"]],
            "parabola_coef_fwhm2": [float(c) for c in r["coef"]],
            "a2_curvature": r["a2"],
            "a2_bootstrap_median": r["a2_bootstrap_median"],
            "a2_bootstrap_std": r["a2_bootstrap_std"],
            "curvature_significance_sigma": r["curvature_significance_sigma"],
            "mu_fix_parabola_vertex": r["mu_fix_vertex"],
            "mu_fix_direct_csd": r["mu_fix_direct"],
            "s_rep_band_hz2_per_hz": r["s_rep_band_hz2_per_hz"],
            "soliton_contrast": r["soliton_contrast"],
        }

    return {
        "pump_h0_hz2_per_hz": DW_RECOIL_H0,
        "pump_hm1_hz3_per_hz": DW_RECOIL_HM1,
        "operating_dw_over_kappa": SOLITON_DW_KAPPA,
        "measured_D_int": _pack(m),
        "taylor_D2": _pack(t),
        "a2_ratio_measured_over_taylor": a2_ratio,
        "s_rep_ratio_measured_over_taylor": srep_ratio,
        "mu_fix_vertex_direct_agreement_modes":
            abs(m["mu_fix_vertex"] - m["mu_fix_direct"]),
    }


# ---------------------------------------------------------------------------
# Figure 7: quiet-point sweep
# ---------------------------------------------------------------------------
def fig7_quiet_point(seed: int, n_tau: int, hold_rt: int, dw_grid,
                     f_offset: float) -> dict:
    sweeps = {}
    for name, meas in (("measured D_int", True), ("Taylor D2", False)):
        sweeps[name] = quiet_point_sweep(
            dw_grid, n_tau=n_tau, hold_rt=hold_rt,
            settle_rt=max(hold_rt // 4, 2000), f_offset_hz=f_offset,
            seed=seed, use_measured_dint=meas)

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    met = {}
    for name, marker in (("measured D_int", "o"), ("Taylor D2", "s")):
        s = sweeps[name]
        ax.semilogy(s["dw_over_kappa"], s["S_rep_at_offset"],
                    marker + "-", ms=4, lw=1.0, label=name)
        met[name] = {
            "quiet_point_dw_over_kappa":
                s.get("quiet_point_dw_over_kappa"),
            "quiet_point_S_rep": s.get("quiet_point_S_rep"),
            "quiet_point_depth_ratio": s.get("quiet_point_depth"),
            "dw_over_kappa": [float(x) for x in s["dw_over_kappa"]],
            "S_rep_at_offset": [float(x) for x in s["S_rep_at_offset"]],
        }
    ax.set_xlabel(r"detuning $\delta\omega/\kappa$")
    ax.set_ylabel(rf"$S_{{rep}}$({f_offset:.0e} Hz)  [Hz$^2$/Hz]")
    ax.set_title("Quiet-point sweep: repetition-rate noise vs detuning "
                 "(pump frequency noise ON)")
    ax.invert_xaxis()          # sweep runs high -> low detuning
    ax.legend()
    fig.savefig(FIG_DIR / "quiet_point_sweep.png")
    plt.close(fig)
    met["f_offset_hz"] = f_offset
    return met


# ---------------------------------------------------------------------------
# Performance: probe / FSR overheads + zero-new-ops property
# ---------------------------------------------------------------------------
def perf_overheads(seed: int, n_tau: int = 8192, t_slow: int = 2000) -> dict:
    from simulator.lle_solver import solve_lle_ssfm_jax

    kappa, kappa_c = 1.519e8, 1.215e8
    common = dict(pin=PIN_W, delta_omega=SOLITON_DW_KAPPA * kappa,
                  t_slow=t_slow, beta=[1.578e-18], kappa=kappa,
                  kappa_c=kappa_c, rng_key=jax.random.PRNGKey(seed),
                  n_tau=n_tau, snapshot_interval=max(t_slow // 8, 1))

    def timed(**kw):
        _quiet(solve_lle_ssfm_jax, **common, **kw)      # compile + warm cache
        t0 = time.time()
        _quiet(solve_lle_ssfm_jax, **common, **kw)
        return time.time() - t0

    t_base = timed()
    t_probe = timed(mode_probe_indices=(-150, -100, -50, 50, 100, 150))
    t_fsr = timed(fsr_delta_d1_override=np.full(t_slow, 1.0))
    # Structural evidence for the zero-new-ops claim: primitive counts of
    # the traced scan body with the channels off/on. The flags-off count is
    # the pre-change solver's count (the legacy path was separately verified
    # bit-identical to git HEAD output-for-output).
    from tests.test_quantum_noise import _scan_body_primitives

    n_ops_off = len(_scan_body_primitives(qnoise_enabled=False))
    out = {
        "n_tau": n_tau,
        "t_slow": t_slow,
        "base_s": t_base,
        "probes_s": t_probe,
        "fsr_s": t_fsr,
        "probe_overhead_pct": 100.0 * (t_probe / t_base - 1.0),
        "fsr_overhead_pct": 100.0 * (t_fsr / t_base - 1.0),
        "scan_body_primitives_all_new_channels_off": n_ops_off,
    }
    print(f"[perf] n_tau={n_tau}: base {t_base:.2f}s, probes {t_probe:.2f}s "
          f"(+{out['probe_overhead_pct']:.1f}%), fsr {t_fsr:.2f}s "
          f"(+{out['fsr_overhead_pct']:.1f}%)")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=str, default="",
                    help="comma list of sections to (re)run: fig1,fig234,"
                         "fig5,fig6,fig6dw,fig7,perf; default = all. Other "
                         "sections keep their values from the existing "
                         "JSON (merged).")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    def _want(tag):
        return not only or tag in only

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="noise_report_"))

    if args.quick:
        n_tau_big, settle, run_rt = 512, 200, 400
        n_tau_frep, t_frep = 256, 4096
        n_tau_lw, t_lw = 256, 4096
        frep_mus = (-60, -40, -20, 20, 40, 60)
        lw_mus = (-60, -40, -20, 20, 40, 60)
        n_tau_dw, t_dw = 1024, 12_000
        dw_mus = (-300, -240, -180, -120, -60, 60, 120, 180, 240, 300)
        dw_boot = 60
        qp_grid = np.linspace(10.5, 9.5, 3)
        qp_ntau, qp_hold = 512, 3000
        n_samp = 1 << 14
        numerics = dict(n_substeps=1, dealias_two_thirds=False,
                        edge_absorber=False, dispersion_validity_mask=False)
        perf_ntau, perf_t = 1024, 400
    else:
        n_tau_big, settle, run_rt = 8192, 2000, 6000
        n_tau_frep, t_frep = 1024, 1 << 17
        n_tau_lw, t_lw = 4096, 1 << 16
        frep_mus = (-150, -100, -50, 50, 100, 150)
        lw_mus = (-240, -160, -80, 80, 160, 240)
        n_tau_dw, t_dw = 4096, 50_000
        dw_mus = DW_RECOIL_PROBE_MUS
        dw_boot = 240
        qp_grid = np.linspace(12.5, 7.5, 21)
        qp_ntau, qp_hold = 4096, 12_000
        n_samp = 1 << 17
        numerics = PRODUCTION_NUMERICS
        perf_ntau, perf_t = 8192, 2000

    cfg_off = str(write_noise_off_config())
    # Full stack: quantum + ECDL pump + K-G TRN + FSR noise (T_k stays 300).
    cfg_full = _sidecar(
        CONFIG_PATH, "full",
        quantum_noise_enabled=1, quantum_noise_injection_cadence=1,
        pump_noise_enabled=1, pump_freq_noise_h0_hz2_per_hz=ECDL_H0,
        pump_freq_noise_hm1_hz3_per_hz=1.0e10,
        pump_rin_floor_dbc_per_hz=-160.0,
        trn_psd_model="kondratiev_gorodetsky", **KG_GEOMETRY,
        fsr_noise_enabled=1)
    # Validation-1 stacks: TRN(K-G)+FSR with and without quantum; pump off.
    cfg_trn_q = _sidecar(
        CONFIG_PATH, "trnq",
        quantum_noise_enabled=1, quantum_noise_injection_cadence=1,
        pump_noise_enabled=0,
        trn_psd_model="kondratiev_gorodetsky", **KG_GEOMETRY,
        fsr_noise_enabled=1)
    cfg_trn_only = _sidecar(
        CONFIG_PATH, "trnonly",
        quantum_noise_enabled=0, pump_noise_enabled=0,
        trn_psd_model="kondratiev_gorodetsky", **KG_GEOMETRY,
        fsr_noise_enabled=1)
    # Validation-2 stack: quantum + STRONG white pump noise (in-band
    # beta-line crossing), thermal noise off so the parabola isolates the
    # pump->rep transduction.
    cfg_lw = _sidecar(
        cfg_off, "lw",
        quantum_noise_enabled=1, quantum_noise_injection_cadence=1,
        pump_noise_enabled=1,
        pump_freq_noise_h0_hz2_per_hz=LINEWIDTH_H0)
    # Validation-2 (DW recoil): quantum + flicker-dominated pump noise
    # (thermal off) so the measured-D_int dispersive-wave recoil transduces
    # pump frequency noise into a resolvable repetition-rate parabola while
    # the soliton stays healthy; Taylor D2 is the negative control.
    cfg_dw = _sidecar(
        cfg_off, "dw",
        quantum_noise_enabled=1, quantum_noise_injection_cadence=1,
        pump_noise_enabled=1,
        pump_freq_noise_h0_hz2_per_hz=DW_RECOIL_H0,
        pump_freq_noise_hm1_hz3_per_hz=DW_RECOIL_HM1)

    metrics = {"seed": args.seed, "quick": bool(args.quick)}
    if only and RESULTS_JSON.exists():
        metrics.update(json.load(open(RESULTS_JSON, encoding="utf-8")))
        metrics["seed"], metrics["quick"] = args.seed, bool(args.quick)

    if _want("fig1"):
        print("[1/6] PSD model overlay ...")
        metrics["fig1_psd_models"] = fig1_psd_models(tmp, args.seed, n_samp)

    if _want("fig234"):
        print("[2/6] soliton OFF vs full stack ON (figs 2-4, 8) ...")
        metrics["figs234_8_soliton"] = figs2348_soliton_off_on(
            cfg_off, cfg_full, args.seed, n_tau_big, settle, run_rt, numerics)

    if _want("fig5"):
        print("[3/6] TRN-limited f_rep (fig 5) ...")
        frep_numerics = numerics if args.quick else dict(
            n_substeps=1, dealias_two_thirds=True, edge_absorber=True,
            dispersion_validity_mask=False)
        metrics["fig5_trn_frep"] = fig5_trn_limited_frep(
            cfg_trn_q, cfg_trn_only, args.seed, n_tau_frep, t_frep,
            frep_mus, frep_numerics)

    if _want("fig6"):
        print("[4/6] linewidth-vs-mu parabola (fig 6, white-pump null) ...")
        metrics["fig6_linewidth"] = fig6_linewidth_parabola(
            cfg_lw, args.seed, n_tau_lw, t_lw, lw_mus, numerics)

    if _want("fig6dw"):
        print("[4b] DW-recoil linewidth parabola vs Taylor control (fig 6b) ...")
        metrics["fig6_linewidth_dwrecoil"] = fig6_linewidth_dwrecoil(
            cfg_dw, args.seed, n_tau_dw, t_dw, numerics,
            n_boot=dw_boot, probe_mus=dw_mus)

    if _want("fig7"):
        print("[5/6] quiet-point sweep (fig 7) ...")
        metrics["fig7_quiet_point"] = fig7_quiet_point(
            args.seed, qp_ntau, qp_hold, qp_grid, f_offset=2.0e8)

    if _want("perf"):
        print("[6/6] performance overheads ...")
        metrics["performance"] = perf_overheads(args.seed, perf_ntau, perf_t)

    # Provenance stamp (analysis/_provenance.py; same convention as
    # quantum_noise_report.json): generating script, short git commit, a hash
    # of the resolved physical_parameters block the runs derived their
    # sidecars from, the seed, the quick flag, and a UTC timestamp. Always
    # refreshed so it reflects the latest (possibly --only) regeneration.
    from analysis._provenance import provenance_stamp

    metrics["provenance"] = provenance_stamp(
        "analysis/noise_comparison_report.py", args.seed,
        physical_params=_load_config(str(CONFIG_PATH)),
        quick=bool(args.quick),
    )

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {RESULTS_JSON}")
    for k in metrics:
        print(" ", k)


if __name__ == "__main__":
    main()
