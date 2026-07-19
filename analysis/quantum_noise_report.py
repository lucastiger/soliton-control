#!/usr/bin/env python
"""Quantum-vacuum-noise validation report (arXiv:2604.05897 Sec. V.B.2, Eq. 126).

Seeded CLI driver producing the quantum-noise deliverables:

(a) OFF vs ON single-soliton comparison at delta_omega = 11*kappa (sech-ansatz
    warm start per :mod:`analysis.dks_access` conventions, measured D_int grid,
    ``PRODUCTION_NUMERICS``), 20,000 RT, breather-safe cycle-averaged spectra
    via :func:`analysis.spectral_metrics.average_power_spectrum`. Also measures
    the far-wing spectral floor of the ON run against the half-photon vacuum
    level hbar*omega0/2 per mode, and the OFF run's numerical floor.
(b) Vacuum-equilibrium occupancy: pin = 0, 4 trajectories, <n_mu> vs mu with
    the 0.5 reference, plus the modal-autocorrelation kappa/2 decay fit.
(c) MI-from-vacuum growth snapshot sequence at delta_omega = -2.5*kappa.

Outputs
-------
* ``analysis/figures/quantum_noise_*.png`` (150 dpi, publication style)
* ``analysis/results/quantum_noise_report.json`` — every measured number.

The stochastic detuning channels are disabled with the T_k = 0 sidecar
(:func:`analysis.run_detuning_sweep.write_noise_off_config`, which also forces
the quantum flag off in the config — the ON runs re-enable it via the solver
kwarg), so OFF runs are fully deterministic and ON runs contain ONLY quantum
noise. All RNG flows from the single ``--seed``.

Wing-band note: with ``dealias_two_thirds`` on, modes |mu| > n_tau/3 are
zeroed every kick and are excluded from all statistics; the floor band
2000 <= |mu| <= 2600 sits inside the 2/3 boundary (2730 at n_tau = 8192) yet
far outside the ~190-mode soliton comb bandwidth.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from analysis.dks_access import (  # noqa: E402
    PIN_W,
    PRODUCTION_NUMERICS,
    _run,
    attach_dispersion,
    load_cavity_params,
    sech_soliton_seed,
)
from analysis.plot_utils import apply_pub_style  # noqa: E402
from analysis.run_detuning_sweep import write_noise_off_config  # noqa: E402
from analysis.spectral_metrics import average_power_spectrum  # noqa: E402
from simulator.lle_solver import (  # noqa: E402
    _load_config,
    build_dispersion,
    _build_omega_grid,
    d2_to_beta2_lle,
    hbar_omega0_from_config,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_JSON = Path(__file__).resolve().parent / "results" / "quantum_noise_report.json"

SOLITON_DW_KAPPA = 11.0
# |mu| floor-measurement band as fractions of n_tau: (0.244, 0.317)*n_tau =
# (2000, 2600) at the production n_tau = 8192 (see module docstring).
WING_BAND_FRAC = (0.244, 0.317)


def wing_band(n_tau: int) -> tuple[int, int]:
    return (int(WING_BAND_FRAC[0] * n_tau), int(WING_BAND_FRAC[1] * n_tau))


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ---------------------------------------------------------------------------
# (a) single-soliton OFF vs ON
# ---------------------------------------------------------------------------
def soliton_comparison(cav, config_path, seed: int, n_tau: int, settle_rt: int,
                       compare_rt: int, snap_int: int) -> dict:
    dw = SOLITON_DW_KAPPA * cav.kappa
    hbw = hbar_omega0_from_config(_load_config(config_path))
    seed_field = sech_soliton_seed(dw, cav, n_tau=n_tau, pin=PIN_W)

    out = {}
    for label, qn in (("off", False), ("on", True)):
        settle = _quiet(_run, dw, settle_rt, cav, e0=seed_field, seed=seed,
                        n_tau=n_tau, pin=PIN_W, snapshot_interval=settle_rt,
                        config_path=config_path, quantum_noise_enabled=qn,
                        **PRODUCTION_NUMERICS)
        sol = _quiet(_run, dw, compare_rt, cav,
                     e0=np.asarray(settle["e_final"])[0],
                     delta_t0=settle["delta_t_final"], seed=seed + 1,
                     n_tau=n_tau, pin=PIN_W, snapshot_interval=snap_int,
                     config_path=config_path, quantum_noise_enabled=qn,
                     **PRODUCTION_NUMERICS)
        snaps = np.asarray(sol["E_snapshots"])[0]
        # Breather-safe: cycle/slow-time average in LINEAR power over the final
        # quarter of the hold (>= many breathing periods; 11*kappa is
        # stationary, and the averaging beats down the ON run's stochastic
        # per-snapshot floor fluctuations).
        window = snaps[3 * snaps.shape[0] // 4:]
        mu, p_mu = average_power_spectrum(window)
        out[label] = {
            "mu": mu,
            "P_mu": p_mu,
            "U_int": np.asarray(sol["U_int_history"])[0],
            "final_rt": np.abs(np.asarray(sol["e_final"])[0]) ** 2,
            "label_history": np.asarray(sol["label_history"])[0],
        }

    # Far-wing floor: modal ENERGY U_mu = P_mu / n_tau^2 [J] against the
    # symmetric-ordered vacuum hbar*omega0/2, and in dB relative to the pump.
    band = wing_band(n_tau)
    # PRODUCTION_NUMERICS runs with dealias_two_thirds ON: modes |mu| > n_tau/3
    # are zeroed every kick and are deliberately under-occupied, so the floor
    # band must sit strictly inside the dealias window.
    assert PRODUCTION_NUMERICS.get("dealias_two_thirds") is not True or (
        band[1] <= n_tau / 3.0
    ), f"wing band {band} extends past the 2/3 dealias boundary {n_tau/3:.0f}"
    wing = (np.abs(out["on"]["mu"]) >= band[0]) & (
        np.abs(out["on"]["mu"]) <= band[1]
    )
    pump_idx = int(np.where(out["on"]["mu"] == 0)[0][0])
    floor = {}
    for label in ("off", "on"):
        p_mu = out[label]["P_mu"]
        u_wing = float(np.mean(p_mu[wing])) / n_tau**2                    # J/mode
        floor[label] = {
            "wing_mean_energy_j": u_wing,
            "wing_over_half_photon": u_wing / (hbw / 2.0),
            "wing_db_rel_pump": 10.0 * math.log10(
                max(np.mean(p_mu[wing]), 1e-300) / p_mu[pump_idx]
            ),
        }
    out["floor"] = floor
    out["hbar_omega0_j"] = hbw
    return out


# ---------------------------------------------------------------------------
# (b) vacuum equilibrium
# ---------------------------------------------------------------------------
def vacuum_equilibrium(cav, config_path, seed: int, n_tau: int = 1024,
                       t_slow: int = 6000, snap_int: int = 2) -> dict:
    phys = _load_config(config_path)
    hbw = hbar_omega0_from_config(phys)
    beta2 = d2_to_beta2_lle(phys["d2_rad_per_s2"], phys["fsr_hz"])
    from simulator.lle_solver import solve_lle_ssfm_jax

    sol = _quiet(
        solve_lle_ssfm_jax,
        pin=0.0,
        delta_omega=np.zeros((4, t_slow)),
        t_slow=t_slow,
        beta=[beta2],
        kappa=cav.kappa,
        kappa_c=cav.kappa_c,
        rng_key=jax.random.PRNGKey(seed),
        n_tau=n_tau,
        config_path=config_path,
        snapshot_interval=snap_int,
        quantum_noise_enabled=True,
    )
    snaps = np.asarray(sol["E_snapshots"])
    modes = np.fft.fft(snaps, axis=-1)
    win = modes[:, modes.shape[1] // 2:, :]
    n_mu = (np.abs(win) ** 2).mean(axis=(0, 1)) / (n_tau**2 * hbw)
    mu = np.fft.fftfreq(n_tau) * n_tau

    # Phase-corrected all-modes decay estimator: every mode decays at kappa/2
    # under a KNOWN linear phase exp(-i*D_int(mu)*tau); rotating it out lets
    # C_mu(tau) average coherently over modes (no |.| bias).
    t_r = cav.t_r
    disp = np.asarray(
        build_dispersion(_build_omega_grid(n_tau, t_r), (beta2,))
    )
    keep = np.abs(mu) <= n_tau / 3.0
    lt = 1.0 / (cav.kappa * t_r)
    lags = np.arange(max(int(round(0.2 * lt / snap_int)), 1),
                     int(round(2.0 * lt / snap_int)) + 1,
                     max(int(round(lt / snap_int / 40)), 1))
    wk = win[:, :, keep]
    dphi = disp[keep] * snap_int * t_r
    corr = np.array([
        float(np.mean(np.real(
            np.mean(wk[:, l:, :] * np.conj(wk[:, : wk.shape[1] - l, :]),
                    axis=(0, 1)) * np.exp(1j * dphi * l)
        )))
        for l in lags
    ])
    tau_s = lags * snap_int * t_r
    rate = float(-np.polyfit(tau_s, np.log(corr), 1)[0])
    return {
        "mu": mu, "n_mu": n_mu,
        "grand_mean": float(np.mean(n_mu)),
        "grand_mean_mu_le_third": float(np.mean(n_mu[keep])),
        "tau_s": tau_s, "corr": corr,
        "decay_rate_rad_s": rate,
        "rate_over_half_kappa": rate / (cav.kappa / 2.0),
    }


# ---------------------------------------------------------------------------
# (c) MI growth from vacuum
# ---------------------------------------------------------------------------
def mi_growth(cav, config_path, seed: int, n_tau: int = 1024,
              t_slow: int = 6000, snap_int: int = 25) -> dict:
    phys = _load_config(config_path)
    hbw = hbar_omega0_from_config(phys)
    gamma = float(phys["gamma_LLE_per_J_per_s"])
    d2 = float(phys["d2_rad_per_s2"])
    beta2 = d2_to_beta2_lle(d2, phys["fsr_hz"])
    p_th = cav.kappa**3 / (8.0 * gamma * cav.kappa_c)
    mu_eq62 = math.sqrt((cav.kappa / d2) * (1.0 + math.sqrt(PIN_W / p_th - 1.0)))
    from simulator.lle_solver import solve_lle_ssfm_jax

    sol = _quiet(
        solve_lle_ssfm_jax,
        pin=PIN_W,
        delta_omega=-2.5 * cav.kappa,
        t_slow=t_slow,
        beta=[beta2],
        kappa=cav.kappa,
        kappa_c=cav.kappa_c,
        rng_key=jax.random.PRNGKey(seed),
        n_tau=n_tau,
        config_path=config_path,
        snapshot_interval=snap_int,
        quantum_noise_enabled=True,
    )
    snaps = np.asarray(sol["E_snapshots"])[0]
    n_mode = np.abs(np.fft.fft(snaps, axis=-1)) ** 2 / (n_tau**2 * hbw)
    mu = np.fft.fftfreq(n_tau) * n_tau
    band = (np.abs(mu) >= 10) & (np.abs(mu) <= n_tau / 3.0)
    peak = n_mode[:, band].max(axis=-1)
    idx = int(np.argmax(peak > 1e5)) if (peak > 1e5).any() else snaps.shape[0] - 1
    mu_star = abs(float(mu[band][np.argmax(n_mode[idx][band])]))
    return {
        "mu": mu, "n_mode": n_mode, "snap_rt": np.arange(snaps.shape[0]) * snap_int,
        "detect_idx": idx, "mu_star": mu_star, "mu_eq62": mu_eq62,
        "ratio": mu_star / mu_eq62, "p_th_w": p_th,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(sol_cmp, vac, mig, cav, n_tau_sol: int) -> list[Path]:
    apply_pub_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hbw = sol_cmp["hbar_omega0_j"]
    paths = []

    # (1) optical spectrum OFF vs ON, dB vs mode number, hbar*omega0/2 floor
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pump = sol_cmp["off"]["P_mu"].max()
    for label, color in (("off", "C0"), ("on", "C3")):
        mu, p = sol_cmp[label]["mu"], sol_cmp[label]["P_mu"]
        ax.plot(mu, 10 * np.log10(np.maximum(p / pump, 1e-30)),
                color=color, lw=0.7, alpha=0.85,
                label=f"quantum noise {label.upper()}")
    vac_db = 10 * math.log10((hbw / 2.0) * n_tau_sol**2 / pump)
    ax.axhline(vac_db, color="k", ls="--", lw=1.0,
               label=r"$\hbar\omega_0/2$ vacuum floor")
    ax.set_xlim(-n_tau_sol / 3, n_tau_sol / 3)
    ax.set_ylim(vac_db - 30, 5)
    ax.set_xlabel("relative mode number $\\mu$")
    ax.set_ylabel("cycle-averaged mode power [dB rel. pump line]")
    ax.set_title(
        f"Single-DKS optical spectrum at $\\delta\\omega = {SOLITON_DW_KAPPA:g}"
        f"\\kappa$: quantum noise OFF vs ON")
    ax.legend(loc="upper right")
    paths.append(FIG_DIR / "quantum_noise_spectrum_off_on.png")
    fig.savefig(paths[-1]); plt.close(fig)

    # (2) <n_mu> vs mu with 0.5 reference
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    order = np.argsort(vac["mu"])
    ax.plot(vac["mu"][order], vac["n_mu"][order], lw=0.6, color="C0",
            label=r"measured $\langle n_\mu\rangle$")
    ax.axhline(0.5, color="k", ls="--", lw=1.0, label=r"$\langle n_\mu\rangle = 1/2$")
    ax.set_xlabel("relative mode number $\\mu$")
    ax.set_ylabel(r"mean photon number $\langle n_\mu\rangle$")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Vacuum equilibrium occupancy (pin = 0, 4 trajectories; grand mean "
        f"{vac['grand_mean']:.3f})")
    ax.legend()
    paths.append(FIG_DIR / "quantum_noise_vacuum_occupancy.png")
    fig.savefig(paths[-1]); plt.close(fig)

    # (3) temporal waveform OFF vs ON, last round trip
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    tau_frac = np.linspace(0, 1, sol_cmp["off"]["final_rt"].size, endpoint=False)
    for label, color in (("off", "C0"), ("on", "C3")):
        ax.semilogy(tau_frac, np.maximum(sol_cmp[label]["final_rt"], 1e-30),
                    color=color, lw=0.8, label=f"quantum noise {label.upper()}")
    ax.set_xlabel("fast time $\\tau / t_r$")
    ax.set_ylabel("$|E(\\tau)|^2$ [J]")
    ax.set_title("Single-DKS temporal waveform, final round trip")
    ax.legend()
    paths.append(FIG_DIR / "quantum_noise_waveform_off_on.png")
    fig.savefig(paths[-1]); plt.close(fig)

    # (4) intracavity energy time series OFF vs ON
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for label, color in (("off", "C0"), ("on", "C3")):
        u = sol_cmp[label]["U_int"]
        ax.plot(np.arange(u.size), u, color=color, lw=0.7,
                label=f"quantum noise {label.upper()}")
    ax.set_xlabel("round trip")
    ax.set_ylabel("$U_{int}$ [J·$t_r$] (stored solver convention)")
    ax.set_title("Intracavity energy history, single-DKS hold")
    ax.legend()
    paths.append(FIG_DIR / "quantum_noise_uint_off_on.png")
    fig.savefig(paths[-1]); plt.close(fig)

    # (5) modal autocorrelation with kappa/2 fit
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.semilogy(vac["tau_s"] * 1e6, vac["corr"] / vac["corr"][0], "o", ms=3,
                color="C0", label="phase-corrected modal autocorrelation")
    ax.semilogy(vac["tau_s"] * 1e6,
                np.exp(-vac["decay_rate_rad_s"] * (vac["tau_s"] - vac["tau_s"][0]))
                * (vac["corr"][0] / vac["corr"][0]) *
                math.exp(-vac["decay_rate_rad_s"] * 0), "-", color="C3",
                label=(f"fit: rate = {vac['rate_over_half_kappa']:.3f}"
                       r"$\times\kappa/2$"))
    ax.set_xlabel(r"lag $\tau$ [$\mu$s]")
    ax.set_ylabel(r"$|C(\tau)|/|C(\tau_0)|$")
    ax.set_title("Vacuum-mode autocorrelation decay vs the cavity linewidth")
    ax.legend()
    paths.append(FIG_DIR / "quantum_noise_autocorrelation.png")
    fig.savefig(paths[-1]); plt.close(fig)

    # (6, report part c) MI-from-vacuum growth snapshot sequence
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    idxs = np.unique(np.clip(np.array(
        [mig["detect_idx"] - 3 * k for k in range(5)][::-1] + [mig["detect_idx"] + 6]
    ), 0, mig["n_mode"].shape[0] - 1))
    order = np.argsort(mig["mu"])
    for i in idxs:
        ax.semilogy(mig["mu"][order], np.maximum(mig["n_mode"][i][order], 1e-3),
                    lw=0.7, label=f"RT {mig['snap_rt'][i]}")
    ax.axhline(0.5, color="k", ls="--", lw=0.9, label=r"vacuum $n_\mu = 1/2$")
    for s in (+1, -1):
        ax.axvline(s * mig["mu_eq62"], color="gray", ls=":", lw=0.9)
    ax.set_xlim(-mig["mu"].size / 3, mig["mu"].size / 3)
    ax.set_xlabel("relative mode number $\\mu$")
    ax.set_ylabel(r"modal photon number $n_\mu$")
    ax.set_title(
        f"MI growth from vacuum at $\\delta\\omega = -2.5\\kappa$ "
        f"(measured $|\\mu^*|$ = {mig['mu_star']:.0f}, Eq. 62: "
        f"{mig['mu_eq62']:.0f})")
    ax.legend(ncols=2, fontsize=7)
    paths.append(FIG_DIR / "quantum_noise_mi_growth.png")
    fig.savefig(paths[-1]); plt.close(fig)

    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tau", type=int, default=8192,
                    help="FFT grid of the single-soliton comparison")
    ap.add_argument("--settle-rt", type=int, default=4000)
    ap.add_argument("--compare-rt", type=int, default=20000)
    ap.add_argument("--quick", action="store_true",
                    help="reduced sizes for a smoke run (not publication data)")
    args = ap.parse_args(argv)

    n_tau = 1024 if args.quick else args.n_tau
    settle_rt = 500 if args.quick else args.settle_rt
    compare_rt = 1000 if args.quick else args.compare_rt
    vac_t = 2000 if args.quick else 6000
    mi_t = 2000 if args.quick else 6000

    config_path = str(write_noise_off_config())   # T_k = 0, quantum forced off;
    # the ON runs re-enable via the solver kwarg
    cav = load_cavity_params()
    cav = attach_dispersion(cav, n_tau)

    print(f"[report] (a) single-DKS OFF vs ON at {SOLITON_DW_KAPPA:g} kappa ...")
    sol_cmp = soliton_comparison(cav, config_path, args.seed, n_tau,
                                 settle_rt, compare_rt,
                                 snap_int=max(compare_rt // 2000, 1))
    print(f"[report] (b) vacuum equilibrium ...")
    vac = vacuum_equilibrium(cav, config_path, args.seed + 100, t_slow=vac_t)
    print(f"[report] (c) MI growth from vacuum ...")
    mig = mi_growth(cav, config_path, args.seed + 200, t_slow=mi_t)

    figs = make_figures(sol_cmp, vac, mig, cav, n_tau)

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Provenance per the repo's artifact conventions (cf.
    # scripts/artifact_manifest.py and dks_artifact_provenance.json): the
    # generating script, the commit the run was made from, a hash of the BASE
    # config the sidecar was derived from, and the seed.
    import hashlib
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    cfg_sha = hashlib.sha256(
        (repo_root / "config" / "sin_params.yaml").read_bytes()
    ).hexdigest()

    report = {
        "provenance": {
            "script": "analysis/quantum_noise_report.py",
            "git_commit": commit,
            "base_config_sha256": cfg_sha,
            "seed": args.seed,
        },
        "seed": args.seed,
        "quick": bool(args.quick),
        "hbar_omega0_j": sol_cmp["hbar_omega0_j"],
        "soliton_comparison": {
            "delta_omega_over_kappa": SOLITON_DW_KAPPA,
            "n_tau": n_tau,
            "compare_rt": compare_rt,
            "wing_band_mu": list(wing_band(n_tau)),
            "floor": sol_cmp["floor"],
            "u_int_mean_off": float(np.mean(sol_cmp["off"]["U_int"])),
            "u_int_mean_on": float(np.mean(sol_cmp["on"]["U_int"])),
            "label_mode_off": int(np.bincount(
                sol_cmp["off"]["label_history"]).argmax()),
            "label_mode_on": int(np.bincount(
                sol_cmp["on"]["label_history"]).argmax()),
        },
        "vacuum_equilibrium": {
            "grand_mean_n_mu": vac["grand_mean"],
            "grand_mean_mu_le_third": vac["grand_mean_mu_le_third"],
            "decay_rate_rad_s": vac["decay_rate_rad_s"],
            "rate_over_half_kappa": vac["rate_over_half_kappa"],
        },
        "mi_growth": {
            "mu_star": mig["mu_star"],
            "mu_eq62": mig["mu_eq62"],
            "ratio": mig["ratio"],
            "p_th_w": mig["p_th_w"],
        },
        "figures": [str(p) for p in figs],
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] wrote {RESULTS_JSON}")
    for p in figs:
        print(f"[report] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
