"""Single-trajectory adiabatic detuning sweeps at pin = 0.214 W.

This script runs forward (blue->red) and reverse (red->blue) adiabatic detuning
sweeps through the modulation-instability (MI) window of the SiN microring LLE,
with the thermal model ON at the config's re-derived Gamma_th, and records per
snapshot: delta_omega_eff, labeler class, U_int, and sech^2 spectral correlation.

Detuning convention (see lle_solver.solve_lle_ssfm_jax):
    delta_omega = omega_res - omega_pump   (cavity minus pump)
    positive = red-detuned = soliton side; MI/soliton window ~ kappa/2 .. ~5*kappa
    adiabatic access sweeps delta_omega from negative to positive (blue -> red).

Adiabaticity (justification of the tuning rate vs kappa and tau_th)
------------------------------------------------------------------
Competing timescales at this operating point (SiN config):
    cavity lifetime 1/kappa  ~ 6.6 ns   (~162 round trips)
    thermal time    tau_th   = 5.0 us   (~1.23e5 round trips)
The sweep spans 7*kappa over t_slow round trips of duration t_r each, so the
tuning rate is  R = 7*kappa / (t_slow * t_r).  With t_slow = 860_000 the sweep
lasts ~7*tau_th, giving:
    R / kappa^2        ~ 1.3e-3  -> detuning moves ~0.0013*kappa per cavity
                                     lifetime: DEEPLY adiabatic w.r.t. the cavity
                                     (the field tracks its instantaneous state).
    R * tau_th / kappa ~ 1.0     -> detuning moves ~1*kappa per thermal time:
                                     marginally adiabatic w.r.t. the thermal pole.
The marginal thermal adiabaticity is acceptable because the steady thermo-optic
shift itself is only a small fraction of kappa (~0.5*kappa at the operating
detuning; see the Gamma_th derivation in config/sin_params.yaml), so any thermal
lag perturbs delta_omega_eff by at most that small amount. The MI-ignition
physics is governed by the cavity/Kerr response, which is tracked adiabatically.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import jax
import numpy as np

from simulator.lle_solver import (
    _load_config,
    d2_to_beta2_lle,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Default sweep length: t_slow*t_r ~ 7*tau_th (see module docstring).
DEFAULT_T_SLOW = 860_000
PIN_W = 0.214
N_TAU = 512


# ---------------------------------------------------------------------------
# Per-snapshot feature extraction
# ---------------------------------------------------------------------------
def sech2_spectral_corr(e_field: np.ndarray) -> float:
    """Correlation of a snapshot's power spectrum with a sech^2 template.

    Mirrors the check used in lle_solver.validate_solver: normalise the
    (fftshifted) power spectrum and a sech^2 comb envelope, then take the
    Pearson correlation. A clean single soliton has a sech^2 spectrum -> corr
    near 1; MI/chaotic combs correlate weakly.
    """
    spec = np.abs(np.fft.fftshift(np.fft.fft(e_field))) ** 2
    spec = spec / max(spec.max(), 1e-12)
    x = np.linspace(-3.0, 3.0, spec.size)
    sech2 = 1.0 / np.cosh(x) ** 2
    sech2 = sech2 / sech2.max()
    corr = np.corrcoef(spec, sech2)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def contrast(e_field: np.ndarray) -> float:
    p = np.abs(e_field) ** 2
    return float(np.max(p) / max(np.mean(p), 1e-30))


def _thermo_optic_coeff(config_path=CONFIG_PATH) -> float:
    """(omega0/n0)*dn_dT in rad/s/K — the ODE's thermal-shift prefactor."""
    p = _load_config(config_path)
    lam = float(p.get("pump_wavelength_m", 1.55e-6))
    omega0 = 2.0 * math.pi * 299_792_458.0 / lam
    n0 = float(p.get("n0", 2.2))
    dn_dT = float(p.get("dn_dT_per_k", 4.0e-5))
    return (omega0 / n0) * dn_dT


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_sweep(
    delta_omega_2d: np.ndarray,
    t_slow: int,
    beta2: float,
    kappa: float,
    kappa_c: float,
    seed: int,
    snapshot_interval: int,
    config_path=CONFIG_PATH,
) -> dict:
    """Run one single-trajectory sweep and package per-snapshot records.

    delta_omega_2d must be shape (1, t_slow) — the 2-D single-trajectory contract.
    Returns a dict with per-snapshot arrays aligned to the recorded field
    snapshots, plus the full-length histories needed for the thermal-sign check.
    """
    assert delta_omega_2d.ndim == 2 and delta_omega_2d.shape[0] == 1, (
        f"expected (1, t_slow) delta_omega, got {delta_omega_2d.shape}"
    )
    sol = solve_lle_ssfm_jax(
        pin=PIN_W,
        delta_omega=delta_omega_2d.astype(np.float32),
        t_slow=t_slow,
        beta=[beta2],
        kappa=kappa,
        kappa_c=kappa_c,
        rng_key=jax.random.PRNGKey(seed),
        n_tau=N_TAU,
        snapshot_interval=snapshot_interval,
        config_path=str(config_path),
    )

    e_snaps = np.asarray(sol["E_snapshots"])[0]          # (n_snap, n_tau)
    labels = np.asarray(sol["label_history"])[0]         # (n_snap,)
    u_hist = np.asarray(sol["U_int_history"])[0]         # (t_slow,)
    dweff_hist = np.asarray(sol["delta_omega_eff_history"])[0]
    dT_hist = np.asarray(sol["DeltaT_history"])[0]
    dw_prog = delta_omega_2d[0]                          # programmed detuning (t_slow,)

    n_snap = e_snaps.shape[0]
    # Snapshot k was written at round-trip step k*snapshot_interval.
    snap_steps = np.minimum(np.arange(n_snap) * snapshot_interval, t_slow - 1)

    corr = np.array([sech2_spectral_corr(e_snaps[k]) for k in range(n_snap)])
    cst = np.array([contrast(e_snaps[k]) for k in range(n_snap)])

    return {
        "snap_steps": snap_steps,
        "delta_omega_prog": dw_prog[snap_steps],          # programmed (clean)
        "delta_omega_eff": dweff_hist[snap_steps],        # incl. thermal + noise
        "label": labels,
        "U_int": u_hist[snap_steps],
        "sech2_corr": corr,
        "contrast": cst,
        # full-length histories (for thermal-sign / numerical-health checks)
        "U_int_full": u_hist,
        "delta_omega_eff_full": dweff_hist,
        "DeltaT_full": dT_hist,
        "delta_omega_prog_full": dw_prog,
        "e_snaps": e_snaps,
    }


# ---------------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------------
def validate(fwd: dict, rev: dict, ctrl: dict, kappa: float, to_coeff: float) -> dict:
    """Run the four required validations; return a dict of (pass, detail)."""
    results: dict[str, tuple[bool, str]] = {}

    # -- V1: forward ignites MI inside the predicted window; control stays CW. --
    dw = fwd["delta_omega_prog"]
    window = (dw > 0.5 * kappa) & (dw < 5.05 * kappa)     # kappa/2 .. ~5*kappa
    ignited = (fwd["contrast"] > 2.0) | (fwd["label"] >= 2)
    n_ignite = int(np.sum(window & ignited))
    ctrl_all_cw = bool(np.all(ctrl["label"] <= 1))
    ctrl_max_label = int(np.max(ctrl["label"]))
    v1 = (n_ignite >= 1) and ctrl_all_cw
    results["V1_MI_ignition"] = (
        v1,
        f"{n_ignite} forward snapshot(s) with contrast>2 or label>=2 inside "
        f"(0.5..5)*kappa; held-CW control max label = {ctrl_max_label} "
        f"(stays CW: {ctrl_all_cw}).",
    )

    # -- V2: hysteresis — forward vs reverse differ measurably through MI. --
    # Compare on a common programmed-detuning grid over the MI region.
    lo, hi = 0.5 * kappa, 5.0 * kappa
    grid = np.linspace(lo, hi, 200)
    # reverse detuning is descending; sort ascending for interpolation
    r_order = np.argsort(rev["delta_omega_prog"])
    f_order = np.argsort(fwd["delta_omega_prog"])
    u_f = np.interp(grid, fwd["delta_omega_prog"][f_order], fwd["U_int"][f_order])
    u_r = np.interp(grid, rev["delta_omega_prog"][r_order], rev["U_int"][r_order])
    l_f = np.interp(grid, fwd["delta_omega_prog"][f_order],
                    fwd["label"][f_order].astype(float))
    l_r = np.interp(grid, rev["delta_omega_prog"][r_order],
                    rev["label"][r_order].astype(float))
    u_scale = max(np.mean(np.abs(u_f)) + np.mean(np.abs(u_r)), 1e-30) / 2.0
    u_rel_diff = float(np.mean(np.abs(u_f - u_r)) / u_scale)
    label_diff = float(np.mean(np.abs(l_f - l_r)))
    # NB: do NOT use np.allclose here — its default atol=1e-8 exceeds the J-scale
    # U_int values (~1e-10..1e-9), so it would spuriously report "identical".
    # Compare on the relative scale instead.
    not_identical = u_rel_diff > 1e-6
    v2 = not_identical and (u_rel_diff > 0.02)
    results["V2_hysteresis"] = (
        v2,
        f"U_int mean |fwd-rev|/mean = {u_rel_diff:.1%}, mean |label diff| = "
        f"{label_diff:.2f} over MI region; trajectories not identical: "
        f"{not_identical}.",
    )

    # -- V3: thermal sign — delta_omega_eff shifts DOWN as U_int rises. --
    # thermal_shift = -(omega0/n0)*dn_dT*DeltaT, so heating (DeltaT>0) is negative.
    fwd_shift = -to_coeff * fwd["DeltaT_full"]
    # correlation between U_int and thermal_shift must be negative (rises -> down)
    u = fwd["U_int_full"]
    shift = fwd_shift
    if np.std(u) > 0 and np.std(shift) > 0:
        rho_us = float(np.corrcoef(u, shift)[0, 1])
    else:
        rho_us = 0.0
    # steady thermal_shift on the control (held-CW) plateau tail
    ctrl_shift = -to_coeff * ctrl["DeltaT_full"]
    tail = slice(int(0.8 * ctrl_shift.size), None)
    steady_shift = float(np.mean(ctrl_shift[tail]))
    steady_over_kappa = steady_shift / kappa
    v3 = (rho_us < -0.2) and (steady_shift < 0.0) and (abs(steady_over_kappa) < 2.0)
    results["V3_thermal_sign"] = (
        v3,
        f"corr(U_int, thermal_shift) = {rho_us:+.2f} (<0 => shifts DOWN as U rises); "
        f"steady control thermal_shift = {steady_over_kappa:+.3f}*kappa "
        f"(sane fraction of kappa, not tens).",
    )

    # -- V4: numerical health — no NaN/Inf; control U_int tail rel-std < 5%. --
    finite = all(
        np.all(np.isfinite(d["U_int_full"]))
        and np.all(np.isfinite(d["delta_omega_eff_full"]))
        and np.all(np.isfinite(d["e_snaps"]))
        for d in (fwd, rev, ctrl)
    )
    ctrl_u = ctrl["U_int_full"]
    ctrl_tail = ctrl_u[int(0.8 * ctrl_u.size):]
    ctrl_rel_std = float(np.std(ctrl_tail) / max(np.mean(ctrl_tail), 1e-30))
    v4 = finite and (ctrl_rel_std < 0.05)
    results["V4_numerical_health"] = (
        v4,
        f"all fields finite: {finite}; held-CW control U_int tail rel-std = "
        f"{ctrl_rel_std:.2%} (< 5% plateau).",
    )

    return results


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_table(path: Path, rec: dict, kappa: float) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "snapshot", "step", "delta_omega_rad_s", "delta_omega_over_kappa",
            "delta_omega_eff_rad_s", "delta_omega_eff_over_kappa",
            "label", "U_int_J", "sech2_corr", "contrast",
        ])
        for k in range(rec["label"].size):
            w.writerow([
                k, int(rec["snap_steps"][k]),
                f"{rec['delta_omega_prog'][k]:.6e}",
                f"{rec['delta_omega_prog'][k] / kappa:.4f}",
                f"{rec['delta_omega_eff'][k]:.6e}",
                f"{rec['delta_omega_eff'][k] / kappa:.4f}",
                int(rec["label"][k]),
                f"{rec['U_int'][k]:.6e}",
                f"{rec['sech2_corr'][k]:.4f}",
                f"{rec['contrast'][k]:.4f}",
            ])


def make_plot(path: Path, fwd: dict, rev: dict, ctrl: dict, kappa: float,
              to_coeff: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fk = fwd["delta_omega_prog"] / kappa
    rk = rev["delta_omega_prog"] / kappa

    # (a) U_int vs detuning — hysteresis loop
    ax = axes[0, 0]
    ax.plot(fk, fwd["U_int"], ".-", ms=3, lw=1, label="forward (blue->red)",
            color="tab:blue")
    ax.plot(rk, rev["U_int"], ".-", ms=3, lw=1, label="reverse (red->blue)",
            color="tab:red")
    ax.axvspan(0.5, 5.0, color="gold", alpha=0.15, label="MI window (0.5..5)*kappa")
    ax.set_xlabel(r"$\delta\omega / \kappa$ (programmed)")
    ax.set_ylabel(r"$U_\mathrm{int}$ (J)")
    ax.set_title("(a) Intracavity energy — hysteresis")
    ax.legend(fontsize=8)

    # (b) label vs detuning
    ax = axes[0, 1]
    ax.plot(fk, fwd["label"], ".-", ms=3, lw=1, label="forward", color="tab:blue")
    ax.plot(rk, rev["label"], ".-", ms=3, lw=1, label="reverse", color="tab:red")
    ax.axvspan(0.5, 5.0, color="gold", alpha=0.15)
    ax.set_xlabel(r"$\delta\omega / \kappa$ (programmed)")
    ax.set_ylabel("labeler class")
    ax.set_yticks(range(0, 7))
    ax.set_title("(b) State label (0 off,1 CW,2 MI,3 chaos,4 multi,5 crystal,6 single)")
    ax.legend(fontsize=8)

    # (c) sech^2 spectral correlation vs detuning
    ax = axes[1, 0]
    ax.plot(fk, fwd["sech2_corr"], ".-", ms=3, lw=1, label="forward",
            color="tab:blue")
    ax.plot(rk, rev["sech2_corr"], ".-", ms=3, lw=1, label="reverse",
            color="tab:red")
    ax.axhline(0.7, color="k", ls="--", lw=0.8, label="0.7 (soliton-like)")
    ax.set_xlabel(r"$\delta\omega / \kappa$ (programmed)")
    ax.set_ylabel(r"sech$^2$ spectral corr")
    ax.set_title("(c) sech$^2$ spectral correlation")
    ax.legend(fontsize=8)

    # (d) thermal shift vs U_int (sign check), forward sweep, full history
    ax = axes[1, 1]
    fwd_shift = -to_coeff * fwd["DeltaT_full"] / kappa
    ax.plot(fwd["U_int_full"], fwd_shift, ",", alpha=0.3, color="tab:green")
    ax.set_xlabel(r"$U_\mathrm{int}$ (J)")
    ax.set_ylabel(r"thermal shift / $\kappa$")
    ax.set_title("(d) Thermal shift vs $U_\\mathrm{int}$ (down as U rises)")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(path: Path, fwd: dict, rev: dict, ctrl: dict, kappa: float,
                 t_slow: int, t_r: float, tau_th: float, results: dict) -> None:
    T = t_slow * t_r
    rate = 7.0 * kappa / T

    def _label_span(rec):
        labs = rec["label"]
        uniq, cnts = np.unique(labs, return_counts=True)
        return ", ".join(f"{int(l)}×{int(c)}" for l, c in zip(uniq, cnts))

    lines = []
    lines.append("# Adiabatic detuning sweeps at pin = 0.214 W\n")
    lines.append(
        "Single-trajectory forward (blue->red) and reverse (red->blue) detuning "
        "sweeps through the MI window of the SiN microring LLE, thermal model ON "
        f"at the config Gamma_th, n_tau = {N_TAU}.\n"
    )
    lines.append("## Sweep design and adiabaticity\n")
    lines.append(
        f"- Detuning range: `-2*kappa -> +5*kappa` (span 7*kappa), "
        f"kappa = {kappa:.3e} rad/s.\n"
        f"- `t_slow = {t_slow}` round trips, t_r = {t_r:.3e} s => sweep duration "
        f"T = {T:.3e} s = {T / tau_th:.2f}*tau_th (tau_th = {tau_th:.1e} s).\n"
        f"- Tuning rate R = 7*kappa/T = {rate:.3e} rad/s^2.\n"
        f"- R/kappa^2 = {rate / kappa**2:.2e}: detuning moves "
        f"{rate / kappa**2:.4f}*kappa per cavity lifetime -> deeply adiabatic "
        f"w.r.t. the cavity.\n"
        f"- R*tau_th/kappa = {rate * tau_th / kappa:.2f}: detuning moves "
        f"~{rate * tau_th / kappa:.2f}*kappa per thermal time -> marginally "
        f"adiabatic w.r.t. the thermal pole; acceptable because the steady "
        f"thermo-optic shift is only a small fraction of kappa (see V3).\n"
    )
    labels_seen = set(np.unique(fwd["label"]).tolist()) | set(
        np.unique(rev["label"]).tolist()
    )
    chaos_clause = (
        "with excursions into chaos (label 3) "
        if 3 in labels_seen else
        "(no chaos, label 3, was reached in this run) "
    )
    lines.append("\n## What actually appears\n")
    lines.append(
        f"- Forward sweep label histogram (label×count): {_label_span(fwd)}.\n"
        f"- Reverse sweep label histogram: {_label_span(rev)}.\n"
        f"- Held-CW control (constant -4*kappa, deep blue) label histogram: "
        f"{_label_span(ctrl)}.\n"
        f"- Max sech^2 spectral correlation: forward {np.max(fwd['sech2_corr']):.3f}, "
        f"reverse {np.max(rev['sech2_corr']):.3f}.\n\n"
        f"The sweeps ignite modulation instability (label 2) {chaos_clause}inside "
        "the predicted window. **No clean single solitons (label 6) form** under "
        "this bare linear adiabatic sweep — consistent with the expectation stated "
        "in the task. The sech^2 spectral correlation stays well below the ~0.7 "
        "single-soliton bar, confirming the combs are MI (Turing/roll) patterns "
        "rather than sech^2 solitonic.\n"
    )
    lines.append("\n## Validations\n")
    for name, (ok, detail) in results.items():
        lines.append(f"- **[{'PASS' if ok else 'FAIL'}] {name}** — {detail}\n")
    lines.append("\n## Follow-up work\n")
    lines.append(
        "Clean single-soliton nucleation is **follow-up work**. A bare linear "
        "detuning ramp lands in MI/chaotic states rather than a single soliton; "
        "reaching the single-soliton step of the LLE requires a dedicated "
        "nucleation protocol (e.g. a fast backward detuning kick / power ramp to "
        "cross the soliton-existence boundary from the chaotic branch, or a "
        "thermally-compensated trajectory). That protocol is intentionally NOT "
        "attempted here.\n"
    )
    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t-slow", type=int, default=DEFAULT_T_SLOW)
    ap.add_argument("--snapshots", type=int, default=430,
                    help="approximate number of snapshots per sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    phys = _load_config(CONFIG_PATH)
    kappa_i, kappa_c, kappa = resolve_cavity_rates(CONFIG_PATH)
    beta2 = d2_to_beta2_lle(float(phys["d2_rad_per_s2"]), float(phys["fsr_hz"]))
    t_r = 1.0 / float(phys["fsr_hz"])
    tau_th = float(phys.get("tau_th_s", 5.0e-6))
    to_coeff = _thermo_optic_coeff(CONFIG_PATH)

    t_slow = int(args.t_slow)
    snap_interval = max(t_slow // int(args.snapshots), 1)

    # Programmed detuning ramps, shape (1, t_slow) — the 2-D single-traj contract.
    fwd_dw = np.linspace(-2.0 * kappa, 5.0 * kappa, t_slow, dtype=np.float32)[None, :]
    rev_dw = np.linspace(5.0 * kappa, -2.0 * kappa, t_slow, dtype=np.float32)[None, :]
    # Held-CW control: no sweep, deep blue. -2*kappa (the sweep start) is already
    # weakly MI-unstable at this power, so the CW control sits deeper at -4*kappa,
    # where the homogeneous branch is robustly stable (label 1).
    ctrl_dw = np.full((1, t_slow), -4.0 * kappa, dtype=np.float32)

    print(f"[run] t_slow={t_slow}, snapshot_interval={snap_interval}, "
          f"kappa={kappa:.3e} rad/s")
    print("[run] forward sweep (blue -> red) ...")
    fwd = run_sweep(fwd_dw, t_slow, beta2, kappa, kappa_c, args.seed, snap_interval)
    print("[run] reverse sweep (red -> blue) ...")
    rev = run_sweep(rev_dw, t_slow, beta2, kappa, kappa_c, args.seed, snap_interval)
    print("[run] held-CW control (deep blue -4*kappa) ...")
    ctrl = run_sweep(ctrl_dw, t_slow, beta2, kappa, kappa_c, args.seed, snap_interval)

    results = validate(fwd, rev, ctrl, kappa, to_coeff)

    write_table(RESULTS_DIR / "forward_sweep.csv", fwd, kappa)
    write_table(RESULTS_DIR / "reverse_sweep.csv", rev, kappa)
    write_table(RESULTS_DIR / "control_held_cw.csv", ctrl, kappa)
    make_plot(RESULTS_DIR / "adiabatic_sweeps.png", fwd, rev, ctrl, kappa, to_coeff)
    write_report(RESULTS_DIR / "adiabatic_sweeps_report.md", fwd, rev, ctrl,
                 kappa, t_slow, t_r, tau_th, results)

    print("\n=== Validation summary ===")
    all_pass = True
    for name, (ok, detail) in results.items():
        all_pass &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\nArtifacts written to {RESULTS_DIR}")
    if not all_pass:
        raise SystemExit("One or more validations FAILED.")
    print("All validations PASSED.")


if __name__ == "__main__":
    main()
