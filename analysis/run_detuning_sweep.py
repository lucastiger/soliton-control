#!/usr/bin/env python
"""Quasi-static detuning sweep of the single-DKS branch -> soliton-step figure.

This analysis-layer driver traces the single-dissipative-Kerr-soliton branch in
POWER vs pump-cavity detuning and renders the soliton-step figure
(``analysis/results/soliton_steps.{png,pdf}``).  It uses ONLY the solver's public
API (:func:`simulator.lle_solver.solve_lle_ssfm_jax`, via the
:mod:`analysis.dks_access` helpers) -- no solver / stepping / thermal / noise code
is modified.

Why a warm-continuation step-and-hold sweep (not the bare adiabatic ramp)
------------------------------------------------------------------------
A bare blue->red detuning ramp at this operating point ignites modulation
instability (MI) and never nucleates a clean single soliton (see
``analysis/adiabatic_sweeps.py``); the validated single DKS is instead accessed
deterministically by seeding an analytic sech ansatz (``access_by_seeding``).
This driver therefore SEEDS one soliton on its branch and then sweeps the
detuning quasi-statically, holding each detuning for many photon lifetimes with
the field + thermal state carried over from the previous step (a true adiabatic
continuation of one trajectory), so the soliton follows its branch until it
annihilates -- the soliton step.

Scan direction (DISCOVERED, not assumed)
----------------------------------------
``delta_omega = omega_res - omega_pump`` (+ve = red-detuned = soliton side).  The
scan direction that produces a GRID-CONVERGED step for this system is
*decreasing* detuning, for a physical reason established by running the sweep
both ways:

* Increasing detuning: the warm-continued soliton is extremely robust and does
  not exhibit a converged annihilation within a tractable window.  Its apparent
  high-detuning "collapse" is a finite-FFT TRUNCATION artifact -- the soliton
  narrows (``tau_s ~ 1/sqrt(delta_omega)``) so its comb broadens until the grid
  clips it; the collapse detuning is NOT converged in ``n_tau`` (it moved
  ~28*kappa -> >32*kappa as ``n_tau`` went 2048 -> 8192).  So there is no honest
  soliton step in the increasing direction here.
* Decreasing detuning: the soliton comb NARROWS (well resolved on any grid) and
  the branch terminates at a genuine, grid-converged LOWER existence edge, where
  the single soliton annihilates and the field collapses onto the MI/CW branch.

The sweep therefore runs from high detuning DOWN through the annihilation; the
figure still plots detuning increasing left->right, with the single-DKS existence
region shaded and its lower edge (the soliton step) marked.

Per-detuning observable and averaging
-------------------------------------
For each detuning the observable is the power averaged over the FINAL
``avg_frac`` (default 25%) of a long constant-detuning hold.  This one operation
does two jobs: it discards the per-step re-settling transient AND cycle-averages
the deterministic breather (on part of the branch the attractor is a limit cycle,
so the instantaneous power oscillates and only a slow-time average is a
reproducible observable).  The averaging is done in LINEAR power
(:func:`analysis.spectral_metrics.hold_window_average`), never in dB and never on
the complex field.

Hold length and adiabaticity.  The photon lifetime is ``1/kappa`` ~ 162 round
trips and the breather period is ~150-180 RT.  The default ``hold_rt = 2000`` is
~12 photon lifetimes (the field re-settles to ``exp(-12)`` of any transient after
a step) and ~13 breather periods, with the final 25% (~500 RT) spanning ~3
breather periods -- enough for both transient decay and breather cycle-averaging.
A fully THERMALLY adiabatic hold (``tau_th`` ~ 1.2e5 RT ~ 760 photon lifetimes)
is deliberately NOT used: the steady thermo-optic shift here is only
~0.01-0.03*kappa (negligible), so thermal lag between steps does not move the
branch; ``hold_rt`` is exposed for callers that want to push toward that regime.

Observable and what the step looks like here
--------------------------------------------
The solver returns the through-port (transmitted) power via the exact all-pass
energy balance ``P_trans = P_in - kappa_i * <|E|^2>`` (``P_trans_history``), so
transmitted power IS computable and is recorded, normalised to the cold-cavity
(far-detuned, empty-cavity) level ``P_in``.  The primary staircase is the total
intracavity power ``sum_mu |a_mu|^2`` (proportional to ``U_int`` by Parseval).

At this hard-driven, high-detuning operating point the classic *power* staircase
is muted: the single soliton is replaced at annihilation by a Turing/MI comb of
COMPARABLE intracavity energy, so neither the intracavity power nor the
transmission shows a large discontinuity AT the annihilation -- the soliton step
is primarily a STATE transition (single-peak breathing DKS -> multi-peak MI),
visible here as the collapse of the breathing error bars and the edge of the
single-DKS existence region.  Below the edge the MI/CW power rises steeply toward
resonance; that near-resonance rise is what a first-difference discontinuity
detector keys on, so :func:`detect_power_steps` is reported honestly as marking
the MI/CW power rise, NOT the soliton annihilation.

Thermal / noise configuration
-----------------------------
The thermo-optic model is run ON, matching the validated operating configuration
(``config/sin_params.yaml`` ``Gamma_th``); the effective detuning after thermal
pulling is recorded per step (the steady thermo-optic shift is only
~0.01-0.03*kappa here, so pulling is small).  Stochastic detuning noise is
DISABLED so the branch is clean: a sidecar config with the thermodynamic
temperature ``T_k = 0`` zeroes the thermorefractive/pyro-EO noise variance
(``var_delta_t = k_B*T_k^2/(rho*Cp*V)``) while leaving every deterministic thermal
parameter untouched (``T_k`` does not enter ``_thermal_params``).

Outputs (all under ``analysis/results/``)
-----------------------------------------
* ``detuning_sweep.npz`` -- detuning grid, averaged powers, per-step std
  (breathing-amplitude indicator), transmission, labels and the full sweep
  config, so the figure regenerates without re-running the sweep.
* ``soliton_steps.{png,pdf}`` -- the publication figure.
* ``spectral_metrics.json`` gains a ``soliton_step`` block (single-DKS existence
  region, annihilation detuning, the power-discontinuity detection result, and
  full provenance).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from analysis.dks_access import (  # noqa: E402  (needs the sys.path insert)
    CONFIG_PATH,
    LABEL_SINGLE_SOLITON,
    PIN_W,
    PRODUCTION_NUMERICS,
    RESULTS_DIR,
    _run,
    access_by_seeding,
    attach_dispersion,
    count_temporal_peaks,
    load_cavity_params,
    numpy_label,
)
from analysis.spectral_metrics import (  # noqa: E402
    DEFAULT_STEP_K,
    SOLITON_STEP_DEFINITION,
    detect_power_steps,
    hold_window_average,
    plot_soliton_steps,
    single_dks_region,
)

SWEEP_NPZ = "detuning_sweep.npz"
STEPS_PNG = "soliton_steps.png"
METRICS_JSON = "spectral_metrics.json"


# ---------------------------------------------------------------------------
# Sweep configuration (all tunables in one place; nothing hardcoded in the
# functions below -- they read this object).  Defaults follow the discovered
# branch: a single DKS seeded well inside its branch (~12*kappa, clean and
# stationary) is warm-continued DOWN in detuning through the breather sub-band
# (~9 -> 6.3*kappa) to its annihilation near ~6.1*kappa, with a few points into
# the MI/CW region below to show the post-step branch.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepConfig:
    dw_start_kappa: float = 12.0     # seed / first hold (clean stationary DKS)
    dw_stop_kappa: float = 4.5       # a few points past annihilation, into MI
    n_steps: int = 31                # detuning samples (0.25*kappa spacing)
    settle_rt: int = 6000            # pre-settle at dw_start (seed -> attractor)
    hold_rt: int = 2000              # round trips held per detuning step
    avg_frac: float = 0.25           # average the final fraction of each hold
    n_tau: int = 4096                # FFT grid (resolves the comb core each step)
    seed: int = 0                    # RNG seed (noise is off; kept for provenance)
    pin_w: float = PIN_W             # on-chip pump power (== config pin_w)
    step_k: float = DEFAULT_STEP_K   # power-discontinuity MAD multiple (conservative)
    smooth_display: int = 0          # display-only smoothing window (0 = off)

    def detunings_kappa(self) -> np.ndarray:
        # linspace preserves the sweep order (start -> stop); start > stop gives a
        # decreasing (warm-continuation-down) sweep.
        return np.linspace(self.dw_start_kappa, self.dw_stop_kappa,
                           int(self.n_steps))

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Deterministic (noise-off) config
# ---------------------------------------------------------------------------
def write_noise_off_config(base_config_path=CONFIG_PATH, out_path=None) -> Path:
    """Write a sidecar config identical to ``base`` but with the noise disabled.

    Sets ``physical_parameters.T_k = 0`` so the thermodynamic temperature-
    fluctuation variance ``var_delta_t = k_B*T_k^2/(rho*Cp*V)`` -- the amplitude
    of every stochastic detuning-noise channel (thermorefractive / pyro-EO; the
    SiN TCCR channel is already zero for r33 = 0) -- becomes exactly zero, giving
    a fully deterministic run.  ``T_k`` does NOT appear in
    :func:`simulator.lle_solver._thermal_params`, so the deterministic thermo-
    optic dynamics (Gamma_th, tau_th, the thermal shift) are unchanged.
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("physical_parameters", {})["T_k"] = 0.0
    if out_path is None:
        fd, name = tempfile.mkstemp(prefix="sin_params_noiseoff_", suffix=".yaml")
        out_path = Path(name)
        import os
        os.close(fd)
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def run_detuning_sweep(cav, cfg: SweepConfig, *, config_path) -> dict:
    """Warm-continuation step-and-hold detuning sweep of the single-DKS branch.

    Seeds one soliton at ``cfg.dw_start_kappa`` (pre-settled for ``cfg.settle_rt``
    round trips), then holds each detuning in ``cfg.detunings_kappa()`` for
    ``cfg.hold_rt`` round trips, carrying the field + thermal state forward.  Per
    step it records the final-``avg_frac`` LINEAR-power average (and std) of the
    total intracavity power ``sum_mu |a_mu|^2`` and of the through-port power
    ``P_trans``, plus the mean effective detuning and the state label / peak
    count (which flag the single-DKS existence region).

    ``config_path`` is the (deterministic, noise-off) solver config.  Returns a
    dict of per-step arrays; every solver call uses ``**PRODUCTION_NUMERICS``.
    """
    kappa = cav.kappa
    n_tau = int(cfg.n_tau)
    # sum_mu |a_mu|^2 = n_tau^2 / t_r * U_int  (numpy-FFT Parseval + solver U_int
    # normalisation U_int = sum_tau|E|^2 * t_r/n_tau).
    modes_from_uint = (n_tau ** 2) / cav.t_r

    dws_k = cfg.detunings_kappa()
    print(f"[sweep] seeding single DKS at {cfg.dw_start_kappa:.2f} kappa "
          f"(settle {cfg.settle_rt} RT, n_tau={n_tau}) ...")
    res0 = access_by_seeding(cfg.dw_start_kappa * kappa, cav,
                             t_slow=int(cfg.settle_rt), seed=int(cfg.seed),
                             n_tau=n_tau, pin=cfg.pin_w, config_path=config_path,
                             **PRODUCTION_NUMERICS)
    m0 = res0["metrics"]
    print(f"[sweep] seed settled: single={res0['is_single']} "
          f"label={m0['np_label']} n_peaks={m0['n_peaks']} "
          f"env_corr={m0['sech2_env_corr']:.3f} breather={m0['is_breather']}")
    if not res0["is_single"]:
        print("[sweep] WARNING: pre-settle did not yield a clean single soliton; "
              "the branch start may be off-attractor -- inspect the trace.")

    e_prev = np.asarray(res0["e_final"])
    dt_prev = float(res0["delta_t_final"])

    rows = []
    for i, dwk in enumerate(dws_k):
        t0 = time.time()
        sol = _run(dwk * kappa, int(cfg.hold_rt), cav, e0=e_prev,
                   delta_t0=dt_prev, seed=int(cfg.seed), n_tau=n_tau,
                   pin=cfg.pin_w, snapshot_interval=int(cfg.hold_rt),
                   config_path=config_path, **PRODUCTION_NUMERICS)
        u_hist = np.asarray(sol["U_int_history"])[0]
        p_hist = np.asarray(sol["P_trans_history"])[0]
        dweff_hist = np.asarray(sol["delta_omega_eff_history"])[0]
        e_final = np.asarray(sol["e_final"])[0]
        dt_final = float(np.asarray(sol["delta_t_final"]).reshape(-1)[0])

        u_avg = hold_window_average(u_hist, avg_frac=cfg.avg_frac)
        p_avg = hold_window_average(p_hist, avg_frac=cfg.avg_frac)
        w = slice(u_avg["i_start"], None)
        dweff_mean = float(np.mean(dweff_hist[w]))

        label = numpy_label(e_final, cav, dwk * kappa, pin=cfg.pin_w)
        n_peaks = count_temporal_peaks(e_final)
        # Class 6 from the NumPy labeler already encodes the single temporal peak
        # + sech^2-envelope goodness-of-fit, so (label == 6 and one peak, finite)
        # is a faithful single-DKS flag without recomputing the full bundle.
        single = bool(label == LABEL_SINGLE_SOLITON and n_peaks == 1
                      and np.all(np.isfinite(e_final)))

        rows.append({
            "dw_over_kappa": float(dwk),
            "dw_rad_s": float(dwk * kappa),
            "dw_eff_over_kappa": dweff_mean / kappa,
            "P_intra": modes_from_uint * u_avg["mean"],      # sum_mu|a_mu|^2
            "P_intra_std": modes_from_uint * u_avg["std"],
            "U_int": u_avg["mean"],                          # J
            "U_int_std": u_avg["std"],
            "U_int_relstd": u_avg["std"] / max(u_avg["mean"], 1e-300),
            "P_trans": p_avg["mean"],                        # W
            "P_trans_std": p_avg["std"],
            "np_label": int(label),
            "n_peaks": int(n_peaks),
            "is_single": single,
        })
        e_prev, dt_prev = e_final, dt_final
        print(f"[sweep] {i + 1:2d}/{len(dws_k)}  dw={dwk:6.2f}k  "
              f"P_intra={rows[-1]['P_intra']:.4e}  "
              f"T={rows[-1]['P_trans'] / cfg.pin_w:.5f}  "
              f"U_relstd={rows[-1]['U_int_relstd']:.2%}  lbl={label} npk={n_peaks}  "
              f"({time.time() - t0:.1f}s)")

    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["kappa_rad_s"] = float(kappa)
    out["t_r_s"] = float(cav.t_r)
    out["fsr_hz"] = float(cav.fsr_measured_hz if cav.fsr_measured_hz is not None
                          else cav.fsr_hz)
    out["seed_metrics"] = m0
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_json(json_path: Path, key: str, block: dict) -> None:
    data = {}
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
    data[key] = block
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=float)


def save_sweep_npz(path: Path, sweep: dict, cfg: SweepConfig) -> None:
    """Persist the sweep so the figure regenerates without re-running the solver."""
    np.savez_compressed(
        path,
        dw_over_kappa=sweep["dw_over_kappa"],
        dw_rad_s=sweep["dw_rad_s"],
        dw_eff_over_kappa=sweep["dw_eff_over_kappa"],
        P_intra=sweep["P_intra"],
        P_intra_std=sweep["P_intra_std"],
        U_int=sweep["U_int"],
        U_int_std=sweep["U_int_std"],
        P_trans=sweep["P_trans"],
        P_trans_std=sweep["P_trans_std"],
        np_label=sweep["np_label"],
        n_peaks=sweep["n_peaks"],
        is_single=sweep["is_single"],
        kappa_rad_s=sweep["kappa_rad_s"],
        t_r_s=sweep["t_r_s"],
        fsr_hz=sweep["fsr_hz"],
        pin_w=float(cfg.pin_w),
        sweep_config_json=json.dumps(cfg.as_dict()),
    )


def load_sweep_npz(path: Path):
    """Load a committed ``detuning_sweep.npz`` back into ``(sweep, cfg)``.

    Lets the figure + JSON be regenerated from the saved data alone (no solver
    re-run), per the deliverable convention that outputs are regenerable.
    """
    d = np.load(path, allow_pickle=False)
    cfg = SweepConfig(**json.loads(str(d["sweep_config_json"])))
    keys = ("dw_over_kappa", "dw_rad_s", "dw_eff_over_kappa", "P_intra",
            "P_intra_std", "U_int", "U_int_std", "P_trans", "P_trans_std",
            "np_label", "n_peaks", "is_single", "kappa_rad_s", "t_r_s", "fsr_hz")
    sweep = {k: (d[k] if d[k].shape else float(d[k])) for k in keys}
    sweep["is_single"] = sweep["is_single"].astype(bool)
    return sweep, cfg


def render_and_report(sweep: dict, cfg: SweepConfig) -> Path:
    """Build the soliton-step figure + JSON block from an assembled sweep dict.

    Sorts by detuning, normalises the observables, locates the single-DKS
    existence region / annihilation edge, runs the power-discontinuity detector,
    writes the figure and the ``soliton_step`` block, and returns the figure path.
    Pure post-processing -- no solver -- so it also serves ``--render-only``.
    """
    order = np.argsort(sweep["dw_over_kappa"])
    dwk = np.asarray(sweep["dw_over_kappa"])[order]
    P = np.asarray(sweep["P_intra"])[order]
    P_std = np.asarray(sweep["P_intra_std"])[order]
    T = (np.asarray(sweep["P_trans"]) / float(cfg.pin_w))[order]
    is_single = np.asarray(sweep["is_single"])[order]

    # Primary staircase observable: intracavity power, normalised to the sweep
    # maximum.  Transmission is normalised to the cold-cavity (far-detuned,
    # empty-cavity) level P_in.
    P_ref = float(np.max(P))
    P_norm = P / P_ref
    P_norm_std = P_std / P_ref

    # Single-DKS existence region + its lower edge (the soliton step), from the
    # per-step state flag; and the power-trace discontinuities (which key on the
    # near-resonance MI/CW rise, reported honestly).
    lo_k, hi_k, annih_k = single_dks_region(dwk, is_single)
    steps = detect_power_steps(dwk, P_norm, k=cfg.step_k)

    kappa = float(sweep["kappa_rad_s"])
    caption = (
        f"Single-DKS branch, pin = {cfg.pin_w} W, n_tau = {cfg.n_tau}. Warm "
        f"continuation, seeded at {cfg.dw_start_kappa:g}$\\kappa$, swept DOWN, "
        f"hold {cfg.hold_rt} RT/step, averaged over the final "
        f"{int(100 * cfg.avg_frac)}% (cycle-averages the breather). Thermo-optic "
        f"model ON (deterministic, noise off). Green = single-DKS existence "
        f"region; the soliton annihilates (step) at its lower edge, where the "
        f"breathing error bars collapse. Intracavity power normalised to the "
        f"sweep max; transmission to the cold-cavity level. Dotted lines mark "
        f"first-difference discontinuities of the power trace, which here fall on "
        f"the near-resonance MI/CW rise, not the soliton step.")
    metadata = {"kappa_rad_s": kappa, "caption": caption}
    plot_path = plot_soliton_steps(
        dwk, P_norm, RESULTS_DIR / STEPS_PNG, power_std=P_norm_std,
        transmission=T, soliton_region=(lo_k, hi_k) if lo_k is not None else None,
        annihilation_kappa=annih_k, steps=steps, metadata=metadata,
        smooth_window=cfg.smooth_display)

    provenance = {
        "driver": "analysis/run_detuning_sweep.py",
        "sweep_data": SWEEP_NPZ,
        "figure": STEPS_PNG,
        "config_file": str(CONFIG_PATH.name),
        "config_sha256": _sha256(CONFIG_PATH),
        "noise": "disabled (deterministic; sidecar config with T_k = 0)",
        "thermal_model": "ON (matches validated operating config; Gamma_th)",
        "n_tau": int(cfg.n_tau),
        "pin_w": float(cfg.pin_w),
        "sweep_config": cfg.as_dict(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    block = {
        "metric": "soliton_step",
        "metric_definition": SOLITON_STEP_DEFINITION,
        "observable": "total intracavity power sum_mu |a_mu|^2 (normalised to the "
                      "sweep maximum); transmitted power P_trans/P_in also stored "
                      "in detuning_sweep.npz",
        "sweep_direction": "decreasing detuning (delta_omega down) along the "
                           "single-DKS branch to its lower-edge annihilation; the "
                           "increasing direction gives no grid-converged step "
                           "(the high-detuning collapse is an FFT-truncation "
                           "artifact, non-convergent in n_tau)",
        "n_detunings": int(dwk.size),
        "detuning_range_over_kappa": [float(cfg.dw_stop_kappa),
                                      float(cfg.dw_start_kappa)],
        "single_dks_existence_region_over_kappa": (
            [lo_k, hi_k] if lo_k is not None else None),
        "soliton_step": {
            "annihilation_over_kappa": annih_k,
            "note": "lower edge of the single-DKS existence region (label 6 + "
                    "single temporal peak); the soliton is replaced by a "
                    "comparable-energy MI comb, so the TOTAL power step here is "
                    "small -- the step is primarily a state transition, also "
                    "visible as the collapse of the breathing amplitude.",
        },
        "power_trace_discontinuities": {
            "rule": "|diff(P)[i] - median(diff P)| > k * 1.4826 * MAD(diff P)",
            "k": float(steps["k"]),
            "robust_sigma": float(steps["sigma"]),
            "detected_over_kappa": list(steps["step_x"]),
            "interpretation": "at this operating point these fall on the "
                              "near-resonance MI/CW power rise below the soliton "
                              "branch, NOT on the soliton annihilation",
        },
        "transmission_contrast": {
            "T_min": float(np.min(T)),
            "T_max": float(np.max(T)),
            "note": "near-unity: the cavity is nearly empty at these high "
                    "detunings, so the through-port step is sub-percent",
        },
        "units": {
            "detuning_over_kappa": "kappa (total cavity linewidth)",
            "annihilation_over_kappa": "kappa",
        },
        "provenance": provenance,
    }
    _update_json(RESULTS_DIR / METRICS_JSON, "soliton_step", block)

    print(f"[sweep] single-DKS existence region: "
          + (f"[{lo_k:.2f}, {hi_k:.2f}] kappa" if lo_k is not None else "none"))
    print(f"[sweep] soliton annihilation (step): "
          + (f"{annih_k:.2f} kappa" if annih_k is not None else "not bracketed"))
    print(f"[sweep] power-trace discontinuities (MI/CW rise) at: "
          + (", ".join(f"{xs:.2f} kappa" for xs in steps["step_x"]) or "none"))
    print(f"[sweep] transmission range T in "
          f"[{np.min(T):.5f}, {np.max(T):.5f}] (cold-cavity = 1)")
    print(f"[sweep] plot -> {plot_path} (+ .pdf)")
    print(f"[sweep] json -> {RESULTS_DIR / METRICS_JSON} (soliton_step)")
    return plot_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dw-start", type=float, default=SweepConfig.dw_start_kappa)
    ap.add_argument("--dw-stop", type=float, default=SweepConfig.dw_stop_kappa)
    ap.add_argument("--n-steps", type=int, default=SweepConfig.n_steps)
    ap.add_argument("--settle-rt", type=int, default=SweepConfig.settle_rt)
    ap.add_argument("--hold-rt", type=int, default=SweepConfig.hold_rt)
    ap.add_argument("--avg-frac", type=float, default=SweepConfig.avg_frac)
    ap.add_argument("--n-tau", type=int, default=SweepConfig.n_tau)
    ap.add_argument("--seed", type=int, default=SweepConfig.seed)
    ap.add_argument("--step-k", type=float, default=SweepConfig.step_k)
    ap.add_argument("--smooth-display", type=int, default=SweepConfig.smooth_display,
                    help="display-only moving-average window (0 = off)")
    ap.add_argument("--render-only", action="store_true",
                    help="regenerate the figure + JSON from the committed "
                         "detuning_sweep.npz without re-running the solver")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args.render_only:
        sweep, cfg = load_sweep_npz(RESULTS_DIR / SWEEP_NPZ)
        render_and_report(sweep, cfg)
        print(f"[sweep] re-rendered from {SWEEP_NPZ} in "
              f"{time.time() - t_start:.1f}s")
        return

    cfg = SweepConfig(
        dw_start_kappa=args.dw_start, dw_stop_kappa=args.dw_stop,
        n_steps=args.n_steps, settle_rt=args.settle_rt, hold_rt=args.hold_rt,
        avg_frac=args.avg_frac, n_tau=args.n_tau, seed=args.seed,
        step_k=args.step_k, smooth_display=args.smooth_display)

    cav = attach_dispersion(load_cavity_params(), cfg.n_tau)
    noise_cfg = write_noise_off_config(CONFIG_PATH)
    try:
        print(f"[sweep] deterministic (noise-off) config -> {noise_cfg}")
        sweep = run_detuning_sweep(cav, cfg, config_path=noise_cfg)
    finally:
        try:
            noise_cfg.unlink()
        except OSError:
            pass

    save_sweep_npz(RESULTS_DIR / SWEEP_NPZ, sweep, cfg)
    print(f"[sweep] data -> {RESULTS_DIR / SWEEP_NPZ}")
    render_and_report(sweep, cfg)
    print(f"[sweep] total wall time {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
