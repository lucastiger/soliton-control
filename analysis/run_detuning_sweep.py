#!/usr/bin/env python
"""Quasi-static detuning sweep of a MULTI-SOLITON state -> soliton-staircase figure.

This analysis-layer driver traces an N-soliton state in POWER vs pump-cavity
detuning and renders the soliton-staircase figure
(``analysis/results/soliton_steps.{png,pdf}``).  It uses ONLY the solver's public
API (:func:`simulator.lle_solver.solve_lle_ssfm_jax`, via the
:mod:`analysis.dks_access` helpers) -- no solver / stepping / thermal / noise code
is modified, and neither are the step detector (``detect_power_steps``) nor the
hold averaging (``hold_window_average``).

Multi-soliton staircase protocol
--------------------------------
A bare blue->red detuning ramp at this operating point ignites modulation
instability (MI) and never nucleates a clean soliton state (see
``analysis/adiabatic_sweeps.py``); soliton states are instead accessed
deterministically by seeding analytic sech ansatz pulses
(:func:`analysis.dks_access.sech_soliton_seed`).  This driver seeds
``n_solitons`` (default 5) pulses at ``dw_start_kappa`` with DETERMINISTIC
symmetry-broken positions (``theta_j = 2*pi*j/N + delta_j``, ``delta_j`` drawn
from ``np.random.default_rng(position_seed)`` within
``+/- position_jitter_frac * 2*pi/N``), settles for ``settle_rt`` round trips,
and ASSERTS that exactly ``n_solitons`` temporal peaks survived (all seeds on
the attractor, none merged) before sweeping.  Equal spacing with zero jitter is
forbidden: identical equidistant solitons annihilate simultaneously, which
collapses the staircase into one event.  The detuning is then swept
quasi-statically DOWN (each hold warm-continues the previous field + thermal
state), so the solitons annihilate ONE BY ONE as the branch's lower edge is
approached -- each sequential single-soliton annihilation drops the intracavity
power by roughly one soliton's energy: the soliton staircase.  The staircase
emerges purely from the existing down-sweep dynamics; the detector, the
averaging, and the solver are untouched.

Scan direction (DISCOVERED, not assumed)
----------------------------------------
``delta_omega = omega_res - omega_pump`` (+ve = red-detuned = soliton side).  The
scan direction that produces a GRID-CONVERGED annihilation edge for this system
is *decreasing* detuning, for a physical reason established by running the sweep
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
  the solitons annihilate and the field collapses onto the MI/CW branch.

The sweep therefore runs from high detuning DOWN through the annihilation
cascade; the figure still plots detuning increasing left->right.

Per-detuning observables and averaging
--------------------------------------
For each detuning the PRIMARY observable is the total intracavity power
``sum_mu |a_mu|^2`` averaged over the FINAL ``avg_frac`` (default 25%) of a long
constant-detuning hold.  This one operation does two jobs: it discards the
per-step re-settling transient AND cycle-averages the deterministic breather (on
part of the branch the attractor is a limit cycle, so the instantaneous power
oscillates and only a slow-time average is a reproducible observable).  The
averaging is done in LINEAR power
(:func:`analysis.spectral_metrics.hold_window_average`), never in dB and never on
the complex field.

SECONDARY observable: the pump-excluded comb power
``P_comb = sum_{mu != 0} |a_mu|^2``, computed from the ~8 field snapshots per
hold (``snapshot_interval = max(hold_rt // 8, 1)``) that fall inside the same
final-``avg_frac`` window, and recorded as its mean/std.  Justification: comb
power is the standard experimental staircase observable and removes the
CW-background near-resonance rise that dominated the old trace's first
differences.  The PRIMARY observable remains total intracavity power; P_comb is
adopted as the PLOTTED primary ONLY if its matched-step contrast is strictly
higher.  Matched-step contrast of a trace = min matched |step_dy| / MAD of dy,
where the steps are the UNCHANGED ``detect_power_steps`` detections on that
trace, a detection is "matched" when its edge coincides with a
``soliton_count`` transition edge of the down-sweep (excluding the final
power-muted 1->0 edge -- see Honesty constraints), and MAD is the median
absolute deviation (about the median) of the trace's first differences.  A
trace with no matched detection has contrast 0 (no staircase visibility).  The
decision is made from the regenerated data and documented in the
``soliton_step`` block of ``spectral_metrics.json``, with the other observable
plotted as the second panel.

Per hold the driver additionally records the sorted temporal peak angles
(``peak_positions_rad``, via :func:`analysis.dks_access.temporal_peak_positions`),
a labeler-gated ``soliton_count``, and the schema-v2 breathing fields
(``is_breather``, ``is_stationary``, ``breathing_relstd``,
``breathing_period_rt``) computed with
:func:`analysis.dks_access.breathing_metrics` on the per-round-trip
``U_int_history`` of the hold.

soliton_count gate (documented choice)
--------------------------------------
The 7-class taxonomy of ``simulator/state_labeler.py`` labels soliton states as
class 4 (multi-soliton), class 5 (soliton crystal -- evenly spaced
multi-soliton) and class 6 (single soliton), so ``SOLITON_LABELS = (4, 5, 6)``.
Empirically, however, the taxonomy's multi-soliton classes CANNOT gate the
states this sweep produces: the labeler's spectral-entropy chaos gate
(``entropy_chaotic = 0.5``, tuned for single-soliton discrimination) misroutes
genuine N = 5 soliton states to class 3 across the upper half of the branch
(the settled 5-soliton comb at 12*kappa has norm-entropy ~0.66 -- five combs
carry ~5x the sideband lines of one), and the deeply-breathing edge states fail
the class-6 sech^2 fit.  Gating on labels alone would therefore zero the
soliton count over holds whose fields demonstrably carry five localized pulses
(verified by pulse-prominence inventory against the CW background).  The
driver consequently accepts EITHER the taxonomy gate OR the documented
fallback gate for a labeler-misrouted soliton state::

    soliton_count = n_peaks if (np_label in SOLITON_LABELS
                                or (finite field AND n_peaks >= 1
                                    AND contrast >= labeler contrast floor))
                    else 0

with ``contrast = max|E|^2 / mean|E|^2`` of the end-of-hold field and the floor
= the labeler's soliton contrast threshold (``contrast_high`` = 8).  The
fallback zeroes the post-collapse MI/CW states (contrast ~1-2) without vetoing
genuine solitons; on this deterministic noise-off sweep no genuinely chaotic
high-contrast state occurs (the class-3 holds ARE the misrouted multi-soliton /
breathing states).  CAVEAT (honest limitation, kept as-is): ``n_peaks`` is the
end-of-hold snapshot count at 50% of the max peak, so in the deep-breathing
sub-band -- where the pulse amplitudes desynchronize and individual solitons
transiently dip below half of the momentary maximum -- ``soliton_count``
UNDERCOUNTS even though all pulses persist (the raw ``peak_positions_rad`` and
``P_comb`` are the reliable per-hold records there).  No thresholds were
changed anywhere; this caveat is reported, not patched.

Hold length and adiabaticity.  The photon lifetime is ``1/kappa`` ~ 162 round
trips and the breather period is ~150-180 RT.  The default ``hold_rt = 2000`` is
~12 photon lifetimes (the field re-settles to ``exp(-12)`` of any transient after
a step) and ~13 breathing periods per hold -- enough for breathing-mediated
switching to complete within a hold, with the final 25% (~500 RT) spanning ~3
breather periods for both transient decay and breather cycle-averaging.  A fully
THERMALLY adiabatic hold (``tau_th`` ~ 1.2e5 RT ~ 760 photon lifetimes) is
deliberately NOT used: the steady thermo-optic shift here is only
~0.01-0.03*kappa (negligible), so thermal lag between steps does not move the
branch; ``hold_rt`` is exposed for callers that want to push toward that regime.

Honesty constraints
-------------------
The final 1->0 annihilation at ~6.1κ remains power-muted (comparable-energy MI
comb) and MUST NOT be forced to register as a power step; it is expected to
appear as a soliton_count 1->0 transition without a matched power discontinuity.
The N->N-1 transitions above it are the staircase.  No smoothing, no detector
changes, no re-thresholding.

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
  (breathing-amplitude indicator), comb power, transmission, soliton counts,
  peak positions, breathing fields, labels and the full sweep config, so the
  figure regenerates without re-running the sweep.
* ``soliton_steps.{png,pdf}`` -- the publication figure.
* ``spectral_metrics.json`` gains a ``soliton_step`` block (staircase
  transitions, primary-observable decision, the power-discontinuity detection
  result, and full provenance).
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
    STATIONARY_RELSTD,
    _run,
    attach_dispersion,
    breathing_metrics,
    count_temporal_peaks,
    load_cavity_params,
    numpy_label,
    sech_soliton_seed,
    temporal_peak_positions,
)
from analysis.spectral_metrics import (  # noqa: E402
    DEFAULT_STEP_K,
    SOLITON_STEP_DEFINITION,
    detect_power_steps,
    hold_window_average,
    plot_soliton_steps,
    single_dks_region,
)
from simulator.state_labeler import make_threshold_params  # noqa: E402

SWEEP_NPZ = "detuning_sweep.npz"
STEPS_PNG = "soliton_steps.png"
METRICS_JSON = "spectral_metrics.json"

# Class ids of the 7-class taxonomy (simulator/state_labeler.py) that denote
# soliton states: 4 = multi-soliton, 5 = soliton crystal (evenly spaced
# multi-soliton), 6 = single soliton.  soliton_count is the temporal peak count
# gated by this set OR by the documented fallback for labeler-misrouted soliton
# states (finite field AND n_peaks >= 1 AND contrast >= the labeler's soliton
# contrast floor) -- see the module docstring: the labeler's entropy gate
# empirically misroutes genuine N-soliton states to class 3 on the upper half
# of the branch, so labels alone cannot gate the multi-soliton holds.
SOLITON_LABELS = (4, 5, LABEL_SINGLE_SOLITON)


# ---------------------------------------------------------------------------
# Sweep configuration (all tunables in one place; nothing hardcoded in the
# functions below -- they read this object).  Defaults follow the discovered
# multi-soliton staircase: n_solitons = 5 seeded well inside the branch
# (12*kappa, clean and stationary) are warm-continued DOWN in detuning through
# the breather sub-band, annihilating one by one (the staircase); the sweep
# ends at 5.5*kappa, just past the final (power-muted) 1->0 annihilation near
# ~6.1*kappa.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepConfig:
    dw_start_kappa: float = 12.0     # seed / first hold (clean stationary DKS)
    dw_stop_kappa: float = 5.5       # ends just past the last annihilation edge
    n_steps: int = 27                # detuning samples (0.25*kappa spacing)
    settle_rt: int = 6000            # pre-settle at dw_start (seeds -> attractor)
    hold_rt: int = 2000              # round trips held per detuning step
    avg_frac: float = 0.25           # average the final fraction of each hold
    n_tau: int = 4096                # FFT grid (resolves the comb core each step)
    seed: int = 0                    # RNG seed (noise is off; kept for provenance)
    pin_w: float = PIN_W             # on-chip pump power (== config pin_w)
    step_k: float = DEFAULT_STEP_K   # power-discontinuity MAD multiple (conservative)
    smooth_display: int = 0          # display-only smoothing window (0 = off)
    n_solitons: int = 5              # seeded soliton number N (the staircase top)
    position_seed: int = 1           # RNG seed of the symmetry-breaking jitter
    position_jitter_frac: float = 0.25   # jitter half-range, in units of 2*pi/N

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
    """Warm-continuation step-and-hold detuning sweep of an N-soliton state.

    Seeds ``cfg.n_solitons`` solitons at ``cfg.dw_start_kappa`` with the
    deterministic symmetry-broken placement (pre-settled for ``cfg.settle_rt``
    round trips and VERIFIED to still carry exactly ``cfg.n_solitons`` temporal
    peaks -- the run aborts otherwise, never continuing with an unknown state),
    then holds each detuning in ``cfg.detunings_kappa()`` for ``cfg.hold_rt``
    round trips, carrying the field + thermal state forward.  Per step it
    records the final-``avg_frac`` LINEAR-power average (and std) of the total
    intracavity power ``sum_mu |a_mu|^2`` and of the through-port power
    ``P_trans``; the pump-excluded comb power ``P_comb`` (mean/std over the
    ~8-per-hold field snapshots inside the same window); the mean effective
    detuning; the state label / peak count / peak positions; the labeler-gated
    ``soliton_count``; and the schema-v2 breathing fields from
    ``breathing_metrics`` on the hold's per-RT ``U_int_history``.

    ``config_path`` is the (deterministic, noise-off) solver config.  Returns a
    dict of per-step arrays (``peak_positions_rad`` is NaN-padded to the widest
    step); every solver call uses ``**PRODUCTION_NUMERICS``.
    """
    kappa = cav.kappa
    n_tau = int(cfg.n_tau)
    # sum_mu |a_mu|^2 = n_tau^2 / t_r * U_int  (numpy-FFT Parseval + solver U_int
    # normalisation U_int = sum_tau|E|^2 * t_r/n_tau).
    modes_from_uint = (n_tau ** 2) / cav.t_r

    dws_k = cfg.detunings_kappa()
    print(f"[sweep] seeding {cfg.n_solitons} solitons at "
          f"{cfg.dw_start_kappa:.2f} kappa (position_seed={cfg.position_seed}, "
          f"jitter={cfg.position_jitter_frac:g}; settle {cfg.settle_rt} RT, "
          f"n_tau={n_tau}) ...")
    seed_field = sech_soliton_seed(
        cfg.dw_start_kappa * kappa, cav, n_tau=n_tau, pin=cfg.pin_w,
        n_solitons=int(cfg.n_solitons), position_seed=int(cfg.position_seed),
        position_jitter_frac=float(cfg.position_jitter_frac))
    sol0 = _run(cfg.dw_start_kappa * kappa, int(cfg.settle_rt), cav,
                e0=seed_field, seed=int(cfg.seed), n_tau=n_tau, pin=cfg.pin_w,
                snapshot_interval=int(cfg.settle_rt), config_path=config_path,
                **PRODUCTION_NUMERICS)
    e_prev = np.asarray(sol0["e_final"])[0]
    dt_prev = float(np.asarray(sol0["delta_t_final"]).reshape(-1)[0])
    u0_hist = np.asarray(sol0["U_int_history"])[0]

    n_settled = count_temporal_peaks(e_prev)
    settled_positions = temporal_peak_positions(e_prev)
    print(f"[sweep] seed settled: peaks={n_settled} (target {cfg.n_solitons}) "
          f"positions={np.round(settled_positions, 3)}")
    if n_settled != int(cfg.n_solitons):
        print(f"[sweep] ABORT: settled peak count {n_settled} != "
              f"n_solitons {cfg.n_solitons} -- seeds merged or died during the "
              f"pre-settle, so the branch start is an UNKNOWN state.")
        raise RuntimeError(
            f"multi-soliton pre-settle failed: {n_settled} peaks survived out "
            f"of {cfg.n_solitons} seeded. Raise the pairwise separation "
            f"(different position_seed / larger position_jitter_frac spread) "
            f"or lower n_solitons; never continue with an unknown state.")
    seed_metrics = {
        "n_peaks": int(n_settled),
        "peak_positions_rad": settled_positions,
        "u_int_final": float(u0_hist[-1]),
        **breathing_metrics(u0_hist),
    }

    # Labeler's soliton contrast floor (contrast_high = 8) for the documented
    # fallback gate; taken from the same threshold bundle the NumPy labeler uses.
    contrast_floor = float(make_threshold_params(
        cav.kappa, cav.kappa_c, cfg.pin_w,
        abs(cfg.dw_start_kappa * kappa))["contrast_high"])

    snap_int = max(int(cfg.hold_rt) // 8, 1)
    rows = []
    for i, dwk in enumerate(dws_k):
        t0 = time.time()
        sol = _run(dwk * kappa, int(cfg.hold_rt), cav, e0=e_prev,
                   delta_t0=dt_prev, seed=int(cfg.seed), n_tau=n_tau,
                   pin=cfg.pin_w, snapshot_interval=snap_int,
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

        # Pump-excluded comb power from the snapshots inside the final-avg_frac
        # window (snapshot k is taken at round trip k * snap_int).
        snaps = np.asarray(sol["E_snapshots"])[0]
        snap_rt = np.arange(snaps.shape[0]) * snap_int
        in_window = snap_rt >= u_avg["i_start"]
        if not in_window.any():
            in_window[-1] = True        # degenerate hold: use the last snapshot
        spec_pow = np.abs(np.fft.fftshift(
            np.fft.fft(snaps[in_window], axis=-1), axes=-1)) ** 2
        p_comb = spec_pow.sum(axis=-1) - spec_pow[:, n_tau // 2]  # mu != 0

        label = numpy_label(e_final, cav, dwk * kappa, pin=cfg.pin_w)
        n_peaks = count_temporal_peaks(e_final)
        peak_positions = temporal_peak_positions(e_final)
        # Class 6 from the NumPy labeler already encodes the single temporal peak
        # + sech^2-envelope goodness-of-fit, so (label == 6 and one peak, finite)
        # is a faithful single-DKS flag without recomputing the full bundle.
        finite = bool(np.all(np.isfinite(e_final)))
        single = bool(label == LABEL_SINGLE_SOLITON and n_peaks == 1 and finite)
        # soliton_count: taxonomy gate OR the documented fallback gate for
        # labeler-misrouted soliton states (see the module docstring).
        p_final = np.abs(e_final) ** 2
        field_contrast = float(p_final.max() / max(p_final.mean(), 1e-300))
        is_soliton_state = bool(
            label in SOLITON_LABELS
            or (finite and n_peaks >= 1 and field_contrast >= contrast_floor))
        soliton_count = int(n_peaks) if is_soliton_state else 0

        # Schema-v2 breathing fields (V6) on the hold's per-RT U_int history.
        v6 = breathing_metrics(u_hist)

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
            "P_comb": float(np.mean(p_comb)),                # sum_{mu!=0}|a_mu|^2
            "P_comb_std": float(np.std(p_comb)),
            "np_label": int(label),
            "n_peaks": int(n_peaks),
            "is_single": single,
            "contrast": field_contrast,
            "soliton_count": soliton_count,
            "peak_positions_rad": peak_positions,
            "is_breather": bool(v6["is_breather"]),
            "is_stationary": bool(v6["breathing_relstd"] < STATIONARY_RELSTD),
            "breathing_relstd": float(v6["breathing_relstd"]),
            "breathing_period_rt": float(v6["breathing_period_rt"]),
        })
        e_prev, dt_prev = e_final, dt_final
        print(f"[sweep] {i + 1:2d}/{len(dws_k)}  dw={dwk:6.2f}k  "
              f"P_intra={rows[-1]['P_intra']:.4e}  "
              f"P_comb={rows[-1]['P_comb']:.4e}  "
              f"U_relstd={rows[-1]['U_int_relstd']:.2%}  lbl={label} "
              f"npk={n_peaks} N={soliton_count}  ({time.time() - t0:.1f}s)")

    scalar_keys = [k for k in rows[0] if k != "peak_positions_rad"]
    out = {k: np.array([r[k] for r in rows]) for k in scalar_keys}
    # peak_positions_rad is ragged (the peak count varies along the sweep);
    # store it NaN-padded to the widest step, shape (n_steps, max_peaks>=1).
    max_np = max((r["peak_positions_rad"].size for r in rows), default=0)
    pos = np.full((len(rows), max(max_np, 1)), np.nan)
    for i, r in enumerate(rows):
        pos[i, :r["peak_positions_rad"].size] = r["peak_positions_rad"]
    out["peak_positions_rad"] = pos
    out["kappa_rad_s"] = float(kappa)
    out["t_r_s"] = float(cav.t_r)
    out["fsr_hz"] = float(cav.fsr_measured_hz if cav.fsr_measured_hz is not None
                          else cav.fsr_hz)
    out["seed_metrics"] = seed_metrics
    return out


# ---------------------------------------------------------------------------
# Staircase helpers (post-processing only; the detector itself is untouched)
# ---------------------------------------------------------------------------
def staircase_transition_edges(soliton_count) -> tuple:
    """Ascending-detuning edge indices of the soliton_count transitions.

    Edge ``i`` joins samples ``i`` and ``i+1`` of the ascending-detuning trace
    (the :func:`detect_power_steps` convention).  A transition is an edge where
    the soliton count drops going DOWN in detuning, i.e.
    ``count[i+1] > count[i]`` in ascending order.  Returns
    ``(all_transitions, matched)`` where ``matched`` additionally requires
    ``count[i] >= 1``: the final ->0 annihilation is EXCLUDED per the honesty
    constraint (it is power-muted -- a comparable-energy MI comb replaces the
    last soliton -- and must not be forced to register as a power step).
    """
    c = np.asarray(soliton_count, dtype=int).ravel()
    all_tr = [int(i) for i in range(c.size - 1) if c[i + 1] > c[i]]
    matched = [i for i in all_tr if c[i] >= 1]
    return all_tr, matched


def matched_step_contrast(y, steps, matched_edges) -> dict:
    """Matched-step contrast of a trace: min matched |step_dy| / MAD of dy.

    ``y`` is an observable on the ascending-detuning grid, ``steps`` the
    UNCHANGED :func:`detect_power_steps` result on that trace, and
    ``matched_edges`` the staircase-transition edge indices from
    :func:`staircase_transition_edges` (final ->0 edge already excluded).  A
    detected step is "matched" when its edge coincides with a transition edge;
    the contrast is the SMALLEST matched |step_dy| divided by the MAD (about
    the median) of ALL first differences of the trace -- i.e. how far even the
    weakest detector-confirmed staircase step stands above the trace's typical
    variation.  Scale-invariant (raw and normalised traces give the same
    value).  A trace with no matched detection has contrast 0.0: its staircase
    is invisible to the detector.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    dy = np.diff(y)
    med = float(np.median(dy))
    mad = float(np.median(np.abs(dy - med)))
    if mad <= 0.0:
        span = float(np.max(y) - np.min(y))
        mad = max(1e-12, 1e-6 * (span if span > 0 else 1.0))
    matched_detected = [int(i) for i in steps["edges"] if i in matched_edges]
    contrast = (float(min(abs(dy[i]) for i in matched_detected) / mad)
                if matched_detected else 0.0)
    return {
        "contrast": contrast,
        "mad_dy": mad,
        "matched_detected_edges": matched_detected,
        "matched_step_dy": [float(dy[i]) for i in matched_detected],
    }


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


# Per-step arrays that every sweep npz carries (legacy schema, still written).
_NPZ_BASE_KEYS = ("dw_over_kappa", "dw_rad_s", "dw_eff_over_kappa", "P_intra",
                  "P_intra_std", "U_int", "U_int_std", "P_trans", "P_trans_std",
                  "np_label", "n_peaks", "is_single", "kappa_rad_s", "t_r_s",
                  "fsr_hz")
# Multi-soliton staircase additions (schema v2).
_NPZ_V2_KEYS = ("P_comb", "P_comb_std", "soliton_count", "peak_positions_rad",
                "contrast", "is_breather", "is_stationary", "breathing_relstd",
                "breathing_period_rt")


def save_sweep_npz(path: Path, sweep: dict, cfg: SweepConfig) -> None:
    """Persist the sweep so the figure regenerates without re-running the solver."""
    arrays = {k: sweep[k] for k in _NPZ_BASE_KEYS + _NPZ_V2_KEYS}
    np.savez_compressed(
        path,
        pin_w=float(cfg.pin_w),
        sweep_config_json=json.dumps(cfg.as_dict()),
        **arrays,
    )


def load_sweep_npz(path: Path):
    """Load a committed ``detuning_sweep.npz`` back into ``(sweep, cfg)``.

    Lets the figure + JSON be regenerated from the saved data alone (no solver
    re-run), per the deliverable convention that outputs are regenerable.  The
    schema-v2 staircase arrays are loaded when present, so a legacy
    (single-soliton) npz still renders with the legacy layout.
    """
    d = np.load(path, allow_pickle=False)
    cfg = SweepConfig(**json.loads(str(d["sweep_config_json"])))
    sweep = {k: (d[k] if d[k].shape else float(d[k])) for k in _NPZ_BASE_KEYS}
    for k in _NPZ_V2_KEYS:
        if k in d.files:
            sweep[k] = d[k]
    sweep["is_single"] = sweep["is_single"].astype(bool)
    for k in ("is_breather", "is_stationary"):
        if k in sweep:
            sweep[k] = sweep[k].astype(bool)
    return sweep, cfg


def render_and_report(sweep: dict, cfg: SweepConfig) -> Path:
    """Build the soliton-staircase figure + JSON block from an assembled sweep.

    Sorts by detuning, normalises the observables, locates the staircase
    (soliton_count transitions) and the single-DKS existence region, makes the
    data-driven primary-observable decision (P_intra vs P_comb by matched-step
    contrast; see the module docstring), runs the power-discontinuity detector
    on the plotted primary, writes the figure and the ``soliton_step`` block,
    and returns the figure path.  Pure post-processing -- no solver -- so it
    also serves ``--render-only`` (including on a legacy npz without the
    staircase arrays, which falls back to the legacy P_intra/transmission
    layout).
    """
    order = np.argsort(sweep["dw_over_kappa"])
    dwk = np.asarray(sweep["dw_over_kappa"])[order]
    P = np.asarray(sweep["P_intra"])[order]
    P_std = np.asarray(sweep["P_intra_std"])[order]
    T = (np.asarray(sweep["P_trans"]) / float(cfg.pin_w))[order]
    is_single = np.asarray(sweep["is_single"])[order]

    has_staircase = "soliton_count" in sweep and "P_comb" in sweep

    # Single-DKS existence region + its lower edge, from the per-step state
    # flag (unchanged from the single-soliton driver).
    lo_k, hi_k, annih_k = single_dks_region(dwk, is_single)

    kappa = float(sweep["kappa_rad_s"])
    metadata = {"kappa_rad_s": kappa}

    if has_staircase:
        counts = np.asarray(sweep["soliton_count"], dtype=int)[order]
        Pc = np.asarray(sweep["P_comb"])[order]
        Pc_std = np.asarray(sweep["P_comb_std"])[order]

        # Staircase transition edges from the soliton counts; the matched set
        # excludes the final power-muted ->0 annihilation.
        all_edges, matched = staircase_transition_edges(counts)

        # Detector (UNCHANGED detect_power_steps) on both normalised traces,
        # then the matched-step-contrast decision (module docstring).
        P_norm_t = P / float(np.max(P))
        Pc_norm_t = Pc / float(np.max(Pc))
        steps_intra = detect_power_steps(dwk, P_norm_t, k=cfg.step_k)
        steps_comb = detect_power_steps(dwk, Pc_norm_t, k=cfg.step_k)
        contrast_intra = matched_step_contrast(P_norm_t, steps_intra, matched)
        contrast_comb = matched_step_contrast(Pc_norm_t, steps_comb, matched)
        use_comb = contrast_comb["contrast"] > contrast_intra["contrast"]

        intra_label = r"intracavity power  $\sum_\mu |a_\mu|^2$  (norm.)"
        comb_label = (r"pump-excluded comb power  "
                      r"$\sum_{\mu\neq 0} |a_\mu|^2$  (norm.)")
        if use_comb:
            y1, y1_std, label1, steps = Pc, Pc_std, comb_label, steps_comb
            y2, label2, name1, name2 = P, intra_label, "P_comb", "P_intra"
        else:
            y1, y1_std, label1, steps = P, P_std, intra_label, steps_intra
            y2, label2, name1, name2 = Pc, comb_label, "P_intra", "P_comb"
    else:
        matched, all_edges = [], []
        contrast_intra = contrast_comb = None
        steps_intra = steps_comb = None
        use_comb = False
        y1, y1_std = P, P_std
        label1 = r"intracavity power  $\sum_\mu |a_\mu|^2$  (norm.)"
        y2, label2 = T, "norm. transmission"
        name1, name2 = "P_intra", "P_trans"
        steps = None

    y1_ref = float(np.max(y1))
    y1_norm, y1_norm_std = y1 / y1_ref, y1_std / y1_ref
    y2_ref = float(np.max(y2))
    y2_norm = y2 / y2_ref if has_staircase else y2  # transmission already norm.

    if steps is None:   # legacy npz: detector on the (only) primary trace
        steps = detect_power_steps(dwk, y1_norm, k=cfg.step_k)
    # Which detected discontinuities coincide with a staircase transition edge?
    steps_matched_flags = ([bool(i in all_edges) for i in steps["edges"]]
                           if has_staircase else [])

    caption = (
        f"{cfg.n_solitons}-soliton staircase, pin = {cfg.pin_w} W, n_tau = "
        f"{cfg.n_tau}. Deterministic symmetry-broken seed "
        f"(position_seed={cfg.position_seed}, jitter="
        f"{cfg.position_jitter_frac:g}) at {cfg.dw_start_kappa:g}$\\kappa$, "
        f"warm continuation swept DOWN, hold {cfg.hold_rt} RT/step, averaged "
        f"over the final {int(100 * cfg.avg_frac)}% (cycle-averages the "
        f"breather). Thermo-optic model ON (deterministic, noise off). The "
        f"solitons annihilate sequentially at the branch's lower edge -- the "
        f"staircase; the final 1->0 annihilation is power-muted in the TOTAL "
        f"power and must not be forced to register as a power step. "
        + ("Green = single-DKS existence region. " if lo_k is not None else "")
        + f"Dotted lines mark first-difference discontinuities of the plotted "
        f"primary trace. Raw data, no smoothing."
        if has_staircase else
        f"Single-DKS branch, pin = {cfg.pin_w} W, n_tau = {cfg.n_tau} (legacy "
        f"npz render).")
    metadata["caption"] = caption

    plot_path = plot_soliton_steps(
        dwk, y1_norm, RESULTS_DIR / STEPS_PNG, power_std=y1_norm_std,
        transmission=y2_norm,
        soliton_region=(lo_k, hi_k) if lo_k is not None else None,
        annihilation_kappa=annih_k, steps=steps, metadata=metadata,
        observable_label=label1,
        second_panel_ylabel=(label2 if has_staircase else "norm. transmission"),
        second_panel_legend=(label2 if has_staircase else
                             "through-port power "
                             "$P_\\mathrm{trans}/P_\\mathrm{in}$"),
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

    def _edge_mid(i):
        return float(0.5 * (dwk[i] + dwk[i + 1]))

    block = {
        "metric": "soliton_step",
        "metric_definition": SOLITON_STEP_DEFINITION,
        "protocol": ("multi-soliton staircase: deterministic symmetry-broken "
                     "N-soliton seed, warm-continuation down-sweep, sequential "
                     "soliton annihilations (the transitions list records the "
                     "actual per-edge count drops)"
                     if has_staircase else "single-DKS branch (legacy data)"),
        "observable": {
            "primary": "total intracavity power sum_mu |a_mu|^2 (the primary "
                       "OBSERVABLE regardless of which trace is plotted first)",
            "secondary": "pump-excluded comb power P_comb = sum_{mu != 0} "
                         "|a_mu|^2, mean/std over the ~8-per-hold field "
                         "snapshots inside the final-avg_frac window",
            "note": "transmission P_trans/P_in also stored in "
                    "detuning_sweep.npz",
        },
        "sweep_direction": "decreasing detuning (delta_omega down) through the "
                           "sequential annihilation cascade to the branch's "
                           "lower edge; the increasing direction gives no "
                           "grid-converged step (the high-detuning collapse is "
                           "an FFT-truncation artifact, non-convergent in "
                           "n_tau)",
        "n_detunings": int(dwk.size),
        "detuning_range_over_kappa": [float(cfg.dw_stop_kappa),
                                      float(cfg.dw_start_kappa)],
        "single_dks_existence_region_over_kappa": (
            [lo_k, hi_k] if lo_k is not None else None),
        "power_trace_discontinuities": {
            "rule": "|diff(P)[i] - median(diff P)| > k * 1.4826 * MAD(diff P)",
            "trace": name1,
            "k": float(steps["k"]),
            "robust_sigma": float(steps["sigma"]),
            "detected_over_kappa": list(steps["step_x"]),
            "coincides_with_soliton_count_transition": steps_matched_flags,
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

    if has_staircase:
        transitions = [{"edge_over_kappa": _edge_mid(i),
                        "from_count": int(counts[i + 1]),
                        "to_count": int(counts[i])} for i in all_edges]
        final_1_to_0 = [t for t in transitions
                        if t["from_count"] >= 1 and t["to_count"] == 0]
        block["staircase"] = {
            "n_solitons_seeded": int(cfg.n_solitons),
            "position_seed": int(cfg.position_seed),
            "position_jitter_frac": float(cfg.position_jitter_frac),
            "soliton_count_by_detuning_over_kappa": {
                f"{float(d):.2f}": int(c) for d, c in zip(dwk, counts)},
            "transitions": transitions,
            "matched_staircase_edges_over_kappa": [
                _edge_mid(i) for i in matched],
            "final_1_to_0_over_kappa": (
                final_1_to_0[0]["edge_over_kappa"] if final_1_to_0 else None),
            "honesty_note": (
                "the final 1->0 annihilation remains power-muted "
                "(comparable-energy MI comb) and MUST NOT be forced to "
                "register as a power step; it is expected to appear as a "
                "soliton_count 1->0 transition without a matched power "
                "discontinuity. The N->N-1 transitions above it are the "
                "staircase. No smoothing, no detector changes, no "
                "re-thresholding."),
            "soliton_count_gate": (
                "n_peaks if (np_label in SOLITON_LABELS = (4, 5, 6)) OR "
                "(finite field AND n_peaks >= 1 AND contrast >= the labeler's "
                "soliton contrast floor contrast_high = 8) else 0. The "
                "taxonomy gate alone empirically zeroes genuine multi-soliton "
                "holds: the labeler's spectral-entropy chaos gate misroutes "
                "bright N-soliton combs to class 3 (see the driver "
                "docstring). CAVEAT: n_peaks is the end-of-hold snapshot "
                "count at 50% of the max peak, so in the deep-breathing "
                "sub-band -- where pulse amplitudes desynchronize -- "
                "soliton_count UNDERCOUNTS even though all pulses persist; "
                "P_comb and peak_positions_rad are the reliable per-hold "
                "records there. Reported, not patched."),
        }
        block["primary_observable_decision"] = {
            "rule": ("P_comb is adopted as the PLOTTED primary ONLY if its "
                     "matched-step contrast (min matched |step_dy| / MAD of "
                     "dy) is strictly higher than total intracavity power's. "
                     "Steps are the unchanged detect_power_steps detections "
                     "on each normalised trace; a detection is matched when "
                     "its edge coincides with a soliton_count transition edge "
                     "(the power-muted final ->0 annihilation is excluded); a "
                     "trace with no matched detection has contrast 0. Decided "
                     "from the regenerated data."),
            "matched_step_contrast_P_intra": contrast_intra["contrast"],
            "matched_step_contrast_P_comb": contrast_comb["contrast"],
            "mad_dy_P_intra": contrast_intra["mad_dy"],
            "mad_dy_P_comb": contrast_comb["mad_dy"],
            "matched_step_dy_P_intra": contrast_intra["matched_step_dy"],
            "matched_step_dy_P_comb": contrast_comb["matched_step_dy"],
            "detected_steps_P_intra_over_kappa": list(steps_intra["step_x"]),
            "detected_steps_P_comb_over_kappa": list(steps_comb["step_x"]),
            "matched_detected_P_intra_over_kappa": [
                _edge_mid(i) for i in contrast_intra["matched_detected_edges"]],
            "matched_detected_P_comb_over_kappa": [
                _edge_mid(i) for i in contrast_comb["matched_detected_edges"]],
            "plotted_primary": name1,
            "second_panel": name2,
        }
        block["soliton_step"] = {
            "annihilation_over_kappa": annih_k,
            "note": "lower edge of the single-DKS existence region (label 6 + "
                    "single temporal peak). The 1->0 event here is power-muted "
                    "(the last soliton is replaced by a comparable-energy MI "
                    "comb) -- it registers as a soliton_count transition, not "
                    "as a power step.",
        }
    else:
        block["soliton_step"] = {
            "annihilation_over_kappa": annih_k,
            "note": "lower edge of the single-DKS existence region (label 6 + "
                    "single temporal peak); legacy single-soliton data.",
        }

    _update_json(RESULTS_DIR / METRICS_JSON, "soliton_step", block)

    if has_staircase:
        print(f"[sweep] soliton counts (ascending dw): "
              f"{[int(c) for c in counts]}")
        print(f"[sweep] staircase transitions at: "
              + (", ".join(f"{_edge_mid(i):.2f}k "
                           f"({counts[i + 1]}->{counts[i]})"
                           for i in all_edges) or "none"))
        print(f"[sweep] matched-step contrast: P_intra="
              f"{contrast_intra['contrast']:.2f}  P_comb="
              f"{contrast_comb['contrast']:.2f}  -> plotted primary: {name1}")
    print(f"[sweep] single-DKS existence region: "
          + (f"[{lo_k:.2f}, {hi_k:.2f}] kappa" if lo_k is not None else "none"))
    print(f"[sweep] power-trace discontinuities ({name1}) at: "
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
    ap.add_argument("--n-solitons", type=int, default=SweepConfig.n_solitons,
                    help="seeded soliton number N (the staircase top)")
    ap.add_argument("--position-seed", type=int,
                    default=SweepConfig.position_seed,
                    help="RNG seed of the deterministic symmetry-breaking "
                         "position jitter")
    ap.add_argument("--position-jitter-frac", type=float,
                    default=SweepConfig.position_jitter_frac,
                    help="jitter half-range as a fraction of the mean spacing "
                         "2*pi/N (must be > 0 for N > 1)")
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
        step_k=args.step_k, smooth_display=args.smooth_display,
        n_solitons=args.n_solitons, position_seed=args.position_seed,
        position_jitter_frac=args.position_jitter_frac)

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
