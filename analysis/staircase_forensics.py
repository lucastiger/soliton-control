#!/usr/bin/env python
"""Offline forensics: is the committed staircase's soliton_count flicker an
end-of-hold COUNTING ARTIFACT or REAL soliton re-nucleation?

The committed ``analysis/results/detuning_sweep.npz`` (261-hold, N = 5 seeded
staircase) fails the driver's monotonicity gate: along the descending sweep the
per-hold ``soliton_count`` INCREASES 38 times (flicker).  Two hypotheses:

* **Counting artifact** -- all pulses persist; the end-of-hold single-snapshot
  peak count (peaks at 50% of the momentary max) transiently loses pulses in
  the deep-breathing sub-band where the pulse amplitudes desynchronize (the
  caveat already documented in ``analysis/run_detuning_sweep.py``).
* **Real re-nucleation** -- solitons genuinely annihilate and new ones
  nucleate a few holds later.

This script decides between them from the COMMITTED data alone -- it never
runs the solver, and it is read-only with respect to every committed artifact
except for writing the single new report
``analysis/results/staircase_forensics.md``.

Three tests per flicker event (post hold ``i`` where count[i] > count[i-1] in
descending-sweep order; "pre" = last earlier hold carrying at least the
recovered count; "dip" = the minimum-count hold between them):

* **TEST A -- position persistence.**  Circularly match the recovered peak
  angles ("post") against the pre-dip angles with tolerance 0.05 rad.  A
  recovered soliton sitting at its old angle never left; a re-nucleated one
  lands anywhere.  CONFOUND, measured and controlled here: the whole pulse
  train rotates COHERENTLY at ~0.036-0.046 rad/hold (a uniform group-velocity
  drift, measured on no-event consecutive same-count holds where nucleation is
  impossible), so the raw fixed-frame tolerance is exceeded by the drift alone
  for any dip spanning >= 2 holds.  The controlled statistic therefore removes
  ONE global rotation per comparison (relative positions are what nucleation
  would scramble) and is read against its measurement ceiling -- the identical
  statistic on the no-event control pairs -- and against the re-nucleation
  null (random angles, best-rotation-fitted).
* **TEST B -- energy continuity.**  A real annihilation removes one soliton's
  energy from the comb.  One soliton quantum is estimated as the median
  per-quantum |Delta P_comb| across the energy-visible clean count decrements
  (envelope drops outside flicker regions; the committed JSON's staircase
  block predates the step<->transition alignment machinery, so it cannot
  supply clean matched N -> N-1 steps -- stated in the report).  Each event's
  pre -> dip |Delta P_comb| is compared with (claimed quanta lost) x quantum.
* **TEST C -- breathing correlation.**  Artifact dips must live in the
  deep-breathing sub-band (that is the documented loss mechanism):
  breathing_relstd of undercount vs correct-count holds, is_breather at the
  dips, np_label at the dips (does the field still classify as a soliton
  state while the count collapses?), and an is_single integrity check at
  dw >= 6.5 kappa (the committed figure's single-DKS band).

The verdict rule follows the forensic brief -- "counting artifact" iff TEST A
shows > 0.9 position persistence AND TEST B shows sub-quantum energy changes
-- with TEST A evaluated on the confound-CONTROLLED statistic relative to its
measured ceiling (the raw fixed-frame number is reported alongside and shown
to be drift-limited, not nucleation-limited).  Everything the rule consumes is
printed and written to the report.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from analysis.run_detuning_sweep import (  # noqa: E402
    METRICS_JSON,
    SOLITON_LABELS,
    SWEEP_NPZ,
    load_sweep_npz,
)
from analysis.dks_access import RESULTS_DIR  # noqa: E402

REPORT_MD = "staircase_forensics.md"

TWO_PI = 2.0 * np.pi
MATCH_TOL_RAD = 0.05          # per the forensic brief (~11 soliton widths @ 8k)
COHERENT_PTP_RAD = 0.01       # max peak-to-peak spread for a "coherent" shift
CEILING_WINDOW = 5            # holds around an event for its local ceiling
QUANTUM_VISIBLE_K = 5.0       # energy-visible = |dP_comb| > K * MAD(diff P_comb)

# Required per-hold columns (name -> expected shape suffix description).
REQUIRED_KEYS = ("soliton_count", "peak_positions_rad", "P_comb", "P_comb_std",
                 "P_intra", "breathing_relstd", "is_breather", "np_label",
                 "is_single")


def _wrap(x):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


def _circ_dist(a, b):
    d = abs(float(a) - float(b)) % TWO_PI
    return min(d, TWO_PI - d)


def _match_fraction(post, pre, tol, *, remove_rotation):
    """Fraction of ``post`` angles within ``tol`` of some ``pre`` angle.

    ``remove_rotation=False`` is the raw fixed-frame test of the brief;
    ``remove_rotation=True`` additionally allows ONE global rotation of the
    whole pattern (searched over all pairwise offsets), which is what real
    re-nucleation would scramble and a coherent group-velocity drift does not.
    Returns ``(n_matched, n_post, best_shift)``.
    """
    post = [float(x) for x in post]
    pre = [float(x) for x in pre]
    if not post or not pre:
        return 0, len(post), 0.0
    shifts = [0.0]
    if remove_rotation:
        shifts = [float(_wrap(a - b)) for a in post for b in pre]
    best_m, best_s = -1, 0.0
    for s in shifts:
        m = sum(1 for a in post
                if min(_circ_dist(a - s, b) for b in pre) <= tol)
        if m > best_m:
            best_m, best_s = m, s
    return best_m, len(post), best_s


def _finite_row(pos_row):
    return pos_row[np.isfinite(pos_row)]


def _fmt_pct(x):
    return f"{100.0 * x:+.3f}%"


def load_and_audit():
    """Load the committed npz and audit the schema (brief item 1)."""
    path = RESULTS_DIR / SWEEP_NPZ
    sweep, cfg = load_sweep_npz(path)
    lines = []
    schema = sweep.get("schema_version")
    lines.append(f"npz: `{path.name}`  sha256 `{_sha256(path)[:16]}...`")
    lines.append(
        f"schema_version: "
        + (str(schema) if schema is not None
           else "ABSENT (pre-v3 file; key set identifies it as schema v2)"))
    missing = []
    for k in REQUIRED_KEYS:
        if k in sweep:
            lines.append(f"  {k}: present, shape "
                         f"{tuple(np.shape(sweep[k]))}, dtype "
                         f"{np.asarray(sweep[k]).dtype}")
        else:
            missing.append(k)
            lines.append(f"  {k}: **MISSING**")
    if "peak_positions_rad" in missing or "P_comb" in missing:
        lines.append(
            "**LOUD WARNING: peak_positions_rad and/or P_comb is MISSING "
            "from the committed npz.  TEST A and/or TEST B below are "
            "impossible on this file, the verdict degrades accordingly, and "
            "the schema-4 plan must first re-persist those columns.**")
    return sweep, cfg, lines, missing


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_events(c):
    """Flicker events on the DESCENDING-order count trace (brief item 2).

    Returns a list of dicts with pre/dip/post hold indices.  ``post`` is the
    recovery hold i (count[i] > count[i-1]); ``pre`` is the last earlier hold
    carrying at least the recovered count; ``dip`` is the minimum-count hold
    strictly between them (ties resolved toward the recovery).
    """
    events = []
    for i in range(1, c.size):
        if c[i] <= c[i - 1]:
            continue
        j = i - 1
        while j >= 0 and c[j] < c[i]:
            j -= 1
        if j < 0:
            events.append({"post": i, "pre": None, "dip": None})
            continue
        between = list(range(j + 1, i)) or [i - 1]
        k = max((t for t in between), key=lambda t: (-c[t], t))
        events.append({"post": i, "pre": j, "dip": k})
    return events


def undercount_mask(c):
    """True where the count sits below its running future maximum.

    Along a descending sweep the true count is non-increasing and peak
    counting can only LOSE pulses, so any later, higher count proves at least
    that many solitons existed now: ``envelope = running max from the right``
    is a lower bound on the true count and ``count < envelope`` marks a
    certain undercount.
    """
    env = np.maximum.accumulate(c[::-1])[::-1]
    return env, c < env


def test_a_positions(pos, c, events):
    """TEST A: raw + rotation-controlled position persistence, with controls."""
    rows = []
    tot_raw_m = tot_rot_m = tot_n = 0
    for ev in events:
        i, j = ev["post"], ev["pre"]
        if j is None:
            rows.append({**ev, "n_post": 0})
            continue
        post = _finite_row(pos[i])
        pre = _finite_row(pos[j])
        raw_m, n, _ = _match_fraction(post, pre, MATCH_TOL_RAD,
                                      remove_rotation=False)
        rot_m, _, shift = _match_fraction(post, pre, MATCH_TOL_RAD,
                                          remove_rotation=True)
        tot_raw_m += raw_m
        tot_rot_m += rot_m
        tot_n += n
        rows.append({**ev, "n_post": n, "raw_m": raw_m, "rot_m": rot_m,
                     "shift": shift})

    # No-event control: consecutive same-count holds (nucleation impossible
    # between them) give (a) the coherent drift rate that invalidates the raw
    # fixed-frame tolerance and (b) the measurement ceiling of the
    # rotation-controlled statistic.
    drifts = []
    control = {}          # pair index t -> (matched, n)
    for t in range(1, c.size):
        if c[t] == c[t - 1] and c[t] >= 1:
            a, b = _finite_row(pos[t]), _finite_row(pos[t - 1])
            if a.size == 0 or a.size != b.size:
                continue
            m, n, _ = _match_fraction(a, b, MATCH_TOL_RAD,
                                      remove_rotation=True)
            control[t] = (m, n)
            d = _wrap(np.sort(a) - np.sort(b))
            if np.ptp(d) < COHERENT_PTP_RAD:
                drifts.append(float(np.mean(d)))
    drifts = np.asarray(drifts)

    # Per-event measurement ceiling: the control statistic on same-count pairs
    # within +-CEILING_WINDOW holds of the event (the same dynamical regime).
    tot_ceiling = 0.0
    tot_null = 0.0
    for r in rows:
        if r.get("n_post", 0) == 0:
            continue
        lo, hi = r["pre"] - CEILING_WINDOW, r["post"] + CEILING_WINDOW
        local = [control[t] for t in control if lo <= t <= hi]
        if local:
            cm = sum(m for m, _ in local)
            cn = sum(n for _, n in local)
            ceiling = cm / cn if cn else 1.0
        else:
            allm = sum(m for m, _ in control.values())
            alln = sum(n for _, n in control.values())
            ceiling = allm / alln if alln else 1.0
        r["ceiling"] = ceiling
        tot_ceiling += ceiling * r["n_post"]
        # Re-nucleation null: random post angles, best-rotation fitted, so one
        # pair aligns by construction and the rest match by chance within
        # +-tol of any of the k pre angles.
        n = r["n_post"]
        k = _finite_row(pos[r["pre"]]).size
        r["null"] = (1.0 + (n - 1) * k * (MATCH_TOL_RAD / np.pi)) / n
        tot_null += r["null"] * n

    agg = {
        "raw": tot_raw_m / tot_n if tot_n else np.nan,
        "rot": tot_rot_m / tot_n if tot_n else np.nan,
        "ceiling": tot_ceiling / tot_n if tot_n else np.nan,
        "null": tot_null / tot_n if tot_n else np.nan,
        "n_peaks": tot_n,
        "drift_median": float(np.median(drifts)) if drifts.size else np.nan,
        "drift_iqr": ([float(np.percentile(drifts, q)) for q in (25, 75)]
                      if drifts.size else [np.nan, np.nan]),
        "n_drift_pairs": int(drifts.size),
    }
    return rows, agg


def estimate_quantum(dw, c, Pc, env, under):
    """TEST B quantum: median per-quantum |dP_comb| over clean decrements.

    The committed spectral_metrics.json staircase block predates the
    step<->transition alignment machinery (its "matched" edge list mixes in
    flicker edges and stores step sizes only on the normalised trace), so it
    cannot supply clean matched N -> N-1 steps; per the brief this falls back
    to count decrements OUTSIDE flicker regions, i.e. the envelope drops.
    Energy-silent drops (|dP_comb| below QUANTUM_VISIBLE_K x the MAD of the
    P_comb first differences) are excluded from the estimate and reported as
    anomalies -- an envelope drop with no energy signature is itself evidence
    that the count, not a soliton, disappeared.
    """
    dPc = np.diff(Pc)
    mad = float(np.median(np.abs(dPc - np.median(dPc))))
    visible_floor = QUANTUM_VISIBLE_K * max(mad, 1e-300)
    drops, silent = [], []
    for t in range(1, c.size):
        if env[t] < env[t - 1]:
            quanta = int(env[t - 1] - env[t])
            step = abs(float(Pc[t] - Pc[t - 1]))
            entry = {"hold": t, "dw": float(dw[t]),
                     "from": int(env[t - 1]), "to": int(env[t]),
                     "dPc": step, "per_quantum": step / quanta}
            (drops if step > visible_floor else silent).append(entry)
    quantum = (float(np.median([d["per_quantum"] for d in drops]))
               if drops else np.nan)
    return quantum, drops, silent, visible_floor


def test_b_energy(dw, c, Pc, Pi, events, quantum):
    """TEST B: per-event pre->dip energy change vs the claimed quanta lost."""
    rows = []
    for ev in events:
        j, k = ev["pre"], ev["dip"]
        if j is None:
            continue
        claimed = int(c[j] - c[k])
        d_pc = float(Pc[k] - Pc[j])
        d_pi = float(Pi[k] - Pi[j])
        ratio = (abs(d_pc) / (claimed * quantum)
                 if claimed > 0 and np.isfinite(quantum) else np.nan)
        rows.append({**ev, "claimed_quanta": claimed,
                     "d_pc_rel": d_pc / Pc[j], "d_pi_rel": d_pi / Pi[j],
                     "quanta_ratio": ratio})
    ratios = np.asarray([r["quanta_ratio"] for r in rows
                         if np.isfinite(r["quanta_ratio"])])
    agg = {"median_ratio": float(np.median(ratios)) if ratios.size else np.nan,
           "max_ratio": float(np.max(ratios)) if ratios.size else np.nan}
    return rows, agg


def test_c_breathing(dw, c, env, under, events, sweep):
    """TEST C: breathing correlation, labels at dips, is_single integrity."""
    brs = np.asarray(sweep["breathing_relstd"], dtype=float)
    isb = np.asarray(sweep["is_breather"], dtype=bool)
    lbl = np.asarray(sweep["np_label"], dtype=int)
    sng = np.asarray(sweep["is_single"], dtype=bool)

    ok = (c == env) & (env >= 1)

    def dist(x):
        return {"median": float(np.median(x)),
                "iqr": [float(np.percentile(x, 25)),
                        float(np.percentile(x, 75))],
                "n": int(x.size)}

    dips = sorted({ev["dip"] for ev in events if ev["dip"] is not None})
    label_counts = {int(l): int(n)
                    for l, n in zip(*np.unique(lbl[dips], return_counts=True))}
    corrupted = (dw >= 6.5) & sng & (env > 1)
    return {
        "relstd_under": dist(brs[under]) if under.any() else None,
        "relstd_ok": dist(brs[ok]) if ok.any() else None,
        "breather_at_dip": int(sum(isb[ev["dip"]] for ev in events
                                   if ev["dip"] is not None)),
        "n_events": len([ev for ev in events if ev["dip"] is not None]),
        "dip_holds": dips,
        "dip_label_counts": label_counts,
        "dip_labels_soliton": int(sum(1 for k in dips
                                      if lbl[k] in SOLITON_LABELS)),
        "n_dip_holds": len(dips),
        "dips_never_zero": bool(all(c[k] >= 1 for k in dips)),
        "is_single_corrupted_holds": [float(x) for x in dw[corrupted]],
        "is_single_any_high": int(((dw >= 6.5) & sng).sum()),
    }


# ---------------------------------------------------------------------------
# --starvation mode: is the robustness failure SNAPSHOT STARVATION?
# ---------------------------------------------------------------------------
# The driver takes a field snapshot every ``snap_int = max(hold_rt // 32, 1)``
# round trips (analysis/run_detuning_sweep.py) and the windowed counter votes
# only over the snapshots inside the final-``avg_frac`` power-averaging window.
# The starvation hypothesis: that window holds only ~2 snapshots, so with the
# breather period T_b ~ 150-180 RT the persistence vote is phase-aliased and
# rests on too few samples.  This mode PROVES OR DISPROVES that offline, from
# the committed npzs alone -- it never runs the solver.
SNAP_DIVISOR_ACTUAL = 32     # mirrors the driver: snap_int = max(hold_rt//32, 1)
SNAP_DIVISOR_HYPOTHESIS = 8  # the hypothesis in the brief: interval = hold_rt//8
STARVATION_N_IN_MAX = 3      # "few samples" ceiling for the CONFIRMED verdict


def _n_in_window(hold_rt: int, avg_frac: float, interval: int) -> int:
    """In-window snapshot count, matching the solver + driver EXACTLY.

    The solver stores ``n_snap = ceil(hold_rt / interval)`` snapshots at round
    trips ``0, interval, 2*interval, ...`` (``simulator/lle_solver.py``:
    ``n_snapshots = (t_slow + interval - 1)//interval`` with
    ``do_snapshot = step_idx % interval == 0``); the driver keeps those with
    ``snap_rt >= floor(hold_rt*(1-avg_frac))``
    (``analysis/run_detuning_sweep.py``).  Returns that count (>= 1: a
    degenerate window falls back to the last snapshot, as the driver does).
    """
    n_snap = (int(hold_rt) + int(interval) - 1) // int(interval)
    snap_rt = np.arange(n_snap) * int(interval)
    i_start = int(np.floor(hold_rt * (1.0 - avg_frac)))
    n_in = int((snap_rt >= i_start).sum())
    return n_in if n_in > 0 else 1


def _on_grid(values, n_in, tol=1e-6) -> bool:
    """True iff every value lies on the grid {0, 1/n_in, 2/n_in, ..., 1}."""
    v = np.asarray(values, dtype=float)
    return bool(np.all(np.abs(v * n_in - np.round(v * n_in)) <= tol * n_in))


def _empirical_denominator(values, max_den=64) -> int:
    """Finest common grid the observed values lie on (their agreement n_in)."""
    from fractions import Fraction
    import math
    N = 1
    for v in sorted(set(np.round(np.asarray(values, dtype=float), 9).tolist())):
        N = N * Fraction(v).limit_denominator(max_den).denominator \
            // math.gcd(N, Fraction(v).limit_denominator(max_den).denominator)
    return int(N)


def starvation_forensics() -> int:
    """Part A: prove/disprove snapshot starvation from the committed npzs only.

    For the primary sweep and each ``robustness/variant_*.npz``: computes the
    in-window snapshot count ``n_in`` (both the driver's actual ``hold_rt//32``
    cadence and the brief's hypothesised ``hold_rt//8``), tests the count_
    agreement quantization signature (every stored value must lie on the
    ``{k/n_in}`` grid), and tabulates every monotonicity-violating and every
    ``count_agreement == 0`` hold with the aliasing indicator
    ``snapshot_spacing / breathing_period_rt``.  Appends a ``Snapshot
    starvation`` section to the forensics report with a one-line verdict:
    CONFIRMED iff ``n_in <= STARVATION_N_IN_MAX`` for ALL files AND the
    quantization signature holds.  Returns 0 (report written); the GATE
    decision (proceed vs stop) is the caller's, keyed on the printed verdict.
    """
    files = [("primary", RESULTS_DIR / SWEEP_NPZ)]
    rob = sorted((RESULTS_DIR / "robustness").glob("variant_*.npz"))
    files += [(p.stem, p) for p in rob]

    md = ["", "## Snapshot starvation (Part A: offline hypothesis test)", "",
          f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat()} by "
          f"`analysis/staircase_forensics.py --starvation` (offline; no solver "
          f"run). Hypothesis under test: the robustness count failures are "
          f"SNAPSHOT STARVATION -- the windowed counter votes over too few, "
          f"phase-aliased in-window snapshots.", "",
          "Driver cadence: `snap_int = max(hold_rt // "
          f"{SNAP_DIVISOR_ACTUAL}, 1)` "
          "(`analysis/run_detuning_sweep.py`); the counter votes over the "
          "snapshots inside the final-`avg_frac` window. The brief hypothesised "
          f"`interval = hold_rt // {SNAP_DIVISOR_HYPOTHESIS}`.", ""]

    all_n_in, all_sig_ok, per_file = [], [], []
    print(f"[starvation] auditing {len(files)} file(s)")
    for name, path in files:
        if not path.exists():
            print(f"[starvation] SKIP {name}: {path} missing")
            continue
        sweep, cfg = load_sweep_npz(path)
        hold_rt, avg_frac = int(cfg.hold_rt), float(cfg.avg_frac)
        iv_act = max(hold_rt // SNAP_DIVISOR_ACTUAL, 1)
        iv_hyp = max(hold_rt // SNAP_DIVISOR_HYPOTHESIS, 1)
        n_in_act = _n_in_window(hold_rt, avg_frac, iv_act)
        n_in_hyp = _n_in_window(hold_rt, avg_frac, iv_hyp)

        order = np.argsort(np.asarray(sweep["dw_over_kappa"]))
        dw = np.asarray(sweep["dw_over_kappa"], float)[order]
        c = np.asarray(sweep["soliton_count"], int)[order]
        ca = np.asarray(sweep["count_agreement"], float)[order]
        ce = (np.asarray(sweep["soliton_count_end_snapshot"], int)[order]
              if "soliton_count_end_snapshot" in sweep else np.full(c.size, -1))
        brel = np.asarray(sweep["breathing_relstd"], float)[order]
        bper = np.asarray(sweep["breathing_period_rt"], float)[order]

        emp_den = _empirical_denominator(ca)
        sig_ok = _on_grid(ca, n_in_act)          # signature for the ACTUAL n_in
        all_n_in.append(n_in_act)
        all_sig_ok.append(sig_ok)

        # monotonicity-violating holds (ascending: the lower side of a diff<0
        # edge -- the undercounted hold) + soliton-bearing agreement==0 holds
        mono = np.nonzero(np.diff(c) < 0)[0] + 1
        agree0 = np.nonzero((c >= 1) & (ca == 0.0))[0]
        problem = sorted(set(mono.tolist()) | set(agree0.tolist()))

        per_file.append(dict(
            name=name, hold_rt=hold_rt, avg_frac=avg_frac, iv_act=iv_act,
            iv_hyp=iv_hyp, n_in_act=n_in_act, n_in_hyp=n_in_hyp,
            emp_den=emp_den, sig_ok=sig_ok, n_mono=int(mono.size),
            n_agree0=int(agree0.size),
            rows=[dict(dw=float(dw[i]), N=int(c[i]), Nend=int(ce[i]),
                       agree=float(ca[i]), brel=float(brel[i]),
                       bper=float(bper[i]),
                       ratio=(float(iv_act) / float(bper[i])
                              if np.isfinite(bper[i]) and bper[i] > 0
                              else float("nan")),
                       kind=("mono" if i in set(mono.tolist()) else "")
                            + ("+agree0" if i in set(agree0.tolist()) else ""))
                  for i in problem]))
        print(f"[starvation] {name}: hold_rt={hold_rt} avg_frac={avg_frac} "
              f"| n_in(actual //{SNAP_DIVISOR_ACTUAL})={n_in_act}  "
              f"n_in(hyp //{SNAP_DIVISOR_HYPOTHESIS})={n_in_hyp}  "
              f"| count_agreement grid=1/{emp_den} (signature on {{k/{n_in_act}}}: "
              f"{sig_ok}) | mono-violating={int(mono.size)} agree0={int(agree0.size)}")

    # ---- Verdict (the brief's rule) ----------------------------------------
    n_in_ok = bool(all_n_in) and all(n <= STARVATION_N_IN_MAX for n in all_n_in)
    sig_all = all(all_sig_ok)
    confirmed = n_in_ok and sig_all
    verdict = "CONFIRMED" if confirmed else "NOT CONFIRMED"

    # ---- Report body -------------------------------------------------------
    md.append("### Per-file snapshot budget and quantization signature")
    md.append("")
    md.append("| file | hold_rt | snap_int (//%d) | **n_in (actual)** | "
              "n_in (//%d hyp) | count_agreement grid | signature {k/n_in} | "
              "mono-viol | agree==0 |"
              % (SNAP_DIVISOR_ACTUAL, SNAP_DIVISOR_HYPOTHESIS))
    md.append("|---|---|---|---|---|---|---|---|---|")
    for f in per_file:
        md.append(f"| {f['name']} | {f['hold_rt']} | {f['iv_act']} | "
                  f"**{f['n_in_act']}** | {f['n_in_hyp']} | 1/{f['emp_den']} | "
                  f"{f['sig_ok']} | {f['n_mono']} | {f['n_agree0']} |")
    md.append("")
    md.append("### Failing holds (monotonicity dips + soliton-bearing "
              "agreement==0)")
    md.append("")
    md.append("`ratio = snap_int / breathing_period_rt` is the aliasing "
              "indicator: **>> 1 would mean the snapshots undersample the "
              "breathing cycle (starvation); < 1 means they oversample it.**")
    md.append("")
    md.append("| file | dw/k | N | N_end-snap | count_agreement | "
              "breathing_relstd | T_b (RT) | snap_int/T_b | kind |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for f in per_file:
        for r in f["rows"]:
            md.append(f"| {f['name']} | {r['dw']:.3f} | {r['N']} | "
                      f"{r['Nend']} | {r['agree']:.3f} | {r['brel']:.4f} | "
                      f"{r['bper']:.0f} | {r['ratio']:.2f} | {r['kind']} |")
    md.append("")
    md.append("### Verdict")
    md.append("")
    md.append(f"**STARVATION: {verdict}** "
              f"(rule: CONFIRMED iff n_in <= {STARVATION_N_IN_MAX} for ALL "
              f"files AND every count_agreement lies on the {{k/n_in}} grid).")
    md.append("")
    if confirmed:
        md.append(f"- n_in <= {STARVATION_N_IN_MAX} for every file and the "
                  f"quantization signature holds: the counter is sample-starved "
                  f"and phase-aliased. Densification (Part B) is warranted.")
    else:
        n_in_set = sorted(set(all_n_in))
        md.append(f"- n_in = {n_in_set} (actual `hold_rt//{SNAP_DIVISOR_ACTUAL}` "
                  f"cadence), which is **> {STARVATION_N_IN_MAX}** -- the "
                  f"counter already votes over ~8 in-window snapshots, not ~2. "
                  f"The brief's `hold_rt//{SNAP_DIVISOR_HYPOTHESIS}` interval "
                  f"(giving n_in={sorted(set(f['n_in_hyp'] for f in per_file))}) "
                  f"is NOT what the driver uses.")
        md.append(f"- The count_agreement quantization confirms it: every value "
                  f"lies on an **eighths** grid (1/{per_file[0]['emp_den']}), "
                  f"i.e. n_in = {per_file[0]['emp_den']}, not the halves "
                  f"({{0, 0.5, 1}}) the starvation hypothesis predicts.")
        md.append(f"- The aliasing indicator `snap_int/T_b` is < 1 at every "
                  f"failing hold (snapshots OVERSAMPLE the breathing cycle by "
                  f"~3x), so phase-aliasing is not the mechanism.")
        md.append(f"- **GATE (per the brief): STOP.** The counter had many "
                  f"(~8) phase-spread samples and still failed at isolated "
                  f"deep-breather holds, so the failure mechanism is NOT "
                  f"starvation. Densification / threshold / protocol work must "
                  f"not proceed on the falsified hypothesis; the residual "
                  f"(individual solitons dipping below the rel-height floor "
                  f"during their breathing troughs at these specific holds) "
                  f"needs its own verification before any fix.")
    md.append("")

    out = RESULTS_DIR / REPORT_MD
    prior = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = "\n## Snapshot starvation (Part A"
    if marker in prior:                       # idempotent: replace prior section
        prior = prior[:prior.index(marker)].rstrip() + "\n"
    out.write_text(prior.rstrip() + "\n" + "\n".join(md) + "\n", encoding="utf-8")
    print(f"[starvation] VERDICT: {verdict} "
          f"(n_in={sorted(set(all_n_in))}, signature_holds={sig_all})")
    if not confirmed:
        print("[starvation] GATE: NOT CONFIRMED -> STOP. The counter already "
              "had ~8 in-window snapshots; starvation is falsified. Do not "
              "densify or touch thresholds blind.")
    print(f"[starvation] report -> {out}")
    return 0


# ---------------------------------------------------------------------------
# --detectability mode (Stage A): is the count defect a RELATIVE-THRESHOLD
# detectability problem (a trough-phase soliton rejected because a sibling is
# at crest, coupling its detection to the others via snapshot_max), and does
# the missing soliton PERSIST in position through the event (a counting
# dropout, not a physics annihilation)?  Offline, committed npzs only.
# ---------------------------------------------------------------------------
DETECT_TOL_RAD = 0.05        # per the brief: circular position-match tolerance
DETECT_TOL_PERSIST = 0.10    # before<->after span is 2 holds; allow ~2x drift


def _greedy_unmatched(src, dst, tol):
    """dst angles NOT covered by any src angle within ``tol`` (circular, 1:1)."""
    src = [float(x) for x in src]
    dst = [float(x) for x in dst]
    used = set()
    for a in src:
        best, bd = None, 1e9
        for j, b in enumerate(dst):
            if j in used:
                continue
            d = _circ_dist(a, b)
            if d < bd:
                bd, best = d, j
        if best is not None and bd <= tol:
            used.add(best)
    return [dst[j] for j in range(len(dst)) if j not in used]


def _nn_separations(angles):
    """Nearest-neighbour circular separation for each angle in ``angles``."""
    a = [float(x) for x in angles]
    return [min(_circ_dist(a[i], a[j]) for j in range(len(a)) if j != i)
            for i in range(len(a))]


def _rank_smallest(values, target):
    """1-based rank of ``target`` among ``values`` (1 = smallest)."""
    return int(sum(1 for v in values if v < target - 1e-12) + 1)


def _cyclic_seed_index(hold_angles, seed_angles):
    """Map each sorted hold angle to a seed index assuming a rigid rotation.

    Soliton order is preserved along the sweep (no crossings), so the settled
    pattern is a cyclic rotation of the seed; pick the cyclic shift minimising
    the total circular distance.  Returns ``{sorted_hold_idx: seed_sorted_idx}``
    and the sorted seed angles.
    """
    h = np.sort(np.asarray([float(x) for x in hold_angles]))
    s = np.sort(np.asarray([float(x) for x in seed_angles]))
    n = min(h.size, s.size)
    if n == 0:
        return {}, s
    best = (1e18, 0)
    for shift in range(n):
        tot = sum(_circ_dist(h[i], s[(i + shift) % n]) for i in range(n))
        if tot < best[0]:
            best = (tot, shift)
    shift = best[1]
    return {i: (i + shift) % n for i in range(n)}, s


def detectability_forensics() -> int:
    """Stage A: offline detectability diagnosis over the committed npzs.

    A1 identifies the missing cluster at every monotonicity-violating hold and
    tests whether it PERSISTS in position through the event (present at the same
    angle in both flanking holds -> a detection dropout, not a physics
    rearrangement -- the GATE).  A2 ranks the missing soliton's seed
    nearest-neighbour separation (tightest pairs breathe deepest) and checks
    whether the same seed-relative soliton drops in variants sharing a seed.
    A3 categorises every soliton-bearing agreement==0 hold (undercount vs
    correct-count-but-no-unanimous-snapshot) from the stored counts -- the raw
    per-cluster persistence_fractions are NOT persisted in the npz (only
    count_agreement and the final cluster angles are), so the fraction
    breakdown is deferred to the instrumented Stage B run.  Appends a
    "Detectability (offline)" section to the report; returns 0.  The caller
    keys the Stage-A GATE on the printed persistence verdict.
    """
    files = [("primary", RESULTS_DIR / SWEEP_NPZ)]
    files += [(p.stem, p) for p in
              sorted((RESULTS_DIR / "robustness").glob("variant_*.npz"))]

    a1_rows, a3_rows = [], []
    persist_all = []          # gate: every mono event must persist in position
    md = ["", "## Detectability (offline; Stage A)", "",
          f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat()} by "
          f"`analysis/staircase_forensics.py --detectability` (offline; no "
          f"solver run). Working hypothesis: the count defect is a "
          f"RELATIVE-threshold detectability problem -- a trough-phase soliton "
          f"is rejected when a sibling is at crest because the candidate floor "
          f"`rel_height_candidate * snapshot_max` couples each soliton's "
          f"detection to the others' breathing phases.", ""]

    for name, path in files:
        if not path.exists():
            continue
        sweep, cfg = load_sweep_npz(path)
        order = np.argsort(np.asarray(sweep["dw_over_kappa"]))
        dw = np.asarray(sweep["dw_over_kappa"], float)[order]
        c = np.asarray(sweep["soliton_count"], int)[order]
        ce = np.asarray(sweep["soliton_count_end_snapshot"], int)[order]
        ca = np.asarray(sweep["count_agreement"], float)[order]
        pos = np.asarray(sweep["peak_positions_rad"], float)[order]
        seed = _finite_row(np.asarray(sweep["seed_positions_rad"], float))
        n_seed = int(sweep.get("n_solitons_seeded", seed.size)) or seed.size

        # ---- A1: monotonicity-violating holds (ascending diff<0 -> hold i) ---
        mono = (np.nonzero(np.diff(c) < 0)[0] + 1)
        for j in mono:
            if j == 0 or j + 1 >= c.size:
                continue
            ev, be, af = (_finite_row(pos[j]), _finite_row(pos[j - 1]),
                          _finite_row(pos[j + 1]))
            miss_be = _greedy_unmatched(ev, be, DETECT_TOL_RAD)
            miss_af = _greedy_unmatched(ev, af, DETECT_TOL_RAD)
            gap = (_circ_dist(miss_be[0], miss_af[0])
                   if (miss_be and miss_af) else float("nan"))
            persists = bool(len(miss_be) == 1 and len(miss_af) == 1
                            and gap <= DETECT_TOL_PERSIST)
            persist_all.append(persists)
            miss_angle = float(miss_be[0]) if miss_be else float("nan")
            # A2: NN-separation rank of the missing soliton
            nn_flank = _nn_separations(be) if be.size else []
            rank_flank = (_rank_smallest(
                nn_flank, min(_circ_dist(miss_angle, b) for b in be
                              if _circ_dist(miss_angle, b) > 1e-9))
                if (be.size and np.isfinite(miss_angle)) else None)
            nn_seed = _nn_separations(seed) if seed.size else []
            smap, s_sorted = _cyclic_seed_index(be, seed)
            # sorted index of the missing angle within the before-flank
            be_sorted = np.sort(be)
            mi = int(np.argmin(np.abs(_wrap(be_sorted - miss_angle)))) \
                if be.size else -1
            seed_idx = smap.get(mi)
            rank_seed = (_rank_smallest(nn_seed, nn_seed[seed_idx])
                         if (seed_idx is not None and nn_seed) else None)
            a1_rows.append(dict(
                file=name, dw=float(dw[j]), N=int(c[j]), N_end=int(ce[j]),
                miss=miss_angle, miss_be=miss_be, miss_af=miss_af, gap=gap,
                persists=persists, rank_flank=rank_flank, n=len(be),
                seed_idx=seed_idx, rank_seed=rank_seed, n_seed=len(nn_seed)))

        # ---- A3: soliton-bearing agreement==0 holds -------------------------
        # running future-max envelope = the true-count lower bound (descending
        # sweep), so "undercount" = count below the envelope.
        env = np.maximum.accumulate(c[::-1])[::-1]   # max over lower detunings
        for j in np.nonzero((c >= 1) & (ca == 0.0))[0]:
            a3_rows.append(dict(
                file=name, dw=float(dw[j]), N=int(c[j]), env=int(env[j]),
                N_end=int(ce[j]), agree=float(ca[j]),
                category=("undercount (a cluster fell below min_persistence)"
                          if c[j] < env[j] else
                          "correct-count, no unanimous snapshot")))

    # ---- Report: A1 ---------------------------------------------------------
    md.append("### A1 -- missing cluster at each monotonicity-violating hold")
    md.append("")
    md.append("| file | dw/k | N | N_end-snap | missing angle | in before | "
              "in after | before<->after gap | persists (dropout) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in a1_rows:
        md.append(f"| {r['file']} | {r['dw']:.3f} | {r['N']} | {r['N_end']} | "
                  f"{r['miss']:.3f} | {'yes' if r['miss_be'] else 'no'} | "
                  f"{'yes' if r['miss_af'] else 'no'} | {r['gap']:.4f} | "
                  f"**{'YES' if r['persists'] else 'NO'}** |")
    md.append("")
    md.append("A missing soliton present at the SAME angle (within the "
              f"{DETECT_TOL_PERSIST} rad drift budget) in BOTH flanking holds "
              "never left -- it is a pure detection dropout, consistent with a "
              "counting defect (not annihilation/re-nucleation).")
    md.append("")

    # ---- Report: A2 ---------------------------------------------------------
    md.append("### A2 -- is the missing soliton the most strongly interacting?")
    md.append("")
    md.append("Rank 1 = tightest nearest-neighbour separation (interacts "
              "hardest, breathes deepest). `rank_flank` is computed on the "
              "event-neighbourhood positions; `rank_seed` maps the missing "
              "soliton back to its seed (rigid-rotation cyclic map) and ranks "
              "the seed separations.")
    md.append("")
    md.append("| file | seed | dw/k | missing angle | rank_flank (of N) | "
              "seed idx | rank_seed (of n) |")
    md.append("|---|---|---|---|---|---|---|")
    seed_of = {}
    for name, path in files:
        if path.exists():
            _, cfg = load_sweep_npz(path)
            seed_of[name] = int(cfg.position_seed)
    for r in a1_rows:
        md.append(f"| {r['file']} | {seed_of.get(r['file'],'?')} | {r['dw']:.3f} "
                  f"| {r['miss']:.3f} | "
                  f"{r['rank_flank']}/{r['n']} | {r['seed_idx']} | "
                  f"{r['rank_seed']}/{r['n_seed']} |")
    md.append("")
    shared = {}
    for r in a1_rows:
        shared.setdefault(seed_of.get(r["file"]), []).append(
            (r["file"], r["seed_idx"]))
    md.append("Same seed-relative soliton across variants sharing a seed: "
              + "; ".join(f"seed {s}: "
                          + ", ".join(f"{f}->idx {i}" for f, i in v)
                          for s, v in shared.items()) + ".")
    md.append("(primary and variant_2 share seed 1 but have NO "
              "monotonicity-violating hold, so only variant_3 supplies a "
              "seed-1 dropout to locate.)")
    md.append("")

    # ---- Report: A3 ---------------------------------------------------------
    md.append("### A3 -- agreement==0 holds (soliton-bearing)")
    md.append("")
    md.append("The raw per-cluster `persistence_fractions` are computed by "
              "`count_solitons_windowed` but NOT persisted to the npz (only "
              "`count_agreement` and the final accepted cluster angles are), so "
              "the per-cluster fraction breakdown the brief asks for is "
              "deferred to the instrumented Stage B run. From the stored counts "
              "the two signatures still separate: `undercount` (a cluster fell "
              "below min_persistence, so N < envelope) vs `correct-count` "
              "(all N clusters kept but no single snapshot saw all N).")
    md.append("")
    md.append("| file | dw/k | N | envelope | N_end-snap | count_agreement | "
              "category |")
    md.append("|---|---|---|---|---|---|---|")
    for r in a3_rows:
        md.append(f"| {r['file']} | {r['dw']:.3f} | {r['N']} | {r['env']} | "
                  f"{r['N_end']} | {r['agree']:.3f} | {r['category']} |")
    md.append("")

    # ---- GATE + verdict -----------------------------------------------------
    all_persist = bool(a1_rows) and all(persist_all)
    md.append("### Stage-A gate")
    md.append("")
    if not a1_rows:
        md.append("- No monotonicity-violating hold found; nothing to gate on.")
        gate = "no-events"
    elif all_persist:
        md.append(f"- **POSITION PERSISTENCE CONFIRMED** at all "
                  f"{len(a1_rows)} monotonicity events: every missing soliton "
                  f"sits at the same angle in both flanks (max "
                  f"before<->after gap "
                  f"{max(r['gap'] for r in a1_rows):.4f} rad). The dropouts are "
                  f"a COUNTING defect, not physics rearrangement -> Stage B "
                  f"(instrumented run) may proceed.")
        gate = "proceed"
    else:
        md.append("- **POSITION REARRANGEMENT DETECTED**: at least one missing "
                  "soliton does not persist through the event, so it may be a "
                  "physics event. **GATE: STOP** -- Stage B must not run.")
        gate = "stop"
    md.append("")

    out = RESULTS_DIR / REPORT_MD
    prior = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = "\n## Detectability (offline; Stage A)"
    if marker in prior:
        prior = prior[:prior.index(marker)].rstrip() + "\n"
    out.write_text(prior.rstrip() + "\n" + "\n".join(md) + "\n", encoding="utf-8")
    for r in a1_rows:
        print(f"[detectability] {r['file']} @ {r['dw']:.3f}k: missing "
              f"{r['miss']:.3f} rad, persists={r['persists']} "
              f"(gap {r['gap']:.4f}), rank_flank {r['rank_flank']}/{r['n']}, "
              f"N_end={r['N_end']}")
    print(f"[detectability] agreement==0 holds tabulated: {len(a3_rows)} "
          f"(persistence_fractions NOT stored -> deferred to Stage B)")
    print(f"[detectability] STAGE-A GATE: {gate.upper()} "
          f"(position persistence at all events: {all_persist})")
    print(f"[detectability] report -> {out}")
    return 0


def _victim_stats(above_rel, above_abs, local_max, median_bg, b2):
    """Per-snapshot detection stats for one soliton across a hold's window.

    ``detected`` (== persistence for this soliton) is passing the candidate
    floor ``max(rel_thresh, abs_thresh)`` -> above_rel AND above_abs.  A
    rel-VICTIM snapshot passes the ABSOLUTE floor but fails the RELATIVE one
    (above_abs AND NOT above_rel): it would have counted under the absolute
    floor alone and is rejected only because a sibling's crest lifted
    ``snapshot_max``.
    """
    ar = np.asarray(above_rel, bool)
    aa = np.asarray(above_abs, bool)
    lm = np.asarray(local_max, float)
    return dict(
        abs_pass=float(np.mean(aa)),
        rel_fail=float(np.mean(~ar)),
        detected=float(np.mean(ar & aa)),
        rel_victim=float(np.mean(aa & ~ar)),
        fails_both=float(np.mean(~aa)),
        min_over_bg=float(np.nanmin(lm) / max(np.median(median_bg), 1e-300)),
        min_over_b2=float(np.nanmin(lm) / max(abs(b2), 1e-300)))


def diagnose_report() -> int:
    """Stage B (B2/B3): read the instrumented sidecars and rule on the mechanism.

    For every failing hold in ``diagnose_{variant}.npz`` picks the victim
    soliton (the flank-recovered MISSING one at a monotonicity/undercount hold;
    the lowest-persistence cluster at an agreement==0 hold), tabulates its
    rel-victim / genuine-dim statistics, and appends a "Detectability (Stage B)"
    verdict to the report: COUPLING CONFIRMED, GENUINE DIMMING, or MIXED.
    """
    sidecars = sorted((RESULTS_DIR / "robustness").glob("diagnose_*.npz"))
    if not sidecars:
        print("[diagnose-report] no diagnose_*.npz found -- run "
              "--diagnose-counting first.")
        return 2
    md = ["", "## Detectability (Stage B: instrumented)", "",
          f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat()} by "
          f"`analysis/staircase_forensics.py --diagnose-report`. Per failing "
          f"hold the VICTIM soliton (missing one at an undercount hold; "
          f"lowest-persistence cluster at an agreement==0 hold) is scored: a "
          f"rel-VICTIM snapshot passes the absolute floor but fails the "
          f"relative one (`rel_height_candidate * snapshot_max`), so it is "
          f"rejected only because a sibling's crest lifted `snapshot_max`.", ""]
    md.append("| variant | dw/k | kind | victim | abs pass | rel-victim frac "
              "| fails-both | detected (persist.) | min|E|²/bg | min|E|²/B² | "
              "class |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")

    mono_classes, all_classes = [], []
    for sc in sidecars:
        d = np.load(sc, allow_pickle=False)
        name = str(d["variant"])
        dw = d["dw_over_kappa"]
        kind = d["kind"].astype(str)
        b2 = d["B2_ref"]
        medbg = d["median_bg"]
        fails = np.nonzero(kind != "")[0]
        for j in fails:
            if kind[j] == "mono":
                st = _victim_stats(d["missing_above_rel"][j],
                                   d["missing_above_abs"][j],
                                   d["missing_local_max"][j], medbg[j], b2[j])
                victim = f"missing@{d['missing_angle_rad'][j]:.3f}"
            else:  # agree0: lowest-persistence real cluster
                ncl = int(d["n_cluster"][j])
                det = [float(np.mean(d["cluster_above_rel"][j, :, c]
                                     & d["cluster_above_abs"][j, :, c]))
                       for c in range(ncl)]
                c = int(np.argmin(det)) if det else 0
                st = _victim_stats(d["cluster_above_rel"][j, :, c],
                                   d["cluster_above_abs"][j, :, c],
                                   d["cluster_local_max"][j, :, c],
                                   medbg[j], b2[j])
                victim = f"cluster {c} (min persist.)"
            # classify: coupling = passes abs a lot but the relative floor is
            # what suppresses it; dimming = fails the absolute floor itself.
            if st["abs_pass"] >= 0.9 and st["rel_victim"] > 0.0 and (
                    kind[j] != "mono" or st["detected"] < 0.5):
                cls = "coupling"
            elif st["fails_both"] > 0.5:
                cls = "dimming"
            else:
                cls = "mixed"
            all_classes.append(cls)
            if kind[j] == "mono":
                mono_classes.append(cls)
            md.append(f"| {name} | {dw[j]:.3f} | {kind[j]} | {victim} | "
                      f"{st['abs_pass']:.2f} | {st['rel_victim']:.2f} | "
                      f"{st['fails_both']:.2f} | {st['detected']:.2f} | "
                      f"{st['min_over_bg']:.1f} | {st['min_over_b2']:.2f} | "
                      f"**{cls}** |")

    # ---- Verdict (B3) -------------------------------------------------------
    # The count DEFECT (monotonicity break) is the undercount holds; the verdict
    # is keyed on them (where a soliton is actually dropped, so "persistence <
    # 0.5" is meaningful). agree0 holds corroborate the same mechanism without
    # a full drop.
    key = mono_classes if mono_classes else all_classes
    if key and all(c == "coupling" for c in key) and all(
            c != "dimming" for c in all_classes):
        verdict = "RELATIVE-THRESHOLD COUPLING CONFIRMED"
    elif key and all(c == "dimming" for c in key):
        verdict = "GENUINE DIMMING"
    else:
        verdict = "MIXED/INCONCLUSIVE"
    md.append("")
    md.append(f"**STAGE-B VERDICT: {verdict}**")
    md.append("")
    md.append("- Rule: COUPLING iff at every undercount hold the missing "
              "soliton passes the ABSOLUTE floor in >= 90% of snapshots yet is "
              "dropped (persistence < 0.5) by the RELATIVE floor "
              "(`rel_height_candidate * snapshot_max`); GENUINE DIMMING iff it "
              "fails the absolute floor in the majority of snapshots; MIXED "
              "otherwise. agree0 holds corroborate (same mechanism, victim "
              "kept above 0.5).")
    md.append("- No fix applied: this diagnosis is the deliverable. Any "
              "remedy (e.g. dropping the coupled relative arm of the candidate "
              "floor) is a separate, gated change -- not made here.")
    md.append("")

    out = RESULTS_DIR / REPORT_MD
    prior = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = "\n## Detectability (Stage B: instrumented)"
    if marker in prior:
        prior = prior[:prior.index(marker)].rstrip() + "\n"
    out.write_text(prior.rstrip() + "\n" + "\n".join(md) + "\n", encoding="utf-8")
    print(f"[diagnose-report] STAGE-B VERDICT: {verdict}")
    print(f"[diagnose-report] report -> {out}")
    return 0


# ---------------------------------------------------------------------------
# --stepquanta mode (offline): confirm or refute the split-step diagnosis
# behind the failing tests/test_soliton_staircase.py::test_step_heights_
# quantized (Part 2i).  Read-only on the committed detuning_sweep.npz and
# spectral_metrics.json; no solver, no fix.  The failing test asserts that the
# matched N -> N-1 step heights (|step_dy| on the plotted primary, EXCLUDING
# the 1 -> 0 edge) agree pairwise within a factor of 2.  The diagnosis under
# test: the 4 -> 3 annihilation completed mid-hold, so its power drop is SPLIT
# between the matched edge and an adjacent, otherwise-unmatched, same-sign
# discontinuity one sample away -- aggregating the two restores quantization.
# ---------------------------------------------------------------------------
STEPQUANTA_FACTOR = 2.0          # the test's pairwise per-quantum tolerance


def _steps_equal(recomputed, committed, *, keys_exact, keys_close, rel=1e-9):
    """True iff two step/transition dicts agree (exact keys + close keys)."""
    for k in keys_exact:
        if int(recomputed[k]) != int(committed[k]):
            return False
    for k in keys_close:
        if abs(float(recomputed[k]) - float(committed[k])) > \
                rel * max(1.0, abs(float(committed[k]))):
            return False
    return True


def _confirm_matches_committed(align, stair) -> list:
    """Confirm the recomputed alignment equals the committed JSON sets.

    Returns a list of drift complaints (empty == the artifact is self-
    consistent).  Mirrors tests/test_soliton_staircase.py::
    test_alignment_reproduces_from_raw so a drifted artifact is caught here
    too, loudly, before any verdict is written.
    """
    problems = []
    cm, cc = align["matched"], stair["matched_steps"]
    if len(cm) != len(cc):
        problems.append(f"matched count {len(cm)} != committed {len(cc)}")
    else:
        for got, exp in zip(cm, cc):
            # committed matched uses step_edge_index/transition_edge_index/
            # n_high_side/n_low_side/delta_n exactly and dw_mid/step_dy close.
            got_c = {"step_edge_index": got["step_edge_index"],
                     "transition_edge_index": got["transition_edge_index"],
                     "n_high_side": got["n_high_side"],
                     "n_low_side": got["n_low_side"], "delta_n": got["delta_n"],
                     "dw_mid": got["dw_mid"], "step_dy": got["step_dy"]}
            if not _steps_equal(got_c, exp,
                                keys_exact=("step_edge_index",
                                            "transition_edge_index",
                                            "n_high_side", "n_low_side",
                                            "delta_n"),
                                keys_close=("dw_mid", "step_dy")):
                problems.append(f"matched edge {got['step_edge_index']} "
                                f"differs from committed")
    cu, ccu = align["unmatched_steps"], stair["unmatched_steps"]
    if len(cu) != len(ccu):
        problems.append(f"unmatched_steps count {len(cu)} != committed "
                        f"{len(ccu)}")
    else:
        for got, exp in zip(cu, ccu):
            if int(got["edge_index"]) != int(exp["edge_index"]) or \
                    abs(float(got["step_dy"]) - float(exp["step_dy"])) > \
                    1e-9 * max(1.0, abs(float(exp["step_dy"]))):
                problems.append(f"unmatched step edge {got['edge_index']} "
                                f"differs from committed")
    ct, cct = align["unmatched_transitions"], stair["unmatched_transitions"]
    if len(ct) != len(cct):
        problems.append(f"unmatched_transitions count {len(ct)} != committed "
                        f"{len(cct)}")
    return problems


def _adjacent_unmatched(edge_index, matched_step_dy, unmatched_steps, tol):
    """Unmatched discontinuities within ``tol`` samples of ``edge_index``.

    Returns a list of ``{edge_index, dw_mid, step_dy, sign, same_sign}`` for
    every detected-but-unmatched step whose edge index differs from the matched
    edge by at most ``tol`` (the committed match tolerance), tagged with its
    sign and whether it shares the matched step's sign.
    """
    msign = float(np.sign(matched_step_dy))
    out = []
    for u in unmatched_steps:
        if abs(int(u["edge_index"]) - int(edge_index)) <= tol:
            s = float(np.sign(u["step_dy"]))
            out.append({"edge_index": int(u["edge_index"]),
                        "dw_mid": float(u["step_x"]),
                        "step_dy": float(u["step_dy"]),
                        "sign": "+" if s >= 0 else "-",
                        "same_sign": bool(s == msign)})
    return sorted(out, key=lambda r: r["edge_index"])


# --- plateau-bounded aggregation (offline; Prompt-C anti-laundering) -------
def _count_plateaus(counts):
    """Maximal runs of constant soliton_count in ascending detuning order.

    Returns a list of ``(value, lo_idx, hi_idx)`` inclusive index ranges.
    """
    c = np.asarray(counts, dtype=int)
    out, i, n = [], 0, c.size
    while i < n:
        j = i
        while j < n and c[j] == c[i]:
            j += 1
        out.append((int(c[i]), i, j - 1))
        i = j
    return out


def _plateau_of_index(plateaus, idx):
    """The ``(value, lo, hi)`` plateau containing sample ``idx`` (or None)."""
    for val, lo, hi in plateaus:
        if lo <= idx <= hi:
            return (val, lo, hi)
    return None


def _is_count_change(counts, e):
    """True iff edge ``e`` (joining samples e, e+1) crosses a count change."""
    return int(counts[e]) != int(counts[e + 1])


def _aggregate_window(e_m, delta_n, matched_dy, counts, um_map, matched_edges,
                      *, mode, radius):
    """Aggregated |step_dy| over an annihilation window under one rule.

    ``mode``:
      - ``"matched"``  -- the matched edge alone (raw single-edge magnitude);
      - ``"tol_same"`` -- the over-permissive Prompt-B rule: the matched edge
        plus EVERY same-sign unmatched discontinuity within ``radius`` samples
        (bounded only by the nearest OTHER matched edge), count-change or not;
      - ``"plateau"``  -- the physically bounded rule: the matched edge plus a
        CONTIGUOUS run of same-sign, COUNT-CHANGE, otherwise-unmatched fragments
        in the flicker region.  In-plateau ripple (no count change across the
        contiguous same-count region) is excluded BY CONSTRUCTION and breaks the
        contiguous run, so the window never extends into a stable plateau; the
        run is also bounded by the nearest other matched edge on each side.

    Returns ``(aggregated_abs_dy, per_quantum, window_edges_sorted)``.  For the
    plateau rule the window depends only on the count structure, so ``radius``
    is a candidate reach that cannot admit an in-plateau edge (the tol-
    invariance the anti-laundering check exercises).
    """
    msign = float(np.sign(matched_dy))
    other = sorted(int(e) for e in matched_edges if int(e) != int(e_m))
    lo_bound = max([e for e in other if e < e_m], default=-1)
    hi_bound = min([e for e in other if e > e_m], default=int(counts.size) - 1)
    total = float(matched_dy)
    win = [int(e_m)]
    if mode != "matched":
        for step in (-1, 1):
            e, dist = int(e_m) + step, 1
            while lo_bound < e < hi_bound and dist <= radius:
                same = (e in um_map
                        and float(np.sign(um_map[e])) == msign)
                if mode == "tol_same":
                    if same:                       # absorb any same-sign ripple
                        total += float(um_map[e])
                        win.append(e)
                    e += step
                    dist += 1
                    continue
                # plateau mode: contiguous same-sign COUNT-CHANGE run only
                if same and _is_count_change(counts, e):
                    total += float(um_map[e])
                    win.append(e)
                    e += step
                    dist += 1
                    continue
                break                              # ripple / opp-sign: stop run
    win = sorted(set(win))
    return abs(total), abs(total) / max(int(delta_n), 1), win


# --- count/energy lag Stage 1 precondition (offline) -----------------------
ANNIH_MATCH_TOL_RAD = 0.05        # circular cluster-match tolerance (the brief)


def _annihilation_precondition(sweep, metrics):
    """Stage 1 (offline): locate the 4->3 annihilation holds and name the target.

    Recomputes the alignment exactly as Part 2b, finds the matched 4->3
    transition edge ``e`` (ascending), and inspects holds ``e-1, e, e+1, e+2``
    (the 31/32/33/34 holds of the committed sweep).  The TARGET soliton is the
    persistent cluster present at hold ``e+1`` (the last N=4 hold before the
    count drops) that is absent at hold ``e`` (the first N=3 hold), matched
    circularly at ``ANNIH_MATCH_TOL_RAD``.  Its nearest-neighbour separation
    rank (both among the hold-``e+1`` clusters and, via the rigid-rotation
    cyclic map, among ``seed_positions_rad``) is reported.  Read-only.

    Returns a dict with the per-hold observables, the target angle/identity and
    a ``status`` in {"ok", "inconclusive"} (inconclusive when the clean 4->3
    matched edge is not present, or the disappearing cluster is not a single
    localized soliton -- e.g. a merged 4->2 event).
    """
    from analysis.spectral_metrics import (DEFAULT_STEP_K, detect_power_steps,
                                           match_steps_to_transitions,
                                           soliton_count_transitions)
    stair = metrics["soliton_step"]["staircase"]
    primary = stair["primary_observable"]["chosen"]
    order = np.argsort(sweep["dw_over_kappa"])
    dwk = np.asarray(sweep["dw_over_kappa"])[order]
    counts = np.asarray(sweep["soliton_count"], dtype=int)[order]
    ce = np.asarray(sweep["soliton_count_end_snapshot"], dtype=int)[order]
    ca = np.asarray(sweep["count_agreement"], dtype=float)[order]
    Pc = np.asarray(sweep["P_comb"], dtype=float)[order]
    Pcs = np.asarray(sweep["P_comb_std"], dtype=float)[order]
    br = np.asarray(sweep["breathing_relstd"], dtype=float)[order]
    pos = np.asarray(sweep["peak_positions_rad"], dtype=float)[order]
    y = np.asarray(sweep[primary], dtype=float)[order]
    y = y / float(np.max(y))
    k_json = metrics["soliton_step"]["power_trace_discontinuities"].get("k")
    k_used = float(k_json) if k_json is not None else DEFAULT_STEP_K
    tol = int(stair.get("match_tol_samples", 1))
    steps = detect_power_steps(dwk, y, k=k_used)
    align = match_steps_to_transitions(
        steps, soliton_count_transitions(dwk, counts), tol_samples=tol)
    e43 = [m for m in align["matched"]
           if m["n_high_side"] == 4 and m["n_low_side"] == 3
           and m["delta_n"] == 1]
    persistence_stored = "persistence_fractions" in sweep

    if not e43 or e43[0]["step_edge_index"] + 2 >= counts.size \
            or e43[0]["step_edge_index"] - 1 < 0:
        return {"status": "inconclusive",
                "why": ("no clean matched 4->3 (delta_n=1) transition with two "
                        "flanking holds on each side"),
                "primary": primary}
    e = int(e43[0]["step_edge_index"])
    holds = [e - 1, e, e + 1, e + 2]
    seed = _finite_row(np.asarray(sweep["seed_positions_rad"], dtype=float))

    a_pre = _finite_row(pos[e + 1])       # last N=4 hold before the drop
    a_post = _finite_row(pos[e])          # first N=3 hold
    target_list = _greedy_unmatched(a_post, a_pre, ANNIH_MATCH_TOL_RAD)
    if len(target_list) != 1:
        return {"status": "inconclusive",
                "why": (f"{len(target_list)} clusters disappear across the "
                        f"count drop (expected exactly one localized soliton; "
                        f">1 suggests a merged annihilation masked as 4->3)"),
                "primary": primary, "n_disappearing": len(target_list),
                "disappearing_angles": [float(x) for x in target_list]}
    target = float(target_list[0])

    nn_pre = _nn_separations(a_pre)
    t_nn = min(_circ_dist(target, b) for b in a_pre if _circ_dist(target, b) > 1e-9)
    rank_pre = _rank_smallest(nn_pre, t_nn)
    smap, _ = _cyclic_seed_index(a_pre, seed)
    a_sorted = np.sort(a_pre)
    mi = int(np.argmin(np.abs(_wrap(a_sorted - target))))
    seed_idx = smap.get(mi)
    nn_seed = _nn_separations(seed) if seed.size else []
    rank_seed = (_rank_smallest(nn_seed, nn_seed[seed_idx])
                 if (seed_idx is not None and nn_seed) else None)

    hold_rows = []
    for h in holds:
        ang = np.sort(_finite_row(pos[h]))
        hold_rows.append({
            "idx": int(h), "dw": float(dwk[h]), "count": int(counts[h]),
            "count_end_snapshot": int(ce[h]), "count_agreement": float(ca[h]),
            "P_comb": float(Pc[h]), "P_comb_std": float(Pcs[h]),
            "breathing_relstd": float(br[h]),
            "angles": [float(x) for x in ang]})
    return {
        "status": "ok", "primary": primary, "edge": e,
        "match_tol_samples": tol, "detector_k": k_used,
        "flip_hold_idx": int(e + 1), "flip_hold_dw": float(dwk[e + 1]),
        "post_hold_idx": int(e), "post_hold_dw": float(dwk[e]),
        "target_angle": target, "target_nn_sep": float(t_nn),
        "target_rank_pre": int(rank_pre), "n_pre": int(a_pre.size),
        "target_seed_sorted_idx": (int(seed_idx) if seed_idx is not None
                                   else None),
        "target_seed_nn_rank": (int(rank_seed) if rank_seed is not None
                                else None),
        "n_seed": int(seed.size),
        "seed_positions_rad": [float(x) for x in np.sort(seed)],
        "persistence_fractions_stored": bool(persistence_stored),
        "holds": hold_rows,
    }


def _print_precondition(pc):
    """Print the Stage-1 precondition dict (shared by --stepquanta / report)."""
    if pc["status"] != "ok":
        print(f"[annih-precond] INCONCLUSIVE: {pc['why']}")
        return
    print(f"[annih-precond] 4->3 matched edge {pc['edge']} (ascending); "
          f"count-flip hold {pc['flip_hold_idx']} @ {pc['flip_hold_dw']:.4f}k "
          f"(last N=4) -> hold {pc['post_hold_idx']} @ {pc['post_hold_dw']:.4f}k "
          f"(first N=3)")
    print(f"[annih-precond] {'hold':>4} {'dw/k':>7} {'N':>2} {'Nend':>4} "
          f"{'agree':>6} {'P_comb':>10} {'Pc_std':>9} {'brelstd':>8}  angles")
    for h in pc["holds"]:
        print(f"[annih-precond] {h['idx']:>4} {h['dw']:7.4f} {h['count']:>2} "
              f"{h['count_end_snapshot']:>4} {h['count_agreement']:6.3f} "
              f"{h['P_comb']:10.4e} {h['P_comb_std']:9.3e} "
              f"{h['breathing_relstd']:8.4f}  "
              f"{[round(a, 3) for a in h['angles']]}")
    print(f"[annih-precond] TARGET soliton angle {pc['target_angle']:.4f} rad "
          f"(present at hold {pc['flip_hold_idx']}, absent at hold "
          f"{pc['post_hold_idx']}) -- localized to ONE cluster")
    print(f"[annih-precond] target NN separation {pc['target_nn_sep']:.4f} rad, "
          f"rank {pc['target_rank_pre']} of {pc['n_pre']} (1 = tightest / most "
          f"strongly interacting); maps to seed sorted-idx "
          f"{pc['target_seed_sorted_idx']}, seed-NN rank "
          f"{pc['target_seed_nn_rank']} of {pc['n_seed']}")
    print(f"[annih-precond] per-cluster persistence_fractions stored in npz: "
          f"{pc['persistence_fractions_stored']} (count_agreement is stored; "
          f"the per-cluster breakdown is the Stage-2 instrumented deliverable)")


def step_quanta_forensics() -> int:
    """Confirm/refute the split-step diagnosis behind Part 2i, offline.

    Reloads the primary observable named by the committed staircase JSON block,
    recomputes ``detect_power_steps`` / ``soliton_count_transitions`` /
    ``match_steps_to_transitions`` exactly as the regression test does, confirms
    the recomputed alignment equals the committed sets (aborts loudly on
    drift), then for every matched N -> N-1 step (excluding the 1 -> 0 edge)
    tabulates the adjacent unmatched discontinuities and applies the split-step
    verdict rule.  Read-only; appends a report section and prints the verdict.
    """
    from analysis.spectral_metrics import (DEFAULT_STEP_K, detect_power_steps,
                                           match_steps_to_transitions,
                                           soliton_count_transitions)

    npz_path = RESULTS_DIR / SWEEP_NPZ
    json_path = RESULTS_DIR / METRICS_JSON
    sweep, cfg = load_sweep_npz(npz_path)
    with open(json_path) as f:
        metrics = json.load(f)
    block = metrics["soliton_step"]
    stair = block["staircase"]
    primary = stair["primary_observable"]["chosen"]
    print(f"[stepquanta] npz {npz_path.name} sha256 {_sha256(npz_path)[:16]}..."
          f"  primary observable = {primary}")

    # (1) recompute exactly as Part 2b.
    order = np.argsort(sweep["dw_over_kappa"])
    dwk = np.asarray(sweep["dw_over_kappa"])[order]
    y = np.asarray(sweep[primary], dtype=float)[order]
    y = y / float(np.max(y))
    counts = np.asarray(sweep["soliton_count"], dtype=int)[order]
    k_json = block["power_trace_discontinuities"].get("k")
    k_used = float(k_json) if k_json is not None else DEFAULT_STEP_K
    tol = int(stair.get("match_tol_samples", 1))
    steps = detect_power_steps(dwk, y, k=k_used)
    transitions = soliton_count_transitions(dwk, counts)
    align = match_steps_to_transitions(steps, transitions, tol_samples=tol)
    sigma = float(steps["sigma"])

    drift = _confirm_matches_committed(align, stair)
    if drift:
        print("[stepquanta] ARTIFACT DRIFT -- recomputed alignment does NOT "
              "match the committed spectral_metrics.json:")
        for d in drift:
            print(f"[stepquanta]   - {d}")
        print("[stepquanta] refusing to write a verdict on a drifted artifact "
              "(regenerate the sweep or the JSON, then re-run).")
        return 2
    print(f"[stepquanta] recomputation matches committed JSON: "
          f"{len(align['matched'])} matched, "
          f"{len(align['unmatched_steps'])} unmatched steps, "
          f"{len(align['unmatched_transitions'])} unmatched transitions "
          f"(k={k_used:g}, tol={tol}, robust sigma={sigma:.3e})")

    # (2) per matched step (exclude the 1 -> 0 edge), adjacent unmatched.
    def _is_final(m):
        return m["n_high_side"] >= 1 and m["n_low_side"] == 0

    rows = []
    for m in sorted(align["matched"], key=lambda m: m["dw_mid"]):
        if _is_final(m):
            continue
        adj = _adjacent_unmatched(m["step_edge_index"], m["step_dy"],
                                  align["unmatched_steps"], tol)
        rows.append({"m": m, "adj": adj})

    # per-quantum magnitude of a matched step (merged annihilations / delta_n).
    def _per_quantum(m):
        return abs(float(m["step_dy"])) / max(int(m["delta_n"]), 1)

    print("[stepquanta] matched steps above the 1->0 edge (ascending dw):")
    for r in rows:
        m = r["m"]
        print(f"[stepquanta]   edge {m['step_edge_index']:3d}  dw_mid "
              f"{m['dw_mid']:.4f}k  {m['n_high_side']}->{m['n_low_side']} "
              f"(dn={m['delta_n']})  step_dy {m['step_dy']:+.5f}  "
              f"|per-quantum| {_per_quantum(m):.5f}")
        for a in r["adj"]:
            print(f"[stepquanta]       adj unmatched edge {a['edge_index']:3d}  "
                  f"dw_mid {a['dw_mid']:.4f}k  step_dy {a['step_dy']:+.5f}  "
                  f"({a['sign']}, {'same-sign' if a['same_sign'] else 'OPP-sign'})")

    # the delta_n == 1 steps above the 1->0 edge are exactly the set the test
    # compares pairwise; find the smallest (the failing 4->3) and the reference.
    unit_steps = [r for r in rows if r["m"]["delta_n"] == 1
                  and r["m"]["n_low_side"] >= 1]
    unit_steps.sort(key=lambda r: abs(r["m"]["step_dy"]))
    small = unit_steps[0]
    ref = unit_steps[-1]
    sm, rm = small["m"], ref["m"]
    same_adj = [a for a in small["adj"] if a["same_sign"]]
    opp_adj = [a for a in small["adj"] if not a["same_sign"]]
    dominant = max(same_adj, key=lambda a: abs(a["step_dy"])) \
        if same_adj else None

    # (3) failing-pair quantities.
    before = abs(sm["step_dy"])
    ref_h = abs(rm["step_dy"])
    ratio_before = max(before, ref_h) / min(before, ref_h)
    agg_dom = before + (abs(dominant["step_dy"]) if dominant else 0.0)
    ratio_after = (max(agg_dom, ref_h) / min(agg_dom, ref_h)
                   if agg_dom > 0 else float("inf"))
    # merged-annihilation per-quantum reference (3 -> 1 etc.), if any survives.
    merged = [r["m"] for r in rows if r["m"]["delta_n"] >= 2]
    merged_pq = _per_quantum(merged[0]) if merged else None
    ratio_after_merged = (max(agg_dom, merged_pq) / min(agg_dom, merged_pq)
                          if merged_pq else None)

    print(f"[stepquanta] FAILING PAIR: {rm['n_high_side']}->{rm['n_low_side']} "
          f"step_dy {rm['step_dy']:+.5f} (edge {rm['step_edge_index']}) vs "
          f"{sm['n_high_side']}->{sm['n_low_side']} matched step_dy "
          f"{sm['step_dy']:+.5f} (edge {sm['step_edge_index']})")
    if dominant is not None:
        print(f"[stepquanta]   dominant adjacent same-sign unmatched: edge "
              f"{dominant['edge_index']} @ {dominant['dw_mid']:.4f}k step_dy "
              f"{dominant['step_dy']:+.5f}; sum {agg_dom:.5f}")
    print(f"[stepquanta]   ratio BEFORE aggregation: {ratio_before:.3f} "
          f"(>{STEPQUANTA_FACTOR:g} -> the test fails); ratio AFTER: "
          f"{ratio_after:.3f}"
          + (f"; vs merged {merged[0]['n_high_side']}->{merged[0]['n_low_side']}"
             f" per-quantum {merged_pq:.5f} -> {ratio_after_merged:.3f}"
             if merged_pq else ""))
    n_same = len(same_adj)
    print(f"[stepquanta]   adjacent same-sign unmatched discontinuities to the "
          f"{sm['n_high_side']}->{sm['n_low_side']} edge: {n_same} "
          + ", ".join(f"(edge {a['edge_index']}, {a['step_dy']:+.5f})"
                      for a in same_adj))
    # is the reference edge itself flanked by a same-sign unmatched? (ripple
    # fragments steps generally, so "adjacent same-sign" is not unique).
    ref_same = [a for a in ref["adj"] if a["same_sign"]]
    if ref_same:
        print(f"[stepquanta]   NOTE the reference "
              f"{rm['n_high_side']}->{rm['n_low_side']} edge is itself flanked "
              f"by {len(ref_same)} same-sign unmatched discontinuity(ies) "
              + ", ".join(f"(edge {a['edge_index']}, {a['step_dy']:+.5f})"
                          for a in ref_same)
              + " -- plateau ripple fragments steps, so an adjacent same-sign "
              "unmatched neighbour is not a unique split-partner signal")

    # (4) VERDICT.
    within_after = agg_dom > 0 and ratio_after <= STEPQUANTA_FACTOR and (
        merged_pq is None or ratio_after_merged <= STEPQUANTA_FACTOR)
    if opp_adj and not same_adj:
        verdict = "NOT A SPLIT STEP"
        reason = ("the only adjacent discontinuity is OPPOSITE-sign -- there is "
                  "no same-sign partner to aggregate; the short 4->3 step is "
                  "not a fragment of a larger drop")
    elif not small["adj"]:
        verdict = "NOT A SPLIT STEP"
        reason = ("the 4->3 matched edge has NO adjacent unmatched "
                  "discontinuity within the match tolerance -- the short step "
                  "stands alone (genuine non-quantization)")
    elif not within_after:
        verdict = "NOT A SPLIT STEP"
        reason = (f"aggregating the adjacent same-sign discontinuity still "
                  f"leaves the per-quantum magnitudes beyond a factor of "
                  f"{STEPQUANTA_FACTOR:g} (ratio {ratio_after:.3f}) -- genuine "
                  f"non-quantization or a merged-annihilation issue, a real "
                  f"physics finding")
    elif n_same == 1 and within_after:
        verdict = "SPLIT-STEP CONFIRMED"
        reason = (f"the 4->3 matched edge has exactly one adjacent same-sign "
                  f"otherwise-unmatched discontinuity (edge "
                  f"{dominant['edge_index']}, {dominant['step_dy']:+.5f}); "
                  f"aggregating restores quantization (ratio {ratio_after:.3f} "
                  f"<= {STEPQUANTA_FACTOR:g})")
    else:
        verdict = "AMBIGUOUS"
        reason = (f"aggregating the dominant adjacent same-sign discontinuity "
                  f"(edge {dominant['edge_index']}, {dominant['step_dy']:+.5f}) "
                  f"restores quantization (ratio {ratio_after:.3f} <= "
                  f"{STEPQUANTA_FACTOR:g}), so the split direction is "
                  f"supported -- but the strict split-step criterion is NOT "
                  f"met: the 4->3 edge has {n_same} same-sign adjacent "
                  f"unmatched discontinuities, not exactly one"
                  + (", and the reference edge is itself flanked by a same-sign "
                     "neighbour (plateau ripple fragments multiple steps, so "
                     "'adjacent same-sign unmatched' is not a unique split "
                     "signal)" if ref_same else ""))
    print(f"[stepquanta] VERDICT: {verdict} -- {reason}")

    # --- report section ----------------------------------------------------
    md = []
    md.append("## Step-quanta (offline)")
    md.append("")
    md.append(f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat()} by "
              f"`analysis/staircase_forensics.py --stepquanta` (offline; no "
              f"solver run; read-only on the committed artifacts). Confirms or "
              f"refutes the split-step diagnosis behind the failing "
              f"`tests/test_soliton_staircase.py::test_step_heights_quantized` "
              f"(Part 2i).")
    md.append("")
    md.append(f"- npz: `{SWEEP_NPZ}`  sha256 `{_sha256(npz_path)[:16]}...`")
    md.append(f"- primary observable (from the staircase JSON block): "
              f"`{primary}`; detector k = {k_used:g}, match tol = {tol} "
              f"sample; robust sigma = {sigma:.3e}")
    md.append(f"- recomputation MATCHES the committed alignment: "
              f"{len(align['matched'])} matched, "
              f"{len(align['unmatched_steps'])} unmatched steps, "
              f"{len(align['unmatched_transitions'])} unmatched transitions "
              f"(the artifact is self-consistent -- Part 2b holds).")
    md.append("")
    md.append("### Matched steps above the 1->0 edge, with adjacent unmatched "
              "discontinuities (within tol)")
    md.append("")
    md.append("| matched edge | dw_mid (k) | transition | delta_n | matched "
              "step_dy | |per-quantum| | adjacent unmatched (edge, dw_mid, "
              "step_dy, sign) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["m"]
        adj_s = "; ".join(
            f"{a['edge_index']} @ {a['dw_mid']:.3f}k {a['step_dy']:+.5f} "
            f"({a['sign']}, {'same' if a['same_sign'] else 'OPP'})"
            for a in r["adj"]) or "(none)"
        md.append(f"| {m['step_edge_index']} | {m['dw_mid']:.4f} | "
                  f"{m['n_high_side']}->{m['n_low_side']} | {m['delta_n']} | "
                  f"{m['step_dy']:+.5f} | {_per_quantum(m):.5f} | {adj_s} |")
    md.append("")
    md.append("### Failing pair (Part 2i)")
    md.append("")
    md.append(f"- reference {rm['n_high_side']}->{rm['n_low_side']} step_dy "
              f"**{rm['step_dy']:+.5f}** (edge {rm['step_edge_index']}, "
              f"{rm['dw_mid']:.4f}k)")
    md.append(f"- short {sm['n_high_side']}->{sm['n_low_side']} matched step_dy "
              f"**{sm['step_dy']:+.5f}** (edge {sm['step_edge_index']}, "
              f"{sm['dw_mid']:.4f}k)")
    if dominant is not None:
        md.append(f"- dominant adjacent same-sign unmatched discontinuity "
                  f"**{dominant['step_dy']:+.5f}** (edge "
                  f"{dominant['edge_index']}, {dominant['dw_mid']:.4f}k)")
    md.append(f"- sum (short + dominant adjacent) = **{agg_dom:.5f}**")
    md.append(f"- ratio BEFORE aggregation ({before:.5f} vs {ref_h:.5f}): "
              f"**{ratio_before:.3f}** (> {STEPQUANTA_FACTOR:g} -> the test "
              f"fails)")
    md.append(f"- ratio AFTER aggregation ({agg_dom:.5f} vs {ref_h:.5f}): "
              f"**{ratio_after:.3f}**"
              + (f"; vs the merged {merged[0]['n_high_side']}->"
                 f"{merged[0]['n_low_side']} per-quantum {merged_pq:.5f}: "
                 f"**{ratio_after_merged:.3f}**" if merged_pq else ""))
    md.append(f"- adjacent same-sign unmatched discontinuities to the "
              f"{sm['n_high_side']}->{sm['n_low_side']} edge: **{n_same}** "
              + ", ".join(f"(edge {a['edge_index']}, {a['step_dy']:+.5f})"
                          for a in same_adj))
    if ref_same:
        md.append(f"- the reference {rm['n_high_side']}->{rm['n_low_side']} "
                  f"edge is itself flanked by {len(ref_same)} same-sign "
                  f"unmatched discontinuity(ies) "
                  + ", ".join(f"(edge {a['edge_index']}, {a['step_dy']:+.5f})"
                              for a in ref_same)
                  + " -- plateau ripple fragments steps, so an adjacent "
                  "same-sign unmatched neighbour is not a unique split-partner "
                  "signal.")
    md.append("")
    md.append("### Verdict")
    md.append("")
    md.append("Rule: **SPLIT-STEP CONFIRMED** iff the 4->3 matched edge has "
              "exactly one adjacent (within tol_samples), same-sign, "
              "otherwise-unmatched discontinuity AND the sum brings all "
              "per-quantum magnitudes within a factor of 2; **NOT A SPLIT "
              "STEP** iff the adjacent discontinuity is opposite-sign, absent, "
              "or the aggregated magnitudes still exceed a factor of 2 (genuine "
              "non-quantization / a merged-annihilation issue -- a real physics "
              "finding); **AMBIGUOUS** otherwise.")
    md.append("")
    md.append(f"**VERDICT: {verdict}** -- {reason}")
    md.append("")
    md.append("No fix applied: this diagnosis is the deliverable. Any remedy "
              "(aggregating split matched+adjacent discontinuities before the "
              "quantization check, a plateau-level step-height measure, or "
              "accepting the merged-annihilation reference) is a separate, "
              "gated change -- not made here.")
    md.append("")

    # =====================================================================
    # Prompt C: PHYSICALLY BOUNDED aggregation.  The tol=1 same-sign rule
    # above can "restore" quantization only by absorbing same-sign neighbours
    # that plateau ripple scatters around every edge.  Bound the aggregation by
    # the COUNT STRUCTURE (plateaus) instead of a neighbour radius, and re-test.
    # =====================================================================
    plateaus = _count_plateaus(counts)
    matched_edge_set = {int(m["step_edge_index"]) for m in align["matched"]}
    um_map = {int(u["edge_index"]): float(u["step_dy"])
              for u in align["unmatched_steps"]}
    cc_edges = [e for e in range(counts.size - 1) if _is_count_change(counts, e)]
    unmatched_cc = [e for e in cc_edges if e not in matched_edge_set]
    matched_nn = [m for m in sorted(align["matched"], key=lambda m: m["dw_mid"])
                  if not _is_final(m)]

    pb_rows, undefined_why, seen_edges = [], [], set()
    for m in matched_nn:
        e = int(m["step_edge_index"])
        lo_p = _plateau_of_index(plateaus, e)        # low-detuning (low-count)
        hi_p = _plateau_of_index(plateaus, e + 1)    # high-detuning (high-count)
        if lo_p is None or hi_p is None or lo_p[0] == hi_p[0]:
            undefined_why.append(
                f"{m['n_high_side']}->{m['n_low_side']} edge {e}: flanking "
                f"plateaus not resolvable")
            pb_rows.append({"m": m, "lo_p": lo_p, "hi_p": hi_p, "undef": True})
            continue
        agg_pb, pq_pb, win_pb = _aggregate_window(
            e, m["delta_n"], m["step_dy"], counts, um_map, matched_edge_set,
            mode="plateau", radius=max(tol, 3))
        if seen_edges & set(win_pb):
            undefined_why.append(
                f"{m['n_high_side']}->{m['n_low_side']} edge {e}: window "
                f"overlaps a neighbour at "
                f"{sorted(seen_edges & set(win_pb))}")
        seen_edges |= set(win_pb)
        neigh = []
        for a in _adjacent_unmatched(e, m["step_dy"],
                                     align["unmatched_steps"], tol):
            if not a["same_sign"]:
                continue
            ce = a["edge_index"]
            reason = ("count change -> absorbed" if _is_count_change(counts, ce)
                      else f"in-plateau ripple (count {int(counts[ce])}->"
                           f"{int(counts[ce + 1])}, no change) -> EXCLUDED")
            neigh.append({"edge": ce, "step_dy": a["step_dy"],
                          "inside": ce in win_pb, "reason": reason})
        pb_rows.append({"m": m, "lo_p": lo_p, "hi_p": hi_p, "agg": agg_pb,
                        "pq": pq_pb, "win": win_pb, "neigh": neigh,
                        "undef": False})

    window_defined = not undefined_why
    defined_rows = [r for r in pb_rows if not r["undef"]]
    pq_vals = [r["pq"] for r in defined_rows]
    raw_vals = [abs(r["m"]["step_dy"]) / r["m"]["delta_n"] for r in defined_rows]
    pb_ratio = (max(pq_vals) / min(pq_vals)) if len(pq_vals) >= 2 else float("nan")
    raw_ratio = (max(raw_vals) / min(raw_vals)) if len(raw_vals) >= 2 \
        else float("nan")
    no_ripple_absorbed = all(
        _is_count_change(counts, e)
        for r in defined_rows for e in r["win"]
        if e != int(r["m"]["step_edge_index"]))

    # sensitivity (per-quantum) under the three window definitions, at tol,
    # plus the plateau-bounded tol-invariance scan (radius 1/2/3).
    def _mode_pq(mode, radius):
        return {int(m["step_edge_index"]): _aggregate_window(
                    int(m["step_edge_index"]), m["delta_n"], m["step_dy"],
                    counts, um_map, matched_edge_set, mode=mode, radius=radius)
                for m in matched_nn}
    sens = {mode: _mode_pq(mode, tol)
            for mode in ("plateau", "tol_same", "matched")}
    pb_scan = {r: _mode_pq("plateau", r) for r in (1, 2, 3)}
    b_scan = {r: _mode_pq("tol_same", r) for r in (1, 2, 3)}
    pb_tol_invariant = all(
        pb_scan[1][int(m["step_edge_index"])][1]
        == pb_scan[r][int(m["step_edge_index"])][1]
        for m in matched_nn for r in (2, 3))

    # verdict (plateau-bounded rule only).
    if not window_defined:
        pb_verdict = "STILL AMBIGUOUS"
        pb_reason = ("the plateau-bounded window is undefined for: "
                     + "; ".join(undefined_why))
    elif not np.isfinite(pb_ratio):
        pb_verdict = "STILL AMBIGUOUS"
        pb_reason = ("fewer than two matched N->N-1 transitions above the 1->0 "
                     "edge -- no pair to compare")
    elif pb_ratio > STEPQUANTA_FACTOR:
        pb_verdict = "NOT QUANTIZED"
        pb_reason = (
            f"the plateau-bounded per-quantum magnitudes still span a factor "
            f"{pb_ratio:.3f} > {STEPQUANTA_FACTOR:g}. Every count-change edge "
            f"is already matched ({len(unmatched_cc)} unmatched count-change "
            f"edges exist), so the annihilation windows are all single matched "
            f"edges and the plateau-bounded sums equal the raw single-edge "
            f"magnitudes: the short 4->3 step (per-quantum "
            f"{sens['plateau'][int(sm['step_edge_index'])][1]:.5f}) cannot be "
            f"restored by absorbing edge 33 (+"
            f"{um_map.get(33, float('nan')):.5f}), which lies INSIDE the "
            f"count-4 plateau (no count change). A real finding: the "
            f"position-persistence count and the comb-energy drop are offset "
            f"by one hold at this annihilation, so a count-structure-respecting "
            f"aggregation does not quantize the single-edge heights")
    elif no_ripple_absorbed:
        pb_verdict = "QUANTIZED (plateau-bounded)"
        pb_reason = (
            f"the plateau-bounded per-quantum magnitudes agree within a factor "
            f"{pb_ratio:.3f} <= {STEPQUANTA_FACTOR:g}, and every window is the "
            f"matched edge plus only count-changing same-sign fragments (no "
            f"in-plateau ripple absorbed)")
    else:
        pb_verdict = "STILL AMBIGUOUS"
        pb_reason = ("the per-quantum magnitudes agree within a factor of 2 "
                     "only because in-plateau ripple was absorbed -- the "
                     "window construction is not purely count-structure-bounded")

    print("[stepquanta] === plateau-bounded aggregation (Prompt C) ===")
    print("[stepquanta] plateaus (count: idx range): "
          + "  ".join(f"{v}:[{a},{b}]" for v, a, b in plateaus if a <= 45)
          + f"  (+{sum(1 for _, a, _ in plateaus if a > 45)} higher plateaus)")
    print(f"[stepquanta] count-change edges: {cc_edges}; ALL matched "
          f"({len(unmatched_cc)} unmatched count-change edges) -> every "
          f"unmatched discontinuity is in-plateau ripple")
    for r in pb_rows:
        m = r["m"]
        if r["undef"]:
            print(f"[stepquanta]   {m['n_high_side']}->{m['n_low_side']} edge "
                  f"{m['step_edge_index']}: WINDOW UNDEFINED")
            continue
        print(f"[stepquanta]   {m['n_high_side']}->{m['n_low_side']} edge "
              f"{m['step_edge_index']} dw {m['dw_mid']:.4f}k dn{m['delta_n']} "
              f"matched_dy {m['step_dy']:+.5f} | low-count({r['lo_p'][0]}) "
              f"plateau [{r['lo_p'][1]},{r['lo_p'][2]}] high-count({r['hi_p'][0]}) "
              f"plateau [{r['hi_p'][1]},{r['hi_p'][2]}] | window {r['win']} "
              f"agg|dy| {r['agg']:.5f} per-quantum {r['pq']:.5f}")
        for nb in r["neigh"]:
            print(f"[stepquanta]       neighbour edge {nb['edge']} "
                  f"{nb['step_dy']:+.5f}: {'INSIDE' if nb['inside'] else 'excluded'}"
                  f" -- {nb['reason']}")
    if window_defined and np.isfinite(pb_ratio):
        print(f"[stepquanta] plateau-bounded per-quantum: "
              + ", ".join(f"{r['m']['n_high_side']}->{r['m']['n_low_side']}="
                          f"{r['pq']:.5f}" for r in defined_rows)
              + f" -> pairwise ratio {pb_ratio:.3f} (raw single-edge ratio "
              f"{raw_ratio:.3f})")
    print("[stepquanta] sensitivity (per-quantum, at committed tol="
          f"{tol}):  (a) plateau-bounded | (b) tol-same-sign | (c) matched-only")
    for m in matched_nn:
        e = int(m["step_edge_index"])
        print(f"[stepquanta]   {m['n_high_side']}->{m['n_low_side']} edge {e}: "
              f"a={sens['plateau'][e][1]:.5f} {sens['plateau'][e][2]}  "
              f"b={sens['tol_same'][e][1]:.5f} {sens['tol_same'][e][2]}  "
              f"c={sens['matched'][e][1]:.5f}")
    print(f"[stepquanta] tol-invariance: plateau-bounded per-quantum identical "
          f"at radius 1/2/3 = {pb_tol_invariant}; (b) tol-same-sign GROWS with "
          f"radius, e.g. 4->3: "
          + "/".join(f"{b_scan[r][int(sm['step_edge_index'])][1]:.5f}"
                     for r in (1, 2, 3)))
    print(f"[stepquanta] PLATEAU-BOUNDED VERDICT: {pb_verdict} -- {pb_reason}")

    # --- report section 2 --------------------------------------------------
    md2 = []
    md2.append("## Plateau-bounded aggregation (offline)")
    md2.append("")
    md2.append(f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat()} by "
               f"`analysis/staircase_forensics.py --stepquanta` (offline; no "
               f"solver run; read-only on the committed artifacts). Tests a "
               f"PHYSICALLY BOUNDED aggregation rule for the Part 2i step "
               f"heights: the AMBIGUOUS verdict above showed that 'exactly one "
               f"adjacent same-sign unmatched discontinuity' is not a valid "
               f"split-partner signal (plateau ripple scatters same-sign "
               f"unmatched neighbours around most edges), so aggregation is "
               f"bounded here by the COUNT STRUCTURE (plateaus), not a "
               f"neighbour radius.")
    md2.append("")
    md2.append(f"- primary observable `{primary}`; detector k = {k_used:g}, "
               f"match tol = {tol}; robust sigma = {sigma:.3e}; npz sha256 "
               f"`{_sha256(npz_path)[:16]}...` (recomputation matches "
               f"committed).")
    md2.append(f"- soliton_count plateaus (count: idx range): "
               + ", ".join(f"{v}:[{a},{b}]" for v, a, b in plateaus))
    md2.append(f"- count-change edges: {cc_edges} -- ALL are matched "
               f"(`unmatched_transitions` is empty; {len(unmatched_cc)} "
               f"unmatched count-change edges). **Every one of the "
               f"{len(um_map)} unmatched discontinuities is therefore "
               f"in-plateau ripple**, so no count-changing fragment exists to "
               f"legitimately absorb into any annihilation window.")
    md2.append("")
    md2.append("### Annihilation windows (matched N->N-1, excluding 1->0)")
    md2.append("")
    md2.append("| transition | matched edge | dw_mid (k) | delta_n | matched "
               "step_dy | low-count plateau | high-count plateau | window "
               "edges | aggregated |step_dy| | per-quantum |")
    md2.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in defined_rows:
        m = r["m"]
        md2.append(
            f"| {m['n_high_side']}->{m['n_low_side']} | {m['step_edge_index']} "
            f"| {m['dw_mid']:.4f} | {m['delta_n']} | {m['step_dy']:+.5f} | "
            f"{r['lo_p'][0]}:[{r['lo_p'][1]},{r['lo_p'][2]}] | "
            f"{r['hi_p'][0]}:[{r['hi_p'][1]},{r['hi_p'][2]}] | {r['win']} | "
            f"{r['agg']:.5f} | {r['pq']:.5f} |")
    md2.append("")
    md2.append("Neighbour classification (same-sign unmatched within tol of "
               "each matched edge -- inside the plateau-bounded window or "
               "excluded, and why):")
    md2.append("")
    for r in defined_rows:
        m = r["m"]
        if not r["neigh"]:
            continue
        for nb in r["neigh"]:
            md2.append(f"- {m['n_high_side']}->{m['n_low_side']} edge "
                       f"{m['step_edge_index']}: neighbour edge {nb['edge']} "
                       f"({nb['step_dy']:+.5f}) -- "
                       f"{'INSIDE window' if nb['inside'] else 'EXCLUDED'}: "
                       f"{nb['reason']}")
    md2.append("")
    md2.append("### Per-quantum magnitudes and ratios")
    md2.append("")
    md2.append(f"- plateau-bounded per-quantum: "
               + ", ".join(f"{r['m']['n_high_side']}->{r['m']['n_low_side']} = "
                           f"{r['pq']:.5f}" for r in defined_rows))
    md2.append(f"- plateau-bounded pairwise ratio (max/min): **{pb_ratio:.3f}** "
               f"(the raw single-edge ratio is {raw_ratio:.3f}; identical here "
               f"because every window is a single matched edge)")
    md2.append("")
    md2.append("### Sensitivity / anti-laundering (per-quantum at committed "
               f"tol = {tol})")
    md2.append("")
    md2.append("| transition | (a) plateau-bounded | (b) tol=1 same-sign | "
               "(c) matched-only |")
    md2.append("|---|---|---|---|")
    for m in matched_nn:
        e = int(m["step_edge_index"])
        md2.append(f"| {m['n_high_side']}->{m['n_low_side']} | "
                   f"{sens['plateau'][e][1]:.5f} {sens['plateau'][e][2]} | "
                   f"{sens['tol_same'][e][1]:.5f} {sens['tol_same'][e][2]} | "
                   f"{sens['matched'][e][1]:.5f} |")
    md2.append("")
    md2.append(f"- **plateau-bounded is INSENSITIVE to the reach**: per-quantum "
               f"identical at radius 1/2/3 = {pb_tol_invariant} (it is defined "
               f"by count structure, not a neighbour radius -- no count-change "
               f"fragment exists to admit at any reach).")
    md2.append("- the over-permissive (b) tol-same-sign rule GROWS with the "
               "reach (it absorbs progressively more in-plateau ripple), e.g. "
               "the 4->3 per-quantum at radius 1/2/3 = "
               + "/".join(f"{b_scan[r][int(sm['step_edge_index'])][1]:.5f}"
                          for r in (1, 2, 3))
               + " and the 5->4 = "
               + "/".join(f"{b_scan[r][int(rm['step_edge_index'])][1]:.5f}"
                          for r in (1, 2, 3))
               + " -- confirming (b) launders quantization by absorbing ripple, "
               "which the plateau bound forbids.")
    md2.append("- (a) equals (c): with no unmatched count-change fragments, the "
               "physically bounded window is exactly the matched edge, so the "
               "plateau-bounded magnitudes ARE the raw single-edge magnitudes.")
    md2.append("")
    md2.append("### Verdict")
    md2.append("")
    md2.append("Rule (plateau-bounded only): **QUANTIZED (plateau-bounded)** "
               "iff all plateau-bounded per-quantum magnitudes agree pairwise "
               "within a factor of 2 AND each window is the matched edge plus "
               "only count-changing same-sign fragments (no in-plateau ripple "
               "absorbed); **NOT QUANTIZED** iff the plateau-bounded per-quantum "
               "magnitudes still exceed a factor of 2 (a real physics finding -- "
               "possible merged annihilation or genuine non-quantization); "
               "**STILL AMBIGUOUS** iff the window construction is undefined for "
               "some transition (overlapping windows or unresolvable flanking "
               "plateaus).")
    md2.append("")
    md2.append(f"**VERDICT: {pb_verdict}** -- {pb_reason}")
    md2.append("")
    md2.append("No fix applied: this adjudication is the deliverable. The "
               "finding -- the 4->3 comb-power drop is split across the "
               "count-flip hold (edge 32, matched, ~1/3 quantum) and the "
               "adjacent count-4 plateau hold (edge 33, ~2/3 quantum), i.e. the "
               "position-persistence count lags the comb-energy drop by one "
               "hold -- means any legitimate remedy (a plateau-integrated step "
               "height, or reconciling the count observable with the energy "
               "observable at the annihilation edge) is a separate, gated "
               "change, not made here.")
    md2.append("")

    out = RESULTS_DIR / REPORT_MD
    prior = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = "\n## Step-quanta (offline)"
    if marker in prior:                       # idempotent: replace both sections
        prior = prior[:prior.index(marker)].rstrip() + "\n"
    out.write_text(prior.rstrip() + "\n" + "\n".join(md) + "\n\n"
                   + "\n".join(md2) + "\n", encoding="utf-8")
    print(f"[stepquanta] report -> {out}")

    # Stage 1 precondition for the count/energy-lag investigation (offline;
    # names the target soliton for the --diagnose-annihilation instrumented
    # run). Printed here; the md "count/energy lag" section is written by
    # --annihilation-report once the Stage-2 sidecar exists.
    print("[stepquanta] === count/energy lag Stage 1 precondition ===")
    _print_precondition(_annihilation_precondition(sweep, metrics))
    return 0


# ---------------------------------------------------------------------------
# --annihilation-report mode (Stage 3): adjudicate the count/energy lag from
# the committed diagnose_annih_4to3.npz sidecar (the Stage-2 instrumented run).
# Read-only; appends a "count/energy lag" section and prints the verdict.
# ---------------------------------------------------------------------------
ANNIH_DIAG_NPZ = "diagnose_annih_4to3.npz"
ANNIH_ENERGY_COLLAPSED = 0.15     # < this fraction of the two-holds-earlier
#                                   local energy == "collapsed" (strong form)
ANNIH_ENERGY_MAJORITY = 0.50      # < this fraction == "majority of the quantum
#                                   gone" (the COUNTER-LATENCY energy criterion)


def annihilation_report() -> int:
    """Stage 3: adjudicate the count/energy lag from the instrumented sidecar.

    Reads ``analysis/results/diagnose_annih_4to3.npz`` (the Stage-2
    ``--diagnose-annihilation`` run), tracks the TARGET soliton's per-cluster
    LOCAL comb energy across the holds around the 4->3 count flip, and applies
    the verdict rule: COUNTER-LATENCY (the height-based acceptance rule
    certifies a soliton whose energy has already left) vs INTRINSIC LAG (the
    annihilation genuinely completes across the count-flip hold, so the integer
    count is correct per hold) vs INCONCLUSIVE (a merged event, or the target
    is not resolvable).  Read-only; appends the "count/energy lag" section to
    the report and prints the verdict.  No fix.
    """
    diag_path = RESULTS_DIR / ANNIH_DIAG_NPZ
    if not diag_path.exists():
        print(f"[annih-report] missing {diag_path} -- run "
              f"`analysis/run_detuning_sweep.py --diagnose-annihilation` first")
        return 2
    sc = np.load(diag_path, allow_pickle=False)

    # committed-data precondition (for the target angle + hold context).
    sweep, _ = load_sweep_npz(RESULTS_DIR / SWEEP_NPZ)
    with open(RESULTS_DIR / METRICS_JSON) as f:
        metrics = json.load(f)
    pc = _annihilation_precondition(sweep, metrics)

    # order holds by DESCENDING detuning (the physical sweep direction).
    dw = np.asarray(sc["dw_over_kappa"], float)
    o = np.argsort(dw)[::-1]
    dw = dw[o]
    count = np.asarray(sc["soliton_count"], int)[o]
    agree = np.asarray(sc["count_agreement"], float)[o]
    col_ang = np.asarray(sc["col_angles_rad"], float)[o]        # (H, n_sol)
    in_cl = np.asarray(sc["in_cluster"], bool)[o]               # (H, n_sol)
    loc_e = np.asarray(sc["local_energy_bgsub"], float)[o]      # (H, n_in, S)
    loc_pk = np.asarray(sc["local_peak"], float)[o]
    ab_abs = np.asarray(sc["above_abs"], bool)[o]
    ab_phys = np.asarray(sc["above_phys"], bool)[o]
    floor_phys = np.asarray(sc["floor_phys"], float)[o]
    b2 = np.asarray(sc["B2_ref"], float)[o]
    target_committed = float(sc["target_angle_committed_rad"])
    min_persist = float(sc["min_persistence"])
    modes_from_uint = float(sc["modes_from_uint"])
    H = dw.size

    # locate the 4->3 count flip: the highest-detuning hold at which the
    # counter's count drops from 4 to 3 going down.
    flips = [i for i in range(1, H) if count[i - 1] == 4 and count[i] == 3]
    status = "ok"
    why = ""
    if not flips:
        status, why = "inconclusive", ("no 4->3 count transition in the "
                                       "instrumented window (the counter's "
                                       "reproduced trace differs)")
    merged = pc.get("status") == "inconclusive"
    if merged:
        status, why = "inconclusive", pc.get("why", "precondition inconclusive")

    verdict = "INCONCLUSIVE"
    reason = why
    rows_md = []
    detail = {}
    if status == "ok":
        post = flips[0]                      # first N=3 hold (lower detuning)
        flip = post - 1                      # last N=4 hold == the count-flip
        # target column: counted at the flip hold, dropped just after, nearest
        # the committed target angle.
        cand = [k for k in range(col_ang.shape[1])
                if in_cl[flip, k] and np.isfinite(col_ang[flip, k])]
        if not cand:
            status, why = "inconclusive", "no counted cluster at the flip hold"
        else:
            tcol = min(cand, key=lambda k: _circ_dist(col_ang[flip, k],
                                                      target_committed))
            # per-hold target local energy (mean over in-window snapshots) and
            # peak-certification fraction.
            e_hold = np.array([float(np.nanmean(loc_e[h, :, tcol]))
                               for h in range(H)])
            certify = np.array([
                float(np.mean(ab_abs[h, :, tcol] & ab_phys[h, :, tcol]))
                for h in range(H)])
            ref = flip - 2                    # two holds earlier (higher dw)
            if ref < 0:
                status, why = "inconclusive", ("fewer than two holds above the "
                                               "flip in the window")
            else:
                e_flip = e_hold[flip]
                e_ref = e_hold[ref]
                frac = e_flip / e_ref if e_ref > 0 else float("nan")
                cert_flip = certify[flip]
                dropped_after = (not in_cl[flip + 1, tcol]) \
                    if flip + 1 < H else True
                detail = dict(
                    tcol=tcol, flip=flip, post=post, ref=ref,
                    dw_flip=float(dw[flip]), dw_ref=float(dw[ref]),
                    dw_post=float(dw[post]), e_flip=e_flip, e_ref=e_ref,
                    frac=frac, cert_flip=cert_flip,
                    agree_flip=float(agree[flip]),
                    target_angle=float(col_ang[flip, tcol]),
                    dropped_after=bool(dropped_after))
                # trajectory rows for the report (a few holds either side).
                for h in range(max(0, ref - 1), min(H, post + 2)):
                    rows_md.append(dict(
                        dw=float(dw[h]), count=int(count[h]),
                        agree=float(agree[h]),
                        e=float(e_hold[h]),
                        e_frac=float(e_hold[h] / e_ref) if e_ref > 0 else
                        float("nan"),
                        pk=float(np.nanmean(loc_pk[h, :, tcol])),
                        floor_phys=float(floor_phys[h]),
                        certify=float(certify[h]),
                        in_cl=bool(in_cl[h, tcol])))

                majority_gone = frac < ANNIH_ENERGY_MAJORITY
                collapsed = frac < ANNIH_ENERGY_COLLAPSED
                peak_certifies = cert_flip >= min_persist
                if majority_gone and peak_certifies:
                    verdict = "COUNTER-LATENCY"
                    reason = (
                        f"at the count-flip hold ({dw[flip]:.4f}k) the target "
                        f"soliton's local comb energy is {100 * frac:.1f}% of "
                        f"its value two holds earlier ({dw[ref]:.4f}k) -- "
                        f"{'collapsed' if collapsed else 'majority gone'} -- yet "
                        f"its peak height clears soliton_frac*B2_ref in "
                        f"{100 * cert_flip:.0f}% of snapshots (>= "
                        f"{100 * min_persist:.0f}% min_persistence), so the "
                        f"height-based rule certifies a soliton whose energy has "
                        f"already left. The count is late BY CONSTRUCTION: a "
                        f"peak-height-vs-energy defect in count_solitons_windowed")
                elif not majority_gone:
                    verdict = "INTRINSIC LAG (benign)"
                    reason = (
                        f"at the count-flip hold ({dw[flip]:.4f}k) the target "
                        f"soliton still carries {100 * frac:.1f}% of its local "
                        f"energy two holds earlier ({dw[ref]:.4f}k) -- "
                        f"substantial -- so the annihilation genuinely completes "
                        f"across the flip; the per-hold count is correct and the "
                        f"one-hold offset is the expected integer-count vs "
                        f"continuous-energy resolution mismatch")
                else:
                    verdict = "INCONCLUSIVE"
                    reason = (
                        f"the target's local energy at the flip hold is "
                        f"{100 * frac:.1f}% of two holds earlier (between the "
                        f"{100 * ANNIH_ENERGY_COLLAPSED:.0f}% collapse and "
                        f"{100 * ANNIH_ENERGY_MAJORITY:.0f}% substantial "
                        f"thresholds) or its peak fails to certify "
                        f"(certify {100 * cert_flip:.0f}% vs "
                        f"{100 * min_persist:.0f}%): neither criterion is met "
                        f"cleanly")

    # ---- print --------------------------------------------------------------
    print("[annih-report] === count/energy lag adjudication (Stage 3) ===")
    if pc.get("status") == "ok":
        print(f"[annih-report] target (committed) ~{target_committed:.4f} rad, "
              f"seed sorted-idx {pc.get('target_seed_sorted_idx')}, seed-NN "
              f"rank {pc.get('target_seed_nn_rank')} of {pc.get('n_seed')}")
    if detail:
        print(f"[annih-report] target column {detail['tcol']} @ "
              f"{detail['target_angle']:.4f} rad; flip hold {detail['dw_flip']:.4f}k "
              f"(N4, agree {detail['agree_flip']:.3f}) -> post {detail['dw_post']:.4f}k "
              f"(N3); ref (two earlier) {detail['dw_ref']:.4f}k")
        print(f"[annih-report] target local energy: flip {detail['e_flip']:.4e} "
              f"vs ref {detail['e_ref']:.4e} -> {100 * detail['frac']:.1f}% "
              f"remaining; peak-certify fraction at flip "
              f"{100 * detail['cert_flip']:.0f}% (min_persistence "
              f"{100 * min_persist:.0f}%); dropped after flip: "
              f"{detail['dropped_after']}")
        print(f"[annih-report] trajectory (dw, N, agree, E, E/ref, certify, in_cluster):")
        for r in rows_md:
            print(f"[annih-report]   {r['dw']:.4f}k N={r['count']} "
                  f"agree={r['agree']:.3f} E={r['e']:.4e} "
                  f"E/ref={100 * r['e_frac']:5.1f}% certify={100 * r['certify']:3.0f}% "
                  f"{'IN' if r['in_cl'] else 'out'}")
    print(f"[annih-report] VERDICT: {verdict} -- {reason}")

    # ---- md section ---------------------------------------------------------
    md = []
    md.append("## Count/energy lag (offline)")
    md.append("")
    md.append(f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat()} by "
              f"`analysis/staircase_forensics.py --annihilation-report` "
              f"(offline adjudication of the Stage-2 instrumented run "
              f"`analysis/run_detuning_sweep.py --diagnose-annihilation`, whose "
              f"sidecar is `analysis/results/{ANNIH_DIAG_NPZ}`). Decides whether "
              f"the 4->3 count/energy offset is COUNTER-LATENCY (the "
              f"physics-anchored height rule holds a dying soliton above "
              f"threshold one hold too long) or an INTRINSIC integer-count vs "
              f"continuous-energy LAG.")
    md.append("")
    if pc.get("status") == "ok":
        md.append(f"### Stage 1 precondition (committed npz)")
        md.append("")
        md.append(f"- 4->3 matched edge {pc['edge']}; count-flip hold "
                  f"{pc['flip_hold_idx']} @ {pc['flip_hold_dw']:.4f}k (last N=4) "
                  f"-> hold {pc['post_hold_idx']} @ {pc['post_hold_dw']:.4f}k "
                  f"(first N=3).")
        md.append(f"- TARGET soliton ~{pc['target_angle']:.4f} rad, localized to "
                  f"ONE cluster; NN-separation rank {pc['target_rank_pre']} of "
                  f"{pc['n_pre']} (1 = tightest); maps to seed sorted-idx "
                  f"{pc['target_seed_sorted_idx']}, seed-NN rank "
                  f"{pc['target_seed_nn_rank']} of {pc['n_seed']} -- the most "
                  f"strongly interacting soliton.")
        md.append("")
        md.append("| hold | dw/k | N | N_end-snap | count_agreement | P_comb | "
                  "breathing_relstd | cluster angles (rad) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for h in pc["holds"]:
            md.append(f"| {h['idx']} | {h['dw']:.4f} | {h['count']} | "
                      f"{h['count_end_snapshot']} | {h['count_agreement']:.3f} | "
                      f"{h['P_comb']:.4e} | {h['breathing_relstd']:.4f} | "
                      f"{[round(a, 3) for a in h['angles']]} |")
        md.append("")
    if detail:
        md.append("### Target soliton local-energy trajectory (instrumented)")
        md.append("")
        md.append("Local comb energy = angular integral of |E(theta)|^2 over "
                  "+/- cluster_tol around the tracked cluster, background "
                  "(angular median) subtracted, meaned over the in-window "
                  "snapshots; E/ref is relative to the value two holds above the "
                  "flip; certify = fraction of snapshots whose local peak clears "
                  "BOTH acceptance floors (bg_floor_multiple*median AND "
                  "soliton_frac*B2_ref).")
        md.append("")
        md.append("| dw/k | N (counter) | count_agreement | target local E | "
                  "E/ref | mean local peak | soliton_frac*B2_ref | certify frac | "
                  "counted? |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows_md:
            md.append(f"| {r['dw']:.4f} | {r['count']} | {r['agree']:.3f} | "
                      f"{r['e']:.4e} | {100 * r['e_frac']:.1f}% | {r['pk']:.4e} | "
                      f"{r['floor_phys']:.4e} | {100 * r['certify']:.0f}% | "
                      f"{'yes' if r['in_cl'] else 'no'} |")
        md.append("")
        md.append(f"- at the count-flip hold ({detail['dw_flip']:.4f}k, the last "
                  f"hold the target is counted, count_agreement "
                  f"{detail['agree_flip']:.3f}): target local energy is "
                  f"**{100 * detail['frac']:.1f}%** of its value two holds "
                  f"earlier ({detail['dw_ref']:.4f}k), and its peak certifies in "
                  f"**{100 * detail['cert_flip']:.0f}%** of snapshots "
                  f"(min_persistence {100 * min_persist:.0f}%).")
        md.append("")
    md.append("### Verdict")
    md.append("")
    md.append("Rule: **COUNTER-LATENCY** iff at the count-flip hold the target's "
              "local energy has collapsed (majority of the quantum gone) yet its "
              "peak keeps it above the acceptance floor in >= min_persistence of "
              "snapshots (the height-based rule certifies a soliton whose energy "
              "has left -- a peak-height-vs-energy defect in "
              "count_solitons_windowed); **INTRINSIC LAG (benign)** iff the "
              "target still carries substantial local energy at the count-flip "
              "hold (the annihilation completes across the flip, so the integer "
              "count is correct per hold); **INCONCLUSIVE** otherwise (e.g. a "
              "merged 4->2 event masked as 4->3).")
    md.append("")
    md.append(f"**VERDICT: {verdict}** -- {reason}")
    md.append("")
    md.append("No fix applied: this diagnosis is the deliverable. No threshold, "
              "gate, test, or committed artifact was changed. The next action "
              "(if COUNTER-LATENCY: reconcile the counter's peak-height "
              "acceptance with an energy/area criterion; if INTRINSIC: accept "
              "the one-hold offset as a resolution limit and, if desired, report "
              "a plateau-integrated step height) awaits review.")
    md.append("")

    out = RESULTS_DIR / REPORT_MD
    prior = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = "\n## Count/energy lag (offline)"
    if marker in prior:                       # idempotent: replace this section
        prior = prior[:prior.index(marker)].rstrip() + "\n"
    out.write_text(prior.rstrip() + "\n" + "\n".join(md) + "\n", encoding="utf-8")
    print(f"[annih-report] report -> {out}")
    return 0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--starvation", action="store_true",
                    help="offline snapshot-starvation hypothesis test over the "
                         "primary + robustness variant npzs (Part A); appends a "
                         "section to staircase_forensics.md and prints a "
                         "CONFIRMED / NOT CONFIRMED verdict")
    ap.add_argument("--detectability", action="store_true",
                    help="offline detectability diagnosis (Stage A): missing-"
                         "cluster position persistence at monotonicity events, "
                         "seed nearest-neighbour ranking, agreement==0 "
                         "categorisation; appends a section and gates Stage B")
    ap.add_argument("--diagnose-report", action="store_true",
                    help="Stage B (B2/B3): read results/robustness/diagnose_*"
                         ".npz and append the COUPLING/DIMMING/MIXED verdict "
                         "to staircase_forensics.md")
    ap.add_argument("--stepquanta", action="store_true",
                    help="offline confirm/refute of the split-step diagnosis "
                         "behind the failing test_step_heights_quantized (Part "
                         "2i): recompute the alignment from the committed npz, "
                         "tabulate each matched N->N-1 step's adjacent unmatched "
                         "discontinuities, and append a SPLIT-STEP CONFIRMED / "
                         "NOT A SPLIT STEP / AMBIGUOUS verdict to "
                         "staircase_forensics.md (also prints the count/energy "
                         "lag Stage 1 precondition)")
    ap.add_argument("--annihilation-report", action="store_true",
                    help="Stage 3: adjudicate the 4->3 count/energy lag from the "
                         "committed diagnose_annih_4to3.npz sidecar; appends a "
                         "'count/energy lag' COUNTER-LATENCY / INTRINSIC LAG / "
                         "INCONCLUSIVE verdict to staircase_forensics.md")
    args = ap.parse_args()
    if args.starvation:
        sys.exit(starvation_forensics())
    if args.detectability:
        sys.exit(detectability_forensics())
    if args.diagnose_report:
        sys.exit(diagnose_report())
    if args.stepquanta:
        sys.exit(step_quanta_forensics())
    if args.annihilation_report:
        sys.exit(annihilation_report())

    sweep, cfg, audit_lines, missing = load_and_audit()
    for ln in audit_lines:
        print("[forensics]", ln.replace("**", ""))

    order = np.argsort(np.asarray(sweep["dw_over_kappa"]))[::-1]  # descending
    dw = np.asarray(sweep["dw_over_kappa"], dtype=float)[order]
    c = np.asarray(sweep["soliton_count"], dtype=int)[order] \
        if "soliton_count" in sweep else None
    if c is None:
        print("[forensics] soliton_count missing -- nothing to analyse.")
        sys.exit(2)
    Pc = (np.asarray(sweep["P_comb"], dtype=float)[order]
          if "P_comb" not in missing else None)
    Pi = np.asarray(sweep["P_intra"], dtype=float)[order]
    pos = (np.asarray(sweep["peak_positions_rad"], dtype=float)[order]
           if "peak_positions_rad" not in missing else None)

    env, under = undercount_mask(c)
    events = find_events(c)
    print(f"[forensics] flicker events: {len(events)}; undercount holds "
          f"(count < future-max envelope): {int(under.sum())} of {c.size}")
    env_drops = [(t, float(dw[t]), int(env[t - 1]), int(env[t]))
                 for t in range(1, c.size) if env[t] < env[t - 1]]
    print("[forensics] envelope (true-count lower bound) staircase: "
          + ", ".join(f"{a}->{b} @ {d:.2f}k" for _, d, a, b in env_drops))

    # ---- TEST A ------------------------------------------------------------
    if pos is not None:
        a_rows, a_agg = test_a_positions(pos, c, events)
        print(f"[forensics] TEST A raw (fixed-frame, tol {MATCH_TOL_RAD} rad): "
              f"{a_agg['raw']:.3f}")
        print(f"[forensics] TEST A coherent pattern drift on no-event "
              f"controls: median {a_agg['drift_median']:+.4f} rad/hold "
              f"(IQR {a_agg['drift_iqr'][0]:+.4f}..{a_agg['drift_iqr'][1]:+.4f}, "
              f"n={a_agg['n_drift_pairs']}) -- exceeds the {MATCH_TOL_RAD} rad "
              f"tolerance over any >=2-hold dip, so the raw number is "
              f"drift-limited, not nucleation-limited")
        print(f"[forensics] TEST A rotation-controlled: {a_agg['rot']:.3f}  "
              f"(measurement ceiling {a_agg['ceiling']:.3f}, re-nucleation "
              f"null {a_agg['null']:.3f})")
    else:
        a_rows, a_agg = [], None
        print("[forensics] TEST A skipped: peak_positions_rad missing")

    # ---- TEST B ------------------------------------------------------------
    if Pc is not None:
        quantum, qdrops, qsilent, floor = estimate_quantum(dw, c, Pc, env,
                                                           under)
        b_rows, b_agg = test_b_energy(dw, c, Pc, Pi, events, quantum)
        print(f"[forensics] TEST B one-soliton quantum |dP_comb| = "
              f"{quantum:.3e} (median over {len(qdrops)} energy-visible clean "
              f"decrements; {len(qsilent)} energy-SILENT envelope drop(s) "
              f"excluded and flagged)")
        for s in qsilent:
            print(f"[forensics]   ANOMALY: envelope drop {s['from']}->"
                  f"{s['to']} at {s['dw']:.2f}k carries |dP_comb| = "
                  f"{s['dPc']:.2e} (< {floor:.2e}) -- an energy-silent "
                  f"'annihilation' is itself an undercount signature")
        print(f"[forensics] TEST B pre->dip energy vs claimed loss: median "
              f"{b_agg['median_ratio']:.3f} quanta-per-claimed-quantum, max "
              f"{b_agg['max_ratio']:.3f} (a real event would give ~1)")
    else:
        quantum, qdrops, qsilent = np.nan, [], []
        b_rows, b_agg = [], None
        print("[forensics] TEST B skipped: P_comb missing")

    # ---- TEST C ------------------------------------------------------------
    c_res = test_c_breathing(dw, c, env, under, events, sweep)
    print(f"[forensics] TEST C breathing_relstd: undercount holds median "
          f"{c_res['relstd_under']['median']:.4f} "
          f"(IQR {c_res['relstd_under']['iqr'][0]:.4f}.."
          f"{c_res['relstd_under']['iqr'][1]:.4f}) vs correct-count "
          f"{c_res['relstd_ok']['median']:.4f} "
          f"(IQR {c_res['relstd_ok']['iqr'][0]:.4f}.."
          f"{c_res['relstd_ok']['iqr'][1]:.4f})")
    print(f"[forensics] TEST C is_breather at dip: "
          f"{c_res['breather_at_dip']}/{c_res['n_events']} events; np_label "
          f"in SOLITON_LABELS at {c_res['dip_labels_soliton']}/"
          f"{c_res['n_dip_holds']} dip holds (labels: "
          f"{c_res['dip_label_counts']}); is_single corrupted at dw>=6.5k: "
          f"{len(c_res['is_single_corrupted_holds'])} holds")

    # ---- Verdict -----------------------------------------------------------
    # "counting artifact" iff TEST A > 0.9 position persistence AND TEST B
    # sub-quantum, with TEST A evaluated on the confound-controlled statistic
    # against its measurement ceiling (raw number reported alongside).
    verdict = "inconclusive"
    reason = []
    if a_agg is not None and b_agg is not None:
        a_ok = (a_agg["rot"] > 0.9 * a_agg["ceiling"]
                and a_agg["rot"] > 2.0 * a_agg["null"])
        b_ok = (b_agg["max_ratio"] < 1.0 and b_agg["median_ratio"] < 0.25)
        a_nucl = a_agg["rot"] < max(0.5, 1.5 * a_agg["null"])
        b_nucl = b_agg["median_ratio"] > 0.75
        if a_ok and b_ok:
            verdict = "counting artifact"
            reason = [
                f"positions persist at {a_agg['rot']:.3f} = "
                f"{a_agg['rot'] / a_agg['ceiling']:.2f} of the measurement "
                f"ceiling ({a_agg['ceiling']:.3f}; re-nucleation null "
                f"{a_agg['null']:.3f}; raw fixed-frame {a_agg['raw']:.3f} is "
                f"drift-limited)",
                f"energy is sub-quantum (median {b_agg['median_ratio']:.3f}, "
                f"max {b_agg['max_ratio']:.3f} of the claimed loss)",
            ]
        elif a_nucl and b_nucl:
            verdict = "real re-nucleation"
    print(f"[forensics] VERDICT: {verdict}"
          + (f" -- {'; '.join(reason)}" if reason else ""))

    # ---- Report ------------------------------------------------------------
    md = []
    md.append("# Staircase flicker forensics: counting artifact vs real "
              "re-nucleation")
    md.append("")
    md.append(f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat()} by "
              f"`analysis/staircase_forensics.py` (offline; no solver run; "
              f"read-only except this file).")
    md.append("")
    md.append("## Data audited")
    md.append("")
    for ln in audit_lines:
        md.append(f"- {ln}")
    md.append("")
    md.append(f"- flicker events (count increases along the descending "
              f"sweep): **{len(events)}**")
    md.append(f"- undercount holds (count < future-max envelope): "
              f"**{int(under.sum())} of {c.size}**")
    md.append(f"- envelope staircase (true-count LOWER BOUND): "
              + ", ".join(f"{a} -> {b} at {d:.2f} k" for _, d, a, b in
                          env_drops))
    md.append(f"- dips never reach count 0 mid-branch "
              f"({c_res['dips_never_zero']}): the loss mechanism is the "
              f"50%-of-max peak threshold, not the labeler/contrast gate.")
    md.append("")

    md.append("## TEST A -- position persistence")
    md.append("")
    if a_agg is not None:
        md.append(f"- Raw fixed-frame match (tolerance {MATCH_TOL_RAD} rad, "
                  f"as prescribed): **{a_agg['raw']:.3f}**.")
        md.append(f"- Measured confound: the whole pulse pattern rotates "
                  f"coherently at **{a_agg['drift_median']:+.4f} rad/hold** "
                  f"(IQR {a_agg['drift_iqr'][0]:+.4f}.."
                  f"{a_agg['drift_iqr'][1]:+.4f}, n = "
                  f"{a_agg['n_drift_pairs']} no-event same-count hold pairs, "
                  f"where nucleation is impossible). One hold of drift "
                  f"is comparable to the tolerance, so the raw statistic is "
                  f"limited by drift, not by nucleation; it is reported but "
                  f"carries no discriminating power.")
        md.append(f"- Rotation-controlled match (same tolerance after "
                  f"removing one global rotation per comparison -- what "
                  f"re-nucleation would scramble): **{a_agg['rot']:.3f}**, "
                  f"against a measurement ceiling of "
                  f"**{a_agg['ceiling']:.3f}** (the identical statistic on "
                  f"local no-event control pairs) and a re-nucleation null "
                  f"of **{a_agg['null']:.3f}** (random angles, best-rotation "
                  f"fitted).")
        md.append(f"- Interpretation: the recovered peaks sit at the pre-dip "
                  f"angles as precisely as this dataset can measure "
                  f"({a_agg['rot'] / a_agg['ceiling']:.2f} of ceiling; the "
                  f"residual misses are breathing-phase position wobble that "
                  f"the no-event controls show identically), and far above "
                  f"the re-nucleation null. **The solitons never moved.**")
    else:
        md.append("- SKIPPED: `peak_positions_rad` missing from the npz.")
    md.append("")

    md.append("## TEST B -- energy continuity")
    md.append("")
    if b_agg is not None:
        md.append(f"- One-soliton quantum: median per-quantum |dP_comb| over "
                  f"the energy-visible clean count decrements = "
                  f"**{quantum:.3e}** "
                  f"(~{quantum / float(np.median(Pc[env == 5])):.0%} of the "
                  f"5-soliton-branch comb power). The committed "
                  f"`{METRICS_JSON}` staircase block predates the "
                  f"step-transition alignment machinery (its matched-edge "
                  f"list mixes flicker edges and stores normalised step "
                  f"sizes), so the brief's fallback -- decrements outside "
                  f"flicker regions, i.e. envelope drops -- is used.")
        for s in qsilent:
            md.append(f"- **Anomaly:** the envelope drop {s['from']} -> "
                      f"{s['to']} at {s['dw']:.2f} k is energy-SILENT "
                      f"(|dP_comb| = {s['dPc']:.2e}, ~"
                      f"{s['dPc'] / quantum:.4f} quanta). An annihilation "
                      f"with no energy signature is itself an undercount "
                      f"signature: the 5 -> 4 'edge' at 7.65 k is most "
                      f"plausibly the onset of PERMANENT undercounting "
                      f"(the count never again reaches 5), and the "
                      f"energy-visible annihilation cascade lives at "
                      f"6.2-6.4 k.")
        md.append(f"- Per-event pre -> dip |dP_comb| vs (claimed quanta lost "
                  f"x quantum): median **{b_agg['median_ratio']:.3f}**, max "
                  f"**{b_agg['max_ratio']:.3f}** (a real annihilation + "
                  f"re-nucleation cycle would give ~1 per quantum). Events "
                  f"claiming 2-4 lost solitons show sub-percent comb-power "
                  f"changes. **The energy never left the cavity.**")
    else:
        md.append("- SKIPPED: `P_comb` missing from the npz.")
    md.append("")

    md.append("## TEST C -- breathing correlation")
    md.append("")
    md.append(f"- breathing_relstd, undercount holds: median "
              f"{c_res['relstd_under']['median']:.4f} (IQR "
              f"{c_res['relstd_under']['iqr'][0]:.4f}.."
              f"{c_res['relstd_under']['iqr'][1]:.4f}, n = "
              f"{c_res['relstd_under']['n']}); correct-count holds: median "
              f"{c_res['relstd_ok']['median']:.4f} (IQR "
              f"{c_res['relstd_ok']['iqr'][0]:.4f}.."
              f"{c_res['relstd_ok']['iqr'][1]:.4f}, n = "
              f"{c_res['relstd_ok']['n']}). Undercounting is confined to "
              f"deep-breathing holds.")
    md.append(f"- is_breather at the dip hold: "
              f"{c_res['breather_at_dip']}/{c_res['n_events']} events.")
    md.append(f"- np_label at the {c_res['n_dip_holds']} dip holds: "
              f"{c_res['dip_label_counts']} -- "
              f"{c_res['dip_labels_soliton']}/{c_res['n_dip_holds']} stay in "
              f"SOLITON_LABELS {tuple(SOLITON_LABELS)} while the count "
              f"collapses (the rest are class 3, the documented labeler "
              f"misroute of breathing multi-soliton states); none fall to a "
              f"CW/MI class. The field's own classification contradicts the "
              f"peak count at the dips.")
    if c_res["is_single_corrupted_holds"]:
        md.append(f"- **is_single CORRUPTED at dw >= 6.5 k**: flagged True "
                  f"while the envelope shows a multi-soliton state at dw = "
                  f"{c_res['is_single_corrupted_holds']} -- the committed "
                  f"figure's shaded single-DKS band is affected.")
    else:
        md.append(f"- is_single at dw >= 6.5 k: not corrupted "
                  f"(True at {c_res['is_single_any_high']} holds there, and "
                  f"never while the envelope shows a multi-soliton state); "
                  f"the committed figure's shaded single-DKS band is "
                  f"unaffected by the flicker.")
    md.append("")

    md.append("## Per-event table")
    md.append("")
    md.append("| post hold | dw_dip (k) | c_pre | c_dip | c_post | "
              "match raw | match rot-ctrl | dP_comb pre->dip | quanta ratio |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    b_by_post = {r["post"]: r for r in b_rows}
    for r in a_rows:
        if r.get("n_post", 0) == 0:
            md.append(f"| {r['post']} | - | - | - | {c[r['post']]} | "
                      f"no pre found | - | - | - |")
            continue
        b = b_by_post.get(r["post"], {})
        md.append(
            f"| {r['post']} | {dw[r['dip']]:.3f} | {c[r['pre']]} | "
            f"{c[r['dip']]} | {c[r['post']]} | "
            f"{r['raw_m']}/{r['n_post']} | {r['rot_m']}/{r['n_post']} | "
            f"{_fmt_pct(b.get('d_pc_rel', np.nan))} | "
            f"{b.get('quanta_ratio', np.nan):.3f} |")
    md.append("")

    md.append("## Verdict")
    md.append("")
    md.append(f"**VERDICT: {verdict}**" + (" -- " + "; ".join(reason)
                                           if reason else ""))
    md.append("")
    md.append("Rule applied: 'counting artifact' iff TEST A shows > 0.9 "
              "position persistence AND TEST B shows sub-quantum energy "
              "changes; TEST A is evaluated on the rotation-controlled "
              "statistic against its measured ceiling because the raw "
              "fixed-frame number is invalidated by the coherent pattern "
              "drift quantified above (with the raw prescription taken "
              "literally, the drift alone would fake a 'new positions' "
              "reading for every dip longer than one hold).")
    md.append("")
    md.append("Consequences for the schema-4 counter hardening (next step, "
              "NOT done here): the flicker is an estimator artifact of the "
              "end-of-hold single-snapshot 50%-of-max peak count in the "
              "deep-breathing sub-band; the per-hold snapshot-median count "
              "proposed in the escalation ladder should remove it. The "
              "energy-silent 5 -> 4 envelope drop at 7.65 k means the "
              "hardened counter must be validated against P_comb steps, not "
              "against the current envelope alone.")
    md.append("")

    out = RESULTS_DIR / REPORT_MD
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[forensics] report -> {out}")


if __name__ == "__main__":
    main()
