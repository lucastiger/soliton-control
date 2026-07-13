"""Committed-artifact validation for the multi-soliton staircase pipeline.

FAST tier, NO solver: every test here reads the COMMITTED artifacts under
``analysis/results/`` -- the schema-4 sweep npz, the robustness variant npzs,
the metrics / provenance JSONs and the single-soliton spectrum npz -- and pins
them to each other and to the raw data.  A regenerated npz with a stale JSON,
a silently edited threshold, or a broken provenance chain fails here.

The suite PINS behaviour; it never retunes it.  If any committed artifact
fails a test in this file that is a FINDING: report which artifact and which
assertion -- do not adjust the test, any library threshold, or the artifact to
force a pass (the single authorized artifact touch was the one-time
``sweep_npz_sha256`` provenance refresh via the driver's ``--render-only``
path).

Conventions: "ascending" means ascending detuning; "descending sweep order"
means the physical sweep direction (high -> low detuning).  All step / edge
indices follow the :func:`analysis.spectral_metrics.detect_power_steps`
convention on the ascending grid (edge ``i`` joins ascending samples ``i`` and
``i + 1``; ``step_dy = y[i+1] - y[i]``), so a power DROP along the descending
sweep is a POSITIVE ascending ``step_dy``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from analysis.run_detuning_sweep import NPZ_SCHEMA_VERSION, load_sweep_npz
from analysis.spectral_metrics import (
    DEFAULT_STEP_K,
    detect_power_steps,
    hold_window_average,
    match_steps_to_transitions,
    soliton_count_transitions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "analysis" / "results"

# committed artifact -> the script that regenerates it (named in skip messages)
REQUIRED_ARTIFACTS = (
    ("detuning_sweep.npz", "analysis/run_detuning_sweep.py"),
    ("spectral_metrics.json",
     "analysis/run_detuning_sweep.py (soliton_step / staircase_robustness) + "
     "analysis/compute_spectral_metrics.py (three_db_span / "
     "conversion_efficiency)"),
    ("dks_artifact_provenance.json", "scripts/regenerate_dks_artifacts.py"),
    ("dks_single_soliton_spectrum.npz", "scripts/regenerate_dks_artifacts.py"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def art():
    """All committed artifacts, or a skip naming the missing file + generator.

    Skipping (not failing) keeps partial checkouts usable; a FULL checkout in
    which any assertion below fails is a finding.
    """
    for name, script in REQUIRED_ARTIFACTS:
        if not (RESULTS / name).exists():
            pytest.skip(f"missing committed artifact analysis/results/{name} "
                        f"(regenerate with: {script})")
    variant_paths = sorted((RESULTS / "robustness").glob("variant_*.npz"))
    if not variant_paths:
        pytest.skip("missing committed artifacts analysis/results/robustness/"
                    "variant_*.npz (regenerate with: "
                    "analysis/run_detuning_sweep.py --robustness)")
    sweep, cfg = load_sweep_npz(RESULTS / "detuning_sweep.npz")
    with open(RESULTS / "spectral_metrics.json") as f:
        metrics = json.load(f)
    with open(RESULTS / "dks_artifact_provenance.json") as f:
        provenance = json.load(f)
    return {
        "sweep": sweep,
        "cfg": cfg,
        "raw_npz": np.load(RESULTS / "detuning_sweep.npz",
                           allow_pickle=False),
        "metrics": metrics,
        "provenance": provenance,
        "variants": [(p, *load_sweep_npz(p)) for p in variant_paths],
    }


@pytest.fixture(scope="module")
def recomputed(art):
    """The 2b recomputation, shared: detector + transitions + alignment on the
    plotted primary, all recomputed from the RAW npz arrays exactly as
    ``render_and_report`` does (sort ascending, normalise to the trace max,
    stock ``detect_power_steps``, ``soliton_count_transitions``,
    ``match_steps_to_transitions``)."""
    block = art["metrics"]["soliton_step"]
    stair = block["staircase"]
    primary = stair["primary_observable"]["chosen"]     # read, never assumed
    assert primary in ("P_intra", "P_comb"), primary

    sweep = art["sweep"]
    order = np.argsort(sweep["dw_over_kappa"])
    dwk = np.asarray(sweep["dw_over_kappa"])[order]
    y = np.asarray(sweep[primary], dtype=np.float64)[order]
    y = y / float(np.max(y))
    counts = np.asarray(sweep["soliton_count"], dtype=int)[order]

    k_json = block["power_trace_discontinuities"].get("k")
    k_used = float(k_json) if k_json is not None else DEFAULT_STEP_K
    steps = detect_power_steps(dwk, y, k=k_used)
    transitions = soliton_count_transitions(dwk, counts)
    tol = int(stair.get("match_tol_samples", 1))
    align = match_steps_to_transitions(steps, transitions, tol_samples=tol)
    return {
        "primary": primary, "dwk": dwk, "y": y, "counts": counts,
        "k_used": k_used, "k_json": k_json, "steps": steps,
        "transitions": transitions, "align": align, "tol": tol,
        "sigma": float(steps["sigma"]),
    }


# ---------------------------------------------------------------------------
# 2a
# ---------------------------------------------------------------------------
def test_schema_and_validation_flag(art):
    """Schema v4, validated flag, parseable config, seeding + counting
    provenance.  The snapshot cadence is deliberately NOT a stored field --
    it is the driver's ``hold_rt // 32`` rule, checked by consistency in
    ``test_count_quality_and_agreement_consistency``."""
    d = art["raw_npz"]
    assert int(d["schema_version"]) == NPZ_SCHEMA_VERSION == 4
    assert bool(d["staircase_validated"]) is True

    cfg_dict = json.loads(str(d["sweep_config_json"]))     # must parse
    assert isinstance(cfg_dict, dict) and cfg_dict["n_solitons"] >= 2

    # seeding provenance: shapes and dtypes
    assert np.issubdtype(d["n_solitons_seeded"].dtype, np.integer)
    assert np.issubdtype(d["position_seed"].dtype, np.integer)
    assert np.issubdtype(d["position_jitter_frac"].dtype, np.floating)
    n_seeded = int(d["n_solitons_seeded"])
    assert d["seed_positions_rad"].shape == (n_seeded,)
    assert np.all(np.isfinite(d["seed_positions_rad"]))

    # counting provenance sub-block in the staircase JSON block
    counting = art["metrics"]["soliton_step"]["staircase"]["counting"]
    assert counting["method"] == "position_persistence_v2_physics_anchor"
    params = counting["parameters"]
    for key in ("soliton_frac", "bg_floor_multiple", "min_persistence"):
        assert isinstance(params[key], (int, float)), (key, params.get(key))


# ---------------------------------------------------------------------------
# 2b
# ---------------------------------------------------------------------------
def test_alignment_reproduces_from_raw(art, recomputed):
    """The committed figure + JSON are pinned to the RAW npz: recomputing the
    detector, the count transitions and the alignment from the stored arrays
    must reproduce the committed matched / unmatched sets exactly.  A
    regenerated npz with a stale JSON fails here."""
    stair = art["metrics"]["soliton_step"]["staircase"]
    block = art["metrics"]["soliton_step"]

    # the JSON's recorded detector k is what the recomputation used
    assert float(block["power_trace_discontinuities"]["k"]) \
        == recomputed["k_used"]
    # the recorded trace is the primary the recomputation ran on
    assert block["power_trace_discontinuities"]["trace"] \
        == recomputed["primary"]
    assert float(block["power_trace_discontinuities"]["robust_sigma"]) \
        == pytest.approx(recomputed["sigma"], rel=1e-12)

    align = recomputed["align"]
    assert len(align["matched"]) >= 2            # the hard gate's floor

    committed = stair["matched_steps"]
    assert len(align["matched"]) == len(committed)
    for got, exp in zip(align["matched"], committed):
        assert got["step_edge_index"] == exp["step_edge_index"]
        assert got["transition_edge_index"] == exp["transition_edge_index"]
        assert got["delta_n"] == exp["delta_n"]              # exact
        assert got["n_high_side"] == exp["n_high_side"]
        assert got["n_low_side"] == exp["n_low_side"]
        assert got["dw_mid"] == pytest.approx(exp["dw_mid"], rel=1e-12)
        assert got["step_dy"] == pytest.approx(exp["step_dy"], rel=1e-9)

    committed_um = stair["unmatched_steps"]
    assert len(align["unmatched_steps"]) == len(committed_um)
    for got, exp in zip(align["unmatched_steps"], committed_um):
        assert got["edge_index"] == exp["edge_index"]
        assert got["step_x"] == pytest.approx(exp["dw_mid"], rel=1e-12)
        assert got["step_dy"] == pytest.approx(exp["step_dy"], rel=1e-9)

    committed_ut = stair["unmatched_transitions"]
    assert len(align["unmatched_transitions"]) == len(committed_ut)
    for got, exp in zip(align["unmatched_transitions"], committed_ut):
        assert got["edge_index"] == exp["edge_index"]
        assert got["n_high_side"] == exp["n_high_side"]
        assert got["n_low_side"] == exp["n_low_side"]
        assert got["dw_mid"] == pytest.approx(exp["dw_mid"], rel=1e-12)


# ---------------------------------------------------------------------------
# 2c
# ---------------------------------------------------------------------------
def test_monotone_annihilation(art):
    """``soliton_count`` is monotone non-increasing along the DESCENDING sweep
    (equivalently non-decreasing in ascending detuning): on a warm-continued
    down-sweep solitons only annihilate.

    ``soliton_count_end_snapshot`` must be PRESENT but is explicitly NOT
    required monotone: it is the retained legacy end-of-hold single-snapshot
    diagnostic, whose breathing-phase dropouts are the documented defect
    (forensics 4-F, "counting artifact") that motivated the windowed counter.
    Its non-monotonicity is the preserved evidence, not a bug."""
    sweep = art["sweep"]
    order = np.argsort(sweep["dw_over_kappa"])
    counts = np.asarray(sweep["soliton_count"], dtype=int)[order]
    assert np.all(np.diff(counts) >= 0), (
        "soliton_count increases going down in detuning -- the hardened "
        "windowed count regressed to the flickering legacy behaviour")
    assert "soliton_count_end_snapshot" in sweep
    assert np.issubdtype(
        np.asarray(sweep["soliton_count_end_snapshot"]).dtype, np.integer)


# ---------------------------------------------------------------------------
# 2d
# ---------------------------------------------------------------------------
def test_regime_classification(art, recomputed):
    """Stationary upper branch, breather sub-band, matched steps inside it.

    The hold nearest 10 kappa must be STATIONARY with soliton_count ==
    n_solitons_seeded: this sweep never visits N = 1 at 10 kappa -- the
    single-soliton 10-kappa state belongs to the SPECTRUM path
    (scripts/regenerate_dks_artifacts.py / dks_single_soliton_spectrum.npz),
    not to the staircase sweep, and the two must not be conflated."""
    sweep = art["sweep"]
    order = np.argsort(sweep["dw_over_kappa"])
    dwk = np.asarray(sweep["dw_over_kappa"])[order]
    counts = np.asarray(sweep["soliton_count"], dtype=int)[order]
    is_stat = np.asarray(sweep["is_stationary"]).astype(bool)[order]
    is_brth = np.asarray(sweep["is_breather"]).astype(bool)[order]

    # every hold at dw >= 9.5 kappa is stationary
    assert np.all(is_stat[dwk >= 9.5]), (
        f"non-stationary hold(s) at dw >= 9.5k: "
        f"{dwk[(dwk >= 9.5) & ~is_stat]}")

    # at least one breather hold in the 6.5-9.0 kappa sub-band
    assert np.any(is_brth[(dwk >= 6.5) & (dwk <= 9.0)])

    # every matched step's dw_mid inside the npz's OWN breather sub-band,
    # guarded by half a grid step on each side
    assert is_brth.any()
    half_step = 0.5 * float(np.min(np.diff(dwk)))
    lo = float(dwk[is_brth].min()) - half_step - 1e-9
    hi = float(dwk[is_brth].max()) + half_step + 1e-9
    for m in recomputed["align"]["matched"]:
        assert lo <= m["dw_mid"] <= hi, (
            f"matched step at {m['dw_mid']}k outside the breather sub-band "
            f"[{lo}, {hi}]k derived from the is_breather flags")

    # the hold nearest 10 kappa: stationary, all seeded solitons alive
    i10 = int(np.argmin(np.abs(dwk - 10.0)))
    assert is_stat[i10]
    assert counts[i10] == int(art["sweep"]["n_solitons_seeded"])


# ---------------------------------------------------------------------------
# 2e
# ---------------------------------------------------------------------------
def test_final_edge_structural(art, recomputed):
    """Exactly one 1 -> 0 transition, inside [5.75, 6.5] kappa, with its
    matched/unmatched status RECORDED in the JSON (which one is data-decided
    per the honesty constraint -- never gated here).

    Sign convention: ``step_dy`` is stored on the ascending-detuning grid
    (``detect_power_steps``: ``dy = y[i+1] - y[i]``), so the collapse of the
    comb power at the last annihilation -- a power DROP along the physical
    DESCENDING sweep -- appears as a POSITIVE ascending ``step_dy``; the
    descending-direction step is its negation."""
    one_to_zero = [t for t in recomputed["transitions"]
                   if t["n_high_side"] >= 1 and t["n_low_side"] == 0]
    assert len(one_to_zero) == 1, one_to_zero
    edge = one_to_zero[0]
    assert 5.75 <= edge["dw_mid"] <= 6.5, edge

    stair = art["metrics"]["soliton_step"]["staircase"]
    matched_rec = [m for m in stair["matched_steps"]
                   if m["transition_edge_index"] == edge["edge_index"]]
    unmatched_rec = [t for t in stair["unmatched_transitions"]
                     if t["edge_index"] == edge["edge_index"]]
    # the status must be recorded exactly once, one way or the other
    assert len(matched_rec) + len(unmatched_rec) == 1, (
        "the 1->0 edge's matched/unmatched status is not recorded in the "
        "staircase JSON block")

    if matched_rec:
        (m,) = matched_rec
        step_dy_descending = -float(m["step_dy"])   # physical sweep direction
        assert step_dy_descending < 0, (
            "matched 1->0 edge must be a power DROP along the descending "
            f"sweep; got ascending step_dy = {m['step_dy']}")
        assert abs(m["step_dy"]) > 3.0 * recomputed["sigma"]
        note = stair["final_edge_note"].lower()
        # observable dependence: real step on the pump-excluded comb power,
        # muted on total intracavity power by the comparable-energy MI comb
        assert "comb power" in note
        assert "intracavity power" in note


# ---------------------------------------------------------------------------
# 2f
# ---------------------------------------------------------------------------
def _n_in_window(cfg) -> int:
    """In-window snapshot count implied by a sweep config.

    Mirrors the driver's cadence expression ``snap_int = max(hold_rt // 32,
    1)`` (analysis/run_detuning_sweep.py, run_detuning_sweep()) and the
    solver's snapshot layout ``n_snap = ceil(t_slow / snapshot_interval)``
    with snapshots at round trips ``0, snap_int, 2*snap_int, ...``
    (simulator/lle_solver.py, ``n_snapshots = (t_slow + snapshot_interval - 1)
    // snapshot_interval``); the window start comes from the REAL
    ``hold_window_average``.  If the driver's cadence rule changes, this
    helper -- and the grid assertion built on it -- breaks loudly, which is
    the point."""
    hold_rt = int(cfg.hold_rt)
    snap_int = max(hold_rt // 32, 1)                 # the driver's rule
    n_snap = (hold_rt + snap_int - 1) // snap_int    # the solver's rule
    snap_rt = np.arange(n_snap) * snap_int
    i_start = hold_window_average(np.zeros(hold_rt),
                                  avg_frac=float(cfg.avg_frac))["i_start"]
    n_in = int((snap_rt >= i_start).sum())
    return max(n_in, 1)          # degenerate window: driver keeps the last


def _assert_agreement_on_grid(agreement, n_in, label):
    """Every stored count_agreement lies on {k/n_in : k = 0..n_in} to 1e-9."""
    a = np.asarray(agreement, dtype=float)
    assert np.all((a >= 0.0) & (a <= 1.0)), label
    off = np.abs(a * n_in - np.round(a * n_in)) / n_in
    bad = np.nonzero(off > 1e-9)[0]
    assert bad.size == 0, (
        f"{label}: {bad.size} count_agreement value(s) off the k/{n_in} grid "
        f"implied by the config + cadence rule, e.g. {a[bad[:5]]} -- the "
        f"stored agreements, the sweep config and the hold_rt // 32 cadence "
        f"are no longer mutually consistent")


def test_count_quality_and_agreement_consistency(art):
    """Count quality on soliton-bearing holds, then the self-consistency check
    that REPLACES the falsified snapshot-starvation signature (forensics 5-R:
    the counting window holds ~8 snapshots at cadence hold_rt // 32, so
    agreement is quantized in eighths -- healthy, not starved).  No particular
    n_in value is asserted: the invariant is that the stored agreement values,
    the stored config, and the cadence rule agree with EACH OTHER."""
    sweep, cfg = art["sweep"], art["cfg"]
    order = np.argsort(sweep["dw_over_kappa"])
    counts = np.asarray(sweep["soliton_count"], dtype=int)[order]
    agree = np.asarray(sweep["count_agreement"], dtype=float)[order]

    sb = counts >= 1
    assert sb.any()
    assert float(np.median(agree[sb])) >= 0.5
    assert not np.any(agree[sb] == 0.0), (
        f"soliton-bearing hold(s) with count_agreement == 0 -- the "
        f"persistence machinery, not the physics, is making the count "
        f"there: dw = {np.asarray(sweep['dw_over_kappa'])[order][sb][agree[sb] == 0.0]}")

    _assert_agreement_on_grid(agree, _n_in_window(cfg), "detuning_sweep.npz")

    # each robustness variant against ITS OWN sweep_config_json (variant 2
    # has hold_rt 1600)
    for path, vsweep, vcfg in art["variants"]:
        _assert_agreement_on_grid(np.asarray(vsweep["count_agreement"],
                                             dtype=float),
                                  _n_in_window(vcfg), path.name)


# ---------------------------------------------------------------------------
# 2g
# ---------------------------------------------------------------------------
def test_robustness_block(art):
    """The three one-at-a-time perturbation variants all pass, in BOTH JSONs,
    and the pre-fix failed runs are preserved under "history" -- the record
    that two variants failed under the deprecated relative-threshold rule is
    part of the audit trail, never to be garbage-collected."""
    base = art["cfg"]
    for source_name, block in (
            ("spectral_metrics.json",
             art["metrics"].get("staircase_robustness")),
            ("dks_artifact_provenance.json",
             art["provenance"].get("staircase_robustness"))):
        assert block is not None, f"staircase_robustness missing in {source_name}"
        variants = block["variants"]
        assert len(variants) == 3, source_name
        for v in variants:
            assert v["pass"] is True, (source_name, v)
            assert len(v["matched_dw_mid_over_kappa"]) > 0, (source_name, v)
            assert float(v["median_count_agreement"]) >= 0.5, (source_name, v)
        history = block.get("history")
        assert history, f"{source_name}: robustness history is empty -- the " \
                        f"pre-fix failure record was dropped"
        assert any(v.get("pass") is False
                   for h in history for v in h.get("variants", [])), (
            f"{source_name}: history no longer contains a failed variant run")

    # the expected one-at-a-time perturbations, against the accepted config
    by_index = {v["index"]: v for v
                in art["metrics"]["staircase_robustness"]["variants"]}
    v1, v2, v3 = by_index[1], by_index[2], by_index[3]
    assert v1["config"]["position_seed"] == base.position_seed + 1
    assert (v1["config"]["hold_rt"], v1["config"]["n_steps"]) \
        == (base.hold_rt, base.n_steps)
    assert v2["config"]["hold_rt"] == 1600 and base.hold_rt == 2000
    assert (v2["config"]["position_seed"], v2["config"]["n_steps"]) \
        == (base.position_seed, base.n_steps)
    assert v3["config"]["n_steps"] == 2 * base.n_steps
    assert (v3["config"]["position_seed"], v3["config"]["hold_rt"]) \
        == (base.position_seed, base.hold_rt)


# ---------------------------------------------------------------------------
# 2h
# ---------------------------------------------------------------------------
def test_provenance_staleness_chain(art):
    """Regression guard for the stale-8k metrics bug (three_db_span and
    conversion_efficiency once silently computed from a superseded spectrum
    npz): every hash recorded in the JSON provenance must equal the sha256 of
    the committed input file it claims to describe."""
    # the exact key compute_spectral_metrics.py records the input hash under
    src = (REPO_ROOT / "analysis" / "compute_spectral_metrics.py").read_text()
    m = re.search(r'"([A-Za-z0-9_]*sha256[A-Za-z0-9_]*)":\s*'
                  r'_sha256\(args\.spectrum\)', src)
    assert m, ("compute_spectral_metrics.py no longer records the input "
               "spectrum's hash -- the staleness chain is broken")
    hash_key = m.group(1)

    spectrum_sha = _sha256(RESULTS / "dks_single_soliton_spectrum.npz")
    for block_name in ("three_db_span", "conversion_efficiency"):
        prov = art["metrics"][block_name]["provenance"]
        assert prov.get(hash_key) == spectrum_sha, (
            f"{block_name}.provenance.{hash_key} does not match the committed "
            f"dks_single_soliton_spectrum.npz -- the block was computed from "
            f"a superseded spectrum (the stale-8k bug)")

    # the staircase block pins the sweep npz the same way
    sweep_sha = _sha256(RESULTS / "detuning_sweep.npz")
    recorded = art["metrics"]["soliton_step"]["provenance"].get(
        "sweep_npz_sha256")
    assert recorded == sweep_sha, (
        "soliton_step.provenance.sweep_npz_sha256 does not match the "
        "committed detuning_sweep.npz -- refresh the JSON from the committed "
        "npz via analysis/run_detuning_sweep.py --render-only")

    # spectrum-path operating point pins
    prov = art["provenance"]
    assert prov["settle"]["dw_kappa"] == 10.0
    assert prov["measured"]["stationarity"] == "stationary"


# ---------------------------------------------------------------------------
# 2i
# ---------------------------------------------------------------------------
def test_step_heights_quantized(recomputed):
    """The steps are soliton energy quanta: over the matched N -> N-1 steps
    (delta_n == 1), EXCLUDING the 1 -> 0 edge (which includes background
    reorganization and is covered by test_final_edge_structural), all
    |step_dy| values agree pairwise within a factor of 2 and each exceeds
    5x the detector's robust sigma.  Near-equal heights well above the
    plateau ripple are what distinguish a physical staircase from detector
    noise."""
    quanta = [(m["dw_mid"], abs(float(m["step_dy"])))
              for m in recomputed["align"]["matched"]
              if m["delta_n"] == 1 and m["n_low_side"] >= 1]
    assert quanta, "no matched N -> N-1 steps above the 1->0 edge"

    sigma = recomputed["sigma"]
    for dw_mid, h in quanta:
        assert h > 5.0 * sigma, (
            f"matched step at {dw_mid}k: |step_dy| = {h} is within 5x the "
            f"robust sigma {sigma} -- indistinguishable from plateau ripple")
    for i, (dw_i, h_i) in enumerate(quanta):
        for dw_j, h_j in quanta[i + 1:]:
            ratio = max(h_i, h_j) / min(h_i, h_j)
            assert ratio <= 2.0, (
                f"matched N -> N-1 step heights are not quantized: "
                f"|step_dy| = {h_i} at {dw_i}k vs {h_j} at {dw_j}k "
                f"(ratio {ratio:.2f} > 2)")
