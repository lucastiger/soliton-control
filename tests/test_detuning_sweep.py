"""Minimal integration test for the detuning-sweep driver (runs the solver).

This is the SLOW-tier counterpart to the pure-synthetic unit tests in
``tests/test_soliton_steps.py`` (which never touch the solver).  It mirrors the
plumbing-only style of ``tests/test_adiabatic_sweeps.py``: a *reduced-length*
warm-continuation MULTI-SOLITON sweep (few steps, short holds, small ``n_tau``,
N = 3 seeds) just deep enough to exercise the driver end-to-end -- deterministic
multi-soliton seeding with the settled-peak-count assertion, continuation,
linear-power hold-window averaging, per-hold comb power / peak positions /
breathing fields, npz round-trip, and the staircase / step helpers -- without
asserting the full production physics.  It writes only to ``tmp_path`` so the
committed ``analysis/results`` artifacts are never clobbered.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.dks_access import attach_dispersion, load_cavity_params
from analysis.run_detuning_sweep import (
    ESCALATION_LADDER,
    NPZ_SCHEMA_VERSION,
    ROBUSTNESS_COUNT_AGREEMENT_MEDIAN_FLOOR,
    ROBUSTNESS_MUTED_EDGE_WINDOW_KAPPA,
    ROBUSTNESS_STATIONARY_EDGE_KAPPA,
    SOLITON_LABELS,
    StaircaseValidationError,
    SweepConfig,
    analyze_sweep_staircase,
    assess_robustness_variant,
    load_sweep_npz,
    matched_step_contrast,
    robustness_variant_specs,
    run_detuning_sweep,
    save_sweep_npz,
    staircase_transition_edges,
    validate_staircase_alignment,
    write_noise_off_config,
)
from analysis.spectral_metrics import (
    detect_power_steps,
    match_steps_to_transitions,
    single_dks_region,
    soliton_count_transitions,
)

# Reduced-length sweep: enough to settle 3 seeded solitons and take a couple of
# warm-continuation steps; small n_tau keeps it CI-cheap.
CFG = SweepConfig(dw_start_kappa=10.0, dw_stop_kappa=8.0, n_steps=3,
                  settle_rt=1500, hold_rt=800, n_tau=1024, n_solitons=3)


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    cav = attach_dispersion(load_cavity_params(), CFG.n_tau)
    noise_cfg = write_noise_off_config()
    try:
        out = run_detuning_sweep(cav, CFG, config_path=noise_cfg)
    finally:
        noise_cfg.unlink(missing_ok=True)
    return out


def test_sweep_arrays_shape_and_finiteness(sweep):
    for key in ("dw_over_kappa", "P_intra", "P_intra_std", "P_trans",
                "U_int", "is_single", "np_label", "n_peaks",
                "P_comb", "P_comb_std", "soliton_count", "contrast",
                "soliton_count_end_snapshot", "count_agreement",
                "is_breather", "is_stationary", "breathing_relstd"):
        assert key in sweep, key
        assert np.asarray(sweep[key]).shape == (CFG.n_steps,)
    for key in ("dw_over_kappa", "P_intra", "P_trans", "U_int", "P_comb"):
        assert np.all(np.isfinite(sweep[key]))
    # detunings are the requested decreasing grid
    assert np.allclose(sweep["dw_over_kappa"], np.linspace(10.0, 8.0, 3))
    # intracavity power is positive and transmission below the pump power
    assert np.all(sweep["P_intra"] > 0)
    assert np.all(sweep["P_trans"] < CFG.pin_w)


def test_multi_soliton_observables(sweep):
    # comb power is positive and strictly below the total (the pump line is out)
    assert np.all(sweep["P_comb"] > 0)
    assert np.all(sweep["P_comb"] < sweep["P_intra"])
    # the settle assertion guarantees N seeds survived; on this short in-branch
    # sweep the peak count stays positive and bounded by N
    assert np.all(sweep["n_peaks"] >= 1)
    assert np.all(sweep["soliton_count"] >= 0)
    assert np.all(sweep["soliton_count"] <= CFG.n_solitons)
    # peak positions: NaN-padded (n_steps, max) PERSISTENT-CLUSTER angles
    pos = sweep["peak_positions_rad"]
    assert pos.ndim == 2 and pos.shape[0] == CFG.n_steps
    finite = pos[np.isfinite(pos)]
    assert np.all((finite >= 0.0) & (finite < 2.0 * np.pi))
    # per-row finite-position count == the hardened soliton_count
    assert np.array_equal(np.isfinite(pos).sum(axis=1),
                          sweep["soliton_count"])
    # F5 contract: the settled seed's hardened count survives the first hold
    # and the windowed counter is self-consistent there
    assert sweep["soliton_count"][0] == CFG.n_solitons
    assert sweep["count_agreement"][0] >= 0.5
    assert np.all((sweep["count_agreement"] >= 0.0)
                  & (sweep["count_agreement"] <= 1.0))


def test_noise_off_config_zeroes_temperature():
    import yaml
    p = write_noise_off_config()
    try:
        cfg = yaml.safe_load(p.read_text())["physical_parameters"]
        assert float(cfg["T_k"]) == 0.0
    finally:
        p.unlink(missing_ok=True)


def test_npz_roundtrip_preserves_arrays_and_config(sweep, tmp_path):
    path = tmp_path / "detuning_sweep.npz"
    save_sweep_npz(path, sweep, CFG)
    loaded, cfg2 = load_sweep_npz(path)
    assert cfg2 == CFG
    assert np.allclose(loaded["P_intra"], sweep["P_intra"])
    assert np.allclose(loaded["P_comb"], sweep["P_comb"])
    assert np.array_equal(loaded["soliton_count"], sweep["soliton_count"])
    assert np.allclose(loaded["peak_positions_rad"],
                       sweep["peak_positions_rad"], equal_nan=True)
    assert loaded["is_single"].dtype == bool
    assert loaded["is_breather"].dtype == bool
    assert loaded["is_stationary"].dtype == bool
    assert float(loaded["kappa_rad_s"]) == pytest.approx(sweep["kappa_rad_s"])


def test_region_and_step_helpers_run_on_real_sweep(sweep):
    dwk = sweep["dw_over_kappa"]
    order = np.argsort(dwk)
    P = sweep["P_intra"][order]
    # helpers must run without error on the real (short) trace
    lo, hi, annih = single_dks_region(dwk[order], sweep["is_single"][order])
    steps = detect_power_steps(dwk[order], P / P.max())
    assert steps["n_steps"] >= 0
    # this reduced sweep sits on the soliton branch (8-10 kappa), so the
    # region helper must at least not crash -- assert the contract, not a
    # specific band.
    assert (lo is None) or (lo <= hi)


# ---------------------------------------------------------------------------
# Pure-synthetic staircase-helper tests (no solver)
# ---------------------------------------------------------------------------
def test_soliton_labels_taxonomy():
    # 4 = multi-soliton, 5 = soliton crystal, 6 = single soliton
    assert SOLITON_LABELS == (4, 5, 6)


def test_staircase_transition_edges_excludes_final_annihilation():
    counts = [0, 0, 1, 1, 3, 3, 5, 5]        # ascending detuning
    all_tr, matched = staircase_transition_edges(counts)
    assert all_tr == [1, 3, 5]               # 0->1, 1->3, 3->5
    assert matched == [3, 5]                 # the ->0 edge (i=1) is excluded


def test_matched_step_contrast_detected_and_matched_step_wins():
    # staircase with two genuine steps on a gently rippling baseline
    x = np.arange(12, dtype=float)
    y = np.concatenate([np.zeros(4), np.full(4, 1.0), np.full(4, 2.2)])
    y = y + 0.001 * np.sin(x)
    counts = np.concatenate([np.ones(4), 2 * np.ones(4), 3 * np.ones(4)])
    steps = detect_power_steps(x, y, k=6.0)
    all_tr, matched = staircase_transition_edges(counts)
    res = matched_step_contrast(y, steps, matched)
    assert res["matched_detected_edges"]      # both steps detected & matched
    assert res["contrast"] > 6.0              # far above the ripple MAD


def test_matched_step_contrast_zero_when_no_matched_detection():
    # smooth ramp: the detector finds nothing, so the contrast must be 0
    x = np.arange(10, dtype=float)
    y = 0.1 * x
    counts = np.concatenate([np.ones(5), 2 * np.ones(5)])
    steps = detect_power_steps(x, y, k=6.0)
    _, matched = staircase_transition_edges(counts)
    res = matched_step_contrast(y, steps, matched)
    assert res["contrast"] == 0.0
    assert res["matched_detected_edges"] == []


# ---------------------------------------------------------------------------
# Synthetic full-schema sweep (no solver): npz v3 round-trip, the validation
# gate, and the render/abort paths
# ---------------------------------------------------------------------------
def _synthetic_staircase_sweep(*, monotone=True, n_power_steps=2):
    """A full-schema fake sweep dict + config, in sweep (descending) order.

    Ascending-detuning design (40 samples, 5.5 -> 12 kappa): counts
    0,0,...,1,1,...,2,2,...,3 in four 10-sample plateaus (transition edges 9,
    19, 29).  Power: the 0->1 edge is power-MUTED (equal plateau levels, like
    the real MI-comb final annihilation), the 1->2 and 2->3 edges carry clear
    jumps, and sample 0 carries an MI/CW-rise discontinuity with NO count
    change.  ``monotone=False`` dips one sample of the N=2 plateau to 1
    (breaks the descending monotonicity like the real deep-breathing
    undercount); ``n_power_steps=1`` mutes the 2->3 jump as well (starves the
    matched count).
    """
    n = 40
    dw = np.linspace(5.5, 12.0, n)               # ascending
    counts = np.repeat([0, 1, 2, 3], 10).astype(np.int64)
    if not monotone:
        counts[25] = 1                           # transient undercount dip
    levels = {0: 0.55, 1: 0.55, 2: 0.75, 3: 1.0}     # 0->1 muted
    if n_power_steps == 1:
        levels[3] = levels[2]                    # 2->3 muted too
    y = np.array([levels[min(c, 3)] for c in np.repeat([0, 1, 2, 3], 10)])
    y = y + 0.001 * np.sin(np.arange(n))
    y[0] = 0.9                                   # near-resonance MI/CW rise

    kappa = 1.5e8
    sweep = {
        "dw_over_kappa": dw,
        "dw_rad_s": dw * kappa,
        "dw_eff_over_kappa": dw.copy(),
        "P_intra": y.copy(),
        "P_intra_std": np.full(n, 0.002),
        "U_int": y * 1e-21,
        "U_int_std": np.full(n, 1e-24),
        "P_trans": np.full(n, 0.19),
        "P_trans_std": np.full(n, 1e-4),
        "np_label": np.where(counts == 1, 6,
                             np.where(counts > 1, 4, 3)).astype(np.int64),
        "n_peaks": counts.copy(),
        "is_single": counts == 1,
        "kappa_rad_s": kappa,
        "t_r_s": 6.4e-11,
        "fsr_hz": 15.6e9,
        "P_comb": y.copy(),
        "P_comb_std": np.full(n, 0.003),
        "soliton_count": counts,
        "contrast": np.where(counts >= 1, 10.0, 1.5),
        "soliton_count_end_snapshot": np.maximum(counts - (np.arange(n) % 7 == 3),
                                                 0).astype(np.int64),
        "count_agreement": np.where(np.arange(n) % 7 == 3, 0.75, 1.0),
        "count_min_persistence": 0.5,
        "count_rel_height_candidate": 0.25,
        "count_bg_floor_multiple": 5.0,
        "is_breather": np.zeros(n, dtype=bool),
        "is_stationary": np.ones(n, dtype=bool),
        "breathing_relstd": np.full(n, 1e-4),
        "breathing_period_rt": np.full(n, np.nan),
        "seed_metrics": {"n_peaks": 3,
                         "peak_positions_rad": np.array([0.5, 2.6, 4.7]),
                         "u_int_final": 1e-21},
    }
    pos = np.full((n, 3), np.nan)
    for i, c in enumerate(counts):
        pos[i, :c] = np.linspace(0.5, 4.7, 3)[:c]
    sweep["peak_positions_rad"] = pos
    # store in sweep (descending) order, as the driver does
    for k, v in sweep.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            sweep[k] = v[::-1].copy()
    cfg = SweepConfig(dw_start_kappa=12.0, dw_stop_kappa=5.5, n_steps=n,
                      n_solitons=3)
    return sweep, cfg


def test_npz_v4_roundtrip_synthetic(tmp_path):
    sweep, cfg = _synthetic_staircase_sweep()
    path = tmp_path / "detuning_sweep.npz"
    save_sweep_npz(path, sweep, cfg, staircase_validated=True)
    loaded, cfg2 = load_sweep_npz(path)
    assert cfg2 == cfg
    # v4 stamp + seeding provenance
    assert loaded["schema_version"] == NPZ_SCHEMA_VERSION == 4
    # v4 counter provenance + validation flag round-trip
    assert loaded["staircase_validated"] is True
    assert np.issubdtype(loaded["soliton_count_end_snapshot"].dtype,
                         np.integer)
    assert np.allclose(loaded["count_agreement"], sweep["count_agreement"])
    assert loaded["count_min_persistence"] == 0.5
    assert loaded["count_rel_height_candidate"] == 0.25
    assert loaded["count_bg_floor_multiple"] == 5.0
    assert loaded["n_solitons_seeded"] == 3
    assert loaded["position_seed"] == cfg.position_seed
    assert loaded["position_jitter_frac"] == cfg.position_jitter_frac
    assert loaded["seed_positions_rad"].shape == (3,)
    assert np.allclose(loaded["seed_positions_rad"], [0.5, 2.6, 4.7])
    # dtypes survive the round trip
    assert np.issubdtype(loaded["soliton_count"].dtype, np.integer)
    assert np.issubdtype(loaded["np_label"].dtype, np.integer)
    for k in ("is_single", "is_breather", "is_stationary"):
        assert loaded[k].dtype == bool
    # NaN padding intact
    assert np.allclose(loaded["peak_positions_rad"],
                       sweep["peak_positions_rad"], equal_nan=True)
    assert np.array_equal(np.isfinite(loaded["peak_positions_rad"]),
                          np.isfinite(sweep["peak_positions_rad"]))
    # v1/v2 arrays unchanged
    for k in ("dw_over_kappa", "P_intra", "P_comb", "P_comb_std"):
        assert np.allclose(loaded[k], sweep[k])
    # a re-save of the LOADED dict (no seed_metrics) keeps the seed positions
    path2 = tmp_path / "resaved.npz"
    save_sweep_npz(path2, loaded, cfg2)
    reloaded, _ = load_sweep_npz(path2)
    assert np.allclose(reloaded["seed_positions_rad"], [0.5, 2.6, 4.7])
    assert reloaded["staircase_validated"] is False   # not asserted -> False


def test_loader_accepts_schema_3_file_without_v4_keys(tmp_path):
    # simulate a committed pre-v4 npz: save v4, strip the v4 keys, restamp v3
    import json as _json

    sweep, cfg = _synthetic_staircase_sweep()
    path4 = tmp_path / "v4.npz"
    save_sweep_npz(path4, sweep, cfg, staircase_validated=True)
    d = dict(np.load(path4, allow_pickle=False))
    for k in ("soliton_count_end_snapshot", "count_agreement",
              "count_min_persistence", "count_rel_height_candidate",
              "count_bg_floor_multiple", "staircase_validated"):
        d.pop(k)
    d["schema_version"] = 3
    path3 = tmp_path / "v3.npz"
    np.savez_compressed(path3, **d)
    loaded, cfg3 = load_sweep_npz(path3)          # must not raise
    assert cfg3 == cfg
    assert loaded["schema_version"] == 3
    for k in ("soliton_count_end_snapshot", "count_agreement",
              "count_min_persistence", "staircase_validated"):
        assert k not in loaded                    # absent, not fabricated
    assert np.array_equal(loaded["soliton_count"], sweep["soliton_count"])
    _json  # quiet linters


def test_validate_staircase_alignment_gate():
    sweep, _ = _synthetic_staircase_sweep()
    order = np.argsort(sweep["dw_over_kappa"])
    dwk = sweep["dw_over_kappa"][order]
    counts = sweep["soliton_count"][order]
    y = sweep["P_intra"][order]
    steps = detect_power_steps(dwk, y / y.max(), k=6.0)
    align = match_steps_to_transitions(
        steps, soliton_count_transitions(dwk, counts))
    assert validate_staircase_alignment(counts, align) == []   # passes
    # < 2 matched steps fails
    starved = dict(align, matched=align["matched"][:1])
    problems = validate_staircase_alignment(counts, starved)
    assert len(problems) == 1 and "state-verified" in problems[0]
    # a count increase going down (descending) fails
    bad = counts.copy()
    bad[25] = 1
    problems = validate_staircase_alignment(bad, align)
    assert len(problems) == 1 and "monotonically" in problems[0]


def test_render_and_report_writes_validated_staircase(tmp_path, monkeypatch):
    import json

    import analysis.run_detuning_sweep as rds

    monkeypatch.setattr(rds, "RESULTS_DIR", tmp_path)
    sweep, cfg = _synthetic_staircase_sweep()
    rds.render_and_report(sweep, cfg)
    assert (tmp_path / rds.STEPS_PNG).exists()
    assert (tmp_path / rds.STEPS_PNG).with_suffix(".pdf").exists()
    block = json.loads((tmp_path / rds.METRICS_JSON).read_text())["soliton_step"]
    stair = block["staircase"]
    assert stair["n_seeded"] == 3
    # two state-verified steps (1->2 and 2->3), each with the required fields
    assert len(stair["matched_steps"]) == 2
    for m in stair["matched_steps"]:
        assert set(m) >= {"dw_mid", "delta_n", "step_dy"}
        assert m["delta_n"] == 1
    # the MI/CW rise stays an honest unmatched discontinuity
    assert len(stair["unmatched_steps"]) == 1
    assert "NOT a soliton step" in stair["unmatched_steps"][0]["label"]
    # the muted 1->0 edge shows up as a transition without a power step
    assert any(t["n_high_side"] == 1 and t["n_low_side"] == 0
               for t in stair["unmatched_transitions"])
    assert "power-muted" in stair["final_edge_note"]
    assert stair["primary_observable"]["chosen"] in ("P_intra", "P_comb")
    assert "matched_step_contrast_P_intra" in stair["primary_observable"]
    assert "matched_step_contrast_P_comb" in stair["primary_observable"]
    assert stair["validation"]["n_matched_steps"] == 2
    assert block["any_soliton_region_over_kappa"] is not None
    # counting-method provenance (schema-v4 sweeps carry count_agreement)
    counting = stair["counting"]
    assert counting["method"] == "position_persistence"
    assert counting["parameters"]["min_persistence"] == 0.5
    assert 0.0 <= counting["count_agreement"]["median"] <= 1.0
    assert "forensics" in counting


@pytest.mark.parametrize("kwargs", [{"monotone": False},
                                    {"n_power_steps": 1}])
def test_render_and_report_hard_fails_without_artifacts(tmp_path, monkeypatch,
                                                        kwargs):
    import analysis.run_detuning_sweep as rds

    monkeypatch.setattr(rds, "RESULTS_DIR", tmp_path)
    sweep, cfg = _synthetic_staircase_sweep(**kwargs)
    with pytest.raises(StaircaseValidationError):
        rds.render_and_report(sweep, cfg)
    assert not (tmp_path / rds.STEPS_PNG).exists()      # nothing was written
    assert not (tmp_path / rds.METRICS_JSON).exists()


def test_render_only_exits_nonzero_and_prints_ladder(tmp_path, monkeypatch,
                                                     capsys):
    import sys as _sys

    import analysis.run_detuning_sweep as rds

    monkeypatch.setattr(rds, "RESULTS_DIR", tmp_path)
    sweep, cfg = _synthetic_staircase_sweep(monotone=False)
    save_sweep_npz(tmp_path / rds.SWEEP_NPZ, sweep, cfg)
    monkeypatch.setattr(_sys, "argv", ["run_detuning_sweep.py", "--render-only"])
    with pytest.raises(SystemExit) as excinfo:
        rds.main()
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "STAIRCASE VALIDATION FAILED" in out
    assert "Staircase validation failed. Apply IN ORDER" in out
    assert "persisted" in out and "unvalidated" in out
    assert not (tmp_path / rds.STEPS_PNG).exists()
    assert not (tmp_path / rds.METRICS_JSON).exists()


def test_escalation_ladder_is_canonical():
    # the canonical ladder: measurement rungs first, protocol rungs in order,
    # and an explicit refusal to weaken the gate
    assert ESCALATION_LADDER.startswith("Staircase validation failed.")
    for rung in ("M1", "M2", "P1", "P2", "P3", "P4", "P5", "STOP"):
        assert rung in ESCALATION_LADDER
    assert "Never weaken detect_power_steps" in ESCALATION_LADDER
    assert ESCALATION_LADDER.index("M1") < ESCALATION_LADDER.index("P1")


# ---------------------------------------------------------------------------
# Robustness harness (--robustness): perturbation specs, the standalone
# analyzer (pinned to the primary render path) and the variant assessment
# ---------------------------------------------------------------------------
def test_robustness_variant_specs_are_one_at_a_time():
    """The three perturbations each change exactly ONE field of the base cfg."""
    base = SweepConfig(position_seed=1, hold_rt=2000, n_steps=261)
    specs = robustness_variant_specs(base)
    assert [i for i, *_ in specs] == [1, 2, 3]
    by_key = {key: cfg for _, key, _, cfg in specs}
    # (i) position_seed + 1, nothing else moves
    v1 = by_key["position_seed+1"]
    assert v1.position_seed == base.position_seed + 1
    assert (v1.hold_rt, v1.n_steps) == (base.hold_rt, base.n_steps)
    # (ii) hold_rt -> 1600, nothing else moves
    v2 = by_key["hold_rt_2000_to_1600"]
    assert v2.hold_rt == 1600
    assert (v2.position_seed, v2.n_steps) == (base.position_seed, base.n_steps)
    # (iii) n_steps doubled, nothing else moves
    v3 = by_key["n_steps_doubled"]
    assert v3.n_steps == 2 * base.n_steps
    assert (v3.position_seed, v3.hold_rt) == (base.position_seed, base.hold_rt)


def test_analyze_sweep_staircase_matches_render_path():
    """The robustness analyzer reproduces render_and_report's decision exactly.

    Pins :func:`analyze_sweep_staircase` to the primary path on the SAME
    synthetic sweep the render/validation tests use, so it can never silently
    diverge (same primary-observable choice, same matched dw_mids).
    """
    sweep, cfg = _synthetic_staircase_sweep()
    a = analyze_sweep_staircase(sweep, cfg)
    # ascending order, monotone non-decreasing counts, >= 2 matched steps
    assert list(a["counts"]) == sorted(a["counts"])
    assert not validate_staircase_alignment(a["counts"], a["align"])
    assert a["name1"] in ("P_intra", "P_comb")
    # the matched set equals match_steps_to_transitions on the plotted primary
    matched_here = match_steps_to_transitions(
        a["steps"], soliton_count_transitions(a["dwk"], a["counts"]))
    assert ([m["dw_mid"] for m in a["align"]["matched"]]
            == [m["dw_mid"] for m in matched_here["matched"]])


def test_assess_robustness_flags_soliton_bearing_zero_agreement():
    """A soliton-bearing hold with count_agreement == 0 is a hard violation.

    Rationale under test: an ``agreement == 0`` hold is one where NO single
    snapshot's raw count matched the persistent count, so the count there is
    counter-made, not physics-made -- the exact failure mode the invariant
    guards.  A clean sweep passes; injecting one zero at a soliton-bearing hold
    (and nothing else) flips only that check.
    """
    sweep, cfg = _synthetic_staircase_sweep()
    # This synthetic sweep breathes nowhere and its muted 0->1 (== the 1->0
    # of a real run) sits outside the [5.75, 6.5] window, so restrict the test
    # to the count_agreement invariant by patching the sweep into that window
    # is unnecessary: assess records every check independently.
    a = analyze_sweep_staircase(sweep, cfg)
    clean = dict(sweep)
    clean["count_agreement"] = np.ones_like(np.asarray(sweep["count_agreement"],
                                                       dtype=float))
    res_clean = assess_robustness_variant(1, "k", "desc", clean, cfg, a, "x.npz")
    assert res_clean["min_count_agreement_soliton_bearing"] == 1.0
    assert not any("count_agreement" in v for v in res_clean["violations"])

    # inject a single zero at a soliton-bearing hold -> one agreement violation
    order = np.argsort(sweep["dw_over_kappa"])
    counts_asc = np.asarray(sweep["soliton_count"])[order]
    sb_idx_asc = int(np.nonzero(counts_asc >= 1)[0][0])
    ag = np.ones_like(counts_asc, dtype=float)
    ag[sb_idx_asc] = 0.0
    dirty = dict(sweep)
    dirty["count_agreement"] = ag[np.argsort(order)]     # back to sweep order
    res_dirty = assess_robustness_variant(1, "k", "desc", dirty, cfg, a, "x.npz")
    assert res_dirty["min_count_agreement_soliton_bearing"] == 0.0
    assert res_dirty["soliton_bearing_zero_agreement_holds"]
    assert any("count_agreement == 0" in v for v in res_dirty["violations"])


def test_assess_robustness_muted_edge_matched_is_not_a_failure():
    """A 1->0 edge that resolves as a power step (matched) is recorded, not failed.

    Per the honesty constraint the matched-vs-unmatched status of the muted
    1->0 edge is DATA-decided; a matched edge (resolved naturally on the
    plotted primary) is a stronger outcome than an unmatched one and must never
    count as a robustness violation.  The invariant is only that the edge's
    STRUCTURE persists in the window.
    """
    # A sweep whose 1->0 annihilation lands inside the [5.75, 6.5] window and
    # carries a clear power step, so the detector matches it.
    n = 40
    dw = np.linspace(5.0, 11.0, n)                       # ascending
    counts = np.where(dw < 6.2, 0, np.where(dw < 6.6, 1, 2)).astype(np.int64)
    y = np.where(counts == 0, 0.20, np.where(counts == 1, 0.60, 1.0))
    y = y + 1e-4 * np.sin(np.arange(n))
    sweep = {
        "dw_over_kappa": dw, "P_intra": y.copy(), "P_intra_std": np.full(n, 1e-3),
        "P_comb": y.copy(), "P_comb_std": np.full(n, 1e-3),
        "soliton_count": counts,
        "count_agreement": np.ones(n),
        "is_stationary": np.ones(n, dtype=bool),
        "is_breather": (counts >= 1),
    }
    cfg = SweepConfig(dw_start_kappa=11.0, dw_stop_kappa=5.0, n_steps=n,
                      n_solitons=2)
    a = analyze_sweep_staircase(sweep, cfg)
    res = assess_robustness_variant(1, "k", "desc", sweep, cfg, a, "x.npz")
    edge = res["muted_1to0_edge"]
    lo_w, hi_w = ROBUSTNESS_MUTED_EDGE_WINDOW_KAPPA
    assert lo_w <= edge["dw_mid_over_kappa"] <= hi_w
    assert edge["status"] in ("matched", "unmatched")     # data-decided
    # a matched 1->0 edge is NOT a violation
    assert not any("muted edge" in v for v in res["violations"])


def test_robustness_thresholds_are_the_documented_fixed_floors():
    """The robustness floors are the fixed values in the task spec (not tuned)."""
    assert ROBUSTNESS_COUNT_AGREEMENT_MEDIAN_FLOOR == 0.5
    assert ROBUSTNESS_STATIONARY_EDGE_KAPPA == 9.5
    assert tuple(ROBUSTNESS_MUTED_EDGE_WINDOW_KAPPA) == (5.75, 6.5)
