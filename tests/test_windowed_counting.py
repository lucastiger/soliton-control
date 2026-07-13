"""Unit tests for the position-persistence soliton counter (solver-free).

``analysis.dks_access.count_solitons_windowed`` replaces the end-of-hold
single-snapshot peak count for the per-hold ``soliton_count`` observable.  The
forensic analysis of the committed sweep (``analysis/staircase_forensics.py``,
verdict: counting artifact) pinned the failure mode these tests encode: with
desynchronized breathers the momentary maximum is set by whichever pulse is
near its crest, so a 0.5-of-max single-snapshot rule drops trough pulses even
though every pulse persists (positions frozen, comb energy continuous).  All
fields here are synthetic -- no solver is run.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.dks_access import count_solitons_windowed, count_temporal_peaks

N_TAU = 2048
BG_AMP = 0.05                     # CW background amplitude (power 2.5e-3)
SECH_W = 15 * 2.0 * np.pi / N_TAU  # soliton angular width (~15 grid cells)
THETA = 2.0 * np.pi * np.arange(N_TAU) / N_TAU


def _wrap(x):
    return np.angle(np.exp(1j * np.asarray(x)))


def _field(center_amp_pairs):
    """CW background + sech pulses at (center_rad, amplitude) with wrap."""
    f = np.full(N_TAU, BG_AMP, dtype=complex)
    for c, a in center_amp_pairs:
        f += a / np.cosh(_wrap(THETA - c) / SECH_W)
    return f


def _breathing_snapshots(n_snap=8, depths=(0.55, 0.3, 0.3, 0.3, 0.3), seed=0):
    """Five sech pulses whose heights breathe sinusoidally, random phases.

    Depths are chosen so that at some breathing phases a pulse falls below
    0.5 of the MOMENTARY max (the exact observed bug) while every pulse stays
    far above the CW background floor at all times.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([0.5, 1.7, 3.0, 4.2, 5.5])
    phis = rng.uniform(0.0, 2.0 * np.pi, centers.size)
    return np.array([
        _field([(c, 1.0 + d * np.sin(2.0 * np.pi * k / n_snap + ph))
                for c, d, ph in zip(centers, depths, phis)])
        for k in range(n_snap)
    ]), centers


# ---------------------------------------------------------------------------
# F1: regression encoding the exact observed bug
# ---------------------------------------------------------------------------
def test_single_snapshot_undercounts_where_windowed_counter_does_not():
    snaps, centers = _breathing_snapshots()
    # the legacy 0.5-of-momentary-max single-snapshot count undercounts at
    # some breathing phases ...
    single = [count_temporal_peaks(s) for s in snaps]
    assert min(single) < 5, single
    # ... while the windowed position-persistence count over >= 8
    # phase-spread snapshots recovers exactly 5, and honestly reports that
    # the single-snapshot counts disagreed (count_agreement < 1)
    res = count_solitons_windowed(snaps, cluster_tol_rad=0.05)
    assert res["count"] == 5
    assert res["count_agreement"] < 1.0
    assert len(res["per_snapshot_counts"]) == snaps.shape[0]
    # the persistent cluster angles recover the true pulse positions
    assert res["cluster_angles_rad"].shape == (5,)
    assert np.allclose(np.sort(res["cluster_angles_rad"]),
                       np.sort(centers), atol=0.02)
    # persistence diagnostics cover every candidate cluster and the accepted
    # ones all clear the default threshold
    assert len(res["persistence_fractions"]) >= 5
    assert sum(f >= 0.5 for f in res["persistence_fractions"]) == 5


def test_every_pulse_stays_above_the_background_floor():
    # the bug is a RELATIVE-threshold artifact: even at its trough each pulse
    # is far above the absolute background floor the windowed counter uses
    snaps, _ = _breathing_snapshots()
    for s in snaps:
        p = np.abs(s) ** 2
        floor = 5.0 * np.median(p)
        trough_peak = (1.0 - 0.55) ** 2          # weakest pulse power
        assert trough_peak > 10 * floor


# ---------------------------------------------------------------------------
# F2: circular wrap-around
# ---------------------------------------------------------------------------
def test_soliton_straddling_the_seam_is_one_cluster():
    # pulse center alternates +-0.01 rad around theta = 0, so its candidates
    # land on BOTH sides of the 0/2*pi seam across the snapshots
    snaps = np.array([
        _field([((0.01 if k % 2 == 0 else -0.01) % (2 * np.pi), 1.0),
                (3.0, 1.0)])
        for k in range(8)
    ])
    res = count_solitons_windowed(snaps, cluster_tol_rad=0.05)
    assert res["count"] == 2                      # seam pulse + control pulse
    # the seam cluster's circular-mean angle sits at the seam, not at pi
    seam = [a for a in res["cluster_angles_rad"]
            if min(a, 2 * np.pi - a) < 0.05]
    assert len(seam) == 1


# ---------------------------------------------------------------------------
# F3: persistence rejection
# ---------------------------------------------------------------------------
def test_transient_peak_in_2_of_8_snapshots_is_not_counted():
    rng = np.random.default_rng(3)
    ghost_angles = rng.uniform(0.5, 5.5, 2)       # a new angle each time
    snaps = []
    for k in range(8):
        pulses = [(3.0, 1.0)]                     # the one persistent soliton
        if k in (2, 5):                           # ghost present in 2/8 only
            pulses.append((float(ghost_angles[0 if k == 2 else 1]), 0.9))
        snaps.append(_field(pulses))
    res = count_solitons_windowed(np.array(snaps), cluster_tol_rad=0.05)
    assert res["count"] == 1
    assert np.allclose(res["cluster_angles_rad"], [3.0], atol=0.02)
    # the ghost clusters exist as candidates but fail min_persistence
    assert any(f < 0.5 for f in res["persistence_fractions"])


def test_min_persistence_is_a_real_knob():
    # the same 2-of-8 transient IS counted if the caller lowers the bar
    snaps = []
    for k in range(8):
        pulses = [(3.0, 1.0)]
        if k in (2, 5):
            pulses.append((1.0, 0.9))             # same angle both times
        snaps.append(_field(pulses))
    snaps = np.array(snaps)
    assert count_solitons_windowed(snaps, cluster_tol_rad=0.05)["count"] == 1
    assert count_solitons_windowed(snaps, cluster_tol_rad=0.05,
                                   min_persistence=0.25)["count"] == 2


# ---------------------------------------------------------------------------
# input validation / degenerate cases
# ---------------------------------------------------------------------------
def test_validation_and_degenerate_inputs():
    with pytest.raises(ValueError):               # not 2D
        count_solitons_windowed(np.zeros(16, dtype=complex),
                                cluster_tol_rad=0.05)
    with pytest.raises(ValueError):               # empty
        count_solitons_windowed(np.zeros((0, 16), dtype=complex),
                                cluster_tol_rad=0.05)
    with pytest.raises(ValueError):               # bad persistence
        count_solitons_windowed(np.ones((2, 16), dtype=complex),
                                cluster_tol_rad=0.05, min_persistence=0.0)
    with pytest.raises(ValueError):               # no tolerance and no cav
        count_solitons_windowed(np.ones((2, 16), dtype=complex))
    # dark field: zero count, sane agreement bookkeeping
    res = count_solitons_windowed(np.zeros((4, 64), dtype=complex),
                                  cluster_tol_rad=0.05)
    assert res["count"] == 0
    assert res["count_agreement"] == 1.0
    assert res["cluster_angles_rad"].size == 0


def test_cluster_tolerance_from_cavity_parameters():
    # the default tolerance derives from the soliton width; a synthetic cav
    # carrying only d2/d2_local exercises that path
    class _Cav:
        d2 = 2.0e5
        d2_local = None
    snaps, _ = _breathing_snapshots()
    res = count_solitons_windowed(snaps, delta_omega=1.5e9, cav=_Cav())
    assert res["count"] == 5


# ---------------------------------------------------------------------------
# Snapshot-budget arithmetic behind the offline starvation test.  The driver's
# actual cadence (hold_rt // 32) puts EIGHT snapshots in the counting window,
# not the two the starvation hypothesis (hold_rt // 8) assumed -- so
# count_agreement is quantized in eighths, and that is the empirical proof the
# failures are NOT sample starvation.
# ---------------------------------------------------------------------------
def test_in_window_snapshot_budget_actual_vs_hypothesis():
    from analysis.staircase_forensics import _n_in_window
    # actual driver cadence: snap_int = hold_rt // 32 -> 8 in-window snapshots
    assert _n_in_window(2000, 0.25, max(2000 // 32, 1)) == 8
    assert _n_in_window(1600, 0.25, max(1600 // 32, 1)) == 8
    # the brief's hypothesised cadence: hold_rt // 8 -> only 2 in-window
    assert _n_in_window(2000, 0.25, max(2000 // 8, 1)) == 2


def test_count_agreement_quantization_signature():
    from analysis.staircase_forensics import _empirical_denominator, _on_grid
    eighths = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]   # observed in the npzs
    assert _empirical_denominator(eighths) == 8         # => n_in = 8, not 2
    assert _on_grid(eighths, 8) is True
    assert _on_grid(eighths, 2) is False                # not halves {0,0.5,1}


def test_victim_stats_separates_coupling_from_dimming():
    # The Stage-B discriminator: a rel-VICTIM passes the absolute floor but
    # fails the relative one; genuine dimming fails the absolute floor.
    from analysis.staircase_forensics import _victim_stats
    bg = np.ones(8)
    # coupling: passes abs every snapshot, fails rel in 6/8 -> dropped by rel
    coup = _victim_stats(above_rel=[1, 1, 0, 0, 0, 0, 0, 0],
                         above_abs=[1, 1, 1, 1, 1, 1, 1, 1],
                         local_max=np.full(8, 3.0), median_bg=bg, b2=1.0)
    assert coup["abs_pass"] == 1.0 and coup["detected"] == 0.25
    assert coup["rel_victim"] == 0.75 and coup["fails_both"] == 0.0
    # dimming: fails the absolute floor in the majority of snapshots
    dim = _victim_stats(above_rel=[1, 1, 1, 1, 1, 1, 1, 1],
                        above_abs=[1, 1, 0, 0, 0, 0, 0, 0],
                        local_max=np.array([3, 3, .5, .5, .5, .5, .5, .5]),
                        median_bg=bg, b2=1.0)
    assert dim["fails_both"] == 0.75 and dim["abs_pass"] == 0.25


def test_detectability_missing_cluster_and_nn_rank():
    # A soliton present in the flank but dropped at the event: greedy match
    # against the flank leaves exactly that angle unmatched (a dropout).
    from analysis.staircase_forensics import (_greedy_unmatched,
                                              _nn_separations, _rank_smallest)
    flank = [0.05, 1.10, 2.40, 3.65, 4.44]     # 5 solitons
    event = [1.10, 2.40, 3.65, 4.44]           # the 0.05 one dropped
    missing = _greedy_unmatched(event, flank, tol=0.05)
    assert len(missing) == 1 and abs(missing[0] - 0.05) < 1e-9
    # nearest-neighbour separations + rank of the (tightest) missing soliton
    nn = _nn_separations(flank)
    # the 0.05 <-> 4.44 pair straddles the seam (gap ~1.9 rad); 1.10 is closest
    # to nothing tighter than ~1.0 -- just assert ranking is well-formed
    assert _rank_smallest(nn, min(nn)) == 1
    assert 1 <= _rank_smallest(nn, nn[0]) <= len(flank)
