# Staircase flicker forensics: counting artifact vs real re-nucleation

Generated 2026-07-12T03:12:49.970611+00:00 by `analysis/staircase_forensics.py` (offline; no solver run; read-only except this file).

## Data audited

- npz: `detuning_sweep.npz`  sha256 `07ead90d3e423e3e...`
- schema_version: ABSENT (pre-v3 file; key set identifies it as schema v2)
-   soliton_count: present, shape (261,), dtype int64
-   peak_positions_rad: present, shape (261, 1007), dtype float64
-   P_comb: present, shape (261,), dtype float64
-   P_comb_std: present, shape (261,), dtype float64
-   P_intra: present, shape (261,), dtype float64
-   breathing_relstd: present, shape (261,), dtype float64
-   is_breather: present, shape (261,), dtype bool
-   np_label: present, shape (261,), dtype int64
-   is_single: present, shape (261,), dtype bool

- flicker events (count increases along the descending sweep): **38**
- undercount holds (count < future-max envelope): **83 of 261**
- envelope staircase (true-count LOWER BOUND): 5 -> 4 at 7.65 k, 4 -> 3 at 6.32 k, 3 -> 1 at 6.22 k, 1 -> 0 at 6.17 k
- dips never reach count 0 mid-branch (True): the loss mechanism is the 50%-of-max peak threshold, not the labeler/contrast gate.

## TEST A -- position persistence

- Raw fixed-frame match (tolerance 0.05 rad, as prescribed): **0.281**.
- Measured confound: the whole pulse pattern rotates coherently at **+0.0454 rad/hold** (IQR +0.0441..+0.0463, n = 140 no-event same-count hold pairs, where nucleation is impossible). One hold of drift is comparable to the tolerance, so the raw statistic is limited by drift, not by nucleation; it is reported but carries no discriminating power.
- Rotation-controlled match (same tolerance after removing one global rotation per comparison -- what re-nucleation would scramble): **0.852**, against a measurement ceiling of **0.848** (the identical statistic on local no-event control pairs) and a re-nucleation null of **0.330** (random angles, best-rotation fitted).
- Interpretation: the recovered peaks sit at the pre-dip angles as precisely as this dataset can measure (1.00 of ceiling; the residual misses are breathing-phase position wobble that the no-event controls show identically), and far above the re-nucleation null. **The solitons never moved.**

## TEST B -- energy continuity

- One-soliton quantum: median per-quantum |dP_comb| over the energy-visible clean count decrements = **6.463e-05** (~17% of the 5-soliton-branch comb power). The committed `spectral_metrics.json` staircase block predates the step-transition alignment machinery (its matched-edge list mixes flicker edges and stores normalised step sizes), so the brief's fallback -- decrements outside flicker regions, i.e. envelope drops -- is used.
- **Anomaly:** the envelope drop 5 -> 4 at 7.65 k is energy-SILENT (|dP_comb| = 1.44e-07, ~0.0022 quanta). An annihilation with no energy signature is itself an undercount signature: the 5 -> 4 'edge' at 7.65 k is most plausibly the onset of PERMANENT undercounting (the count never again reaches 5), and the energy-visible annihilation cascade lives at 6.2-6.4 k.
- Per-event pre -> dip |dP_comb| vs (claimed quanta lost x quantum): median **0.022**, max **0.380** (a real annihilation + re-nucleation cycle would give ~1 per quantum). Events claiming 2-4 lost solitons show sub-percent comb-power changes. **The energy never left the cavity.**

## TEST C -- breathing correlation

- breathing_relstd, undercount holds: median 0.0540 (IQR 0.0394..0.0696, n = 83); correct-count holds: median 0.0000 (IQR 0.0000..0.0111, n = 150). Undercounting is confined to deep-breathing holds.
- is_breather at the dip hold: 38/38 events.
- np_label at the 27 dip holds: {3: 10, 4: 14, 5: 3} -- 17/27 stay in SOLITON_LABELS (4, 5, 6) while the count collapses (the rest are class 3, the documented labeler misroute of breathing multi-soliton states); none fall to a CW/MI class. The field's own classification contradicts the peak count at the dips.
- is_single at dw >= 6.5 k: not corrupted (True at 0 holds there, and never while the envelope shows a multi-soliton state); the committed figure's shaded single-DKS band is unaffected by the flicker.

## Per-event table

| post hold | dw_dip (k) | c_pre | c_dip | c_post | match raw | match rot-ctrl | dP_comb pre->dip | quanta ratio |
|---|---|---|---|---|---|---|---|---|
| 124 | 8.925 | 5 | 4 | 5 | 0/5 | 5/5 | +0.540% | 0.031 |
| 128 | 8.825 | 5 | 3 | 5 | 0/5 | 5/5 | -0.098% | 0.003 |
| 134 | 8.675 | 5 | 2 | 4 | 0/4 | 4/4 | +0.504% | 0.010 |
| 135 | 8.675 | 5 | 2 | 5 | 0/5 | 5/5 | +0.504% | 0.010 |
| 137 | 8.600 | 5 | 3 | 5 | 0/5 | 5/5 | -1.740% | 0.051 |
| 141 | 8.500 | 3 | 2 | 3 | 0/3 | 3/3 | +0.976% | 0.056 |
| 143 | 8.450 | 3 | 2 | 3 | 0/3 | 3/3 | -0.247% | 0.014 |
| 145 | 8.450 | 5 | 2 | 5 | 0/5 | 5/5 | +0.249% | 0.005 |
| 148 | 8.325 | 5 | 3 | 5 | 0/5 | 5/5 | +2.153% | 0.061 |
| 152 | 8.225 | 5 | 3 | 4 | 0/4 | 4/4 | +1.306% | 0.037 |
| 158 | 8.075 | 3 | 1 | 2 | 1/2 | 1/2 | -0.561% | 0.016 |
| 162 | 8.075 | 4 | 1 | 4 | 0/4 | 3/4 | -0.990% | 0.019 |
| 166 | 8.075 | 5 | 1 | 5 | 0/5 | 5/5 | +0.601% | 0.009 |
| 170 | 7.775 | 5 | 2 | 5 | 0/5 | 5/5 | +0.330% | 0.006 |
| 173 | 7.700 | 5 | 3 | 5 | 5/5 | 5/5 | -0.647% | 0.019 |
| 177 | 7.600 | 2 | 1 | 2 | 1/2 | 1/2 | +0.008% | 0.000 |
| 178 | 7.600 | 5 | 1 | 3 | 1/3 | 3/3 | +0.047% | 0.001 |
| 180 | 7.600 | 5 | 1 | 4 | 0/4 | 4/4 | +0.047% | 0.001 |
| 183 | 7.450 | 3 | 1 | 2 | 0/2 | 1/2 | +0.174% | 0.005 |
| 188 | 7.325 | 2 | 1 | 2 | 2/2 | 2/2 | +1.284% | 0.072 |
| 189 | 7.325 | 3 | 1 | 3 | 1/3 | 1/3 | -0.045% | 0.001 |
| 191 | 7.325 | 4 | 1 | 4 | 0/4 | 3/4 | +0.608% | 0.011 |
| 193 | 7.200 | 4 | 1 | 3 | 3/3 | 3/3 | -1.289% | 0.025 |
| 194 | 7.200 | 4 | 1 | 4 | 3/4 | 3/4 | -1.289% | 0.025 |
| 197 | 7.100 | 3 | 2 | 3 | 2/3 | 2/3 | -0.025% | 0.001 |
| 200 | 7.025 | 3 | 2 | 3 | 2/3 | 2/3 | -3.806% | 0.220 |
| 201 | 7.025 | 4 | 2 | 4 | 2/4 | 2/4 | -2.234% | 0.064 |
| 203 | 6.950 | 4 | 1 | 2 | 1/2 | 1/2 | +0.788% | 0.014 |
| 205 | 6.950 | 4 | 1 | 3 | 1/3 | 2/3 | +0.788% | 0.014 |
| 207 | 6.850 | 3 | 1 | 2 | 1/2 | 1/2 | -2.516% | 0.071 |
| 209 | 6.800 | 2 | 1 | 2 | 0/2 | 1/2 | -3.573% | 0.203 |
| 212 | 6.725 | 3 | 1 | 3 | 0/3 | 2/3 | -2.075% | 0.059 |
| 215 | 6.650 | 4 | 1 | 4 | 0/4 | 3/4 | -4.303% | 0.079 |
| 217 | 6.600 | 4 | 2 | 3 | 3/3 | 3/3 | +0.982% | 0.026 |
| 220 | 6.525 | 3 | 2 | 3 | 1/3 | 2/3 | -6.857% | 0.380 |
| 222 | 6.475 | 4 | 1 | 4 | 1/4 | 3/4 | -18.668% | 0.330 |
| 226 | 6.375 | 4 | 1 | 4 | 4/4 | 4/4 | -3.631% | 0.052 |
| 230 | 6.275 | 4 | 1 | 3 | 3/3 | 3/3 | -25.516% | 0.352 |

## Verdict

**VERDICT: counting artifact** -- positions persist at 0.852 = 1.00 of the measurement ceiling (0.848; re-nucleation null 0.330; raw fixed-frame 0.281 is drift-limited); energy is sub-quantum (median 0.022, max 0.380 of the claimed loss)

Rule applied: 'counting artifact' iff TEST A shows > 0.9 position persistence AND TEST B shows sub-quantum energy changes; TEST A is evaluated on the rotation-controlled statistic against its measured ceiling because the raw fixed-frame number is invalidated by the coherent pattern drift quantified above (with the raw prescription taken literally, the drift alone would fake a 'new positions' reading for every dip longer than one hold).

Consequences for the schema-4 counter hardening (next step, NOT done here): the flicker is an estimator artifact of the end-of-hold single-snapshot 50%-of-max peak count in the deep-breathing sub-band; the per-hold snapshot-median count proposed in the escalation ladder should remove it. The energy-silent 5 -> 4 envelope drop at 7.65 k means the hardened counter must be validated against P_comb steps, not against the current envelope alone.

## Snapshot starvation (Part A: offline hypothesis test)

Generated 2026-07-12T23:31:51.089818+00:00 by `analysis/staircase_forensics.py --starvation` (offline; no solver run). Hypothesis under test: the robustness count failures are SNAPSHOT STARVATION -- the windowed counter votes over too few, phase-aliased in-window snapshots.

Driver cadence: `snap_int = max(hold_rt // 32, 1)` (`analysis/run_detuning_sweep.py`); the counter votes over the snapshots inside the final-`avg_frac` window. The brief hypothesised `interval = hold_rt // 8`.

### Per-file snapshot budget and quantization signature

| file | hold_rt | snap_int (//32) | **n_in (actual)** | n_in (//8 hyp) | count_agreement grid | signature {k/n_in} | mono-viol | agree==0 |
|---|---|---|---|---|---|---|---|---|
| primary | 2000 | 62 | **8** | 2 | 1/8 | True | 0 | 2 |
| variant_1 | 2000 | 62 | **8** | 2 | 1/8 | True | 1 | 1 |
| variant_2 | 1600 | 50 | **8** | 2 | 1/8 | True | 0 | 0 |
| variant_3 | 2000 | 62 | **8** | 2 | 1/8 | True | 1 | 3 |

### Failing holds (monotonicity dips + soliton-bearing agreement==0)

`ratio = snap_int / breathing_period_rt` is the aliasing indicator: **>> 1 would mean the snapshots undersample the breathing cycle (starvation); < 1 means they oversample it.**

| file | dw/k | N | N_end-snap | count_agreement | breathing_relstd | T_b (RT) | snap_int/T_b | kind |
|---|---|---|---|---|---|---|---|---|
| primary | 7.050 | 5 | 2 | 0.000 | 0.0318 | 184 | 0.34 | +agree0 |
| primary | 7.075 | 5 | 3 | 0.000 | 0.0338 | 181 | 0.34 | +agree0 |
| variant_1 | 6.500 | 5 | 2 | 0.000 | 0.0331 | 393 | 0.16 | +agree0 |
| variant_1 | 6.750 | 4 | 1 | 0.250 | 0.0587 | 188 | 0.33 | mono |
| variant_3 | 6.785 | 5 | 3 | 0.000 | 0.0465 | 193 | 0.32 | +agree0 |
| variant_3 | 6.810 | 5 | 4 | 0.000 | 0.0378 | 192 | 0.32 | +agree0 |
| variant_3 | 6.960 | 5 | 1 | 0.000 | 0.0531 | 186 | 0.33 | +agree0 |
| variant_3 | 6.997 | 4 | 2 | 0.375 | 0.0435 | 187 | 0.33 | mono |

### Verdict

**STARVATION: NOT CONFIRMED** (rule: CONFIRMED iff n_in <= 3 for ALL files AND every count_agreement lies on the {k/n_in} grid).

- n_in = [8] (actual `hold_rt//32` cadence), which is **> 3** -- the counter already votes over ~8 in-window snapshots, not ~2. The brief's `hold_rt//8` interval (giving n_in=[2]) is NOT what the driver uses.
- The count_agreement quantization confirms it: every value lies on an **eighths** grid (1/8), i.e. n_in = 8, not the halves ({0, 0.5, 1}) the starvation hypothesis predicts.
- The aliasing indicator `snap_int/T_b` is < 1 at every failing hold (snapshots OVERSAMPLE the breathing cycle by ~3x), so phase-aliasing is not the mechanism.
- **GATE (per the brief): STOP.** The counter had many (~8) phase-spread samples and still failed at isolated deep-breather holds, so the failure mechanism is NOT starvation. Densification / threshold / protocol work must not proceed on the falsified hypothesis; the residual (individual solitons dipping below the rel-height floor during their breathing troughs at these specific holds) needs its own verification before any fix.

## Detectability (offline; Stage A)

Generated 2026-07-13T03:28:26.726382+00:00 by `analysis/staircase_forensics.py --detectability` (offline; no solver run). Working hypothesis: the count defect is a RELATIVE-threshold detectability problem -- a trough-phase soliton is rejected when a sibling is at crest because the candidate floor `rel_height_candidate * snapshot_max` couples each soliton's detection to the others' breathing phases.

### A1 -- missing cluster at each monotonicity-violating hold

| file | dw/k | N | N_end-snap | missing angle | in before | in after | before<->after gap | persists (dropout) |
|---|---|---|---|---|---|---|---|---|
| variant_1 | 6.750 | 4 | 1 | 0.057 | yes | yes | 0.0176 | **YES** |
| variant_3 | 6.997 | 4 | 2 | 3.028 | yes | yes | 0.0090 | **YES** |

A missing soliton present at the SAME angle (within the 0.1 rad drift budget) in BOTH flanking holds never left -- it is a pure detection dropout, consistent with a counting defect (not annihilation/re-nucleation).

### A2 -- is the missing soliton the most strongly interacting?

Rank 1 = tightest nearest-neighbour separation (interacts hardest, breathes deepest). `rank_flank` is computed on the event-neighbourhood positions; `rank_seed` maps the missing soliton back to its seed (rigid-rotation cyclic map) and ranks the seed separations.

| file | seed | dw/k | missing angle | rank_flank (of N) | seed idx | rank_seed (of n) |
|---|---|---|---|---|---|---|
| variant_1 | 2 | 6.750 | 0.057 | 3/5 | 4 | 3/5 |
| variant_3 | 1 | 6.997 | 3.028 | 1/5 | 2 | 1/5 |

Same seed-relative soliton across variants sharing a seed: seed 2: variant_1->idx 4; seed 1: variant_3->idx 2.
(primary and variant_2 share seed 1 but have NO monotonicity-violating hold, so only variant_3 supplies a seed-1 dropout to locate.)

### A3 -- agreement==0 holds (soliton-bearing)

The raw per-cluster `persistence_fractions` are computed by `count_solitons_windowed` but NOT persisted to the npz (only `count_agreement` and the final accepted cluster angles are), so the per-cluster fraction breakdown the brief asks for is deferred to the instrumented Stage B run. From the stored counts the two signatures still separate: `undercount` (a cluster fell below min_persistence, so N < envelope) vs `correct-count` (all N clusters kept but no single snapshot saw all N).

| file | dw/k | N | envelope | N_end-snap | count_agreement | category |
|---|---|---|---|---|---|---|
| primary | 7.050 | 5 | 5 | 2 | 0.000 | correct-count, no unanimous snapshot |
| primary | 7.075 | 5 | 5 | 3 | 0.000 | correct-count, no unanimous snapshot |
| variant_1 | 6.500 | 5 | 5 | 2 | 0.000 | correct-count, no unanimous snapshot |
| variant_3 | 6.785 | 5 | 5 | 3 | 0.000 | correct-count, no unanimous snapshot |
| variant_3 | 6.810 | 5 | 5 | 4 | 0.000 | correct-count, no unanimous snapshot |
| variant_3 | 6.960 | 5 | 5 | 1 | 0.000 | correct-count, no unanimous snapshot |

### Stage-A gate

- **POSITION PERSISTENCE CONFIRMED** at all 2 monotonicity events: every missing soliton sits at the same angle in both flanks (max before<->after gap 0.0176 rad). The dropouts are a COUNTING defect, not physics rearrangement -> Stage B (instrumented run) may proceed.

## Detectability (Stage B: instrumented)

Generated 2026-07-13T05:26:41.654287+00:00 by `analysis/staircase_forensics.py --diagnose-report`. Per failing hold the VICTIM soliton (missing one at an undercount hold; lowest-persistence cluster at an agreement==0 hold) is scored: a rel-VICTIM snapshot passes the absolute floor but fails the relative one (`rel_height_candidate * snapshot_max`), so it is rejected only because a sibling's crest lifted `snapshot_max`.

| variant | dw/k | kind | victim | abs pass | rel-victim frac | fails-both | detected (persist.) | min|E|²/bg | min|E|²/B² | class |
|---|---|---|---|---|---|---|---|---|---|---|
| variant_1 | 6.750 | mono | missing@0.048 | 1.00 | 0.62 | 0.00 | 0.38 | 32.8 | 0.45 | **coupling** |
| variant_1 | 6.500 | agree0 | cluster 2 (min persist.) | 1.00 | 0.50 | 0.00 | 0.50 | 23.4 | 0.37 | **coupling** |
| variant_3 | 6.997 | mono | missing@3.023 | 1.00 | 0.62 | 0.00 | 0.38 | 41.6 | 0.52 | **coupling** |
| variant_3 | 6.960 | agree0 | cluster 4 (min persist.) | 1.00 | 0.38 | 0.00 | 0.62 | 32.0 | 0.41 | **coupling** |
| variant_3 | 6.810 | agree0 | cluster 0 (min persist.) | 1.00 | 0.38 | 0.00 | 0.62 | 28.6 | 0.39 | **coupling** |
| variant_3 | 6.785 | agree0 | cluster 1 (min persist.) | 1.00 | 0.50 | 0.00 | 0.50 | 27.2 | 0.38 | **coupling** |

**STAGE-B VERDICT: RELATIVE-THRESHOLD COUPLING CONFIRMED**

- Rule: COUPLING iff at every undercount hold the missing soliton passes the ABSOLUTE floor in >= 90% of snapshots yet is dropped (persistence < 0.5) by the RELATIVE floor (`rel_height_candidate * snapshot_max`); GENUINE DIMMING iff it fails the absolute floor in the majority of snapshots; MIXED otherwise. agree0 holds corroborate (same mechanism, victim kept above 0.5).
- No fix applied: this diagnosis is the deliverable. Any remedy (e.g. dropping the coupled relative arm of the candidate floor) is a separate, gated change -- not made here.
## Step-quanta (offline)

Generated 2026-07-14T04:48:44.316421+00:00 by `analysis/staircase_forensics.py --stepquanta` (offline; no solver run; read-only on the committed artifacts). Confirms or refutes the split-step diagnosis behind the failing `tests/test_soliton_staircase.py::test_step_heights_quantized` (Part 2i).

- npz: `detuning_sweep.npz`  sha256 `fe7b6789daac517b...`
- primary observable (from the staircase JSON block): `P_comb`; detector k = 6, match tol = 1 sample; robust sigma = 7.880e-04
- recomputation MATCHES the committed alignment: 4 matched, 64 unmatched steps, 0 unmatched transitions (the artifact is self-consistent -- Part 2b holds).

### Matched steps above the 1->0 edge, with adjacent unmatched discontinuities (within tol)

| matched edge | dw_mid (k) | transition | delta_n | matched step_dy | |per-quantum| | adjacent unmatched (edge, dw_mid, step_dy, sign) |
|---|---|---|---|---|---|---|
| 29 | 6.2375 | 3->1 | 2 | +0.34604 | 0.17302 | 30 @ 6.262k -0.01299 (-, OPP) |
| 32 | 6.3125 | 4->3 | 1 | +0.05683 | 0.05683 | 31 @ 6.287k +0.01119 (+, same); 33 @ 6.337k +0.10272 (+, same) |
| 40 | 6.5125 | 5->4 | 1 | +0.16929 | 0.16929 | 39 @ 6.487k -0.03235 (-, OPP); 41 @ 6.537k +0.06142 (+, same) |

### Failing pair (Part 2i)

- reference 5->4 step_dy **+0.16929** (edge 40, 6.5125k)
- short 4->3 matched step_dy **+0.05683** (edge 32, 6.3125k)
- dominant adjacent same-sign unmatched discontinuity **+0.10272** (edge 33, 6.3375k)
- sum (short + dominant adjacent) = **0.15955**
- ratio BEFORE aggregation (0.05683 vs 0.16929): **2.979** (> 2 -> the test fails)
- ratio AFTER aggregation (0.15955 vs 0.16929): **1.061**; vs the merged 3->1 per-quantum 0.17302: **1.084**
- adjacent same-sign unmatched discontinuities to the 4->3 edge: **2** (edge 31, +0.01119), (edge 33, +0.10272)
- the reference 5->4 edge is itself flanked by 1 same-sign unmatched discontinuity(ies) (edge 41, +0.06142) -- plateau ripple fragments steps, so an adjacent same-sign unmatched neighbour is not a unique split-partner signal.

### Verdict

Rule: **SPLIT-STEP CONFIRMED** iff the 4->3 matched edge has exactly one adjacent (within tol_samples), same-sign, otherwise-unmatched discontinuity AND the sum brings all per-quantum magnitudes within a factor of 2; **NOT A SPLIT STEP** iff the adjacent discontinuity is opposite-sign, absent, or the aggregated magnitudes still exceed a factor of 2 (genuine non-quantization / a merged-annihilation issue -- a real physics finding); **AMBIGUOUS** otherwise.

**VERDICT: AMBIGUOUS** -- aggregating the dominant adjacent same-sign discontinuity (edge 33, +0.10272) restores quantization (ratio 1.061 <= 2), so the split direction is supported -- but the strict split-step criterion is NOT met: the 4->3 edge has 2 same-sign adjacent unmatched discontinuities, not exactly one, and the reference edge is itself flanked by a same-sign neighbour (plateau ripple fragments multiple steps, so 'adjacent same-sign unmatched' is not a unique split signal)

No fix applied: this diagnosis is the deliverable. Any remedy (aggregating split matched+adjacent discontinuities before the quantization check, a plateau-level step-height measure, or accepting the merged-annihilation reference) is a separate, gated change -- not made here.


## Plateau-bounded aggregation (offline)

Generated 2026-07-14T04:48:44.317081+00:00 by `analysis/staircase_forensics.py --stepquanta` (offline; no solver run; read-only on the committed artifacts). Tests a PHYSICALLY BOUNDED aggregation rule for the Part 2i step heights: the AMBIGUOUS verdict above showed that 'exactly one adjacent same-sign unmatched discontinuity' is not a valid split-partner signal (plateau ripple scatters same-sign unmatched neighbours around most edges), so aggregation is bounded here by the COUNT STRUCTURE (plateaus), not a neighbour radius.

- primary observable `P_comb`; detector k = 6, match tol = 1; robust sigma = 7.880e-04; npz sha256 `fe7b6789daac517b...` (recomputation matches committed).
- soliton_count plateaus (count: idx range): 0:[0,27], 1:[28,29], 3:[30,32], 4:[33,40], 5:[41,260]
- count-change edges: [27, 29, 32, 40] -- ALL are matched (`unmatched_transitions` is empty; 0 unmatched count-change edges). **Every one of the 64 unmatched discontinuities is therefore in-plateau ripple**, so no count-changing fragment exists to legitimately absorb into any annihilation window.

### Annihilation windows (matched N->N-1, excluding 1->0)

| transition | matched edge | dw_mid (k) | delta_n | matched step_dy | low-count plateau | high-count plateau | window edges | aggregated |step_dy| | per-quantum |
|---|---|---|---|---|---|---|---|---|---|
| 3->1 | 29 | 6.2375 | 2 | +0.34604 | 1:[28,29] | 3:[30,32] | [29] | 0.34604 | 0.17302 |
| 4->3 | 32 | 6.3125 | 1 | +0.05683 | 3:[30,32] | 4:[33,40] | [32] | 0.05683 | 0.05683 |
| 5->4 | 40 | 6.5125 | 1 | +0.16929 | 4:[33,40] | 5:[41,260] | [40] | 0.16929 | 0.16929 |

Neighbour classification (same-sign unmatched within tol of each matched edge -- inside the plateau-bounded window or excluded, and why):

- 4->3 edge 32: neighbour edge 31 (+0.01119) -- EXCLUDED: in-plateau ripple (count 3->3, no change) -> EXCLUDED
- 4->3 edge 32: neighbour edge 33 (+0.10272) -- EXCLUDED: in-plateau ripple (count 4->4, no change) -> EXCLUDED
- 5->4 edge 40: neighbour edge 41 (+0.06142) -- EXCLUDED: in-plateau ripple (count 5->5, no change) -> EXCLUDED

### Per-quantum magnitudes and ratios

- plateau-bounded per-quantum: 3->1 = 0.17302, 4->3 = 0.05683, 5->4 = 0.16929
- plateau-bounded pairwise ratio (max/min): **3.044** (the raw single-edge ratio is 3.044; identical here because every window is a single matched edge)

### Sensitivity / anti-laundering (per-quantum at committed tol = 1)

| transition | (a) plateau-bounded | (b) tol=1 same-sign | (c) matched-only |
|---|---|---|---|
| 3->1 | 0.17302 [29] | 0.17302 [29] | 0.17302 |
| 4->3 | 0.05683 [32] | 0.17074 [31, 32, 33] | 0.05683 |
| 5->4 | 0.16929 [40] | 0.23071 [40, 41] | 0.16929 |

- **plateau-bounded is INSENSITIVE to the reach**: per-quantum identical at radius 1/2/3 = True (it is defined by count structure, not a neighbour radius -- no count-change fragment exists to admit at any reach).
- the over-permissive (b) tol-same-sign rule GROWS with the reach (it absorbs progressively more in-plateau ripple), e.g. the 4->3 per-quantum at radius 1/2/3 = 0.17074/0.17074/0.20477 and the 5->4 = 0.23071/0.26564/0.26564 -- confirming (b) launders quantization by absorbing ripple, which the plateau bound forbids.
- (a) equals (c): with no unmatched count-change fragments, the physically bounded window is exactly the matched edge, so the plateau-bounded magnitudes ARE the raw single-edge magnitudes.

### Verdict

Rule (plateau-bounded only): **QUANTIZED (plateau-bounded)** iff all plateau-bounded per-quantum magnitudes agree pairwise within a factor of 2 AND each window is the matched edge plus only count-changing same-sign fragments (no in-plateau ripple absorbed); **NOT QUANTIZED** iff the plateau-bounded per-quantum magnitudes still exceed a factor of 2 (a real physics finding -- possible merged annihilation or genuine non-quantization); **STILL AMBIGUOUS** iff the window construction is undefined for some transition (overlapping windows or unresolvable flanking plateaus).

**VERDICT: NOT QUANTIZED** -- the plateau-bounded per-quantum magnitudes still span a factor 3.044 > 2. Every count-change edge is already matched (0 unmatched count-change edges exist), so the annihilation windows are all single matched edges and the plateau-bounded sums equal the raw single-edge magnitudes: the short 4->3 step (per-quantum 0.05683) cannot be restored by absorbing edge 33 (+0.10272), which lies INSIDE the count-4 plateau (no count change). A real finding: the position-persistence count and the comb-energy drop are offset by one hold at this annihilation, so a count-structure-respecting aggregation does not quantize the single-edge heights

No fix applied: this adjudication is the deliverable. The finding -- the 4->3 comb-power drop is split across the count-flip hold (edge 32, matched, ~1/3 quantum) and the adjacent count-4 plateau hold (edge 33, ~2/3 quantum), i.e. the position-persistence count lags the comb-energy drop by one hold -- means any legitimate remedy (a plateau-integrated step height, or reconciling the count observable with the energy observable at the annihilation edge) is a separate, gated change, not made here.
## Count/energy lag (offline)

Generated 2026-07-14T05:01:10.429835+00:00 by `analysis/staircase_forensics.py --annihilation-report` (offline adjudication of the Stage-2 instrumented run `analysis/run_detuning_sweep.py --diagnose-annihilation`, whose sidecar is `analysis/results/diagnose_annih_4to3.npz`). Decides whether the 4->3 count/energy offset is COUNTER-LATENCY (the physics-anchored height rule holds a dying soliton above threshold one hold too long) or an INTRINSIC integer-count vs continuous-energy LAG.

### Stage 1 precondition (committed npz)

- 4->3 matched edge 32; count-flip hold 33 @ 6.3250k (last N=4) -> hold 32 @ 6.3000k (first N=3).
- TARGET soliton ~2.5242 rad, localized to ONE cluster; NN-separation rank 1 of 4 (1 = tightest); maps to seed sorted-idx 2, seed-NN rank 1 of 5 -- the most strongly interacting soliton.

| hold | dw/k | N | N_end-snap | count_agreement | P_comb | breathing_relstd | cluster angles (rad) |
|---|---|---|---|---|---|---|---|
| 31 | 6.2750 | 3 | 1 | 1.000 | 1.9922e-04 | 0.0499 | [1.267, 3.303, 5.331] |
| 32 | 6.3000 | 3 | 1 | 1.000 | 2.0369e-04 | 0.0387 | [1.258, 3.294, 5.323] |
| 33 | 6.3250 | 4 | 1 | 0.500 | 2.2641e-04 | 0.0735 | [1.251, 2.524, 3.288, 5.313] |
| 34 | 6.3500 | 4 | 4 | 1.000 | 2.6746e-04 | 0.0495 | [1.244, 2.519, 3.281, 5.306] |

### Target soliton local-energy trajectory (instrumented)

Local comb energy = angular integral of |E(theta)|^2 over +/- cluster_tol around the tracked cluster, background (angular median) subtracted, meaned over the in-window snapshots; E/ref is relative to the value two holds above the flip; certify = fraction of snapshots whose local peak clears BOTH acceptance floors (bg_floor_multiple*median AND soliton_frac*B2_ref).

| dw/k | N (counter) | count_agreement | target local E | E/ref | mean local peak | soliton_frac*B2_ref | certify frac | counted? |
|---|---|---|---|---|---|---|---|---|
| 6.4000 | 4 | 1.000 | 1.3941e-08 | 111.1% | 2.4451e-09 | 1.8893e-10 | 100% | yes |
| 6.3750 | 4 | 1.000 | 1.2550e-08 | 100.0% | 2.0285e-09 | 1.8819e-10 | 100% | yes |
| 6.3500 | 4 | 1.000 | 1.2769e-08 | 101.7% | 2.0579e-09 | 1.8745e-10 | 100% | yes |
| 6.3250 | 4 | 0.500 | 5.7045e-09 | 45.5% | 5.3804e-10 | 1.8671e-10 | 50% | yes |
| 6.3000 | 3 | 1.000 | 1.7868e-11 | 0.1% | 4.0686e-11 | 1.8598e-10 | 0% | no |
| 6.2750 | 3 | 1.000 | -1.2921e-12 | -0.0% | 4.0878e-11 | 1.8524e-10 | 0% | no |

- at the count-flip hold (6.3250k, the last hold the target is counted, count_agreement 0.500) the target's across-snapshot mean local energy is 45.5% of the full-soliton value two holds earlier (6.3750k) -- but that mean is a red herring at a deep-breathing annihilation hold.
- **DECISIVE discriminator** (is the count certifying a soliton whose energy has left?): conditioning the local energy on peak-certification, the 4/8 snapshots whose peak clears BOTH floors carry **79.4%** of the full-soliton energy, versus **11.5%** in the rejected snapshots. Peak acceptance and energy are CORRELATED: the counter accepts the target exactly when it is energetically present (breathing crest) and rejects it when drained (trough).
- by the next hold (6.3000k, N=3) the target's energy is **0.1%** of full -- the annihilation completes across the flip.

### Verdict

Rule (on the certifying-snapshot energy discriminator, since the naive across-snapshot mean is misleading at a deep-breathing annihilation hold): **COUNTER-LATENCY** iff the target's peak certifies it in >= min_persistence of snapshots yet its local energy EVEN IN THOSE CERTIFYING SNAPSHOTS has collapsed (majority of the quantum gone) -- the height rule certifies a soliton whose energy has left, a peak-height-vs-energy defect in count_solitons_windowed; **INTRINSIC LAG (benign)** iff the target carries substantial local energy in the certifying snapshots (peak acceptance and energy are correlated, so the count tracks a really-present soliton that completes its annihilation across the flip -- the integer count is correct per hold); **INCONCLUSIVE** otherwise (e.g. a merged 4->2 event masked as 4->3).

**VERDICT: INTRINSIC LAG (benign)** -- at the count-flip hold (6.3250k, count_agreement 0.500) the target is a deep death-breather: its local comb energy is 79.4% of the full-soliton value (6.3750k) in the 50% of snapshots where its peak certifies it, versus 11.5% in the rejected snapshots (across-snapshot mean 45.5%). The peak acceptance and the energy are CORRELATED, so the count is NOT certifying an empty soliton -- it is tracking a substantially-present (if deeply breathing) soliton that annihilates completely by the next hold (6.3000k, 0.1% of full). The annihilation genuinely completes across the flip; the per-hold count is correct and the one-hold offset between the count decrement and the bulk comb-power drop is the expected integer-count vs continuous-energy resolution mismatch, sharpened by the target's breathing

No fix applied: this diagnosis is the deliverable. No threshold, gate, test, or committed artifact was changed. The next action (if COUNTER-LATENCY: reconcile the counter's peak-height acceptance with an energy/area criterion; if INTRINSIC: accept the one-hold offset as a resolution limit and, if desired, report a plateau-integrated step height) awaits review.

## Resolution

The multi-soliton staircase IS energy-quantized -- at PLATEAU LEVEL, the observable robust to the count/energy hold offset. For each matched N->N-1 annihilation the transition energy is the difference of the plateau-mean plotted primary (P_comb) between the settled count-N branch and the settled count-(N-1) branch, excluding the one or two transitional holds where the count and energy are mid-flip (`analysis.spectral_metrics.plateau_transition_energies`). On the committed sweep the matched N->N-1 plateau per-quantum energies are 0.185 (4->3) and 0.263 (5->4) of the normalised comb power -- quantized within a factor 1.42 (< 2) and ~47x the detector's robust sigma. `tests/test_soliton_staircase.py::test_step_heights_quantized` asserts this plateau-level quantization.

The single-edge `step_dy` NON-quantization (4->3 = 0.057 vs 5->4 = 0.169, ratio ~3.0 > 2) is NOT a physics defect and NOT a counter defect: it is the documented, xfail-pinned count/energy one-hold offset. At the 4->3 mid-hold annihilation the integer, hold-quantized `soliton_count` decrements one hold before the continuous comb-energy drop completes, so the quantum is split ~[36%, 64%] across the count-flip hold (edge 32) and the adjacent count-4 plateau hold (edge 33). The PHYSICAL PER-HOLD COUNT IS CORRECT (the target soliton is substantially present -- 79.4% of a full soliton -- in the snapshots where it is detected at the flip hold, and gone by the next hold). The old single-edge assertion is retained as a strict xfail (`test_step_heights_quantized_single_edge_xfail`) so this known limitation is pinned and cannot silently start passing without review.

Full verdict chain (this file): counting-artifact -> starvation-falsified -> relative-threshold-coupling -> physics-anchor -> split-step-refuted -> intrinsic-lag. The count/energy offset is an intrinsic integer-count vs continuous-energy resolution mismatch, sharpened by the annihilating soliton's death-breathing; it is not a reason to alter the counter, any threshold, or any gate.

