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
