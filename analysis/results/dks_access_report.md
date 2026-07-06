# Single dissipative-Kerr-soliton (DKS) access protocol

## Stale artifacts (post-PR-#40 rebase status)

Classification of every file in `analysis/results/` after the dispersion-layer
fixes on branch `claude/measured-dispersion-grid-px220y` (D1 soliton-rest gauge
from PR #40; `_fit_local_d2` window fix; crossing-derived `dispersive_wave_peaks`).
Nothing in this directory should be trusted without checking this table.
**Validation rerun completed with the corrected numerics defaults** (float64,
n_substeps=4, 2/3 dealias + edge absorber, smooth D_int extrapolation,
fine-cadence mode available, **dispersion-validity mask OFF**): the
spectrum/summary PNGs below are regenerated from that run; see
"Dispersive-wave peak scan" for the measured floor and DW peaks. The earlier
rerun had the |D_int*t_r|-keyed validity mask ON (threshold 1.0 rad), which
amputated real comb spectrum over mu ~ [-2950, -1000] and [+1150, +3050]; the
mask is now keyed to the per-sub-step mismatch phase, defaults OFF, and is
kept only as an opt-in guard for n_substeps=1 runs
(see `simulator/lle_solver.py`). The five spectral-integrity checks V1-V5
(`tests/test_dks_spectral_integrity.py`) all pass on the rerun.

| File | Status | Reason |
|------|--------|--------|
| `dks_access_report.md` | **REGENERATED HERE** | Dispersion sections rebuilt from the corrected code + CSV (pure analysis, no solver): corrected local D2, rest-gauge crossings, crossing-derived DW windows. Solver-derived sections (validated soliton, reproducibility, control, existence band) carried over unchanged because mode powers are gauge-invariant under the D1 fix. |
| `dks_single_soliton_spectrum.png` | **REGENERATED — mask-OFF validation rerun** | Rebuilt with the corrected defaults (float64, n_substeps=4, 2/3 dealias, edge absorber, **validity mask OFF**; n_tau=16384, 12000 RT). The previous version (mask ON at \|D_int*t_r\| > 1) amputated the comb over mu ~ [-2950, -1000] / [+1150, +3050]; the envelope is now smooth and continuous from the pump out to BOTH DW peaks (V2 worst tail-line deficit 6.7 dB blue / 0.9 dB red, limit 25 dB). DW peaks -90.8 dB @ 1096 nm (mu = +3266) and -87.7 dB @ 2529 nm (mu = -3069), within a few dB/modes of the numpy reference (-95.4 @ 1095, -93.1 @ 2529). Floor median -355 dB beyond the 2/3 cutoff. |
| `dks_single_soliton_summary.png` | **REGENERATED — mask-OFF validation rerun** | Same run as the spectrum PNG; comb panel shows the full dynamic range down to the float64 floor. |
| `dks_existence_map.csv` | **VALID — mask-OFF boundary re-probe confirms** | The full map re-run (band **[6.5, 16.0] kappa**, was [7.5, 13.0]) was generated with the buggy validity mask ON; but the mask bands sit at \|mu\| > ~1000 where the comb tail is < -80 dB, so the classification is insensitive to it. Confirmed by re-probing the four band-boundary points (6.0 / 6.5 / 16.0 / 16.5 kappa) with the fixed numerics (mask OFF, same seed, 1 tau_th, n_tau = 8192): labels 1 / 6 / 3 unchanged at every point (6.0 -> 1, 6.5 -> 6, 16.0 -> 6, 16.5 -> 3), no is_single flip, so the map was NOT regenerated. |
| `dks_existence_map.png` | **VALID — mask-OFF boundary re-probe confirms** | Plot of `dks_existence_map.csv` (see that row). |
| `adiabatic_sweeps.png` | **VALID — predates numerics fixes** | Unaffected by these analysis-layer fixes (gauge-invariant dynamics, no scanner dependence). Predates the dealiasing/float64/sub-stepping numerics fixes; re-check if quantitative wing/width numbers are used. |
| `adiabatic_sweeps_report.md` | **VALID — predates numerics fixes** | Same as above. |
| `forward_sweep.csv` | **VALID — predates numerics fixes** | Same as above. |
| `reverse_sweep.csv` | **VALID — predates numerics fixes** | Same as above. |
| `control_held_cw.csv` | **VALID — predates numerics fixes** | Same as above. |

Operating point: pin = 0.214 W, n_tau = 8192, thermal model ON at the config Gamma_th. kappa = 1.519e+08 rad/s, kappa_c = 1.215e+08 rad/s, gamma_LLE = 1.029e+18 J^-1 s^-1, D2 = 3.770e+04 rad/s^2, tau_th = 5.0e-06 s (123000 round trips).

## Access protocol

Two routes were implemented (`analysis/dks_access.py`):

- **(b) Direct single-sech seeding (`access_by_seeding`) — KEPT.** An analytic bright-sech ansatz `B*sech(t/tau_s)` (B = sqrt(2*dw/gamma), tau_s = sqrt(beta2/(2*dw))) on the CW background is injected as the warm-start field (`e0_override`) at a detuning inside the existence window, then integrated to steady state. This is deterministic and reliably yields exactly one soliton.
- **(a) Forward/backward tuning (`access_by_forward_backward`).** Cold forward ramp (blue->red) through MI to a deep red detuning, then a backward tune down to the target detuning to shed excess solitons, held to settle. Carried as one continuous trajectory via the warm-start path. Reported for completeness.

## Validated single soliton

_Solver-derived; carried over from the pre-rebase run. Mode powers are gauge-invariant under the D1 fix, so these numbers are unchanged. The mask-OFF rerun (12000 RT, n_tau = 16384, corrected numerics defaults — see "Dispersive-wave peak scan") re-validates the state: class 6, n_peaks = 1, env corr = 0.9960, U_int tail rel-std = 4.09%._

Route (b) at programmed delta_omega = 8.0 kappa (effective 7.96 kappa after thermal shift), integrated for t_slow = 615000 round trips = 5.0 tau_th:

- single dominant temporal peak: **n_peaks = 1**
- sech^2 spectral (envelope) correlation: **0.9983** (> 0.9 required; r^2 = 0.9967)
- U_int tail rel-std over the long integration: **4.20%** (< 5% required)
- NumPy labeler class: **6** (6 = single soliton)
- finite (no NaN/Inf): **True**
- peak-to-mean contrast: 125.8

## Reproducibility across RNG seeds

Seeds tested: [0, 1, 2]. Single-soliton success rate: **3/3** = 100%.

- seed 0: n_peaks=1, class=6, env_corr=0.998, single=True
- seed 1: n_peaks=1, class=6, env_corr=0.998, single=True
- seed 2: n_peaks=1, class=6, env_corr=0.998, single=True

## Control (no protocol)

Cold start held at the same detuning (8.0 kappa) with NO seed and NO tuning protocol: class = **1**, n_peaks = 2775, sech^2 env corr = nan, contrast = 1.00. The plain (unseeded) run does **not** yield a class-6 single soliton — confirming the protocol is doing the work. (The bare adiabatic forward sweep likewise lands in MI/Turing, never a single soliton; see `analysis/adiabatic_sweeps.py`.)

## Forward/backward route result

`access_by_forward_backward` (forward -1.0->9.0 kappa, back to 8.0 kappa): class = 4, n_peaks = 1, env_corr = 0.996, single = False.

## Existence window (seeded)

**Regenerated post-rebase** (new seed with corrected local D2; numerics: float64, n_substeps=4, 2/3 dealias, edge absorber, validity mask ON — see below; 1 tau_th = 123000 RT per point, n_tau = 8192, rng seed 0). The staged boundary re-probe (7, 7.5, 13, 13.5 kappa) flipped 7.0 and 13.5 kappa to single, so the full map was regenerated over 1.0-18.0 kappa in 0.5 kappa steps (35 points).

**Mask-OFF confirmation:** that map re-run predates the validity-mask fix (the \|D_int*t_r\|-keyed mask amputated the comb at \|mu\| > ~1000, where the tail is already < -80 dB — far below anything the labeler keys on). Re-probing the four band-boundary points (6.0 / 6.5 / 16.0 / 16.5 kappa) with the corrected defaults (`PRODUCTION_NUMERICS`, mask OFF; same seed, 1 tau_th, n_tau = 8192) reproduces every label exactly (1 / 6 / 6 / 3) with no is_single flip, so the **[6.5, 16.0] kappa** band stands and the map was not regenerated.

Class-6 single solitons now appear in a single contiguous detuning band **[6.5, 16.0] kappa** (20 sampled points, contiguous = True). The old map's band was [7.5, 13.0] kappa: the corrected `_fit_local_d2` widens the seed sech by 1.42x (tau_s ∝ sqrt(D2)), which is enough for the seed to survive at both former boundaries.

Note on the band location: at pin = 0.214 W the pump is ~61x the MI threshold, a very hard drive. The single-DKS existence window therefore sits at higher detuning than the generic `kappa/2 < dw < ~5 kappa` estimate. Below the band the seed either sinks into background MI (labels 3, dw <= 3.5 kappa) or collapses to CW (label 1, 4.0-6.0 kappa); above 16.0 kappa the steady state keeps a single temporal peak but the labeler drops it from class 6 (sech^2 env corr falls below threshold, label 3).

## Dispersion: local D2 (corrected)

The near-pump curvature of the measured integrated dispersion `D_int(mu)` (from `config/pyLLE_dispersion_w4400_h800.csv`) sizes the analytic sech seed. `_fit_local_d2` fits `D_int(mu) ~ (D2/2) mu^2` over **5 < |mu| <= 300**, giving

- **CSV local D2 = 4.980e+04 rad/s^2 = 2*pi*7.93 kHz** (measured FSR = 2.4455e+10 Hz).

The fit window excludes the innermost `|mu| <= 5` modes, where a localized pump-neighborhood defect displaces resonances by up to -27 MHz. The old `|mu| <= 40` window sat on top of that defect and returned a curvature biased ~2x high, **2*pi*15.7 kHz**. The corrected window is on the converged plateau (window convergence: +/-100 -> 2*pi*6.4 kHz, +/-300 -> 2*pi*7.9 kHz, +/-400 -> 2*pi*7.8 kHz). Because the fit is degree 2, the (defect-biased) linear D1 term does not affect the recovered quadratic coefficient.

The corrected local D2 is 1.32x the config `d2_rad_per_s2` (3.770e+04), well within the 0.5x-2x consistency band, so the previous **">2x config mismatch" warning no longer fires** for this CSV (the old 2*pi*15.7 kHz was 2.62x the config value and tripped it).

## Dispersive waves: phase-matched crossings (rest gauge)

Working in the soliton-rest gauge (the D1 fixed in PR #40 — a raw central-difference D1 tilted `D_int` and produced a spurious `mu ~ +2400 / 1188 nm` crossing, now removed), the dispersive-wave (Cherenkov) phase-matching condition `D_int(mu) = delta_omega` is scanned for sign changes of `D_int(mu) - delta_omega` restricted to `|mu| > 500` (this excludes the comb-core crossings near `|mu| ~ 200-300`). At **delta_omega = 8 kappa** there are exactly two far-detuned crossings:

| crossing mu | wavelength | side |
|-------------|-----------|------|
| **+3270** | **~1096 nm** | blue |
| **-3051** | **~2520 nm** | red |

Wavelengths use `lambda(mu) = c / (f0 + mu*D1/(2*pi))` with `f0` = the CSV pump frequency at the mu=0 row (1.93589e+14 Hz, ~1548.6 nm) and `D1/(2*pi)` = 2.4455e+10 Hz.

Both true dispersive waves fall **outside** the old scanner's hard-coded `[1120, 1260]` / `[2150, 2400]` nm windows, which is why the old run reported no edge peaks. The rewritten `dispersive_wave_peaks()` derives its search windows from these crossings instead: for each crossing `mu_x` it scans the spectrum over `mu_x +/- 30` modes for the largest dB peak, then reports `(lambda_nm, mu, peak_dB, prominence_dB)`, where prominence is the peak height above a local sech-tail baseline (a line fit in dB over `|mu|` in `[400, 700]` on the same sign side, extrapolated to the peak mode). The `+/-30`-mode window covers the empirical peaks, which land a few modes further out (**+3281 / -3069 = 1095 / 2529 nm**) due to soliton recoil.

## Dispersive-wave peak scan (mask-OFF validation rerun)

Measured on the final field of the validation rerun with the corrected numerics defaults: programmed delta_omega = 8 kappa (kappa = 1.519e8 rad/s), pin = 0.214 W, seeded single soliton (corrected local D2 = 2*pi*7.93 kHz), **n_tau = 16384**, 12000 round trips, solver flags float64 (module-wide x64), `n_substeps=4`, `dealias_two_thirds=True` (cutoff |mu| = 5461, retaining both DW regions), `edge_absorber=True`, **`dispersion_validity_mask=False`**. The previous rerun's mask (keyed to |D_int*t_r| > 1, i.e. |D_int| > ~162 kappa) was amputating real comb spectrum — the linear exponential is exact at any phase; the genuine discrete-map artifact (spurious FWM at mismatch phase ~2*pi per nonlinear kick) sits at ~4070 kappa with n_substeps=4, outside the band — so its -103.5/-100.5 dB DW peaks and -206 dB blue "floor" were artifacts of the amputation. Effective detuning drifts only 8.000 -> 7.996 kappa over this window (tau_th ~ 123k RT), thermal loop ON.

All numbers dB relative to the max mode (matching the V1-V5 checks in `tests/test_dks_spectral_integrity.py`, all of which pass on this run):

- **Numerical floor** (median over |mu| > n_tau/3 + 50 = 5511, past the 2/3 dealias cutoff): **-355.2 dB** (blue) / **-355.5 dB** (red) — at the float64 roundoff limit, <= -300 dB required (V5). The 3200 < |mu| < 3900 window used by the previous rerun is no longer a floor readout: with the mask OFF it contains the real DW shoulders (medians -174.5 / -190.1 dB there).
- **DW peaks** (max within +/-60 modes of the crossings, prominence over the sech-tail baseline — linear dB fit over |mu| in [400, 700] on the same sign side, extrapolated):
  - blue: **-90.8 dB at mu = +3266 (1096 nm)**, prominence **+150 dB** — audit reference -95.4 dB at mu = +3281 (1095 nm): delta +4.6 dB, 15 modes.
  - red: **-87.7 dB at mu = -3069 (2529 nm)**, prominence **+149 dB** — audit reference -93.1 dB at mu = -3069 (2529 nm): delta +5.4 dB, 0 modes.
  - Both clear the >50 dB prominence criterion (V4) by a wide margin.
- **Spectral integrity**: dB-linear sech tail over 400 <= |mu| <= 1500, RMS < 3 dB per side, slope asymmetry < 15% (V1); NO interior hole — everywhere in 300 < |mu| < 2900 the spectrum stays within 25 dB of the fitted tail line, worst deficit 6.7 dB (blue, mu = 2627) / 0.9 dB (red, mu = -2598) (V2); sech^2 core fit over 3 <= |mu| <= 200 with RMS 1.50 dB, width ~124 modes (V3).
- Steady state remains a class-6 single soliton: n_peaks = 1, sech^2 env corr = 0.9960, U_int tail rel-std = 4.09%.

## Labeler note

Both labelers return class 6 for these states. The JAX scan-time labeler (which produces label_history for the training dataset) keys class 6 on a single temporal peak plus a smooth monotonic sech^2 spectral envelope; an earlier 'fraction of power in the top ~32 points' heuristic mislabeled a DKS on a bright CW background as chaotic (class 3) and was replaced. Classification in this study uses the NumPy sech^2-fit labeler.

## Artifacts

See the **Stale artifacts** table at the top for the current status of each file.

- `dks_single_soliton_spectrum.png` — optical power vs wavelength (nm) — regenerated (mask-OFF rerun)
- `dks_single_soliton_summary.png` — waveform, comb, U_int stability — regenerated (mask-OFF rerun)
- `dks_existence_map.png` / `dks_existence_map.csv` — existence window — valid (boundary points re-confirmed with mask OFF)
