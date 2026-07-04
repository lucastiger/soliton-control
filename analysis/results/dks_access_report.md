# Single dissipative-Kerr-soliton (DKS) access protocol

## Stale artifacts (post-PR-#40 rebase status)

Classification of every file in `analysis/results/` after the dispersion-layer
fixes on branch `claude/measured-dispersion-grid-px220y` (D1 soliton-rest gauge
from PR #40; `_fit_local_d2` window fix; crossing-derived `dispersive_wave_peaks`).
Nothing in this directory should be trusted without checking this table. The
solver was **not** re-run in this pass: the D1 fix is gauge-invariant for mode
powers, so any solver-derived artifact regenerated now would be numerically
identical to the stale one but carry a misleading fresh timestamp.

| File | Status | Reason |
|------|--------|--------|
| `dks_access_report.md` | **REGENERATED HERE** | Dispersion sections rebuilt from the corrected code + CSV (pure analysis, no solver): corrected local D2, rest-gauge crossings, crossing-derived DW windows. Solver-derived sections (validated soliton, reproducibility, control, existence band) carried over unchanged because mode powers are gauge-invariant under the D1 fix. |
| `dks_single_soliton_spectrum.png` | **STALE — regenerate post-rebase** | Contains the aliasing-dominated floor (-100..-120 dB) and annotations from the old fixed-window scanner. Regenerate in the post-rebase validation rerun (float64 + dealiasing); expected floor <= -180 dB with DW peaks near 1095 nm (~-95 dB) and 2529 nm (~-93 dB). |
| `dks_single_soliton_summary.png` | **STALE — regenerate post-rebase** | Same aliasing floor and old-scanner annotations as the spectrum PNG; regenerate in the post-rebase validation rerun. |
| `dks_existence_map.csv` | **STALE — regenerate post-rebase** | Mode powers are gauge-invariant under the D1 fix, but the `_fit_local_d2` correction widens the seed sech by 1.42x (tau_s ∝ sqrt(D2), sqrt(15.7/7.8) = 1.42). If the map probes seeded-soliton survival, band edges may shift. Re-validate after the rebase; not re-run now. |
| `dks_existence_map.png` | **STALE — regenerate post-rebase** | Plot of `dks_existence_map.csv`; same seed-width caveat. |
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

_Solver-derived; carried over from the pre-rebase run. Mode powers are gauge-invariant under the D1 fix, so these numbers are unchanged; the fields/PNGs are re-validated in the post-rebase rerun._

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

Class-6 single solitons appear in a single contiguous detuning band **[7.5, 13.0] kappa** (12 sampled points, contiguous = True).

Note on the band location: at pin = 0.214 W the pump is ~61x the MI threshold, a very hard drive. The single-DKS existence window therefore sits at higher detuning than the generic `kappa/2 < dw < ~5 kappa` estimate — the measured lower edge is where the CW background becomes MI-stable enough to hold a soliton, and the upper edge is where the soliton amplitude collapses back to CW. Below the band the seed is swamped by background MI; above it the seed decays to CW. (Seed-width caveat: the `_fit_local_d2` correction widens the seed sech by 1.42x, so band edges are re-validated in the post-rebase rerun; see the stale-artifacts table.)

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

## Dispersive-wave peak scan on the committed field

**Pending post-rebase rerun.** Only the (stale) PNG spectra exist on disk for the committed run — no raw final field / snapshot is stored — so the rewritten `dispersive_wave_peaks()` cannot be re-run against the actual comb here. Expected from the post-rebase validation rerun (float64 + dealiasing): peaks near **1095 nm (~-95 dB)** and **2529 nm (~-93 dB)**, above a <= -180 dB dealiased floor, with prominence measured against the sech-tail baseline.

## Labeler note

Both labelers return class 6 for these states. The JAX scan-time labeler (which produces label_history for the training dataset) keys class 6 on a single temporal peak plus a smooth monotonic sech^2 spectral envelope; an earlier 'fraction of power in the top ~32 points' heuristic mislabeled a DKS on a bright CW background as chaotic (class 3) and was replaced. Classification in this study uses the NumPy sech^2-fit labeler.

## Artifacts

See the **Stale artifacts** table at the top for the current status of each file.

- `dks_single_soliton_spectrum.png` — optical power vs wavelength (nm) — STALE
- `dks_single_soliton_summary.png` — waveform, comb, U_int stability — STALE
- `dks_existence_map.png` / `dks_existence_map.csv` — existence window — STALE
