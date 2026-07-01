# Single dissipative-Kerr-soliton (DKS) access protocol

Operating point: pin = 0.214 W, n_tau = 8192, thermal model ON at the config Gamma_th. kappa = 1.519e+08 rad/s, kappa_c = 1.215e+08 rad/s, gamma_LLE = 1.029e+18 J^-1 s^-1, D2 = 3.770e+04 rad/s^2, tau_th = 5.0e-06 s (123000 round trips).

## Access protocol

Two routes were implemented (`analysis/dks_access.py`):

- **(b) Direct single-sech seeding (`access_by_seeding`) — KEPT.** An analytic bright-sech ansatz `B*sech(t/tau_s)` (B = sqrt(2*dw/gamma), tau_s = sqrt(beta2/(2*dw))) on the CW background is injected as the warm-start field (`e0_override`) at a detuning inside the existence window, then integrated to steady state. This is deterministic and reliably yields exactly one soliton.
- **(a) Forward/backward tuning (`access_by_forward_backward`).** Cold forward ramp (blue->red) through MI to a deep red detuning, then a backward tune down to the target detuning to shed excess solitons, held to settle. Carried as one continuous trajectory via the warm-start path. Reported for completeness.

## Validated single soliton

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

Note on the band location: at pin = 0.214 W the pump is ~61x the MI threshold, a very hard drive. The single-DKS existence window therefore sits at higher detuning than the generic `kappa/2 < dw < ~5 kappa` estimate — the measured lower edge is where the CW background becomes MI-stable enough to hold a soliton, and the upper edge is where the soliton amplitude collapses back to CW. Below the band the seed is swamped by background MI; above it the seed decays to CW.

## Resolution / dispersion note

The solver is driven by the full MEASURED integrated dispersion D_int(mu) (from `config/pyLLE_dispersion_w4400_h800.csv`), not a pure D2 parabola, so it can in principle radiate dispersive waves (Cherenkov peaks) at dispersion-matched band edges rather than showing a plain sech^2 roll-off. Those edges span roughly 1150-2300 nm, thousands of cavity modes from the pump, so the default grid is n_tau = 8192 to resolve the full display window (measured FSR = 2.4455e+10 Hz; CSV local D2 = 9.894e+04 rad/s^2). The analytic sech seed is sized from the local (near-pump) curvature; the full dispersion then reshapes the wings. The measured D_int has a blue-side phase-matching point near mu ~ +2400 (~1180 nm), but at this operating point the warm-started single-sech comb decays to the numerical floor before reaching it, so no edge peaks are populated in the [1120,1260] / [2150,2400] nm windows scanned by the run (the two largest local maxima per window are printed to stdout for lab comparison).

Both labelers return class 6 for these states. The JAX scan-time labeler (which produces label_history for the training dataset) keys class 6 on a single temporal peak plus a smooth monotonic sech^2 spectral envelope; an earlier 'fraction of power in the top ~32 points' heuristic mislabeled a DKS on a bright CW background as chaotic (class 3) and was replaced. Classification in this study uses the NumPy sech^2-fit labeler.

## Artifacts

- `dks_single_soliton_spectrum.png` — optical power vs wavelength (nm)
- `dks_single_soliton_summary.png` — waveform, comb, U_int stability
- `dks_existence_map.png` / `dks_existence_map.csv` — existence window
