# Adiabatic detuning sweeps at pin = 0.214 W
Single-trajectory forward (blue->red) and reverse (red->blue) detuning sweeps through the MI window of the SiN microring LLE, thermal model ON at the config Gamma_th, n_tau = 512.
## Sweep design and adiabaticity
- Detuning range: `-2*kappa -> +5*kappa` (span 7*kappa), kappa = 1.519e+08 rad/s.
- `t_slow = 860000` round trips, t_r = 4.065e-11 s => sweep duration T = 3.496e-05 s = 6.99*tau_th (tau_th = 5.0e-06 s).
- Tuning rate R = 7*kappa/T = 3.041e+13 rad/s^2.
- R/kappa^2 = 1.32e-03: detuning moves 0.0013*kappa per cavity lifetime -> deeply adiabatic w.r.t. the cavity.
- R*tau_th/kappa = 1.00: detuning moves ~1.00*kappa per thermal time -> marginally adiabatic w.r.t. the thermal pole; acceptable because the steady thermo-optic shift is only a small fraction of kappa (see V3).

## What actually appears
- Forward sweep label histogram (label×count): 1×3, 2×427.
- Reverse sweep label histogram: 1×84, 2×346.
- Held-CW control (constant -4*kappa, deep blue) label histogram: 1×430.
- Max sech^2 spectral correlation: forward 0.104, reverse 0.091.

The sweeps ignite modulation instability (label 2) (no chaos, label 3, was reached in this run) inside the predicted window. **No clean single solitons (label 6) form** under this bare linear adiabatic sweep — consistent with the expectation stated in the task. The sech^2 spectral correlation stays well below the ~0.7 single-soliton bar, confirming the combs are MI (Turing/roll) patterns rather than sech^2 solitonic.

## Validations
- **[PASS] V1_MI_ignition** — 276 forward snapshot(s) with contrast>2 or label>=2 inside (0.5..5)*kappa; held-CW control max label = 1 (stays CW: True).
- **[PASS] V2_hysteresis** — U_int mean |fwd-rev|/mean = 34.1%, mean |label diff| = 0.27 over MI region; trajectories not identical: True.
- **[PASS] V3_thermal_sign** — corr(U_int, thermal_shift) = -1.00 (<0 => shifts DOWN as U rises); steady control thermal_shift = -0.096*kappa (sane fraction of kappa, not tens).
- **[PASS] V4_numerical_health** — all fields finite: True; held-CW control U_int tail rel-std = 0.00% (< 5% plateau).

## Follow-up work
Clean single-soliton nucleation is **follow-up work**. A bare linear detuning ramp lands in MI/chaotic states rather than a single soliton; reaching the single-soliton step of the LLE requires a dedicated nucleation protocol (e.g. a fast backward detuning kick / power ramp to cross the soliton-existence boundary from the chaotic branch, or a thermally-compensated trajectory). That protocol is intentionally NOT attempted here.
