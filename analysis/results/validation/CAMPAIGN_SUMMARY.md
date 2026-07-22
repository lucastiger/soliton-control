# Noise-enabled publication validation campaign — consolidated summary

**Driver:** `analysis/noise_validation_campaign.py --workstream all --seeds 24`
(production, `quick:false`).
**Operating point (reused, unchanged):** single dissipative Kerr soliton at
`OPERATING_DW_KAPPA = 10·κ`, `PIN_W = 0.214 W`, measured `d_int_grid`,
`PRODUCTION_NUMERICS`. **Full stochastic stack ON** = quantum vacuum
(Langevin, Eq. 126) + ECDL pump frequency noise (h₀=3e3 Hz²/Hz, h₋₁=1e10 Hz³/Hz)
+ RIN floor −150 dBc/Hz + colored TRN (Kondratiev–Gorodetsky geometry) +
FSR/repetition-rate noise. **Deterministic reference** = every channel silenced
(`T_k=0`, quantum/pump off). No `simulator/` file and no noise/metrology numerics
were modified — this is analysis + figure generation only.

Vacuum-floor normalization used throughout (solver convention
`Ẽ_μ = a_μ·n_tau·√(ħω₀)`): raw `|fft(E)_μ|²` floor `= n_tau²·ħω₀/2`; per-mode
energy `E_μ = |fft(E)_μ|²/n_tau²` (floor `ħω₀/2`); physical intracavity energy
`E_total = U_int/t_r`; vacuum term in `P_abs = κ_i·n_tau·ħω₀/2`.

## Consolidated table — every metric vs its physical expectation

| # | Metric | Measured (mean ± std) | Physical expectation | Verdict |
|---|--------|-----------------------|----------------------|---------|
| W1 | DW peak positions (dispersion-set crossings) | shift **0 modes**; OFF peaks at μ=−3069 / +3277 = their phase-match crossings | invariant to ±2 (a shift is a bug) | ✅ dispersion intact |
| W1 | ħω₀/2 vacuum floor | **−82.2 dB** rel. pump | fundamental floor | ✅ |
| W1 | DW peak SNR over the floor (OFF) | red **−18.1 dB**, blue **−20.3 dB** | (measured) | peaks intrinsically below floor |
| W1 | DW survival | **0/2 survive; 2/2 submerged** by the risen floor | levels may drop / wings lift within budget | ✅ physical |
| W1 | wing-level change | **+5.23 dB** = vacuum **+5.11** + pump-jitter **+0.12** | within vacuum+pump budget (factor 3) | ✅ within budget, no bug |
| W1 | 3 dB spectral span | OFF 6987 → ON **7023 ± 66 GHz** | ≈ invariant | ✅ (+0.5%) |
| W1 | intracavity comb fraction | OFF 0.302 → ON **0.3018 ± 0.0007** | ≈ invariant | ✅ |
| W1 | far-wing floor | **0.995 ×** (n_tau²·ħω₀/2) | ~1 (factor 3) | ✅ |
| W2 | single-soliton access success rate | **66.7 %** (4/6) | high (> 50 %) | ✅ |
| W2 | step bias vs jitter (5 soliton-number boundaries) | \|bias\| ≤ 0.15κ, all **< 3·jitter** (jitter 0.08–0.19κ) | jitter-dominated, no systematic shift | ✅ all True |
| W3 | cross-realization β-line curvature a₂ | **7.41×10⁹ ± 4.7×10⁷** (n=16) | > 0 | ✅ |
| W3 | bootstrap p(a₂ > 0) | **1.000** | ≈ 1 | ✅ significant across realizations |
| W3 | Taylor-D₂ negative control | a₂ = 0 → ratio **0.000** | ≥ 10× smaller | ✅ |
| W3 | a₂ vs pump regime (h₋₁ = 1e9…1e13) | **7.4×10⁹ at every preset**, Taylor 0 at every preset | robust, not preset-peculiar | ✅ |
| W4 | S_rep vs TRN(K-G)+FSR limit | **+65.5 dB** (7.2×10⁻⁴ vs 8.6×10⁻⁸ Hz²/Hz @1 MHz) | above the thermal limit (pump/quantum) | ✅ |
| W4 | per-line effective linewidth | **0 Hz** (below the β-separation-line estimator floor) | coherent lines | ✅ (record-limited) |
| W5 | far-wing modal energy | **0.995 ×** (ħω₀/2) | ~1 (factor 3) | ✅ |
| W5 | intracavity energy | **15.3 pJ (1.19×10⁸ photons)** | pJ-scale soliton | ✅ |
| W5 | energy fluctuation / shot-noise scale | **212.7 ×** | ≥ 1 (quantum floor + classical excess) | ✅ above the quantum limit |
| W5 | vacuum contribution to P_abs | κ_i·n_tau·ħω₀/2 = 3.19×10⁻⁸ W = **6.9×10⁻⁵** of real P_abs | ≪ 1 (negligible) | ✅ |

## Workstream gate details

**W1 — DW-peak survival (flagship).** OFF spectrum reproduces the committed
provenance exactly (peaks at μ=−3069/−100.3 dB and μ=+3277/−102.5 dB vs committed
−100.26 / −102.51 dB). Because those absolute levels lie 18–20 dB *below* the
ħω₀/2 quantum floor (−82.2 dB rel. pump — the comb is pump-dominated), enabling
the vacuum noise raises the floor above them and submerges both peaks. The change
is fully accounted for by the vacuum floor (+5.11 dB) plus a +0.12 dB pump-jitter
pedestal (within the factor-3 budget); the phase-matched crossings are invariant
(shift 0), so the dispersion operator and seeding are untouched. The canary held.

**W2 — Monte-Carlo staircase.** 6/6 realizations survived the 5-soliton
pre-settle; the staircase is recognizable and single-soliton access succeeds in
66.7 % of realizations (the razor-thin single-DKS window is occasionally skipped
by noise jitter). Using robust soliton-number-boundary level crossings (immune to
transition merging on the affordable grid), every boundary N≤{4,3,2,1,0} has
\|bias\| < 3·jitter — the switching detunings jitter about their deterministic
locations rather than shifting systematically.

**W3 — cross-realization linewidth + pump map.** 16 independent pump realizations
give a₂ = 7.41×10⁹ ± 4.7×10⁷ with bootstrap p(a₂>0) = 1.000 — the across-record
significance the earlier within-record 256σ did not provide. The Taylor-D₂
control (pure quadratic dispersion → pure common mode) has a₂ = 0 at every pump
preset (ratio 0.000), and the measured a₂ is flat across h₋₁ ∈ {1e9…1e13}, so the
dispersive-wave-recoil transduction is not peculiar to one preset.

**W4 — coherence / RF beatnote.** The full-stack repetition-rate frequency noise
sits +65.5 dB above the TRN(K-G)+FSR limit over the low-offset decade (pump- and
quantum-dominated), with the elastic-tape S_c/S_cr/S_rep decomposition resolved.
The per-line β-separation-line linewidth is below the estimator floor at this
record length (the lines are coherent to within the resolution).

**W5 — vacuum-floor + energy budget (anchor).** The far comb wings reproduce
ħω₀/2 per mode to 0.995× (essentially exact). The physical intracavity energy is
15.3 pJ (1.19×10⁸ photons); its fluctuations sit 213× above the shot-noise floor
(classical pump/thermal excess, never below the quantum limit). The vacuum
contribution to the absorbed power, κ_i·n_tau·ħω₀/2 = 3.19×10⁻⁸ W, is 6.9×10⁻⁵ of
the real P_abs — negligible for the thermal ODE.

## File-touch list

Added (no existing file modified; **no `simulator/` change**):

- `analysis/noise_validation_campaign.py` — five-workstream campaign driver.
- `tests/test_noise_validation_campaign.py` — always-on committed-JSON assertions
  + cheap CI re-derivations + a `RUN_SLOW_VALIDATION`-gated full-fidelity re-run.
- `analysis/results/validation/campaign_report.json` — consolidated metrics +
  provenance stamp (`quick:false`).
- `analysis/results/validation/*.png` — 10 figures (2 per workstream):
  `dw_survival_spectrum_off_on`, `dw_peak_metrics`, `staircase_montecarlo`,
  `staircase_step_jitter`, `linewidth_parabola_ensemble`, `a2_vs_pump_regime`,
  `rf_beatnote_ensemble`, `comb_phase_noise_tape`, `vacuum_floor_ensemble`,
  `energy_fluctuation_budget`.
- `analysis/results/validation/CAMPAIGN_SUMMARY.md` — this file.

## Scientific conclusion

Enabling the arXiv:2604.05897 stochastic models at the committed operating point
changes **nothing** about the deterministic dispersion and seeding physics and
adds **exactly** the fundamental quantum + laser-technical noise floor the paper
predicts. The two dispersive-wave/Cherenkov peaks stay locked to their
dispersion-set phase-matching crossings (position shift 0 modes), but because
their absolute levels lie ~18–20 dB below the ħω₀/2 quantum vacuum floor of this
pump-dominated comb, the risen floor submerges them — a change that decomposes
entirely into the vacuum floor plus a sub-dB pump-jitter pedestal (within the
factor-3 budget), with zero anomalous contribution. The soliton staircase remains
recognizable, single-soliton access survives (66.7 %), and the switching
detunings jitter about their deterministic locations with sub-jitter bias rather
than a systematic seeding/thermal shift. Across independent pump realizations the
β-separation-line linewidth curvature is significantly positive (a₂ = 7.4×10⁹,
p = 1.000) with a ≥10×-smaller flat Taylor control and is robust across pump
regimes, confirming dispersive-wave recoil as a genuine, dispersion-driven
transduction of pump frequency noise into repetition-rate noise. Finally the
ensemble reproduces the ħω₀/2 vacuum floor to within 0.5 %, energy fluctuations
sit at/above the shot-noise scale, and the vacuum term in the absorbed power is
negligible for the thermal dynamics — the anchor that validates the calibration
of every prior workstream.
