# soliton-control

`soliton-control` is a scientific computing project scaffold for simulating and controlling soliton dynamics in thin-film lithium niobate (TFLN) and silicon nitride microresonators. The repository is organized around four major workflows:

- **Simulation** of Lugiato–Lefever equation (LLE) dynamics with thermal effects and realistic noise models.

  The SSFM solver optionally includes physically normalized **quantum vacuum noise** (Herr, Tikan & Kippenberg, arXiv:2604.05897, Sec. V.B.2, Eq. 126): each cavity mode is driven by the vacuum Langevin input √κ·ξ̂\_μ(t) with ⟨ξ̂\_μ(t)ξ̂†\_μ′(t′)⟩ = δ(t−t′)δ\_μμ′ (both loss baths combined; coherent/vacuum baths only), implemented in the classical truncated-Wigner (symmetric-ordering) limit as an additive complex Gaussian injected in the fast-time domain once per fine step, with per-quadrature std √(ħω₀·κ·n\_tau·dt/4). Its undriven steady state is the symmetric-ordered vacuum occupation of **½ photon per mode** — the same convention used for the optional cold-start seed, which replaces the legacy ad-hoc 1e-3·|E\_cw| noise with a vacuum-scale draw (⟨n\_μ⟩ = ½ at t = 0). The channel is gated behind `quantum_noise_enabled` in `config/sin_params.yaml` (**off by default** — the flag-off solver is bit-identical to the legacy solver), with `quantum_noise_seed_vacuum_init` selecting the vacuum cold-start seed and `hbar_omega0_j` optionally overriding ħω₀ (0 = compute from `pump_wavelength_m`; the pump-mode ħω₀ is used for all comb modes, a <1 % approximation across the comb span). See `analysis/quantum_noise_report.py` for the validation suite (½-photon vacuum equilibrium, cavity-linewidth decay, MI sideband selection from vacuum against paper Eq. 62, and the single-soliton spectral floor at ħω₀/2).

  Config-encoding note: all quantum-noise booleans (and the cadence enum) are encoded as **0/1 integers** rather than YAML `true`/`false`/`null`, because the config regression tests pin every `physical_parameters` leaf to parse as a plain number; the solver validates them as boolean-valued. Two further knobs: `quantum_noise_injection_cadence` selects `0` = one injection per fine step (the exact prescription, default) or `1` = one injection per round trip with the variance rescaled to dt = t\_r — a CPU-performance option valid because κ·t\_r ≈ 6.2×10⁻³ ≪ 1 (steady occupation 0.5015 vs 0.5; bit-identical to the fine cadence when `fine_cadence_M = 1`). When — and only when — the channel is enabled, the state labeler activates its vacuum-floor parameters `labeler_vacuum_floor_margin` (default 10 = +10 dB) and `labeler_envelope_smooth_modes` (default 8): the single-soliton envelope gate smooths the linear spectrum and clips it at margin × n\_tau²ħω₀/2 (the per-mode vacuum level in raw |FFT|² units, whose single-snapshot fluctuation is ≈5.6 dB), and the OFF power floor is lifted to at least margin × n\_tau·ħω₀/2 so a vacuum-filled cavity labels OFF; with the channel disabled the labeler is bit-identical to the legacy one. The vacuum background's contribution to absorbed power is κ\_i·n\_tau·ħω₀/2 ≈ 1.6×10⁻⁸ W at n\_tau = 8192 — negligible for the thermal ODE.

  The solver also optionally includes **pump-laser noise** — frequency noise and relative intensity noise (RIN) — following the same paper (Herr, Tikan & Kippenberg, arXiv:2604.05897, Secs. V.B.4–V.B.5). Both channels are synthesized **host-side in float64** once per trajectory (`PumpNoise` in `simulator/noise_models.py`) and fed into the *existing* equations of motion, so the cavity's transfer function (low-pass filtering and quadrature rotation) and the thermal transduction pathway emerge from the solver itself — no transfer function is hand-implemented. **Frequency noise:** because the solver frame co-rotates with the pump, the instantaneous laser-frequency deviation δν\_p(t) is exactly a detuning noise; since δω ≡ ω\_res − ω\_p, a positive laser-frequency excursion *lowers* δω, so the contribution **−2π·δν\_p(t)** is summed into the per-round-trip detuning-noise sequence (no solver-scan change) and returned as `pump_freq_noise_history`. Its one-sided PSD is S\_δν(f) = h₀ + h₋₁/f on f ∈ [1/(t\_slow·t\_r), 1/(2t\_r)]: a white plateau h₀ carrying the intrinsic Lorentzian linewidth Δν\_L = π·h₀ (i.i.d. per round trip, variance h₀·f\_s/2) plus a flicker term h₋₁/f synthesized by Hermitian FFT (DC bin clamped to the first bin). **RIN:** P\_in(t) = P̄\_in·(1+ε(t)) from S\_ε(f) = 10^(floor/10) + 10^(excess/10)·(f\_c/f) below the corner f\_c (floor-only above), clipped so 1+ε ≥ 0 (a warning fires if >0.01 % of samples clip). The per-round-trip pump-power scale 1+ε is threaded as `pump_scale_sequence`, so the pump kick becomes √(max(κ\_c·P̄\_in·(1+ε), 0))·dt\_sub held constant across the fine steps (RIN bandwidth ≪ FSR ⇒ per-round-trip resolution is exact), and the absorbed-power/thermal pathway then transduces RIN → ΔT → detuning automatically (the paper's thermal-transfer mechanism); `pump_rin_epsilon_history` is returned for diagnostics. Both channels are gated behind `pump_noise_enabled` (0/1, **off by default** — the flag-off solver is bit-identical to the legacy solver, and the RIN-disabled path traces zero extra ops in the scan body via a static `None` sequence). The knobs are `pump_freq_noise_h0_hz2_per_hz` (ECDL ≈ 3×10³ ⇒ Δν\_L ≈ 10 kHz; fiber laser ≈ 30 ⇒ ≈ 100 Hz), `pump_freq_noise_hm1_hz3_per_hz` (flicker, representative ECDL ≈ 10¹⁰), and `pump_rin_floor_dbc_per_hz` / `pump_rin_excess_dbc_per_hz` / `pump_rin_corner_hz`; validation rejects negative h₀/h₋₁ and any RIN value above −80 dBc/Hz (a guard against accidental linear-vs-dB entry), and `enabled = 0` forces every channel inert regardless of the numbers. New per-trajectory PRNG subkeys are *appended* to the existing key chain, so enabling pump noise never perturbs a legacy stream. See `tests/test_pump_noise.py` for the validation suite (PSD fidelity within 3 dB/octave over 3 decades, exact −2πδν sign convention, the linearized CW low-pass transfer at f\_mod ∈ {κ/20, κ/2, 5κ}/2π, RIN energy balance, determinism, and config-validation triggers) and `analysis/pump_noise_report.py` for the physics study, including the **dispersive-wave-recoil** contrast (pump frequency noise is predominantly common-mode with Taylor-D₂-only dispersion, but couples more strongly into repetition-rate wander once the measured `d_int_grid` supplies the DW phase matching) and the RIN → ΔT transduction via R\_th ≈ 0.545 K/W.
- **Data generation** for large-scale supervised/physics-informed learning.
- **Model training** for a physics-informed recurrent network (PI-RNN).
- **Closed-loop control** using model predictive control (MPC) and hardware integration stubs.

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd soliton-control
   ```

2. Create and activate a Python environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage Overview

- Place and tune physical constants in `config/tfln_params.yaml`.
- Implement solvers and noise models under `simulator/`.
- Generate datasets with scripts in `data/`.
- Develop and train PI-RNN models in `model/`.
- Integrate control loops and hardware APIs in `control/`.
- Run evaluation and visualization workflows from `analysis/`.
- Use `notebooks/exploration.ipynb` for exploratory experiments.
- Add validation coverage in `tests/` as modules are implemented.

## Project Layout

```text
tfln-soliton-control/
├── README.md
├── requirements.txt
├── config/
│   └── tfln_params.yaml
├── simulator/
├── data/
├── model/
├── control/
├── analysis/
├── notebooks/
└── tests/
```
