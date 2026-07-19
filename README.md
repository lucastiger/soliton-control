# soliton-control

`soliton-control` is a scientific computing project scaffold for simulating and controlling soliton dynamics in thin-film lithium niobate (TFLN) and silicon nitride microresonators. The repository is organized around four major workflows:

- **Simulation** of Lugiato–Lefever equation (LLE) dynamics with thermal effects and realistic noise models.

  The SSFM solver optionally includes physically normalized **quantum vacuum noise** (Herr, Tikan & Kippenberg, arXiv:2604.05897, Sec. V.B.2, Eq. 126): each cavity mode is driven by the vacuum Langevin input √κ·ξ̂\_μ(t) with ⟨ξ̂\_μ(t)ξ̂†\_μ′(t′)⟩ = δ(t−t′)δ\_μμ′ (both loss baths combined; coherent/vacuum baths only), implemented in the classical truncated-Wigner (symmetric-ordering) limit as an additive complex Gaussian injected in the fast-time domain once per fine step, with per-quadrature std √(ħω₀·κ·n\_tau·dt/4). Its undriven steady state is the symmetric-ordered vacuum occupation of **½ photon per mode** — the same convention used for the optional cold-start seed, which replaces the legacy ad-hoc 1e-3·|E\_cw| noise with a vacuum-scale draw (⟨n\_μ⟩ = ½ at t = 0). The channel is gated behind `quantum_noise_enabled` in `config/sin_params.yaml` (**off by default** — the flag-off solver is bit-identical to the legacy solver), with `quantum_noise_seed_vacuum_init` selecting the vacuum cold-start seed and `hbar_omega0_j` optionally overriding ħω₀ (0 = compute from `pump_wavelength_m`; the pump-mode ħω₀ is used for all comb modes, a <1 % approximation across the comb span). See `analysis/quantum_noise_report.py` for the validation suite (½-photon vacuum equilibrium, cavity-linewidth decay, MI sideband selection from vacuum against paper Eq. 62, and the single-soliton spectral floor at ħω₀/2).

  Config-encoding note: all quantum-noise booleans (and the cadence enum) are encoded as **0/1 integers** rather than YAML `true`/`false`/`null`, because the config regression tests pin every `physical_parameters` leaf to parse as a plain number; the solver validates them as boolean-valued. Two further knobs: `quantum_noise_injection_cadence` selects `0` = one injection per fine step (the exact prescription, default) or `1` = one injection per round trip with the variance rescaled to dt = t\_r — a CPU-performance option valid because κ·t\_r ≈ 6.2×10⁻³ ≪ 1 (steady occupation 0.5015 vs 0.5; bit-identical to the fine cadence when `fine_cadence_M = 1`). When — and only when — the channel is enabled, the state labeler activates its vacuum-floor parameters `labeler_vacuum_floor_margin` (default 10 = +10 dB) and `labeler_envelope_smooth_modes` (default 8): the single-soliton envelope gate smooths the linear spectrum and clips it at margin × n\_tau²ħω₀/2 (the per-mode vacuum level in raw |FFT|² units, whose single-snapshot fluctuation is ≈5.6 dB), and the OFF power floor is lifted to at least margin × n\_tau·ħω₀/2 so a vacuum-filled cavity labels OFF; with the channel disabled the labeler is bit-identical to the legacy one. The vacuum background's contribution to absorbed power is κ\_i·n\_tau·ħω₀/2 ≈ 1.6×10⁻⁸ W at n\_tau = 8192 — negligible for the thermal ODE.
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
