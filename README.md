# soliton-control

`soliton-control` is a scientific computing project scaffold for simulating and controlling soliton dynamics in thin-film lithium niobate (TFLN) and silicon nitride microresonators. The repository is organized around four major workflows:

- **Simulation** of Lugiato–Lefever equation (LLE) dynamics with thermal effects and realistic noise models.
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
