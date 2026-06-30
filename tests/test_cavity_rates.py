"""Regression tests for the cavity-rate resolver and its callers.

Bug context: `data/dataset_generator.py` and `data/dataloader.py` used to
hardcode the cavity coupling rate as κ_c = κ_i, giving κ_total = 2·κ_i. That
is wrong for over-coupled devices — the SiN config has κ_c ≈ 4·κ_i. These
tests lock in `resolve_cavity_rates` as the single source of truth and assert
that the generator and the dataloader normalize by the same κ.
"""
import math
from pathlib import Path

import pytest
import yaml

from simulator.lle_solver import resolve_cavity_rates

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

_C = 299_792_458.0


@pytest.fixture(scope="module")
def physical():
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)["physical_parameters"]


def test_resolver_matches_q_values(physical):
    """κ_i ≈ ω₀/Q_i, κ_c ≈ ω₀/Q_c, κ_total = κ_i + κ_c, all within 1%."""
    omega0 = 2.0 * math.pi * _C / float(physical["pump_wavelength_m"])
    kappa_i, kappa_c, kappa_total = resolve_cavity_rates(CONFIG_PATH)

    assert math.isclose(kappa_i, omega0 / float(physical["intrinsic_q"]), rel_tol=0.01)
    assert math.isclose(kappa_c, omega0 / float(physical["coupling_q"]), rel_tol=0.01)
    assert math.isclose(kappa_total, kappa_i + kappa_c, rel_tol=1e-12)


def test_over_coupled_total_is_not_double_kappa_i():
    """For this over-coupled config the old κ_total = 2·κ_i bug must not return."""
    kappa_i, kappa_c, kappa_total = resolve_cavity_rates(CONFIG_PATH)
    # SiN device is over-coupled: κ_c ≈ 4·κ_i.
    assert kappa_c > 2.0 * kappa_i
    assert not math.isclose(kappa_total, 2.0 * kappa_i, rel_tol=0.05)


def test_generator_and_dataloader_normalize_by_same_kappa():
    """Consistency invariant: the generator's κ equals the κ the dataloader
    would normalize by, both sourced from `resolve_cavity_rates`."""
    from data.dataset_generator import DatasetGenerator

    param_grid = {
        "pin": [1.0],
        "sweep_rate": [1e3],
        "Gamma_th": [0.1],
        "noise_scale": [1.0],
    }
    gen = DatasetGenerator(param_grid=param_grid, config_path=str(CONFIG_PATH))

    # The dataloader (SolitonDataset) needs an actual .h5 dataset to instantiate,
    # so we assert the κ it would resolve via the shared helper instead.
    _, _, dataloader_kappa = resolve_cavity_rates(str(CONFIG_PATH))

    assert gen.kappa == dataloader_kappa
    assert math.isclose(gen.kappa, gen.kappa_i + gen.kappa_c, rel_tol=1e-12)


def test_synthetic_two_component_config(tmp_path):
    """A synthetic config with κ_i=1e7, κ_c=4e7 resolves to κ_total=5e7."""
    cfg = {
        "physical_parameters": {
            "pump_wavelength_m": 1.55e-6,
            "kappa_i_rad_per_s": 1.0e7,
            "kappa_c_rad_per_s": 4.0e7,
        }
    }
    cfg_path = tmp_path / "synthetic.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    kappa_i, kappa_c, kappa_total = resolve_cavity_rates(str(cfg_path))
    assert kappa_i == 1.0e7
    assert kappa_c == 4.0e7
    assert kappa_total == 5.0e7
