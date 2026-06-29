"""Regression tests for config/tfln_params.yaml parsing.

PyYAML's default (1.1) resolver only treats an exponential float as a float
when its exponent carries an explicit sign (e.g. ``2.0e+11``). An unsigned
exponent such as ``2.0e11`` is loaded as the *string* ``'2.0e11'`` instead.
These tests lock in the fix that signs every exponent in the config file so
all physical parameters parse natively as numbers.
"""
import math
from pathlib import Path

import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tfln_params.yaml"


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _str_leaves(node, path=""):
    """Yield (path, value) for every ``str`` leaf scalar in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _str_leaves(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _str_leaves(value, f"{path}[{i}]")
    elif isinstance(node, str):
        yield (path, node)


def test_no_string_leaves_under_physical_parameters(config):
    """No scalar under ``physical_parameters`` should load as a ``str``.

    A string leaf here means an exponent lost its sign and PyYAML failed to
    coerce the value to a float — the exact bug this guards against.
    """
    physical = config["physical_parameters"]
    string_leaves = list(_str_leaves(physical, "physical_parameters"))
    assert not string_leaves, (
        "expected all physical_parameters to parse as numbers, but these "
        f"leaves are strings: {string_leaves}"
    )


def test_all_physical_parameters_are_numeric(config):
    """Every physical parameter is a native ``int`` or ``float``."""
    physical = config["physical_parameters"]
    for key, value in physical.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{key}={value!r} did not parse as a number (type {type(value).__name__})"
        )


@pytest.mark.parametrize(
    "key, expected",
    [
        ("fsr_hz", 2.0e11),
        ("intrinsic_q", 1.0e7),
        ("coupling_q", 1.0e7),
        ("d2_rad_per_s2", 1.2566370614e7),
        ("kappa_i_rad_per_s", 1.215e8),
        ("surface_state_density_per_m2", 2.0e11),
        ("gamma_LLE_per_J_per_s", 3.98e19),
    ],
)
def test_previously_unsigned_exponents_parse_to_expected_floats(config, key, expected):
    """The 7 formerly-unsigned-exponent values parse to their intended floats."""
    value = config["physical_parameters"][key]
    assert isinstance(value, float), f"{key} is {type(value).__name__}, not float"
    assert math.isclose(value, expected, rel_tol=1e-12), (key, value, expected)
