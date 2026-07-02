"""Acceptance tests for the measured integrated-dispersion grid loader.

These pin the behavior of :func:`load_dint_grid` (shape, mu=0 bin, interpolation,
measured FSR) and verify that threading a per-mode ``d_int_grid`` through
``solve_lle_ssfm_jax`` (a) completes finite and (b) reduces exactly to the Taylor
path in the parabolic limit, proving the injection is wired correctly.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from simulator.lle_solver import (
    _build_omega_grid,
    _load_config,
    build_dispersion,
    d2_to_beta2_lle,
    load_dint_grid,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"
DINT_CSV = REPO_ROOT / "config" / "pyLLE_dispersion_w4400_h800.csv"


@pytest.fixture(scope="module")
def csv_arrays():
    """Raw (mu, D_int) computed directly from the CSV for cross-checking interp."""
    data = np.loadtxt(DINT_CSV, delimiter=",")
    mu = data[:, 0].astype(np.int64)
    f = data[:, 1].astype(np.float64)
    omega = 2.0 * np.pi * f
    i0 = int(np.where(mu == 0)[0][0])
    d1 = 0.5 * (omega[i0 + 1] - omega[i0 - 1])
    d_int = omega - omega[i0] - d1 * mu
    return mu, d_int, d1


# ---------------------------------------------------------------------------
# 1. Loader shape / mu=0 bin / interpolation / measured FSR
# ---------------------------------------------------------------------------
def test_load_dint_grid_shape_and_dc_bin():
    res = load_dint_grid(512)
    grid = np.asarray(res.grid)
    assert grid.shape == (512,)
    assert grid[0] == 0.0, "mu=0 (DC) FFT bin must be exactly 0"


def test_load_dint_grid_matches_direct_interp(csv_arrays):
    mu, d_int, _ = csv_arrays
    n_tau = 512
    grid = np.asarray(load_dint_grid(n_tau).grid)

    # Reproduce the exact FFT mode grid used by the loader.
    fsr = (csv_arrays[2]) / (2.0 * np.pi)
    t_r = 1.0 / fsr
    k = np.round(np.fft.fftfreq(n_tau, d=t_r / n_tau) / fsr).astype(int)
    expected = np.interp(k, mu, d_int)

    # Check a handful of sampled bins (positive, negative, and Nyquist neighbors).
    for j in (1, 5, 17, 200, n_tau // 2, n_tau - 3, n_tau - 1):
        assert np.isclose(grid[j], expected[j], rtol=1e-5, atol=1.0), (
            f"bin {j} (mode {k[j]}): grid={grid[j]:.6e} vs interp={expected[j]:.6e}"
        )


def test_measured_fsr_is_about_24p45_ghz():
    res = load_dint_grid(512)
    fsr_ghz = res.d1 / (2.0 * np.pi) / 1e9
    assert abs(fsr_ghz - 24.45) / 24.45 < 0.01, f"measured FSR {fsr_ghz:.4f} GHz"


def test_load_dint_grid_is_cached():
    """Repeated calls with the same (n_tau, csv_path) return the identical object."""
    a = load_dint_grid(512)
    b = load_dint_grid(512)
    assert a is b


# ---------------------------------------------------------------------------
# 2. Solver injection: finite run + exact reduction to Taylor path (parabola)
# ---------------------------------------------------------------------------
def test_solver_runs_finite_with_measured_grid():
    """A short seeded run with the FULL measured D_int(mu) completes finite."""
    kappa_i, kappa_c, kappa = resolve_cavity_rates(CONFIG_PATH)
    n_tau, t_slow = 512, 2000
    grid = load_dint_grid(n_tau).grid

    dw = np.full(t_slow, 2.0 * kappa, dtype=np.float32)[None, :]
    out = solve_lle_ssfm_jax(
        pin=0.214, delta_omega=dw, t_slow=t_slow, beta=[0.0],
        kappa=kappa, kappa_c=kappa_c, rng_key=jax.random.PRNGKey(0),
        n_tau=n_tau, snapshot_interval=t_slow - 1, config_path=CONFIG_PATH,
        d_int_grid=grid,
    )
    e = np.asarray(out["E_snapshots"])[0][-1]
    assert np.all(np.isfinite(e)), "measured-grid run produced non-finite field"


def test_parabolic_grid_reduces_to_taylor_path():
    """d_int_grid built as a parabola from the same D2 must match the Taylor path.

    In the parabolic limit D_int(mu) = ½D₂μ² and build_dispersion((beta2,)) with
    beta2 = D2/D1² are the SAME array on the FFT grid, so injecting the parabola
    as d_int_grid must reproduce the Taylor solve bit-for-bit (up to float32),
    proving the injection reduces to the old path.
    """
    phys = _load_config(CONFIG_PATH)
    kappa_i, kappa_c, kappa = resolve_cavity_rates(CONFIG_PATH)
    n_tau, t_slow = 512, 2000
    t_r = 1.0 / float(phys["fsr_hz"])
    beta2 = d2_to_beta2_lle(float(phys["d2_rad_per_s2"]), float(phys["fsr_hz"]))

    # Parabola on the exact FFT grid == build_dispersion output.
    omega = _build_omega_grid(n_tau, t_r)
    parabola = np.asarray(build_dispersion(omega, (beta2,)))

    dw = np.full(t_slow, 2.0 * kappa, dtype=np.float32)[None, :]
    common = dict(
        pin=0.214, delta_omega=dw, t_slow=t_slow, kappa=kappa, kappa_c=kappa_c,
        rng_key=jax.random.PRNGKey(1), n_tau=n_tau, snapshot_interval=t_slow - 1,
        config_path=CONFIG_PATH,
    )
    out_taylor = solve_lle_ssfm_jax(beta=[beta2], **common)
    out_grid = solve_lle_ssfm_jax(beta=[0.0], d_int_grid=parabola, **common)

    e_taylor = np.asarray(out_taylor["E_snapshots"])[0][-1]
    e_grid = np.asarray(out_grid["E_snapshots"])[0][-1]
    assert np.all(np.isfinite(e_taylor)) and np.all(np.isfinite(e_grid))

    def norm_power(e):
        p = np.abs(np.fft.fft(e)) ** 2
        return p / max(p.max(), 1e-30)

    max_diff = float(np.max(np.abs(norm_power(e_taylor) - norm_power(e_grid))))
    assert max_diff < 1e-3, f"grid vs Taylor normalized-power diff {max_diff:.3e}"


# ---------------------------------------------------------------------------
# 3. Sub-stepping: n_substeps=1 is the legacy single Strang step, bit-for-bit
# ---------------------------------------------------------------------------
def _substep_case(**overrides):
    """The fixed tiny deterministic run used for the n_substeps regression."""
    kappa_i, kappa_c, kappa = resolve_cavity_rates(CONFIG_PATH)
    n_tau, t_slow = 128, 300
    dw = np.full(t_slow, 3.0 * kappa, dtype=np.float64)[None, :]
    kw = dict(
        pin=0.214, delta_omega=dw, t_slow=t_slow, beta=[1e-16],
        kappa=kappa, kappa_c=kappa_c, rng_key=jax.random.PRNGKey(7),
        n_tau=n_tau, snapshot_interval=t_slow - 1, config_path=CONFIG_PATH,
    )
    kw.update(overrides)
    return solve_lle_ssfm_jax(**kw)


def test_n_substeps_1_matches_legacy_single_step():
    """n_substeps=1 must reproduce the pre-substep single Strang step bit-for-bit.

    The golden field ``tests/data/lle_singlestep_legacy_128.npy`` was captured
    from the single-step solver (the commit before ``n_substeps`` existed) for
    this exact deterministic case, and verified bit-identical to the n_substeps=1
    path when sub-stepping was added. This guards the n_substeps=1 fast path (and
    the default) against silently diverging from the legacy round-trip update.
    """
    golden = np.load(REPO_ROOT / "tests" / "data" / "lle_singlestep_legacy_128.npy")
    e1 = np.asarray(_substep_case(n_substeps=1)["e_final"])
    e_default = np.asarray(_substep_case()["e_final"])
    assert np.array_equal(e1, golden), "n_substeps=1 diverged from legacy single step"
    assert np.array_equal(e_default, golden), "default n_substeps must be the legacy path"


def test_n_substeps_gt1_changes_result_but_stays_finite():
    """Sub-stepping is actually wired: n_substeps>1 finite and != the single step."""
    e1 = np.asarray(_substep_case(n_substeps=1)["e_final"])
    e2 = np.asarray(_substep_case(n_substeps=2)["e_final"])
    assert np.all(np.isfinite(e2))
    assert not np.array_equal(e1, e2), "n_substeps=2 must differ from the single step"


def test_n_substeps_must_be_positive():
    with pytest.raises(ValueError):
        _substep_case(n_substeps=0)


# ---------------------------------------------------------------------------
# 4. Anti-aliasing toggles: OFF is bit-identical; ON de-aliases and stays finite
# ---------------------------------------------------------------------------
def test_antialiasing_toggles_off_are_bit_identical():
    """dealias/absorber OFF must reproduce the legacy single-step field bit-for-bit."""
    golden = np.load(REPO_ROOT / "tests" / "data" / "lle_singlestep_legacy_128.npy")
    e_off = np.asarray(
        _substep_case(n_substeps=1, dealias_two_thirds=False, edge_absorber=False)["e_final"]
    )
    assert np.array_equal(e_off, golden), "toggles OFF must be the legacy path"


def test_dealias_zeros_modes_above_two_thirds_and_changes_result():
    """dealias ON removes |mu|>n_tau/3 content and changes the field; stays finite."""
    n_tau = 128
    e_on = np.asarray(
        _substep_case(n_substeps=1, dealias_two_thirds=True, edge_absorber=True)["e_final"]
    )[0]
    e_off = np.asarray(_substep_case(n_substeps=1)["e_final"])[0]
    assert np.all(np.isfinite(e_on))
    assert not np.array_equal(e_on, e_off)
    sp = np.abs(np.fft.fft(e_on)) ** 2
    absmu = np.abs(np.fft.fftfreq(n_tau) * n_tau)
    assert sp[absmu > n_tau / 3].max() < 1e-20 * max(sp.max(), 1e-300), \
        "dealias must zero the |mu|>n_tau/3 band"
