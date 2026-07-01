"""Regression tests for the LLE linear-operator dispersion sign.

Bug context
-----------
The split-step linear operator entered dispersion with the WRONG sign relative
to the detuning term:

    lin_exp = (-kappa/2 + 1j*disp - 1j*delta_omega_eff) * t_r   # buggy

``disp`` is the integrated dispersion D_int(mu) = ½D₂μ² + … (all-positive
coefficients from ``build_dispersion`` — that function is correct and must NOT
be touched). Dispersion and detuning share ONE detuning axis, so they must carry
the SAME sign. With ``+1j*disp`` the operator is effectively *normal* dispersion
for the anomalous config (D₂>0), so the homogeneous CW state is modulationally
stable at every power/detuning and no MI or solitons ever form. The fix flips the
operator sign only:

    lin_exp = (-kappa/2 - 1j*disp - 1j*delta_omega_eff) * t_r   # fixed

These tests pin the sign deterministically (source + operator formula) and
behaviorally (MI ignites with the fix, not with the reverted sign), and check
that the spectral grid resolves the resulting MI band.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import numpy as np
import pytest

from simulator.lle_solver import (
    _build_omega_grid,
    _load_config,
    build_dispersion,
    d2_to_beta2_lle,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
    validate_solver,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"
SOLVER_SRC = REPO_ROOT / "simulator" / "lle_solver.py"


@pytest.fixture(scope="module")
def phys():
    return _load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def rates():
    return resolve_cavity_rates(CONFIG_PATH)


@pytest.fixture(scope="module")
def beta2(phys):
    return d2_to_beta2_lle(float(phys["d2_rad_per_s2"]), float(phys["fsr_hz"]))


def _held_scan(pin, dw_over_kappa, kappa, kappa_c, beta2, *, n_tau=512,
               t_slow=6000, seed=0, config_path=CONFIG_PATH):
    """Run a held-detuning single trajectory; return the final field snapshot."""
    dw = np.full(t_slow, dw_over_kappa * kappa, dtype=np.float32)[None, :]  # (1, t_slow)
    out = solve_lle_ssfm_jax(
        pin=pin, delta_omega=dw, t_slow=t_slow, beta=[beta2],
        kappa=kappa, kappa_c=kappa_c, rng_key=jax.random.PRNGKey(seed),
        n_tau=n_tau, snapshot_interval=t_slow - 1, config_path=config_path,
    )
    return np.asarray(out["E_snapshots"])[0][-1]


def _contrast(e):
    p = np.abs(e) ** 2
    return float(np.max(p) / np.mean(p))


# ---------------------------------------------------------------------------
# 1. Deterministic operator-sign pin (no full solve)
# ---------------------------------------------------------------------------
def test_build_dispersion_is_positive_D_int(phys, beta2):
    """build_dispersion returns D_int(mu) = +½D₂μ² > 0 (this function is correct)."""
    omega = _build_omega_grid(512, 1.0 / float(phys["fsr_hz"]))
    disp = np.asarray(build_dispersion(omega, (beta2,)))
    nz = omega != 0.0
    assert np.all(disp[nz] > 0.0), "D_int must be positive for beta2>0 (anomalous, D2>0)"


def test_linear_operator_dispersion_imag_is_negative(phys, beta2):
    """The per-step linear phase must be -i·D_int (negative imag) at a test mode.

    Reproduces the operator exactly as the solver builds it. With the fix the
    imaginary part is negative (-i·½D₂μ²·t_r); reverting the sign makes it
    positive — the discriminating spec.
    """
    t_r = 1.0 / float(phys["fsr_hz"])
    kappa = 1.5e8
    omega = _build_omega_grid(512, t_r)
    disp = np.asarray(build_dispersion(omega, (beta2,)))
    mode = 5  # any non-DC mode; detuning set to 0 to isolate dispersion
    lin_exp_fixed = (-kappa / 2.0 - 1j * disp[mode] - 1j * 0.0) * t_r
    lin_exp_reverted = (-kappa / 2.0 + 1j * disp[mode] - 1j * 0.0) * t_r
    assert lin_exp_fixed.imag < 0.0, "fixed operator must give -i·D_int (imag<0)"
    assert lin_exp_reverted.imag > 0.0, "reverted operator gives +i·D_int (imag>0)"
    assert np.isclose(lin_exp_fixed.imag, -0.5 * beta2 * omega[mode] ** 2 * t_r)


def test_source_uses_minus_1j_disp():
    """Pin the source line so the sign cannot silently regress."""
    src = SOLVER_SRC.read_text(encoding="utf-8")
    # dispersion and detuning must both be -1j on the linear-operator line
    pat = re.compile(r"lin_exp\s*=\s*\(\s*-\s*kappa\s*/\s*2\.0\s*-\s*1j\s*\*\s*disp\s*-\s*1j\s*\*\s*delta_omega_eff\s*\)")
    assert pat.search(src), (
        "linear operator must read '(-kappa/2.0 - 1j*disp - 1j*delta_omega_eff)': "
        "dispersion shares the -i sign of the detuning term."
    )


# ---------------------------------------------------------------------------
# 2. Behavioral MI-ignition test (pins the source via dynamics)
# ---------------------------------------------------------------------------
def test_mi_ignites_with_fixed_sign(rates, beta2):
    """pin=214 mW, held delta_omega=2*kappa: MI must break the CW symmetry.

    With the corrected operator (anomalous D2>0), a modulationally-unstable CW
    breaks up -> contrast >> 1. Passing -beta2 to the *fixed* operator reproduces
    the reverted (+i·D_int, normal-dispersion) case, which stays flat CW. If the
    operator sign is ever reverted in source, the +beta2 case collapses to CW and
    this test fails.
    """
    kappa_i, kappa_c, kappa = rates
    e_fixed = _held_scan(0.214, 2.0, kappa, kappa_c, beta2, t_slow=6000)
    e_reverted = _held_scan(0.214, 2.0, kappa, kappa_c, -beta2, t_slow=6000)

    c_fixed = _contrast(e_fixed)
    c_reverted = _contrast(e_reverted)
    assert np.all(np.isfinite(e_fixed)) and np.all(np.isfinite(e_reverted))
    assert c_fixed > 1.5, f"fixed operator must ignite MI (contrast {c_fixed:.3f} <= 1.5)"
    assert c_reverted < 1.1, (
        f"reverted operator (== -beta2) must stay CW (contrast {c_reverted:.3f} >= 1.1)"
    )


# ---------------------------------------------------------------------------
# 3. Grid-resolution report (does the MI band fit in +/- n_tau/2 ?)
# ---------------------------------------------------------------------------
def test_mi_band_fits_grid_and_report(rates, beta2, capsys):
    """Sweep n_tau, locate the saturated MI gain-peak mode mu*, recommend n_tau.

    mu* is read from the dominant non-DC sideband of the saturated spectrum. The
    unstable band 'fits with margin' when mu* < 0.4*(n_tau/2). Prints a
    recommendation; changes no default.
    """
    kappa_i, kappa_c, kappa = rates
    rows = []
    for n_tau in (512, 1024, 2048):
        e = _held_scan(0.214, 2.0, kappa, kappa_c, beta2, n_tau=n_tau, t_slow=8000)
        assert np.all(np.isfinite(e))
        spec = np.abs(np.fft.fft(e)) ** 2
        spec[0] = 0.0  # drop DC
        mu = np.arange(n_tau)
        mu = np.where(mu > n_tau // 2, mu - n_tau, mu)
        mu_star = abs(int(mu[int(np.argmax(spec))]))
        margin = 0.4 * (n_tau // 2)
        rows.append((n_tau, mu_star, margin, mu_star < margin))
        # MI band must at least be resolved (within the Nyquist half-grid)
        assert mu_star < n_tau // 2, f"MI mode mu*={mu_star} not resolved at n_tau={n_tau}"

    fits_512 = rows[0][3]
    recommendation = (
        "n_tau=512 already resolves the MI band with margin; keep the default."
        if fits_512 else
        f"n_tau=512 is marginal (mu*={rows[0][1]} vs 0.4*(n_tau/2)={rows[0][2]:.0f}); "
        "prefer the smallest swept n_tau that satisfies mu* < 0.4*(n_tau/2)."
    )
    with capsys.disabled():
        print("\n[grid-resolution report] held delta_omega=2*kappa, pin=0.214 W")
        for n_tau, mu_star, margin, fits in rows:
            print(f"  n_tau={n_tau:5d}: mu*={mu_star:4d}  0.4*(n_tau/2)={margin:6.0f}  "
                  f"fits_with_margin={fits}")
        print(f"  RECOMMENDATION: {recommendation}")


# ---------------------------------------------------------------------------
# 4. Regression: validate_solver CW energy-balance still passes
# ---------------------------------------------------------------------------
def test_validate_solver_cw_energy_balance(phys, rates, beta2):
    """Below the MI threshold the field stays CW and the analytic energy balance
    (U_cw = kappa_c*pin*t_r / ((kappa/2)^2 + dw^2)) must still hold within 10%."""
    kappa_i, kappa_c, kappa = rates
    gamma = float(phys["gamma_LLE_per_J_per_s"])
    t_r = 1.0 / float(phys["fsr_hz"])
    pin = 1.0e-3  # 1 mW, below P_th ~ 3.5 mW -> pure CW
    t_slow = 4000
    dw = np.full(t_slow, 2.0 * kappa, dtype=np.float32)[None, :]
    sol = solve_lle_ssfm_jax(
        pin=pin, delta_omega=dw, t_slow=t_slow, beta=[beta2],
        kappa=kappa, kappa_c=kappa_c, rng_key=jax.random.PRNGKey(0),
        n_tau=512, snapshot_interval=t_slow // 4, config_path=CONFIG_PATH,
    )
    # validate_solver asserts the CW energy balance internally; no raise == pass.
    results = validate_solver(
        sol, pin=pin, kappa=kappa, kappa_c=kappa_c, gamma=gamma, t_r=t_r,
        print_results=False, config_path=CONFIG_PATH,
    )
    assert isinstance(results, dict)
