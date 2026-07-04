"""Regression test: the loaded D_int grid must not carry a spurious linear-in-mu
tilt (a gauge/group-velocity term) from the pump-neighborhood dispersion defect.

Why a *drift* test, and why it catches this class of bug
--------------------------------------------------------
The CSV ``config/pyLLE_dispersion_w4400_h800.csv`` has a localized defect around
the pump (resonances at |mu|<=4 displaced up to ~-27 MHz from the smooth trend).
A plain 3-point central difference of omega at mu=0 is biased by +2*pi*3.35 MHz,
so ``load_dint_grid`` used to return a D_int tilted by -(2*pi*3.35 MHz)*mu. A term
linear in mu is a pure group-velocity offset: it leaves every mode POWER unchanged
(spectrum/power tests stay green) but translates any localized structure in
fast time by a fixed number of samples per round trip.

We expose exactly that translation: seed a single dissipative Kerr soliton on the
analytic CW background, integrate with the real measured grid, and track the
intensity centroid. A linear-in-mu D_int tilt Delta*D1 shows up as a constant
soliton drift of  Delta*D1 * t_r * n_tau / (2*pi)  samples per round trip. For the
pre-fix +2*pi*3.35 MHz bias that is ~ -1.10 samples/step; after replacing the
central difference with the defect-excluding smooth-trend fit the drift collapses
to near zero (limited only by intrinsic odd dispersion), which is what this test
asserts.
"""

from __future__ import annotations

import time

import numpy as np

from analysis.dks_access import (
    CONFIG_PATH,
    PIN_W,
    attach_dispersion,
    load_cavity_params,
    sech_soliton_seed,
)
from simulator.lle_solver import solve_lle_ssfm_jax
import jax

N_TAU = 8192
T_SLOW = 4000
SNAPSHOT_INTERVAL = 100
DW_KAPPA = 8.0            # detuning delta_omega = 8*kappa (inside the soliton window)
DRIFT_TOL = 0.1          # samples per round trip


def _intensity_centroid_phase(e_field: np.ndarray) -> float:
    """Circular centroid of the |E|^4 intensity profile, as an angle in radians.

    ``|E|^4`` weights the sharp soliton peak far above the flat CW background
    (whose uniform contribution sums to zero on the circle), so the phasor tracks
    the soliton's fast-time position.
    """
    n = e_field.shape[0]
    weight = np.abs(e_field) ** 4
    phasor = np.sum(weight * np.exp(2j * np.pi * np.arange(n) / n))
    return float(np.angle(phasor))


def test_measured_grid_has_no_gauge_drift():
    """A seeded single soliton must not drift under the loaded measured D_int grid.

    Pre-fix (central-difference D1) the tilted grid drives a ~-1.10 samples/step
    drift; post-fix (smooth-trend-fit D1) it is near zero. Runs in ~10 s on CPU
    (well under the ~60 s ``slow`` threshold), so it stays in the default suite.
    """
    t0 = time.time()

    cav = load_cavity_params(CONFIG_PATH)
    cav = attach_dispersion(cav, N_TAU)
    delta_omega = DW_KAPPA * cav.kappa

    seed = sech_soliton_seed(delta_omega, cav, n_tau=N_TAU, pin=PIN_W)

    dw = np.full((1, T_SLOW), float(delta_omega), dtype=np.float32)
    sol = solve_lle_ssfm_jax(
        pin=PIN_W,
        delta_omega=dw,
        t_slow=T_SLOW,
        beta=[cav.beta2],          # ignored: the measured grid drives the dispersion
        kappa=cav.kappa,
        kappa_c=cav.kappa_c,
        rng_key=jax.random.PRNGKey(0),
        n_tau=N_TAU,
        snapshot_interval=SNAPSHOT_INTERVAL,
        config_path=str(CONFIG_PATH),
        e0_override=seed,
        d_int_grid=cav.d_int_grid,
    )

    snapshots = np.asarray(sol["E_snapshots"])[0]        # (n_snapshots, n_tau)
    assert np.all(np.isfinite(snapshots)), "measured-grid run produced non-finite field"

    # Snapshot j is the field after round trip j*SNAPSHOT_INTERVAL.
    steps = np.arange(snapshots.shape[0]) * SNAPSHOT_INTERVAL
    phases = np.array([_intensity_centroid_phase(e) for e in snapshots])
    # Unwrap the circular centroid, then convert the angle to fast-time samples.
    centroid_samples = np.unwrap(phases) * N_TAU / (2.0 * np.pi)

    # Slope of centroid vs round-trip index = drift in samples per round trip.
    drift = float(np.polyfit(steps, centroid_samples, 1)[0])

    elapsed = time.time() - t0
    print(f"[gauge-drift] centroid drift = {drift:.4f} samples/step "
          f"(tol {DRIFT_TOL}); runtime {elapsed:.1f} s")

    assert abs(drift) < DRIFT_TOL, (
        f"soliton drift {drift:.4f} samples/step exceeds {DRIFT_TOL}; the loaded "
        f"D_int grid carries a spurious linear-in-mu tilt (pump-neighborhood "
        f"dispersion defect leaking into the central-difference D1)."
    )
