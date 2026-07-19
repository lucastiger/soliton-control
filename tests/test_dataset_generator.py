"""Regression tests for the repaired data/dataset_generator.py surface.

Context: ``simulate_batch`` had gone stale against the batched per-trajectory
solver — it passed 17 of the (then) 25 positional arguments, handed the solver
a ``(B,)``-shaped detuning where ``delta_omega[step]`` indexing needs
``(B, t_seg)``, and called ``jax.random.fold_in`` on a whole key ARRAY (which
modern JAX rejects). The quantum-noise integration repaired all three and also
guarded the pin=0 pre-flight message against ZeroDivisionError. These tests
pin the repaired surface.

All tests run with n_tau = 256, B <= 4, t_seg ~ 100 and 2-3 segments so the
module completes in seconds on CPU.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simulator.lle_solver as lle
from analysis.run_detuning_sweep import write_noise_off_config
from data.dataset_generator import DatasetGenerator
from simulator.lle_solver import (
    _detuning_noise_sequences,
    _legacy_rng_chain,
    hbar_omega0_from_config,
    solve_lle_ssfm_jax,
)
from simulator.noise_models import TotalNoise, _load_config as nm_load_cfg

N_TAU = 256
SEG_RT = 100
HOLD_RT = 50
SNAP_INT = 10
GAMMA_TH = 4.72e-3


def _make_gen(tmp_dir, seed=42, config_path=None, **kw) -> DatasetGenerator:
    gen = DatasetGenerator(
        param_grid={
            "pin": [1e-3],
            "sweep_rate": [1.0],
            "Gamma_th": [GAMMA_TH],
            "noise_scale": [1.0],
        },
        config_path=config_path,
        output_dir=tmp_dir,
        n_tau=N_TAU,
        snapshot_interval=SNAP_INT,
        seed=seed,
        **kw,
    )
    gen.SEGMENT_RT = SEG_RT
    gen.HOLD_RT = HOLD_RT
    return gen


def _sweep_rate_for_two_segments(gen) -> float:
    # ceil(8*kappa / (sr * SEGMENT_RT)) == 2 exactly.
    return 8.0 * gen.kappa / (2.0 * SEG_RT)


def _run_batch(gen, noise_scales, pin=1e-3, batch_global_idx=0):
    sr = _sweep_rate_for_two_segments(gen)
    params = [
        dict(pin=pin, sweep_rate=sr, Gamma_th=GAMMA_TH, noise_scale=s)
        for s in noise_scales
    ]
    with warnings.catch_warnings():
        # the deliberately coarse 2-segment sweep trips the step-size warning
        warnings.simplefilter("ignore")
        return gen.simulate_batch(params, batch_global_idx=batch_global_idx)


class _ConstantScheduleGen(DatasetGenerator):
    """DatasetGenerator with an explicit (constant-detuning) segment schedule.

    Uses the ``_segment_schedule`` hook, so everything else — keys, carry,
    noise regeneration, concatenation — is the production ``simulate_batch``
    code path.
    """

    def __init__(self, schedule, **kw):
        super().__init__(**kw)
        self._schedule = [(float(d), int(t)) for d, t in schedule]

    def _segment_schedule(self, sweep_rate):
        return self._schedule


# ---------------------------------------------------------------------------
# 1. Shape / dtype contract (+ HDF5 attribute round trip)
# ---------------------------------------------------------------------------
def test_output_shapes_dtypes_and_h5_roundtrip(tmp_path):
    gen = _make_gen(tmp_path)
    res = _run_batch(gen, [0.5, 1.0])
    B = 2
    t_total = 2 * SEG_RT + HOLD_RT                      # exactly the segment sum
    n_snap = 2 * (SEG_RT // SNAP_INT) + -(-HOLD_RT // SNAP_INT)

    for key in ("P_trans", "U_int", "DeltaT", "delta_omega_eff"):
        assert res[key].shape == (B, t_total), key
        assert res[key].dtype == np.float32, key
    assert res["labels"].shape == (B, t_total)
    assert res["labels"].dtype == np.int32
    assert res["E_snapshots"].shape == (B, n_snap, N_TAU)
    assert res["E_snapshots"].dtype == np.complex64
    assert np.isfinite(res["U_int"]).all()

    sr = _sweep_rate_for_two_segments(gen)
    params = [
        dict(pin=1e-3, sweep_rate=sr, Gamma_th=GAMMA_TH, noise_scale=s)
        for s in (0.5, 1.0)
    ]
    with h5py.File(tmp_path / "roundtrip.h5", "a") as h5:
        gen.save_batch(res, params, start_idx=0, h5file=h5)
        for i, p in enumerate(params):
            grp = h5[f"sim_{i}"]
            assert grp["E_snapshots"].shape == (n_snap, N_TAU)
            for attr in ("pin", "sweep_rate", "Gamma_th", "noise_scale"):
                assert grp.attrs[attr] == pytest.approx(p[attr])


# ---------------------------------------------------------------------------
# 2. Determinism under the constructor seed
# ---------------------------------------------------------------------------
def test_seed_determinism(tmp_path):
    a = _run_batch(_make_gen(tmp_path, seed=42), [1.0, 2.0])
    b = _run_batch(_make_gen(tmp_path, seed=42), [1.0, 2.0])
    for key in a:
        assert np.array_equal(a[key], b[key]), key

    c = _run_batch(_make_gen(tmp_path, seed=43), [1.0, 2.0])
    assert not np.array_equal(a["E_snapshots"], c["E_snapshots"])


# ---------------------------------------------------------------------------
# 3. Batch independence + B=1 reproducibility of a batch row
# ---------------------------------------------------------------------------
def test_batch_rows_independent_and_b1_reproduces_row0(tmp_path):
    """Per-trajectory keys/noise must be batched correctly.

    ``simulate_batch`` requires a batch to share pin/sweep_rate/Gamma_th, so
    the per-trajectory distinction inside one batch is ``noise_scale`` (plus
    the per-trajectory key rows). The B=1 run must reproduce row 0 of the B=3
    run BIT-IDENTICALLY (jax.random.split(key, B) yields a per-row key stream
    independent of B) — this is the check that would have caught the old
    unbatched ``fold_in(key_arr, seg)`` (which crashes on modern JAX, and on
    any JAX would have collapsed/scrambled the per-row streams).
    """
    r3 = _run_batch(_make_gen(tmp_path), [0.5, 1.0, 2.0])
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.array_equal(
                r3["E_snapshots"][i], r3["E_snapshots"][j]
            ), (i, j)

    r1 = _run_batch(_make_gen(tmp_path), [0.5])
    for key in r1:
        assert np.array_equal(r1[key][0], r3[key][0]), key


# ---------------------------------------------------------------------------
# 4. Segment continuity across the carry boundary (noise-off)
# ---------------------------------------------------------------------------
def test_segment_carry_continuity_noise_off(tmp_path):
    """One 200-RT segment vs two chained 100-RT segments at fixed detuning.

    With the T_k=0 / quantum-off sidecar the dynamics are fully deterministic,
    so the ONLY discrepancy the two-segment run can show is the carry handling
    at the boundary.

    FLAGGED LATENT ISSUE (do not fix on this branch): ``simulate_batch``
    downcasts the carried field/thermal state to complex64/float32 at every
    segment boundary (e_carry / delta_t_carry), while the solver integrates in
    complex128 under the module-wide x64 mandate. The boundary therefore
    injects a ~1e-7 relative rounding step per segment. This test pins the
    agreement at <1e-4 (field) / <1e-5 (energy) rather than bit-identity;
    promoting the carry to complex128/float64 belongs in its own change (it
    alters every multi-segment dataset bit-for-bit).
    """
    noff = str(write_noise_off_config(out_path=tmp_path / "noiseoff.yaml"))
    kw = dict(
        param_grid={
            "pin": [1e-3],
            "sweep_rate": [1.0],
            "Gamma_th": [GAMMA_TH],
            "noise_scale": [1.0],
        },
        config_path=noff,
        output_dir=tmp_path,
        n_tau=N_TAU,
        snapshot_interval=SNAP_INT,
        seed=42,
    )
    kappa = DatasetGenerator(**kw).kappa
    dw0 = 3.0 * kappa
    gx = _ConstantScheduleGen([(dw0, 2 * SEG_RT)], **kw)
    gy = _ConstantScheduleGen([(dw0, SEG_RT), (dw0, SEG_RT)], **kw)
    rx = _run_batch(gx, [1.0])
    ry = _run_batch(gy, [1.0])

    ux, uy = rx["U_int"][0], ry["U_int"][0]
    # Pre-boundary: same integration, single segment each -> identical.
    assert np.array_equal(ux[:SEG_RT], uy[:SEG_RT])
    # Post-boundary: only the complex64/float32 carry downcast separates them.
    u_rel = float(np.max(np.abs(ux[SEG_RT:] - uy[SEG_RT:])) / np.max(np.abs(ux[SEG_RT:])))
    assert u_rel < 1e-5, u_rel

    ex = rx["E_snapshots"][0][-1]
    ey = ry["E_snapshots"][0][-1]
    e_rel = float(np.max(np.abs(ex - ey)) / np.max(np.abs(ex)))
    assert e_rel < 1e-4, e_rel
    print(f"\n[gate V-A] segment-boundary agreement: field rel = {e_rel:.3e} "
          f"(< 1e-4), U_int rel = {u_rel:.3e} (< 1e-5)")


# ---------------------------------------------------------------------------
# 5. Known-flaw tripwire: per-segment AR(1) regeneration restarts at zero
# ---------------------------------------------------------------------------
def test_segment_noise_restarts_at_zero_tripwire(tmp_path):
    """TRIPWIRE (documents current behavior, does NOT endorse it).

    ``simulate_batch`` regenerates the AR(1) detuning-noise sequence per
    segment from x0 = 0 (``_ar1_samples`` starts its scan at zero), so every
    segment boundary restarts the noise far below its stationary variance —
    with tau_th ~ 1.2e5 RT the entire ~100-500 RT segment lives in the
    restart transient. This test asserts the flaw EXISTS (early-segment
    mean-square well below the interior) so the planned
    ``legacy_segment_noise`` migration has a pinned baseline to flip.
    """
    gen = _make_gen(tmp_path)
    _, noise_keys, _ = gen._make_keys(batch_global_idx=0, B=4)
    seg_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(noise_keys, 0)
    nm = TotalNoise(nm_load_cfg(None))
    t_seg = 2000
    seqs = np.asarray(
        gen._segment_noise(nm, seg_keys, jnp.ones(4, jnp.float32), t_seg)
    )
    assert seqs.shape == (4, t_seg)
    early = float(np.mean(seqs[:, : t_seg // 20] ** 2))     # first 5%
    interior = float(np.mean(seqs[:, t_seg // 2 :] ** 2))   # last 50%
    ratio = early / interior
    assert 0.0 <= ratio < 0.9, ratio
    print(f"\n[gate V-A] AR(1) restart tripwire: early/interior mean-square "
          f"ratio = {ratio:.4f} (< 0.9 asserts the per-segment restart flaw)")


# ---------------------------------------------------------------------------
# 6. Quantum-noise-on smoke
# ---------------------------------------------------------------------------
def test_quantum_noise_on_smoke(tmp_path):
    """enable_quantum_noise=True runs end-to-end; occupations are O(0.5).

    Sub-threshold pin, 500 RT (~3 photon lifetimes) at constant detuning:
    the modal occupation relaxes from the legacy seed toward the vacuum
    equilibrium 1/2. Loose bounds [0.2, 1.0] — a smoke test, not physics.
    """
    kw = dict(
        param_grid={
            "pin": [1e-4],
            "sweep_rate": [1.0],
            "Gamma_th": [GAMMA_TH],
            "noise_scale": [1.0],
        },
        output_dir=tmp_path,
        n_tau=N_TAU,
        snapshot_interval=SNAP_INT,
        seed=42,
        enable_quantum_noise=True,
    )
    kappa = DatasetGenerator(**{**kw, "enable_quantum_noise": False}).kappa
    gen = _ConstantScheduleGen([(3.0 * kappa, 500)], **kw)
    assert gen.qnoise_scale > 0.0
    res = _run_batch(gen, [1.0, 1.0], pin=1e-4)
    assert np.isfinite(res["U_int"]).all()

    hbw = hbar_omega0_from_config(gen.config)
    last = res["E_snapshots"][:, -1, :].astype(np.complex128)
    dev = last - last.mean(axis=-1, keepdims=True)          # remove CW (mu=0)
    n_mu = np.abs(np.fft.fft(dev, axis=-1)) ** 2 / (N_TAU**2 * hbw)
    mean_n = float(np.mean(np.delete(n_mu, 0, axis=-1)))
    assert 0.2 < mean_n < 1.0, mean_n


# ---------------------------------------------------------------------------
# 7. Pre-flight pin=0 does not crash and warns cleanly
# ---------------------------------------------------------------------------
def test_preflight_pin_zero_warns_without_exception():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sol = solve_lle_ssfm_jax(
            pin=0.0,
            delta_omega=0.0,
            t_slow=3,
            beta=[1.578e-18],
            kappa=1.519e8,
            kappa_c=1.215e8,
            rng_key=jax.random.PRNGKey(0),
            n_tau=64,
            snapshot_interval=1,
        )
    assert np.isfinite(sol["U_int_history"]).all()
    msgs = [str(w.message) for w in rec if "below the MI threshold" in str(w.message)]
    assert len(msgs) == 1
    assert "inf" not in msgs[0] and "nan" not in msgs[0]


# ---------------------------------------------------------------------------
# 8. Key isolation: quantum channel cannot perturb the legacy noise streams
# ---------------------------------------------------------------------------
def test_key_isolation():
    """The detuning-noise sequences are bit-independent of the quantum flag.

    (a) pins the legacy key derivation (split(rng_key, 3) arity must never
    widen); (b) proves solve_lle_ssfm_jax actually routes through the pinned
    helpers; (c) captures the sequences handed to the scan in quantum-on and
    quantum-off runs of the same rng_key and asserts bit-identity.
    """
    rng = jax.random.PRNGKey(7)

    # (a) exact legacy derivation
    key_arr, noise_keys, key_qnoise = _legacy_rng_chain(rng, 5)
    key, key_field, key_noise = jax.random.split(rng, 3)
    _, expect_qnoise = jax.random.split(key, 2)
    assert np.array_equal(
        jax.random.key_data(key_arr), jax.random.key_data(jax.random.split(key_field, 5))
    )
    assert np.array_equal(
        jax.random.key_data(noise_keys),
        jax.random.key_data(jax.random.split(key_noise, 5)),
    )
    assert np.array_equal(
        jax.random.key_data(key_qnoise), jax.random.key_data(expect_qnoise)
    )

    # (b) + (c): capture the sequences the solver hands to the scan
    captured = {}
    real = _detuning_noise_sequences

    def run(flag, tag):
        def spy(noise_keys_, t_slow_, config_path_=None):
            out = real(noise_keys_, t_slow_, config_path_)
            captured[tag] = np.asarray(out)
            return out

        orig = lle._detuning_noise_sequences
        lle._detuning_noise_sequences = spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solve_lle_ssfm_jax(
                    pin=1e-3,
                    delta_omega=3.0 * 1.519e8,
                    t_slow=20,
                    beta=[1.578e-18],
                    kappa=1.519e8,
                    kappa_c=1.215e8,
                    rng_key=rng,
                    n_tau=64,
                    snapshot_interval=5,
                    quantum_noise_enabled=flag,
                )
        finally:
            lle._detuning_noise_sequences = orig

    run(False, "off")
    run(True, "on")
    assert "off" in captured and "on" in captured, "solver bypassed the pinned helper"
    assert np.array_equal(captured["off"], captured["on"])
    assert captured["off"].shape == (1, 20)

    # structural: the sequence generator cannot even see the quantum flag
    sig = inspect.signature(_detuning_noise_sequences)
    assert all("quantum" not in p for p in sig.parameters)
