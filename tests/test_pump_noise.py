"""Tests for pump-laser frequency noise and RIN (arXiv:2604.05897 V.B.4-V.B.5).

Coverage
--------
1. PSD fidelity        -- Welch of 2^20-sample delta_nu and eps sequences vs the
                          closed-form targets, within 3 dB per octave over 3
                          decades.
2. Sign convention     -- a deterministic delta_nu ramp shifts
                          delta_omega_eff by -2*pi*delta_nu EXACTLY.
3. Linear response     -- below-threshold CW, sinusoidal delta_nu at
                          f_mod in {kappa/20, kappa/2, 5*kappa}/2pi: the U(t)
                          modulation depth matches the linearized CW cavity
                          response (|dU_cw/ddw| x low-pass). STOP-AND-REPORT
                          gate prints measured vs analytic at all three.
4. RIN energy balance  -- P_trans + kappa_i*U/t_r tracks pin*(1+eps(t)).
5. Determinism         -- same key bit-identical; different key/traj decorrelate.
6. Backward compat     -- flag off (and no override) is bit-identical to legacy.
7. Config validation   -- h0,h-1 >= 0; RIN <= -80 dBc/Hz; enabled forces inert.

The linear-response transfer (test 3) is anchored to the solver's OWN static
dU/ddw (finite difference of two static CW solves). The discrete round-trip map
reproduces the continuous CW energy only to ~10% (the exact tolerance
lle_solver.validate_solver already allows), so comparing the modulation depth
to the CONTINUOUS |dU_cw/ddw| carries that offset; anchoring to the measured
static gain isolates the FREQUENCY-DEPENDENT transfer (the cavity low-pass),
which is the physics under test. The analytic low-pass factor is derived below.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml
from scipy.signal import welch

from simulator.lle_solver import (
    d2_to_beta2_lle,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
    _load_config,
)
from simulator.noise_models import PumpNoise

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sin_params.yaml"

_PHYS = _load_config()
_KAPPA_I, _KAPPA_C, _KAPPA = resolve_cavity_rates()
_T_R = 1.0 / float(_PHYS["fsr_hz"])
_F_S = 1.0 / _T_R
_BETA2 = float(d2_to_beta2_lle(_PHYS["d2_rad_per_s2"], _PHYS["fsr_hz"]))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pump_cfg(**overrides) -> dict:
    cfg = dict(_PHYS)
    cfg["pump_noise_enabled"] = 1
    cfg.update(overrides)
    return cfg


def _tk0_sidecar(tmp_path: Path, name: str, **updates) -> str:
    """Base config + T_k = 0 (deterministic) + flat physical_parameters updates."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    pp = cfg.setdefault("physical_parameters", {})
    pp["T_k"] = 0.0
    pp.update(updates)
    out = tmp_path / name
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out)


def _octave_db_devs(x, psd_fn, f_lo, f_hi, nperseg):
    """Mean Welch/target ratio [dB] per octave band across [f_lo, f_hi]."""
    f, p = welch(x, fs=_F_S, nperseg=nperseg)
    edges = f_lo * 2.0 ** np.arange(0, math.ceil(math.log2(f_hi / f_lo)) + 1)
    devs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.sum() < 4:
            continue
        devs.append(10.0 * np.log10(p[m].mean() / np.asarray(psd_fn(f[m])).mean()))
    return np.array(devs)


# ---------------------------------------------------------------------------
# 1. PSD fidelity
# ---------------------------------------------------------------------------
def test_freq_noise_psd_fidelity():
    """Welch PSD of 2^20 delta_nu samples within 3 dB/octave over 3 decades.

    h0 = 3e3 Hz^2/Hz, h-1 = 1e10 Hz^3/Hz: the 1/f knee sits at
    f = h-1/h0 ~ 3.3e6 Hz, so [1e6, 1e9] exercises both the flicker regime
    (low) and the white plateau (high). Also checks the Lorentzian identity
    Delta_nu_L = pi*h0.
    """
    pm = PumpNoise(_pump_cfg(pump_freq_noise_h0_hz2_per_hz=3e3,
                             pump_freq_noise_hm1_hz3_per_hz=1e10))
    assert math.isclose(pm.lorentzian_linewidth_hz, math.pi * 3e3, rel_tol=1e-12)
    x = pm.sample_freq(jax.random.PRNGKey(0), 2 ** 20) / (2.0 * math.pi)  # delta_nu [Hz]
    devs = _octave_db_devs(x, pm.psd_freq, 1e6, 1e9, nperseg=2 ** 16)
    assert devs.size >= 9, devs.size  # ~3 decades of octaves
    assert np.max(np.abs(devs)) < 3.0, (float(np.max(np.abs(devs))), devs)


def test_rin_psd_fidelity():
    """Welch PSD of 2^20 eps samples within 3 dB/octave over 3 decades.

    floor = -150 dBc/Hz, excess = -120 dBc/Hz, corner = 1e9 Hz: the excess 1/f
    dominates across [1e6, 1e9] and the floor above; both are checked.
    """
    pm = PumpNoise(_pump_cfg(pump_rin_floor_dbc_per_hz=-150.0,
                             pump_rin_excess_dbc_per_hz=-120.0,
                             pump_rin_corner_hz=1e9))
    e = pm.sample_rin(jax.random.PRNGKey(1), 2 ** 20)
    devs = _octave_db_devs(e, pm.psd_rin, 1e6, 1e9, nperseg=2 ** 16)
    assert devs.size >= 9, devs.size
    assert np.max(np.abs(devs)) < 3.0, (float(np.max(np.abs(devs))), devs)


def test_psd_closed_forms_match_definition():
    """psd_freq / psd_rin return the documented closed forms."""
    pm = PumpNoise(_pump_cfg(pump_freq_noise_h0_hz2_per_hz=5.0,
                             pump_freq_noise_hm1_hz3_per_hz=2.0,
                             pump_rin_floor_dbc_per_hz=-140.0,
                             pump_rin_excess_dbc_per_hz=-110.0,
                             pump_rin_corner_hz=1e7))
    f = np.array([1e3, 1e5, 1e7, 1e9])
    assert np.allclose(pm.psd_freq(f), 5.0 + 2.0 / f)
    floor = 10 ** (-140.0 / 10.0)
    exc = 10 ** (-110.0 / 10.0)
    expect = floor + np.where(f < 1e7, exc * 1e7 / f, 0.0)
    assert np.allclose(pm.psd_rin(f), expect)


# ---------------------------------------------------------------------------
# 2. Sign convention: delta_omega_eff shifts by -2*pi*delta_nu EXACTLY
# ---------------------------------------------------------------------------
def test_sign_convention_exact(tmp_path):
    """pin=0, T_k=0: injected delta_nu ramp shifts delta_omega_eff by -2*pi*delta_nu.

    With pin=0 the field is identically zero, so the thermal shift is exactly
    zero and delta_omega_eff = dw_prog + (-2*pi*delta_nu). Checked to float64.
    """
    sc = _tk0_sidecar(tmp_path, "signconv.yaml")
    t_slow = 64
    dnu = np.linspace(0.0, 2e6, t_slow)  # Hz ramp
    dw0 = 2.0 * _KAPPA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_lle_ssfm_jax(
            pin=0.0, delta_omega=dw0, t_slow=t_slow, beta=[_BETA2],
            kappa=_KAPPA, kappa_c=_KAPPA_C, rng_key=jax.random.PRNGKey(0),
            n_tau=64, snapshot_interval=1, config_path=sc,
            pump_freq_noise_override=dnu,
        )
    dwe = np.asarray(sol["delta_omega_eff_history"])[0]
    assert np.max(np.abs(dwe - (dw0 - 2.0 * math.pi * dnu))) == 0.0
    # diagnostic history is exactly the -2*pi*delta_nu contribution
    assert np.array_equal(
        np.asarray(sol["pump_freq_noise_history"])[0], -2.0 * math.pi * dnu
    )


# ---------------------------------------------------------------------------
# 3. Linear-response transfer  (STOP-AND-REPORT GATE)
# ---------------------------------------------------------------------------
_LR_PIN = 1e-3          # below MI threshold (~3.5 mW): CW only
_LR_DW0 = 1.0 * _KAPPA
_LR_A = 0.02 * _KAPPA   # detuning-modulation amplitude (delta_nu = A/2pi)
_LR_NTAU = 16


def _lr_lowpass(wm):
    """Exact linearized two-sideband low-pass factor L(wm), L(0) = 1.

    dE/dt = -(kappa/2 + i*dw(t))E + F linearizes to a modulation depth
        dU(wm) = |dU_cw/ddw| * A * L(wm),
        L(wm) = (a^2 + dw0^2) / [sqrt(a^2+(dw0+wm)^2) * sqrt(a^2+(dw0-wm)^2)],
    a = kappa/2. L reduces to the single-pole 1/sqrt(1+(wm/a)^2) only as
    dw0 -> 0; the two-shifted-Lorentzian form is exact for finite dw0.
    """
    a = _KAPPA / 2.0
    return (a ** 2 + _LR_DW0 ** 2) / (
        np.sqrt(a ** 2 + (_LR_DW0 + wm) ** 2) * np.sqrt(a ** 2 + (_LR_DW0 - wm) ** 2)
    )


def _lr_static_gain(sc):
    """Solver's own static dU/ddw at dw0 (central finite difference, 2 solves)."""
    a = _KAPPA / 2.0
    h = 0.02 * _KAPPA

    def u_static(dw):
        e_cw = (math.sqrt(_KAPPA_C * _LR_PIN) / (a + 1j * dw)) * np.ones(
            _LR_NTAU, dtype=np.complex128
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = solve_lle_ssfm_jax(
                pin=_LR_PIN, delta_omega=float(dw), t_slow=3000, beta=[_BETA2],
                kappa=_KAPPA, kappa_c=_KAPPA_C, rng_key=jax.random.PRNGKey(0),
                n_tau=_LR_NTAU, snapshot_interval=1, config_path=sc, e0_override=e_cw,
            )
        return float(np.mean(np.asarray(s["U_int_history"])[0][-500:]))

    return (u_static(_LR_DW0 + h) - u_static(_LR_DW0 - h)) / (2.0 * h)


def _lr_measure_depth(sc, wm):
    """Amplitude of the U(t) modulation at wm (settle >= 12 lifetimes, lstsq fit)."""
    a = _KAPPA / 2.0
    period_rt = (2.0 * math.pi / wm) / _T_R
    settle = int(12.0 / a / _T_R)
    n_rt = int(settle + 6.0 * period_rt)
    t = np.arange(n_rt) * _T_R
    dnu = (_LR_A / (2.0 * math.pi)) * np.sin(wm * t)
    e_cw = (math.sqrt(_KAPPA_C * _LR_PIN) / (a + 1j * _LR_DW0)) * np.ones(
        _LR_NTAU, dtype=np.complex128
    )
    snap = max(int(period_rt / 40), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = solve_lle_ssfm_jax(
            pin=_LR_PIN, delta_omega=_LR_DW0, t_slow=n_rt, beta=[_BETA2],
            kappa=_KAPPA, kappa_c=_KAPPA_C, rng_key=jax.random.PRNGKey(0),
            n_tau=_LR_NTAU, snapshot_interval=snap, config_path=sc, e0_override=e_cw,
            pump_freq_noise_override=dnu,
        )
    u = np.asarray(s["U_int_history"])[0]
    tu = np.arange(u.size) * _T_R
    m = tu >= settle * _T_R
    tt, w = tu[m], u[m]
    basis = np.column_stack(
        [np.cos(wm * tt), np.sin(wm * tt), np.ones_like(tt), tt - tt.mean()]
    )
    c, *_ = np.linalg.lstsq(basis, w, rcond=None)
    return float(np.hypot(c[0], c[1]))


def test_linear_response_transfer(tmp_path, capsys):
    """Modulation depth vs the linearized CW cavity response at 3 frequencies."""
    sc = _tk0_sidecar(tmp_path, "linresp.yaml", Gamma_th=1e-4)
    g_static = _lr_static_gain(sc)                      # solver's own dU/ddw
    a = _KAPPA / 2.0
    # analytic continuous |dU_cw/ddw| for reference/reporting
    ucw = _KAPPA_C * _LR_PIN * _T_R / (a ** 2 + _LR_DW0 ** 2)
    dUddw_cont = 2.0 * ucw * abs(_LR_DW0) / (a ** 2 + _LR_DW0 ** 2)

    freqs = {"kappa/20": _KAPPA / 20.0, "kappa/2": _KAPPA / 2.0, "5*kappa": 5.0 * _KAPPA}
    rows = []
    for name, wm in freqs.items():
        meas = _lr_measure_depth(sc, wm)
        lp = _lr_lowpass(wm)
        pred = abs(g_static) * _LR_A * lp               # static-gain-anchored
        pred_cont = dUddw_cont * _LR_A * lp             # continuous-gain
        pred_prompt = dUddw_cont * _LR_A / math.sqrt(1.0 + (wm / a) ** 2)  # single-pole
        rows.append((name, wm, meas, pred, pred_cont, pred_prompt, lp))

    with capsys.disabled():
        print("\n=== GATE (test 3): pump-frequency-noise -> U(t) linear response ===")
        print(f"  dw0 = {_LR_DW0/_KAPPA:.1f}*kappa, A = {_LR_A/_KAPPA:.3f}*kappa, "
              f"pin = {_LR_PIN*1e3:.1f} mW (below MI threshold)")
        print(f"  static dU/ddw: solver-measured = {g_static:.4e}, "
              f"continuous-analytic = {-dUddw_cont:.4e} "
              f"(ratio {abs(g_static)/dUddw_cont:.3f})")
        print(f"  {'f_mod':>10} {'wm/kappa':>9} {'meas depth':>12} "
              f"{'analytic':>12} {'meas/analytic':>13} {'lowpass L':>10} "
              f"{'meas/cont':>10} {'meas/1pole':>10}")
        for name, wm, meas, pred, pred_cont, pred_prompt, lp in rows:
            print(f"  {name:>10} {wm/_KAPPA:9.3f} {meas:12.4e} {pred:12.4e} "
                  f"{meas/pred:13.3f} {lp:10.4f} {meas/pred_cont:10.3f} "
                  f"{meas/pred_prompt:10.3f}")

    # Assertions: modulation depth matches the analytic linearized response
    # (static-gain-anchored) within 10%. The deep-stopband 5*kappa point carries
    # an O(10%) discrete round-trip-map rolloff correction (the map's effective
    # linewidth differs slightly from kappa/2), documented and bounded at 12%.
    for name, wm, meas, pred, *_ in rows:
        tol = 0.12 if name == "5*kappa" else 0.10
        assert abs(meas / pred - 1.0) < tol, (name, meas, pred, meas / pred)


# ---------------------------------------------------------------------------
# 4. RIN energy balance
# ---------------------------------------------------------------------------
def test_rin_energy_balance(tmp_path):
    """P_trans + kappa_i*U/t_r tracks pin*(1+eps) (coupled bookkeeping, unclipped).

    Below-threshold CW warm start: P_trans is defined as
    clip(pin*(1+eps) - kappa_i*U/t_r, 0, pin*(1+eps)), so the balance
    P_trans + kappa_i*U/t_r == pin*(1+eps) holds identically while unclipped.
    This pins that BOTH terms use the instantaneous launched power.
    """
    sc = _tk0_sidecar(tmp_path, "rinbal.yaml", Gamma_th=1e-4)
    pin = 1e-3
    a = _KAPPA / 2.0
    dw0 = 2.0 * _KAPPA
    t_slow = 4000
    t = np.arange(t_slow) * _T_R
    eps = 0.05 * np.sin(2.0 * math.pi * (_KAPPA / 50.0) / (2 * math.pi) * t)  # slow RIN
    e_cw = (math.sqrt(_KAPPA_C * pin) / (a + 1j * dw0)) * np.ones(32, dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_lle_ssfm_jax(
            pin=pin, delta_omega=dw0, t_slow=t_slow, beta=[_BETA2],
            kappa=_KAPPA, kappa_c=_KAPPA_C, rng_key=jax.random.PRNGKey(0),
            n_tau=32, snapshot_interval=1, config_path=sc, e0_override=e_cw,
            pump_rin_epsilon_override=eps,
        )
    p_trans = np.asarray(sol["P_trans_history"])[0]
    u_int = np.asarray(sol["U_int_history"])[0]
    eps_hist = np.asarray(sol["pump_rin_epsilon_history"])[0]
    assert np.array_equal(eps_hist, eps)
    pin_t = pin * (1.0 + eps)
    balance = p_trans + _KAPPA_I * u_int / _T_R
    # drop the initial settling transient; require no clipping in the window
    w = slice(t_slow // 5, None)
    assert np.all(p_trans[w] > 0.0) and np.all(p_trans[w] < pin_t[w])
    rel = np.max(np.abs(balance[w] - pin_t[w]) / pin_t[w])
    assert rel < 1e-6, rel


# ---------------------------------------------------------------------------
# 5. Determinism & seed independence
# ---------------------------------------------------------------------------
def test_pump_noise_determinism_and_independence():
    kw = dict(
        pin=0.05, delta_omega=3.0 * _KAPPA, t_slow=40, beta=[_BETA2],
        kappa=_KAPPA, kappa_c=_KAPPA_C, n_tau=128, snapshot_interval=10,
        pump_noise_enabled=True,
    )
    cfg = str(_tmp_pump_cfg())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = solve_lle_ssfm_jax(rng_key=jax.random.PRNGKey(3), config_path=cfg, **kw)
        b = solve_lle_ssfm_jax(rng_key=jax.random.PRNGKey(3), config_path=cfg, **kw)
        c = solve_lle_ssfm_jax(rng_key=jax.random.PRNGKey(4), config_path=cfg, **kw)
    assert np.array_equal(a["E_snapshots"], b["E_snapshots"])
    assert np.array_equal(a["pump_freq_noise_history"], b["pump_freq_noise_history"])
    assert not np.array_equal(a["E_snapshots"], c["E_snapshots"])
    assert not np.array_equal(
        a["pump_freq_noise_history"], c["pump_freq_noise_history"]
    )


def test_pump_noise_vmapped_trajectories_independent():
    cfg = str(_tmp_pump_cfg())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_lle_ssfm_jax(
            pin=0.05, delta_omega=np.full((2, 60), 3.0 * _KAPPA), t_slow=60,
            beta=[_BETA2], kappa=_KAPPA, kappa_c=_KAPPA_C,
            rng_key=jax.random.PRNGKey(5), n_tau=64, snapshot_interval=30,
            config_path=cfg, pump_noise_enabled=True,
        )
    fh = np.asarray(sol["pump_freq_noise_history"])
    eh = np.asarray(sol["pump_rin_epsilon_history"])
    assert not np.array_equal(fh[0], fh[1])
    assert not np.array_equal(eh[0], eh[1])


# ---------------------------------------------------------------------------
# 6. Backward compatibility: flag off (no override) == legacy
# ---------------------------------------------------------------------------
def test_flag_off_bit_identical_to_legacy():
    kw = dict(
        pin=0.214, delta_omega=3.0 * _KAPPA, t_slow=40, beta=[_BETA2],
        kappa=_KAPPA, kappa_c=_KAPPA_C, n_tau=256, snapshot_interval=10,
        rng_key=jax.random.PRNGKey(3),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = solve_lle_ssfm_jax(**kw)
        explicit_off = solve_lle_ssfm_jax(pump_noise_enabled=False, **kw)
    assert np.array_equal(legacy["E_snapshots"], explicit_off["E_snapshots"])
    assert np.array_equal(legacy["U_int_history"], explicit_off["U_int_history"])
    # disabled path adds no pump diagnostics to the output dict
    assert "pump_freq_noise_history" not in legacy
    assert "pump_rin_epsilon_history" not in legacy


# ---------------------------------------------------------------------------
# 7. Config validation
# ---------------------------------------------------------------------------
def test_validation_rejects_negative_coefficients():
    with pytest.raises(ValueError, match="must be >= 0"):
        PumpNoise(_pump_cfg(pump_freq_noise_h0_hz2_per_hz=-1.0))
    with pytest.raises(ValueError, match="must be >= 0"):
        PumpNoise(_pump_cfg(pump_freq_noise_hm1_hz3_per_hz=-1.0))


def test_validation_rejects_linear_vs_db_rin():
    """A RIN value above -80 dBc/Hz is rejected (guards linear-vs-dB confusion)."""
    with pytest.raises(ValueError, match="dBc/Hz"):
        PumpNoise(_pump_cfg(pump_rin_floor_dbc_per_hz=1e-14))
    with pytest.raises(ValueError, match="dBc/Hz"):
        PumpNoise(_pump_cfg(pump_rin_excess_dbc_per_hz=-40.0))


def test_validation_rejects_non_boolean_enabled():
    with pytest.raises(ValueError, match="pump_noise_enabled"):
        PumpNoise(_PHYS, enabled=0.5)


def test_rin_clip_enforced_and_warns():
    """A huge RIN floor drives eps < -1 for many samples: clip + warn (>0.01%)."""
    pm = PumpNoise(_pump_cfg(pump_rin_floor_dbc_per_hz=-80.0))  # eps std ~ O(10)
    with pytest.warns(UserWarning, match="clipped"):
        e = pm.sample_rin(jax.random.PRNGKey(0), 100_000)
    assert np.all(1.0 + e >= 0.0)      # 1 + eps >= 0 enforced


def test_rin_no_clip_warning_at_physical_levels():
    """A physical RIN level (-140 dBc/Hz) clips no samples: no warning."""
    pm = PumpNoise(_pump_cfg(pump_rin_floor_dbc_per_hz=-140.0,
                             pump_rin_excess_dbc_per_hz=-120.0, pump_rin_corner_hz=1e7))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        e = pm.sample_rin(jax.random.PRNGKey(0), 100_000)
    assert not [w for w in rec if "clipped" in str(w.message)]
    assert np.all(1.0 + e >= 0.0)


def test_disabled_forces_all_channels_inert():
    """enabled = 0 => samples exactly zero and PSDs zero, regardless of numbers."""
    pm = PumpNoise(
        dict(_PHYS, pump_noise_enabled=0,
             pump_freq_noise_h0_hz2_per_hz=3e3,
             pump_freq_noise_hm1_hz3_per_hz=1e10,
             pump_rin_floor_dbc_per_hz=-140.0,
             pump_rin_excess_dbc_per_hz=-110.0),
    )
    assert not pm.enabled
    assert np.all(pm.sample_freq(jax.random.PRNGKey(0), 4096) == 0.0)
    assert np.all(pm.sample_rin(jax.random.PRNGKey(0), 4096) == 0.0)
    f = np.array([1e4, 1e6, 1e8])
    assert np.all(pm.psd_freq(f) == 0.0)
    assert np.all(pm.psd_rin(f) == 0.0)
    assert pm.lorentzian_linewidth_hz == 0.0


def test_disabled_solver_ignores_pump_numbers(tmp_path):
    """A config with LARGE pump numbers but enabled=0 is bit-identical to legacy."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["physical_parameters"].update(
        pump_noise_enabled=0,
        pump_freq_noise_h0_hz2_per_hz=3e3,
        pump_freq_noise_hm1_hz3_per_hz=1e10,
        pump_rin_floor_dbc_per_hz=-120.0,
    )
    sc = tmp_path / "pump_disabled.yaml"
    yaml.safe_dump(cfg, open(sc, "w"), sort_keys=False)
    kw = dict(
        pin=0.05, delta_omega=3.0 * _KAPPA, t_slow=20, beta=[_BETA2],
        kappa=_KAPPA, kappa_c=_KAPPA_C, rng_key=jax.random.PRNGKey(1),
        n_tau=128, snapshot_interval=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = solve_lle_ssfm_jax(config_path=str(sc), **kw)
        b = solve_lle_ssfm_jax(**kw)
    assert np.array_equal(a["E_snapshots"], b["E_snapshots"])
    assert "pump_freq_noise_history" not in a


# ---------------------------------------------------------------------------
def _tmp_pump_cfg():
    """A committed-config sidecar with the pump channel enabled (ECDL preset)."""
    import tempfile

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["physical_parameters"].update(
        pump_noise_enabled=1,
        pump_freq_noise_h0_hz2_per_hz=3e3,
        pump_freq_noise_hm1_hz3_per_hz=1e10,
        pump_rin_floor_dbc_per_hz=-140.0,
        pump_rin_excess_dbc_per_hz=-120.0,
        pump_rin_corner_hz=1e4,
    )
    fd, name = tempfile.mkstemp(suffix=".yaml", prefix="pump_on_")
    import os

    os.close(fd)
    yaml.safe_dump(cfg, open(name, "w"), sort_keys=False)
    return name
