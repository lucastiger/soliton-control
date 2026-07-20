"""Comb noise metrology: elastic-tape-model analysis of per-mode phase noise.

Implements the noise-metrology suite of Herr, Tikan & Kippenberg,
arXiv:2604.05897 Sec. V.B.1, on the solver's per-mode probe records
(``mode_probe_history``: the complex FFT amplitudes E~_mu(t) of a static set
of probed modes, recorded EVERY round trip — see
``solve_lle_ssfm_jax(mode_probe_indices=...)``):

1. Per-probe unwrapped phase phi_mu(t) and its frequency-noise PSD
   S_dnu,mu(f) [Hz^2/Hz] via Welch on the detrended instantaneous frequency.
2. Repetition-rate phase phi_rep(t) = [phi_mu2(t) - phi_mu1(t)]/(mu2 - mu1)
   and its PSDs (phase and frequency-noise).
3. The elastic-tape ("fixed point") decomposition: per Fourier bin, the
   least-squares fit of

       S_mu(f) = S_c(f) + 2*mu*S_cr(f) + mu^2*S_rep(f)

   across >= 5 probe modes — S_c is the common-mode (carrier) frequency-noise
   PSD, S_rep the repetition-rate PSD and S_cr their cross term — and the
   fix-point index mu_fix(f) = -S_cr(f)/S_rep(f), the comb line where the
   two contributions cancel (paper Sec. V.B.1).
4. The effective (FWHM) linewidth of a comb line from its frequency-noise
   PSD via the beta-separation-line integral (Di Domenico, Schilt & Thomann,
   Appl. Opt. 49, 4801 (2010)): only the spectral region where
   S_dnu(f) > beta(f) = 8*ln(2)*f/pi^2 contributes to the linewidth,
   FWHM = sqrt(8*ln(2)*A) with A = integral of S_dnu over that region.
5. Timing jitter from the temporal peak trajectory theta_max(t) of
   ``E_snapshots`` — an independent cross-check of phi_rep (for a pulse at
   angle theta0 the comb-line phases are phi_mu = phi_c - mu*theta0, so
   phi_rep(t) = -theta_max(t) + const).
6. A quiet-point sweep driver: a warm-continuation detuning scan of a single
   soliton (measured D_int grid, pump frequency noise ON) recording S_rep at
   a fixed offset frequency vs detuning — the paper's Sec. V.B.5 quiet-point
   signature is a minimum of that curve.

Sampling conventions: probes arrive once per round trip (f_s = 1/t_r);
snapshots every ``snapshot_interval`` round trips. All spectra are one-sided.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import welch

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# 1. Phases and per-mode frequency-noise PSDs
# ---------------------------------------------------------------------------
def unwrapped_phases(mode_probe_history: np.ndarray) -> np.ndarray:
    """Unwrapped phase phi_mu(t) [rad] per probe, shape (t_slow, n_probe).

    Accepts (t_slow, n_probe) or a single trajectory sliced from the solver's
    (n_traj, t_slow, n_probe) output. Unwrapping is along time and assumes
    the per-round-trip phase step stays below pi — true for probe modes with
    |D_int(mu) + delta_omega|*t_r < pi (|mu| up to ~2000 on this device).
    """
    probes = np.asarray(mode_probe_history)
    if probes.ndim != 2:
        raise ValueError(
            f"mode_probe_history must be (t_slow, n_probe); got shape "
            f"{probes.shape} (slice a single trajectory first)."
        )
    return np.unwrap(np.angle(probes), axis=0)


def instantaneous_frequency(phase: np.ndarray, t_r: float) -> np.ndarray:
    """delta_nu(t) [Hz] from an unwrapped phase trace: (dphi/dt)/(2*pi).

    First differences at the round-trip cadence (length N-1 for length-N
    phase); the mean (the deterministic line offset in the rotating frame)
    is left in — Welch's detrending removes it.
    """
    phase = np.asarray(phase, dtype=np.float64)
    return np.diff(phase, axis=0) / (2.0 * math.pi * float(t_r))


def frequency_noise_psd(phase: np.ndarray, t_r: float,
                        nperseg: int | None = None,
                        detrend: str = "constant"):
    """One-sided frequency-noise PSD S_dnu(f) [Hz^2/Hz] of a phase trace.

    Welch on the detrended instantaneous frequency (constant detrend per
    segment removes the deterministic line offset; the linear phase ramp is
    already gone after the derivative). Returns ``(f [Hz], S [Hz^2/Hz])``
    with the DC bin dropped.
    """
    dnu = instantaneous_frequency(phase, t_r)
    n = dnu.shape[0]
    if nperseg is None:
        nperseg = min(max(256, n // 8), n)
    f, s = welch(dnu, fs=1.0 / float(t_r), nperseg=int(nperseg),
                 detrend=detrend, axis=0)
    return f[1:], np.take(s, np.arange(1, f.size), axis=0)


def phase_noise_psd(phase: np.ndarray, t_r: float,
                    nperseg: int | None = None):
    """One-sided phase PSD S_phi(f) [rad^2/Hz] of a (linearly detrended) trace."""
    phase = np.asarray(phase, dtype=np.float64)
    n = phase.shape[0]
    if nperseg is None:
        nperseg = min(max(256, n // 8), n)
    f, s = welch(phase, fs=1.0 / float(t_r), nperseg=int(nperseg),
                 detrend="linear", axis=0)
    return f[1:], np.take(s, np.arange(1, f.size), axis=0)


# ---------------------------------------------------------------------------
# 2. Repetition-rate phase
# ---------------------------------------------------------------------------
def rep_rate_phase(phases: np.ndarray, mus, i1: int, i2: int) -> np.ndarray:
    """phi_rep(t) = [phi_mu2(t) - phi_mu1(t)] / (mu2 - mu1)  [rad].

    ``phases`` is the (t_slow, n_probe) unwrapped-phase array, ``mus`` the
    probe mode numbers, and ``i1``/``i2`` the probe COLUMN indices of the two
    lines used. The two probes should be well separated in mu for leverage.
    """
    mus = np.asarray(mus)
    dmu = float(mus[i2] - mus[i1])
    if dmu == 0.0:
        raise ValueError("rep_rate_phase needs two distinct probe modes.")
    return (phases[:, i2] - phases[:, i1]) / dmu


# ---------------------------------------------------------------------------
# 3. Elastic-tape-model decomposition (paper Sec. V.B.1)
# ---------------------------------------------------------------------------
def tape_model_fit(phases: np.ndarray, mus, t_r: float,
                   nperseg: int | None = None) -> dict:
    """Per-bin least-squares fit S_mu(f) = S_c + 2*mu*S_cr + mu^2*S_rep.

    Args:
        phases: (t_slow, n_probe) unwrapped probe phases.
        mus: probe mode numbers (length n_probe, >= 5 distinct values — three
            fit parameters per bin need genuine over-determination for the
            fit to reject per-line estimation noise).
        t_r: round-trip time [s].
        nperseg: Welch segment length (default ~ N/8).

    Returns dict with ``f`` [Hz], the fitted ``S_c``/``S_cr``/``S_rep``
    (frequency-noise units, Hz^2/Hz; ``S_cr`` may be negative), the fix-point
    index ``mu_fix(f) = -S_cr/S_rep`` (NaN where S_rep is not resolved), the
    per-probe PSD matrix ``S_mu`` (n_bins, n_probe), and ``mus``.
    """
    mus = np.asarray(mus, dtype=np.float64)
    if mus.size < 5 or np.unique(mus).size < 5:
        raise ValueError(
            f"tape_model_fit needs >= 5 distinct probe modes (got "
            f"{np.unique(mus).size}): the 3-parameter per-bin fit must be "
            f"over-determined."
        )
    if phases.shape[1] != mus.size:
        raise ValueError(
            f"phases has {phases.shape[1]} probes but mus has {mus.size}."
        )
    f, s_mu = frequency_noise_psd(phases, t_r, nperseg=nperseg)  # (nb, np)
    # Design matrix per probe: S_mu = [1, 2*mu, mu^2] . [S_c, S_cr, S_rep]
    a = np.stack([np.ones_like(mus), 2.0 * mus, mus**2], axis=1)  # (np, 3)
    coef, *_ = np.linalg.lstsq(a, s_mu.T, rcond=None)             # (3, nb)
    s_c, s_cr, s_rep = coef[0], coef[1], coef[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_fix = np.where(s_rep > 0.0, -s_cr / s_rep, np.nan)
    return {
        "f": f,
        "S_c": s_c,
        "S_cr": s_cr,
        "S_rep": s_rep,
        "mu_fix": mu_fix,
        "S_mu": s_mu,
        "mus": mus,
    }


# ---------------------------------------------------------------------------
# 4. Effective linewidth (beta-separation line)
# ---------------------------------------------------------------------------
def effective_linewidth(f: np.ndarray, s_dnu: np.ndarray) -> float:
    """FWHM effective linewidth [Hz] from a one-sided S_dnu(f) [Hz^2/Hz].

    Beta-separation-line convention of Di Domenico, Schilt & Thomann,
    Appl. Opt. 49, 4801 (2010): only the "slow-modulation" region where
    S_dnu(f) exceeds beta(f) = 8*ln(2)*f/pi^2 contributes to the linewidth;
    integrating S_dnu over that region,

        A = integral_{S_dnu(f) > beta(f)} S_dnu(f) df,
        FWHM = sqrt(8*ln(2)*A).

    Returns 0.0 when the PSD never crosses the beta line (linewidth is then
    set by the (unresolved) Lorentzian wings, below this estimator's floor).
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s_dnu, dtype=np.float64)
    beta = 8.0 * _LN2 * f / math.pi**2
    mask = s > beta
    if not mask.any():
        return 0.0
    area = float(np.trapezoid(np.where(mask, s, 0.0), f))
    return math.sqrt(8.0 * _LN2 * max(area, 0.0))


# ---------------------------------------------------------------------------
# 5. Timing jitter from the temporal peak trajectory
# ---------------------------------------------------------------------------
def peak_angle_trajectory(e_snapshots: np.ndarray) -> np.ndarray:
    """theta_max(t) [rad, unwrapped] of |E|^2 per snapshot, sub-cell accurate.

    Circular parabolic interpolation of the |E|^2 maximum on the periodic
    fast-time grid; the trajectory is unwrapped so slow drifts accumulate
    rather than fold at 2*pi.
    """
    snaps = np.asarray(e_snapshots)
    if snaps.ndim != 2:
        raise ValueError(
            f"e_snapshots must be (n_snapshots, n_tau); got {snaps.shape}."
        )
    p = np.abs(snaps) ** 2
    n_tau = p.shape[1]
    idx = np.argmax(p, axis=1)
    rows = np.arange(p.shape[0])
    y0 = p[rows, (idx - 1) % n_tau]
    y1 = p[rows, idx]
    y2 = p[rows, (idx + 1) % n_tau]
    denom = y0 - 2.0 * y1 + y2
    frac = np.where(np.abs(denom) > 0.0,
                    0.5 * (y0 - y2) / np.where(denom == 0.0, 1.0, denom), 0.0)
    theta = (idx + frac) * (2.0 * math.pi / n_tau)
    return np.unwrap(theta)


def timing_jitter(e_snapshots: np.ndarray, snapshot_interval: int,
                  t_r: float, nperseg: int | None = None) -> dict:
    """Timing jitter of the pulse from theta_max(t): trace + PSD.

    The pulse timing deviation is dt(t) = theta_max(t)/(2*pi)*t_r; its PSD
    S_dt(f) [s^2/Hz] is Welch on the linearly detrended trace (removing the
    deterministic drift velocity). This is the independent cross-check of
    phi_rep: phi_rep(t) = -theta_max(t) + const, so
    S_phi_rep(f) == S_theta(f) on the common band. Snapshots are sampled at
    f_s = 1/(snapshot_interval*t_r).
    """
    theta = peak_angle_trajectory(e_snapshots)
    fs = 1.0 / (float(snapshot_interval) * float(t_r))
    n = theta.size
    if nperseg is None:
        nperseg = min(max(64, n // 8), n)
    f, s_theta = welch(theta, fs=fs, nperseg=int(nperseg), detrend="linear")
    dt = theta * (float(t_r) / (2.0 * math.pi))
    return {
        "theta_max": theta,
        "dt_s": dt,
        "f": f[1:],
        "S_theta": s_theta[1:],                          # rad^2/Hz
        "S_dt": s_theta[1:] * (float(t_r) / (2.0 * math.pi)) ** 2,  # s^2/Hz
        "jitter_rms_s": float(np.std(dt - np.polyval(
            np.polyfit(np.arange(n), dt, 1), np.arange(n)))),
    }


# ---------------------------------------------------------------------------
# PSD sampling helper
# ---------------------------------------------------------------------------
def psd_at_offset(f: np.ndarray, s: np.ndarray, f_offset: float) -> float:
    """Log-log interpolated PSD value at a fixed offset frequency."""
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    keep = (f > 0) & (s > 0) & np.isfinite(s)
    if keep.sum() < 2:
        return float("nan")
    return float(10.0 ** np.interp(math.log10(f_offset),
                                   np.log10(f[keep]), np.log10(s[keep])))


# ---------------------------------------------------------------------------
# 6. Quiet-point sweep driver (paper Sec. V.B.5)
# ---------------------------------------------------------------------------
def quiet_point_sweep(
    dw_grid_kappa=None,
    *,
    n_tau: int = 2048,
    hold_rt: int = 20_000,
    settle_rt: int = 4_000,
    probe_mus: tuple = (-75, -50, -25, 25, 50, 75),
    f_offset_hz: float = 2.0e8,
    pump_h0_hz2_per_hz: float = 3.0e3,
    seed: int = 0,
    config_path=None,
    use_measured_dint: bool = True,
) -> dict:
    """Warm-continuation quiet-point scan: S_rep(f_offset) vs detuning.

    Seeds a single soliton at the highest detuning of ``dw_grid_kappa`` (an
    analytic sech ansatz, settled for ``settle_rt`` round trips), then
    warm-continues DOWN the grid, holding each detuning for ``hold_rt``
    round trips with pump frequency noise ON (white plateau
    ``pump_h0_hz2_per_hz``; quantum noise off so the pump channel dominates)
    and the measured ``d_int_grid`` (Taylor-D2 with
    ``use_measured_dint=False`` for the comparison overlay). Per hold the
    probe phases give the tape-model S_rep(f); the recorded observable is
    S_rep at the fixed offset ``f_offset_hz`` — the paper's V.B.5 quiet
    point is a minimum of that curve vs detuning (it may be shallow for
    this device; the driver reports whatever the physics gives).

    Returns per-step arrays plus the located minimum. Reuses the
    run/warm-continuation machinery of analysis/run_detuning_sweep.py via
    analysis.dks_access.
    """
    import dataclasses as _dc

    import jax as _jax

    from analysis.dks_access import (
        CONFIG_PATH, PIN_W, PRODUCTION_NUMERICS, _run, attach_dispersion,
        load_cavity_params, sech_soliton_seed,
    )
    from analysis.run_detuning_sweep import write_noise_off_config

    if dw_grid_kappa is None:
        dw_grid_kappa = np.linspace(11.0, 7.0, 17)
    dw_grid_kappa = np.asarray(dw_grid_kappa, dtype=np.float64)

    base_cfg = Path(config_path) if config_path is not None else CONFIG_PATH
    # Sidecar config: classical/quantum noise OFF, pump FREQUENCY noise ON —
    # the quiet point is a property of the pump-noise transfer, so the scan
    # isolates that channel.
    import tempfile as _tf

    import yaml as _yaml
    noff = write_noise_off_config(base_config_path=base_cfg)
    cfg = _yaml.safe_load(open(noff, encoding="utf-8"))
    pp = cfg["physical_parameters"]
    pp["pump_noise_enabled"] = 1
    pp["pump_freq_noise_h0_hz2_per_hz"] = float(pump_h0_hz2_per_hz)
    fd, name = _tf.mkstemp(prefix="sin_params_quietpoint_", suffix=".yaml")
    import os as _os
    _os.close(fd)
    qp_cfg = Path(name)
    with open(qp_cfg, "w", encoding="utf-8") as fh:
        _yaml.safe_dump(cfg, fh, sort_keys=False)

    cav = load_cavity_params(base_cfg)
    if use_measured_dint:
        cav = attach_dispersion(cav, n_tau)
    kappa = cav.kappa

    dw0 = float(dw_grid_kappa[0]) * kappa
    seed_field = sech_soliton_seed(dw0, cav, n_tau=n_tau, pin=PIN_W)
    sol0 = _run(dw0, int(settle_rt), cav, e0=seed_field, seed=seed,
                n_tau=n_tau, pin=PIN_W, snapshot_interval=int(settle_rt),
                config_path=qp_cfg, **PRODUCTION_NUMERICS)
    e_prev = np.asarray(sol0["e_final"])[0]
    dt_prev = float(np.asarray(sol0["delta_t_final"]).reshape(-1)[0])

    rows = []
    for i, dwk in enumerate(dw_grid_kappa):
        sol = _run(dwk * kappa, int(hold_rt), cav, e0=e_prev,
                   delta_t0=dt_prev, seed=seed + 1 + i, n_tau=n_tau,
                   pin=PIN_W, snapshot_interval=max(hold_rt // 8, 1),
                   config_path=qp_cfg, mode_probe_indices=tuple(probe_mus),
                   **PRODUCTION_NUMERICS)
        e_prev = np.asarray(sol["e_final"])[0]
        dt_prev = float(np.asarray(sol["delta_t_final"]).reshape(-1)[0])
        probes = np.asarray(sol["mode_probe_history"])[0]
        # Discard the re-settling transient (first quarter of the hold).
        phases = unwrapped_phases(probes[hold_rt // 4:])
        fit = tape_model_fit(phases, probe_mus, cav.t_r)
        s_rep_off = psd_at_offset(fit["f"], fit["S_rep"], f_offset_hz)
        s_c_off = psd_at_offset(fit["f"], fit["S_c"], f_offset_hz)
        u_tail = float(np.mean(np.asarray(sol["U_int_history"])[0][-hold_rt // 4:]))
        rows.append({
            "dw_over_kappa": float(dwk),
            "S_rep_at_offset": s_rep_off,
            "S_c_at_offset": s_c_off,
            "mu_fix_at_offset": psd_at_offset(
                fit["f"], np.abs(fit["mu_fix"]), f_offset_hz),
            "U_int": u_tail,
        })
        print(f"[quiet-point] {i + 1:2d}/{dw_grid_kappa.size} "
              f"dw={dwk:5.2f}k  S_rep({f_offset_hz:.1e} Hz) = "
              f"{s_rep_off:.3e} Hz^2/Hz  U={u_tail:.3e} J")
        _jax.clear_caches()

    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    finite = np.isfinite(out["S_rep_at_offset"])
    if finite.any():
        i_min = int(np.nanargmin(np.where(finite, out["S_rep_at_offset"],
                                          np.nan)))
        out["quiet_point_dw_over_kappa"] = float(out["dw_over_kappa"][i_min])
        out["quiet_point_S_rep"] = float(out["S_rep_at_offset"][i_min])
        depth = (np.nanmax(out["S_rep_at_offset"][finite])
                 / max(out["quiet_point_S_rep"], 1e-300))
        out["quiet_point_depth"] = float(depth)
    out["f_offset_hz"] = float(f_offset_hz)
    out["probe_mus"] = np.asarray(probe_mus)
    out["use_measured_dint"] = bool(use_measured_dint)
    return out
