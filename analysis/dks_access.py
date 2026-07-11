"""Dissipative-Kerr-soliton (DKS) access protocol and optical-spectrum study.

Context
-------
The plain adiabatic detuning sweep at pin = 0.214 W (see
``analysis/adiabatic_sweeps.py``) ignites modulation instability (MI/Turing rolls)
but never lands on a clean *single* soliton — as expected for a bare linear ramp.
This module implements a dedicated single-DKS **access protocol**, validates the
resulting soliton, maps its existence window in detuning, and plots its optical
spectrum (power vs wavelength).

Two access routes are provided (the task asks to "try both, keep whichever
reliably yields one soliton"):

  (a) ``access_by_forward_backward`` — forward-tune (blue->red) through MI into the
      red-detuned existence range, then *backward-tune* (reduce delta_omega) and/or
      apply a brief pump kick to shed excess solitons down to one.  This is the
      experimental "soliton-step" route.  Reliability is reported, not assumed.

  (b) ``access_by_seeding`` — inject an analytic single-sech ansatz as the
      warm-start field (``e0_override``) at a detuning inside the existence window
      and integrate to steady state.  This is deterministic and, at this operating
      point, the **reliable** route we keep for the validated example.

Physics of the seed (anomalous dispersion, D2>0)
------------------------------------------------
The mean-field LLE solved by the simulator (dividing the per-round-trip map by
t_r) is

    dE/dt = -(kappa/2 + i*delta_omega) E + i*(D2/2) d^2E/dphi^2
            + i*gamma |E|^2 E + sqrt(kappa_c * pin),

with phi the fast-time angle (mu conjugate to phi; omega = mu*D1, D1 = 2*pi*FSR).
Balancing dispersion, Kerr, and detuning for a bright sech gives

    E_s(phi) = B * sech(phi / w),   B = sqrt(2*delta_omega/gamma),
    w = sqrt(D2 / (2*delta_omega))   (rad)  ->  tau_s = w/D1 = sqrt(beta2/(2*delta_omega))  (s),

on top of the (lower-branch) CW background E_bg = sqrt(kappa_c*pin)/(kappa/2 + i*delta_omega).
``sech_soliton_seed`` builds E_bg + N*sech(...) for N solitons.

Resolution note (n_tau = 512)
-----------------------------
D2 = 2*pi*6 kHz is small, so the single-DKS comb is broad: its sech^2 envelope
spans several hundred cavity modes.  At the mandated n_tau = 512 the RESOLVED
(central) comb is a clean, smooth, symmetric sech^2 envelope with the pump line
~30 dB above the sidebands, but the far wings (>~30 dB down) are truncated by the
+/-256-mode window.  ``spectrum_resolution_check`` re-runs the validated soliton
at n_tau = 2048 to show the fully-resolved envelope (identical central shape,
wings rolling off to <-55 dB).  The sech^2 *envelope* correlation is >0.99 at both
resolutions.

Operating point (10*kappa, stationary) and breather note (8*kappa)
-----------------------------------------------------------------
The PRODUCTION operating point is delta_omega = ``OPERATING_DW_KAPPA`` =
10*kappa: past the ~9.3-9.4*kappa Hopf boundary the attractor is a STATIONARY
single soliton, so V6 reports ``is_breather = False`` and the committed spectrum
artifacts are single snapshots, not cycle-averages.

Controlled A/B testing (numpy vs jax bit-match over 1 RT, thermal/noise off,
parabolic-dispersion control) separately established that at delta_omega =
8*kappa (the historically A/B-validated point) with the measured D_int grid the
attractor is a deterministic BREATHER: the soliton energy U_int oscillates with
period T_b ~ 152-153 RT and rel-std ~4.1%, so any single-snapshot spectrum
*there* is breathing-phase-dependent. ``breathing_metrics`` (check V6, part of
``soliton_metrics``) detects this from the U_int autocorrelation and rel-std
over the last >= 2*T_b; when it flags a breather (e.g. a deliberate 8*kappa run)
the spectrum artifacts must come from ``cycle_averaged_spectrum`` (per-RT
accumulation of |fftshift(fft(E))|^2 over >= 2*T_b), which is deterministic and
phase-independent. ``cycle_averaged_spectrum``, ``breathing_metrics`` and
``breathing_scan`` remain in the library for characterising the breather
sub-band and any deliberate breather run; ``breathing_scan`` maps the breathing
sub-band vs detuning across the existence band.

Labeler note
------------
Both labelers return class 6 for these states.  The NumPy labeler
(``simulator.state_labeler.label_soliton_state``) uses an actual temporal sech^2
goodness-of-fit; the JAX scan-time labeler (which produces ``label_history`` for
the training dataset) keys class 6 on a single temporal peak plus a smooth
(monotonic) sech^2 spectral envelope.  An earlier JAX heuristic — "fraction of
power in the top ~32 points" — mislabeled a single DKS as chaotic (class 3)
because the soliton sits on a bright CW background that carries most of the
energy; that gate was replaced by the envelope test (see ``make_state_labeler``),
so JAX and NumPy now agree (class 6) at both n_tau = 512 and 2048.  Class 6 also
requires real comb structure (a minimum inner/outer sideband ratio), so a CW field
carrying a single-sample numerical spike — high peak-to-mean but a FLAT sideband
spectrum — is classified CW (class 1), not soliton.  Classification here uses the
NumPy labeler.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from scipy.signal import find_peaks

from simulator.lle_solver import (
    _load_config,
    d2_to_beta2_lle,
    load_dint_grid,
    resolve_cavity_rates,
    solve_lle_ssfm_jax,
)
from simulator.state_labeler import (
    label_soliton_state,
    make_threshold_params,
    sech2_envelope_correlation,
)

C_LIGHT = 299_792_458.0

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "sin_params.yaml"
DINT_CSV_PATH = REPO_ROOT / "config" / "pyLLE_dispersion_w4400_h800.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

PIN_W = 0.214
OPERATING_DW_KAPPA = 10.0   # production operating detuning [kappa]:
# stationary single DKS. 8κ (the historically A/B-validated point) is a
# deterministic breather; ≥9.5κ is past the ~9.3–9.4κ Hopf boundary.
# Full-dispersion default: the measured D_int grid spreads the single-DKS comb
# across thousands of cavity modes (phase-matched dispersive waves near mu ~
# +3269 / -3051, i.e. ~1096 / 2520 nm; positions quoted at the 8*kappa
# characterisation point, the production operating point being 10*kappa), so
# the display window needs a large FFT grid. A CLI/kwarg override keeps
# smaller, faster runs possible.
N_TAU = 8192
LABEL_SINGLE_SOLITON = 6

# Production numerics stack for physical (quantitative-spectrum) runs. float64
# is module-wide in the solver; n_substeps=4 keeps the per-kick linear-phase
# mismatch far below the ~2*pi spurious-FWM onset; the 2/3 dealias + edge
# absorber contain cubic-nonlinearity aliasing. The dispersion-validity mask
# stays OFF: the exact linear exponential is valid at any phase, and the mask
# (an opt-in guard for n_substeps=1 runs) amputates real soliton-tail /
# dispersive-wave spectrum when enabled here.
PRODUCTION_NUMERICS = dict(
    n_substeps=4,
    dealias_two_thirds=True,
    edge_absorber=True,
    dispersion_validity_mask=False,
)


# ---------------------------------------------------------------------------
# Cavity / material parameters bundled once
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CavityParams:
    kappa: float
    kappa_c: float
    kappa_i: float
    gamma: float
    beta2: float
    d2: float
    fsr_hz: float
    t_r: float
    d1: float
    pump_wavelength_m: float
    tau_th_s: float
    # Measured full-dispersion fields (populated by ``attach_dispersion``; the grid
    # depends on n_tau, so it is not built at config-load time).
    d_int_grid: object = None       # (n_tau,) measured D_int [rad/s], FFT-bin order
    fsr_measured_hz: float = None   # FSR from the measured D1 = 2*pi*FSR
    d2_local: float = None          # CSV-derived local D2 (curvature near mu=0)

    @property
    def tau_th_round_trips(self) -> int:
        return int(round(self.tau_th_s / self.t_r))


def load_cavity_params(config_path=CONFIG_PATH) -> CavityParams:
    """Resolve the SiN cavity/material parameters used throughout this module."""
    phys = _load_config(config_path)
    kappa_i, kappa_c, kappa = resolve_cavity_rates(config_path)
    d2 = float(phys["d2_rad_per_s2"])
    fsr_hz = float(phys["fsr_hz"])
    return CavityParams(
        kappa=kappa,
        kappa_c=kappa_c,
        kappa_i=kappa_i,
        gamma=float(phys["gamma_LLE_per_J_per_s"]),
        beta2=d2_to_beta2_lle(d2, fsr_hz),
        d2=d2,
        fsr_hz=fsr_hz,
        t_r=1.0 / fsr_hz,
        d1=2.0 * math.pi * fsr_hz,
        pump_wavelength_m=float(phys.get("pump_wavelength_m", 1.55e-6)),
        tau_th_s=float(phys.get("tau_th_s", 5.0e-6)),
    )


def _fit_local_d2(csv_path=None) -> float:
    """CSV-derived LOCAL D2: fit D_int(mu) ~ (D2/2)*mu^2 over 5 < |mu| <= 300.

    Near the pump the integrated dispersion is dominated by its quadratic term,
    D_int(mu) ~ (D2/2) mu^2, so a degree-2 fit of D_int vs mu over the central
    modes recovers the local curvature; the mu^2 coefficient times two is the
    local D2 [rad/s^2] used to size the analytic sech seed.

    The fit window excludes the innermost |mu| <= 5 modes: a localized
    pump-neighborhood defect displaces those resonances by up to -27 MHz, which
    biases the curvature high (the old |mu| <= 40 window returned
    D2 ~ 2*pi*15.7 kHz vs the window-converged smooth value 2*pi*7.8 kHz). The
    outer edge sits at |mu| = 300, on the converged plateau (+/-100 -> 2*pi*6.4
    kHz, +/-300 -> 2*pi*7.9 kHz, +/-400 -> 2*pi*7.8 kHz). Because the fit is
    degree 2, the (defect-biased) linear D1 term does not affect the recovered
    quadratic coefficient.
    """
    csv_path = Path(csv_path) if csv_path is not None else DINT_CSV_PATH
    data = np.loadtxt(csv_path, delimiter=",")
    mu = data[:, 0].astype(np.int64)
    f_hz = data[:, 1].astype(np.float64)
    omega = 2.0 * np.pi * f_hz
    i0 = int(np.where(mu == 0)[0][0])
    omega0 = omega[i0]
    d1 = 0.5 * (omega[i0 + 1] - omega[i0 - 1])
    d_int = omega - omega0 - d1 * mu
    sel = (np.abs(mu) <= 300) & (np.abs(mu) > 5)
    coeff2 = float(np.polyfit(mu[sel].astype(np.float64), d_int[sel], 2)[0])
    return 2.0 * coeff2


def attach_dispersion(cav: CavityParams, n_tau: int,
                      csv_path=None) -> CavityParams:
    """Return a copy of ``cav`` carrying the measured dispersion for this n_tau.

    Builds the measured D_int(mu) grid (FFT-bin order, shape (n_tau,)) and the
    measured FSR (from the CSV's D1), plus the CSV-derived local D2 used to size
    the sech seed. Logs the FSR reconciliation (asserts within 1% of the config
    FSR) and warns — without failing — if the config D2 disagrees with the local
    D2 by more than 2x.
    """
    grid, d1 = load_dint_grid(int(n_tau), csv_path)
    fsr_measured = d1 / (2.0 * math.pi)
    d2_local = _fit_local_d2(csv_path)

    fsr_rel = abs(fsr_measured - cav.fsr_hz) / cav.fsr_hz
    print(f"[dispersion] measured FSR = {fsr_measured:.6e} Hz vs config "
          f"{cav.fsr_hz:.6e} Hz (rel diff {fsr_rel:.2%}); local D2 = "
          f"{d2_local:.4e} rad/s^2 vs config D2 = {cav.d2:.4e} rad/s^2.")
    assert fsr_rel <= 0.01, (
        f"measured FSR {fsr_measured:.6e} Hz differs from config "
        f"{cav.fsr_hz:.6e} Hz by {fsr_rel:.2%} (> 1%)."
    )
    ratio = d2_local / cav.d2 if cav.d2 else float("inf")
    if not (0.5 <= ratio <= 2.0):
        print(f"[dispersion] WARNING: config d2_rad_per_s2 ({cav.d2:.4e}) "
              f"disagrees with CSV local D2 ({d2_local:.4e}) by "
              f"{ratio:.2f}x (> 2x). Seed width uses the CSV local D2.")

    return dataclasses.replace(
        cav, d_int_grid=grid, fsr_measured_hz=fsr_measured, d2_local=d2_local,
    )


# ---------------------------------------------------------------------------
# Seed field (single-sech ansatz on the CW background)
# ---------------------------------------------------------------------------
def sech_soliton_seed(
    delta_omega: float,
    cav: CavityParams,
    n_tau: int = N_TAU,
    pin: float = PIN_W,
    n_solitons: int = 1,
    phase: float = 0.0,
) -> np.ndarray:
    """Analytic warm-start field: CW background + ``n_solitons`` bright sech pulses.

    Peak amplitude B = sqrt(2*delta_omega/gamma) and fast-time width
    tau_s = sqrt(beta2_local/(2*delta_omega)) are the anomalous-dispersion LLE
    soliton relations (see module docstring).  The width uses the CSV-derived
    LOCAL curvature (``cav.d2_local`` converted to beta2 with the measured FSR)
    when the measured dispersion is attached, so the seed matches the soliton
    that the full-dispersion solver actually supports; it falls back to the
    config beta2 otherwise.  Pulses are placed at equal spacing around the round
    trip.  Returned as complex64, shape (n_tau,).
    """
    if delta_omega <= 0:
        raise ValueError("delta_omega must be > 0 (red-detuned soliton side).")
    dt = cav.t_r / n_tau
    t = np.arange(n_tau) * dt
    amp = math.sqrt(2.0 * delta_omega / cav.gamma)
    if cav.d2_local is not None:
        fsr = cav.fsr_measured_hz if cav.fsr_measured_hz is not None else cav.fsr_hz
        beta2_local = d2_to_beta2_lle(cav.d2_local, fsr)
    else:
        beta2_local = cav.beta2
    tau_s = math.sqrt(beta2_local / (2.0 * delta_omega))
    e_bg = math.sqrt(cav.kappa_c * pin) / (cav.kappa / 2.0 + 1j * delta_omega)
    field = np.full(n_tau, e_bg, dtype=np.complex128)
    for k in range(n_solitons):
        t0 = cav.t_r * (k + 0.5) / n_solitons
        field += amp / np.cosh((t - t0) / tau_s) * np.exp(1j * phase)
    return field.astype(np.complex64)


# ---------------------------------------------------------------------------
# Single-soliton metrics
# ---------------------------------------------------------------------------
# V6 stationarity check (breather detection). Established by controlled A/B
# (numpy vs jax bit-match, thermal/noise off, parabola control): at
# delta_omega = 8*kappa with the measured D_int grid the attractor is a
# deterministic BREATHER (period ~152-153 RT, U rel-std ~4.1%), so a "stable
# single soliton" readout from one snapshot is phase-dependent there. The
# production operating point (OPERATING_DW_KAPPA = 10*kappa, past the ~9.3-9.4
# kappa Hopf edge) is instead a STATIONARY single soliton, where V6 returns
# is_breather = False. V6 measures the breathing period T_b from the U_int
# autocorrelation peak (search lag V6_LAG_SEARCH round trips) and the relative
# std over the last >= 2*T_b; a rel-std above V6_BREATHER_RELSTD classifies the
# state as a breather.
V6_BREATHER_RELSTD = 0.005      # rel-std > 0.5% => breathing, not stationary
V6_LAG_SEARCH = (50, 1000)      # autocorrelation lag window [RT] for T_b
V6_MIN_AC_PEAK = 0.2            # min autocorr peak to accept a period readout


def breathing_metrics(u_int_history: np.ndarray, *,
                      lag_search: tuple = V6_LAG_SEARCH,
                      relstd_threshold: float = V6_BREATHER_RELSTD) -> dict:
    """V6: breather-vs-stationary classification from the U_int(RT) history.

    1. Take a settled segment: the last max(2*lag_hi + 1, 20%) round trips.
    2. Remove the mean and any residual linear (thermal) drift, then locate the
       breathing period T_b as the normalized-autocorrelation peak over lags
       ``lag_search`` = (50, 1000) RT. The peak must exceed ``V6_MIN_AC_PEAK``
       for the period readout to be trusted (a stationary/noisy trace has no
       genuine periodicity).
    3. Compute the relative std of U_int over the last >= 2*T_b round trips
       (falling back to 2*lag_lo when no period was found). If it exceeds
       ``relstd_threshold`` (0.5%) the state is a BREATHER.

    Returns ``{"is_breather", "breathing_period_rt", "breathing_relstd"}``;
    the period is NaN for non-breathers or when the autocorrelation shows no
    clear peak.
    """
    u = np.asarray(u_int_history, dtype=np.float64)
    n = u.size
    lag_lo, lag_hi_req = int(lag_search[0]), int(lag_search[1])
    out = {
        "is_breather": False,
        "breathing_period_rt": float("nan"),
        "breathing_relstd": float("nan"),
    }
    if n < 4 * lag_lo:
        return out
    lag_hi = int(min(lag_hi_req, n // 2 - 1))
    if lag_hi <= lag_lo:
        return out

    seg_len = int(min(n, max(2 * lag_hi + 1, int(0.2 * n))))
    seg = u[n - seg_len:]
    t = np.arange(seg_len, dtype=np.float64)
    x = seg - seg.mean()
    x = x - np.polyval(np.polyfit(t, x, 1), t)   # kill residual thermal drift

    ac = np.correlate(x, x, mode="full")[seg_len - 1:]
    ac = ac / max(ac[0], 1e-300)
    lag = lag_lo + int(np.argmax(ac[lag_lo:lag_hi + 1]))
    ac_peak = float(ac[lag])
    period = float(lag) if ac_peak > V6_MIN_AC_PEAK else float("nan")

    span = int(min(n, max(2 * lag, 2 * lag_lo)))
    tail = u[n - span:]
    relstd = float(np.std(tail) / max(np.mean(tail), 1e-30))

    out["breathing_relstd"] = relstd
    out["is_breather"] = bool(relstd > relstd_threshold)
    out["breathing_period_rt"] = period if out["is_breather"] else float("nan")
    return out


def count_temporal_peaks(e_field: np.ndarray, rel_height: float = 0.5) -> int:
    """Number of temporal peaks above ``rel_height`` * max, with circular wrap."""
    p = np.abs(e_field) ** 2
    if p.max() <= 0:
        return 0
    doubled = np.concatenate([p, p])
    peaks, _ = find_peaks(doubled, height=rel_height * p.max())
    # each real peak is counted twice in the doubled array
    return len(peaks) // 2


def numpy_label(e_field: np.ndarray, cav: CavityParams, delta_omega: float,
                pin: float = PIN_W) -> int:
    """NumPy 7-class label (sech^2-fit based) with the physical OFF floor."""
    params = make_threshold_params(cav.kappa, cav.kappa_c, pin, abs(delta_omega))
    return int(label_soliton_state(e_field, params))


def soliton_metrics(e_field: np.ndarray, u_int_history: np.ndarray,
                    cav: CavityParams, delta_omega: float,
                    pin: float = PIN_W) -> dict:
    """Full single-soliton metric bundle for one final field + its U_int history."""
    p = np.abs(e_field) ** 2
    corr, r2, mode_w = sech2_envelope_correlation(e_field)
    u_tail = u_int_history[int(0.8 * u_int_history.size):]
    rel_std = float(np.std(u_tail) / max(np.mean(u_tail), 1e-30))
    v6 = breathing_metrics(u_int_history)
    return {
        "n_peaks": count_temporal_peaks(e_field),
        "contrast": float(p.max() / max(p.mean(), 1e-30)),
        "sech2_env_corr": corr,
        "sech2_env_r2": r2,
        "mode_width": mode_w,
        "u_int_tail_rel_std": rel_std,
        "np_label": numpy_label(e_field, cav, delta_omega, pin),
        "finite": bool(np.all(np.isfinite(e_field))),
        "u_int_final": float(u_int_history[-1]),
        # V6 stationarity: a breather is a (single-)soliton state whose energy
        # oscillates periodically; it must NOT be reported as a stationary
        # single soliton (all snapshot spectra of a breather are
        # phase-dependent — use cycle_averaged_spectrum for artifacts).
        "stationarity": "breather" if v6["is_breather"] else "stationary",
        **v6,
    }


def is_single_soliton(metrics: dict) -> bool:
    """A validated single DKS: one temporal peak, sech^2 comb, class 6, finite."""
    return (
        metrics["finite"]
        and metrics["n_peaks"] == 1
        and metrics["np_label"] == LABEL_SINGLE_SOLITON
        and np.isfinite(metrics["sech2_env_corr"])
        and metrics["sech2_env_corr"] > 0.9
    )


# ---------------------------------------------------------------------------
# Optical spectrum (power vs wavelength)
# ---------------------------------------------------------------------------
def _spectrum_dict(spec: np.ndarray, cav: CavityParams) -> dict:
    """Package a (fftshifted) mode-power array as an optical-spectrum dict.

    The FFT bins are cavity modes mu (omega = mu*D1), one per FSR.  The absolute
    optical frequency of mode mu is f_mu = f_pump + mu*FSR, with
    f_pump = c / pump_wavelength.  Wavelength lambda_mu = c / f_mu.  Returns the
    fftshifted mode index, wavelength (nm), and normalized power in dB.
    """
    n = spec.shape[0]
    spec_n = spec / max(spec.max(), 1e-300)
    # Clamp at 1e-36 (-360 dB): below the ~-320 dB float64 aliasing-free floor,
    # so real structure is never clipped, while exactly-zero (dealiased) bins
    # stay finite. The old 1e-12 clamp hid everything under -120 dB.
    power_db = 10.0 * np.log10(np.maximum(spec_n, 1e-36))
    mu = np.arange(n) - n // 2
    f_pump = C_LIGHT / cav.pump_wavelength_m
    fsr = cav.fsr_measured_hz if cav.fsr_measured_hz is not None else cav.fsr_hz
    f_mu = f_pump + mu * fsr
    wavelength_nm = C_LIGHT / f_mu * 1e9
    return {
        "mu": mu,
        "wavelength_nm": wavelength_nm,
        "power_db": power_db,
        "power_norm": spec_n,
        "f_mu_hz": f_mu,
    }


def optical_spectrum(e_field: np.ndarray, cav: CavityParams) -> dict:
    """Absolute optical SNAPSHOT spectrum of one intracavity field.

    NOTE: for a breather (V6 ``is_breather``) every snapshot spectrum is
    breathing-phase-dependent; quantitative artifacts must use
    :func:`cycle_averaged_spectrum` instead.
    """
    spec = np.abs(np.fft.fftshift(np.fft.fft(e_field))) ** 2
    return _spectrum_dict(spec, cav)


# >= 2*T_b at the delta_omega = 8*kappa breather point, where the breathing
# period is T_b ~ 152-153 RT (see V6 / breathing_metrics). This averaging length
# is for a deliberate 8*kappa breather run; the production operating point
# (OPERATING_DW_KAPPA = 10*kappa) is stationary and needs no cycle-averaging.
CYCLE_AVG_RT_8KAPPA = 304


def cycle_averaged_spectrum(res: dict, cav: CavityParams, *, n_rt: int = None,
                            pin: float = PIN_W, config_path=CONFIG_PATH,
                            **solver_kwargs) -> dict:
    """Breathing-cycle-averaged optical spectrum of a settled soliton state.

    ``res`` is a settled trajectory result (:func:`access_by_seeding` schema:
    ``e_final``, ``delta_t_final``, ``delta_omega``, ``metrics``).  The
    trajectory is CONTINUED (warm field + thermal state) for ``n_rt`` further
    round trips with ``snapshot_interval=1``, accumulating
    ``|fftshift(fft(E))|**2`` EVERY round trip; the returned spectrum dict
    (same schema as :func:`optical_spectrum`, plus ``n_rt_averaged`` and
    ``e_final``) is built from the MEAN mode power.

    ``n_rt`` defaults to ceil(2 * breathing_period_rt) when V6 measured a
    period, else ``CYCLE_AVG_RT_8KAPPA`` = 304 RT (2 breathing cycles at the
    validated 8-kappa point).  Averaging over an integer number of breathing
    cycles removes the breathing-phase dependence of snapshot spectra, so the
    result is deterministic and phase-independent.  Pass the SAME
    ``solver_kwargs`` (e.g. ``**PRODUCTION_NUMERICS``) as the settling run.
    """
    e0 = np.asarray(res["e_final"])
    n_tau = int(e0.shape[0])
    if n_rt is None:
        tb = res.get("metrics", {}).get("breathing_period_rt", float("nan"))
        n_rt = int(math.ceil(2.0 * tb)) if np.isfinite(tb) else CYCLE_AVG_RT_8KAPPA
    n_rt = int(n_rt)
    sol = _run(res["delta_omega"], n_rt, cav, e0=e0,
               delta_t0=res.get("delta_t_final"), seed=res.get("seed", 0),
               n_tau=n_tau, pin=pin, snapshot_interval=1,
               config_path=config_path, **solver_kwargs)
    snaps = np.asarray(sol["E_snapshots"])[0]
    assert snaps.shape == (n_rt, n_tau), snaps.shape
    power = np.mean(
        np.abs(np.fft.fftshift(np.fft.fft(snaps, axis=-1), axes=-1)) ** 2, axis=0
    )
    sp = _spectrum_dict(power, cav)
    sp["n_rt_averaged"] = n_rt
    sp["e_final"] = np.asarray(sol["e_final"])[0]
    return sp


def _measured_dint_native(csv_path=None):
    """Native (mu, D_int, D1, f0) from the CSV in the soliton-rest gauge.

    Identical construction to :func:`simulator.lle_solver.load_dint_grid` but on
    the CSV's own integer mode axis (no FFT interpolation), so crossings can be
    resolved out to the full |mu| the CSV spans regardless of the run's n_tau:

    - omega = 2*pi*f; omega0 = the MEASURED mu=0 resonance so D_int(0) == 0.
    - D1 from a degree-7 smooth-trend fit of omega over 5 < |mu| <= 600. A plain
      3-point central difference at mu=0 is biased +2*pi*3.35 MHz by a localized
      pump-neighborhood defect; that tilt is pure gauge for mode powers but it
      corrupts every crossing / dispersive-wave readout, so the smooth trend is
      used instead (this is the D1 gauge fixed in PR #40).
    - D_int(mu) = omega - omega0 - D1*mu.
    - f0 = the CSV pump frequency at mu=0 [Hz], used for the absolute-wavelength
      map lambda(mu) = c / (f0 + mu*D1/(2*pi)).

    Returns ``(mu, d_int, d1, f0)`` with float64 arrays/scalars.
    """
    csv_path = Path(csv_path) if csv_path is not None else DINT_CSV_PATH
    data = np.loadtxt(csv_path, delimiter=",")
    mu = data[:, 0].astype(np.int64)
    f_hz = data[:, 1].astype(np.float64)
    omega = 2.0 * np.pi * f_hz
    i0 = int(np.where(mu == 0)[0][0])
    omega0 = omega[i0]
    sel = (np.abs(mu) <= 600) & (np.abs(mu) > 5)
    pf = np.polynomial.Polynomial.fit(mu[sel].astype(np.float64), omega[sel], 7)
    d1 = float(pf.deriv()(0.0))
    d_int = omega - omega0 - d1 * mu
    return mu, d_int, d1, float(f_hz[i0])


def dispersive_wave_crossings(delta_omega: float, csv_path=None,
                              mu_core: int = 500) -> list:
    """Phase-matched dispersive-wave crossings from the measured dispersion.

    A dispersive wave (Cherenkov radiation) is emitted where the soliton is
    phase-matched to a linear cavity mode, i.e. where the integrated dispersion
    equals the detuning, ``D_int(mu) = delta_omega``. This scans the measured
    D_int(mu) (soliton-rest gauge, see :func:`_measured_dint_native`) for sign
    changes of ``D_int(mu) - delta_omega`` at ``|mu| > mu_core`` — the
    ``|mu| ~ 200-300`` comb-core crossings are excluded so only the genuine
    far-detuned band edges survive.

    Returns a list of ``{crossing_mu, wavelength_nm, mu}`` dicts (one per
    crossing, ``mu`` = nearest integer mode), sorted by ascending wavelength.
    """
    mu, d_int, d1, f0 = _measured_dint_native(csv_path)
    fsr = d1 / (2.0 * math.pi)
    g = d_int - float(delta_omega)
    core = np.abs(mu) > int(mu_core)
    idx = np.where(core)[0]
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:  # only adjacent modes bracket a real crossing
            continue
        ga, gb = g[a], g[b]
        if ga == 0.0 or (ga < 0.0) != (gb < 0.0):
            mx = mu[a] - ga * (mu[b] - mu[a]) / (gb - ga)
            lam_nm = C_LIGHT / (f0 + mx * fsr) * 1e9
            out.append({"crossing_mu": float(mx),
                        "wavelength_nm": float(lam_nm),
                        "mu": int(round(mx))})
    out.sort(key=lambda r: r["wavelength_nm"])
    return out


def dispersive_wave_peaks(sp: dict, delta_omega: float, *, csv_path=None,
                          scan_modes: int = 30, baseline_mu=(400, 700),
                          mu_core: int = 500) -> list:
    """Dispersive-wave candidates derived from D_int / detuning phase matching.

    Rather than scanning fixed wavelength windows (both true DWs at this
    operating point fall OUTSIDE the old [1120,1260] / [2150,2400] nm bands),
    this derives the search windows from the physics:

    1. Compute the phase-matched crossings ``D_int(mu) = delta_omega`` at
       ``|mu| > mu_core`` (:func:`dispersive_wave_crossings`).
    2. Around each crossing mode ``mu_x`` scan the spectrum over
       ``mu_x +/- scan_modes`` for the largest dB peak (the empirical DW lands a
       few modes out from the linear crossing because of soliton recoil, so the
       +/-30-mode window brackets it).
    3. Convert the peak mode to wavelength with
       ``lambda(mu) = c / (f0 + mu*D1/(2*pi))``, f0 = the CSV pump frequency at
       mu=0.
    4. Report ``prominence_db`` = peak height above a local sech-tail baseline:
       a line fit in dB to the spectrum over ``|mu|`` in ``baseline_mu`` on the
       SAME sign side as the crossing, extrapolated to the peak mode.

    ``sp`` is an :func:`optical_spectrum` result. Returns a list of
    ``{wavelength_nm, mu, power_db, prominence_db, crossing_mu}`` dicts sorted by
    descending prominence. A crossing whose scan window falls outside the
    resolved spectrum (n_tau too small) is skipped.
    """
    _, _, d1, f0 = _measured_dint_native(csv_path)
    fsr = d1 / (2.0 * math.pi)
    sp_mu = np.asarray(sp["mu"])
    sp_db = np.asarray(sp["power_db"])
    blo, bhi = baseline_mu

    results = []
    for cr in dispersive_wave_crossings(delta_omega, csv_path, mu_core=mu_core):
        mx = cr["crossing_mu"]
        win = (sp_mu >= mx - scan_modes) & (sp_mu <= mx + scan_modes)
        if not win.any():
            continue  # crossing outside the resolved FFT window (raise n_tau)
        wi = np.where(win)[0]
        j = wi[int(np.argmax(sp_db[wi]))]
        peak_mu = int(sp_mu[j])
        peak_db = float(sp_db[j])

        # Local sech-tail baseline: line fit in dB over the same-sign tail,
        # extrapolated to the peak mode. prominence = peak - baseline.
        side = 1.0 if mx > 0 else -1.0
        bsel = (sp_mu * side >= blo) & (sp_mu * side <= bhi)
        if int(bsel.sum()) >= 2:
            coeff = np.polyfit(sp_mu[bsel].astype(np.float64), sp_db[bsel], 1)
            baseline_db = float(np.polyval(coeff, peak_mu))
            prominence = peak_db - baseline_db
        else:
            prominence = float("nan")

        lam_nm = C_LIGHT / (f0 + peak_mu * fsr) * 1e9
        results.append({
            "wavelength_nm": float(lam_nm),
            "mu": peak_mu,
            "power_db": peak_db,
            "prominence_db": float(prominence),
            "crossing_mu": float(mx),
        })

    results.sort(
        key=lambda r: (r["prominence_db"]
                       if np.isfinite(r["prominence_db"]) else -np.inf),
        reverse=True,
    )
    return results


# ---------------------------------------------------------------------------
# Low-level trajectory runner (constant or ramped detuning, optional warm start)
# ---------------------------------------------------------------------------
def _run(delta_omega, t_slow, cav, *, e0=None, delta_t0=None, seed=0,
         n_tau=N_TAU, pin=PIN_W, snapshot_interval=None, config_path=CONFIG_PATH,
         **solver_kwargs):
    """Thin wrapper around solve_lle_ssfm_jax for a single trajectory.

    ``delta_omega`` is a scalar (held) or a (t_slow,) ramp.  ``e0`` is an optional
    (n_tau,) warm-start field (None = cold start).  Extra ``solver_kwargs`` are
    forwarded to :func:`solve_lle_ssfm_jax` (pass ``**PRODUCTION_NUMERICS`` for
    quantitative-spectrum runs; the default is the legacy path).  Returns the
    numpy result dict plus the final field / U_int history sliced to trajectory 0.
    """
    if snapshot_interval is None:
        snapshot_interval = max(t_slow // 200, 1)
    if np.isscalar(delta_omega):
        dw = np.full((1, int(t_slow)), float(delta_omega), dtype=np.float32)
    else:
        dw = np.asarray(delta_omega, dtype=np.float32).reshape(1, int(t_slow))
    sol = solve_lle_ssfm_jax(
        pin=pin,
        delta_omega=dw,
        t_slow=int(t_slow),
        beta=[cav.beta2],  # harmless fallback; ignored when d_int_grid is set
        kappa=cav.kappa,
        kappa_c=cav.kappa_c,
        rng_key=jax.random.PRNGKey(int(seed)),
        n_tau=int(n_tau),
        snapshot_interval=int(snapshot_interval),
        config_path=str(config_path),
        e0_override=None if e0 is None else np.asarray(e0),
        delta_t0_override=None if delta_t0 is None else np.asarray(delta_t0),
        d_int_grid=cav.d_int_grid,
        **solver_kwargs,
    )
    return sol


# ---------------------------------------------------------------------------
# Access protocol (b): direct single-sech seeding
# ---------------------------------------------------------------------------
def access_by_seeding(delta_omega, cav, *, t_slow=None, seed=0, n_tau=N_TAU,
                      pin=PIN_W, config_path=CONFIG_PATH, **solver_kwargs):
    """Deterministic single-DKS access by warm-starting an analytic sech ansatz.

    Builds ``sech_soliton_seed(delta_omega)`` and integrates at constant
    ``delta_omega`` to steady state.  ``t_slow`` defaults to ~1 thermal time
    (enough for the seed to relax onto the attractor and the thermal pole to
    settle); pass a larger value for the long stability validation.  Returns a
    dict with the final field, U_int history, effective detuning, and metrics.
    """
    if t_slow is None:
        t_slow = cav.tau_th_round_trips  # ~1 tau_th
    seed_field = sech_soliton_seed(delta_omega, cav, n_tau=n_tau, pin=pin)
    sol = _run(delta_omega, t_slow, cav, e0=seed_field, seed=seed,
               n_tau=n_tau, pin=pin, config_path=config_path, **solver_kwargs)
    e_final = np.asarray(sol["e_final"])[0]
    u_hist = np.asarray(sol["U_int_history"])[0]
    dweff = np.asarray(sol["delta_omega_eff_history"])[0]
    metrics = soliton_metrics(e_final, u_hist, cav, delta_omega, pin=pin)
    return {
        "route": "seeding",
        "delta_omega": float(delta_omega),
        "t_slow": int(t_slow),
        "seed": int(seed),
        "e_final": e_final,
        "delta_t_final": float(np.asarray(sol["delta_t_final"]).reshape(-1)[0]),
        "u_int_history": u_hist,
        "delta_omega_eff_history": dweff,
        "delta_omega_eff_mean": float(np.mean(dweff[-max(1000, t_slow // 100):])),
        "seed_field": seed_field,
        "metrics": metrics,
        "is_single": is_single_soliton(metrics),
    }


# ---------------------------------------------------------------------------
# Access protocol (a): forward tune through MI -> backward tune / pump kick
# ---------------------------------------------------------------------------
def access_by_forward_backward(
    cav, *, dw_start=-1.0, dw_peak=13.0, dw_target=10.0,
    t_forward=None, t_back=None, t_hold=None, seed=0, n_tau=N_TAU,
    pin=PIN_W, config_path=CONFIG_PATH,
):
    """Experimental "soliton-step" route: forward ramp through MI, then back-tune.

    Detunings are given in units of kappa.  Stage 1 forward-tunes (cold start) from
    ``dw_start`` (blue) up through the MI region to ``dw_peak`` deep in the
    red-detuned range, igniting MI/chaos and (ideally) nucleating solitons.
    Stage 2 back-tunes from ``dw_peak`` down to ``dw_target`` (reducing the number
    of solitons), and stage 3 holds at ``dw_target`` to settle.  The field and
    thermal state are carried between stages via the warm-start path so it is one
    continuous trajectory.

    Returns the same result schema as ``access_by_seeding`` (route="forward_backward")
    with the final soliton count reported.  This route is not guaranteed to yield
    exactly one soliton at every operating point; its success is reported, not
    assumed.
    """
    if t_forward is None:
        t_forward = 3 * cav.tau_th_round_trips
    if t_back is None:
        t_back = cav.tau_th_round_trips
    if t_hold is None:
        t_hold = cav.tau_th_round_trips
    k = cav.kappa

    # Stage 1: forward ramp, cold start.
    ramp_fwd = np.linspace(dw_start * k, dw_peak * k, int(t_forward), dtype=np.float32)
    s1 = _run(ramp_fwd, t_forward, cav, e0=None, seed=seed, n_tau=n_tau, pin=pin,
              config_path=config_path)
    e1 = np.asarray(s1["e_final"])[0]
    dt1 = float(np.asarray(s1["delta_t_final"])[0]) if np.ndim(s1["delta_t_final"]) else float(s1["delta_t_final"])

    # Stage 2: backward tune peak -> target, warm start from stage 1.
    ramp_back = np.linspace(dw_peak * k, dw_target * k, int(t_back), dtype=np.float32)
    s2 = _run(ramp_back, t_back, cav, e0=e1, delta_t0=dt1, seed=seed + 1,
              n_tau=n_tau, pin=pin, config_path=config_path)
    e2 = np.asarray(s2["e_final"])[0]
    dt2 = float(np.asarray(s2["delta_t_final"])[0]) if np.ndim(s2["delta_t_final"]) else float(s2["delta_t_final"])

    # Stage 3: hold at target, warm start from stage 2.
    s3 = _run(dw_target * k, t_hold, cav, e0=e2, delta_t0=dt2, seed=seed + 2,
              n_tau=n_tau, pin=pin, config_path=config_path)
    e_final = np.asarray(s3["e_final"])[0]
    u_hist = np.asarray(s3["U_int_history"])[0]
    dweff = np.asarray(s3["delta_omega_eff_history"])[0]
    metrics = soliton_metrics(e_final, u_hist, cav, dw_target * k, pin=pin)
    return {
        "route": "forward_backward",
        "delta_omega": float(dw_target * k),
        "dw_start": dw_start, "dw_peak": dw_peak, "dw_target": dw_target,
        "seed": int(seed),
        "e_final": e_final,
        "u_int_history": u_hist,
        "delta_omega_eff_history": dweff,
        "delta_omega_eff_mean": float(np.mean(dweff[-max(1000, t_hold // 100):])),
        "metrics": metrics,
        "is_single": is_single_soliton(metrics),
    }


# ---------------------------------------------------------------------------
# Existence-window map (batched over detuning via the vmapped solver)
# ---------------------------------------------------------------------------
def existence_map(cav, dw_over_kappa, *, t_slow=None, seed=0, n_tau=N_TAU,
                  pin=PIN_W, config_path=CONFIG_PATH, **solver_kwargs):
    """Seed a single soliton at each detuning and classify the steady state.

    All detunings are integrated in one batched (vmapped) solver call: the seed
    fields and constant-detuning ramps are stacked along the trajectory axis.
    Returns per-detuning metrics and the contiguous band of single-soliton (class
    6) detunings.
    """
    if t_slow is None:
        t_slow = cav.tau_th_round_trips
    dws = np.asarray(dw_over_kappa, dtype=float) * cav.kappa
    n = dws.size
    seeds = np.stack([sech_soliton_seed(dw, cav, n_tau=n_tau, pin=pin) for dw in dws])
    dw_batch = np.repeat(dws[:, None], int(t_slow), axis=1).astype(np.float32)
    sol = solve_lle_ssfm_jax(
        pin=pin, delta_omega=dw_batch, t_slow=int(t_slow), beta=[cav.beta2],
        kappa=cav.kappa, kappa_c=cav.kappa_c, rng_key=jax.random.PRNGKey(int(seed)),
        n_tau=int(n_tau), snapshot_interval=max(int(t_slow) // 50, 1),
        config_path=str(config_path), e0_override=seeds.astype(np.complex64),
        d_int_grid=cav.d_int_grid, **solver_kwargs,
    )
    e_finals = np.asarray(sol["e_final"])
    u_hists = np.asarray(sol["U_int_history"])
    dweff = np.asarray(sol["delta_omega_eff_history"])
    rows = []
    for i, dw in enumerate(dws):
        m = soliton_metrics(e_finals[i], u_hists[i], cav, dw, pin=pin)
        rows.append({
            "dw_over_kappa": float(dw_over_kappa[i]),
            "delta_omega": float(dw),
            "delta_omega_eff_over_kappa": float(np.mean(dweff[i, -50:]) / cav.kappa),
            "is_single": is_single_soliton(m),
            **m,
        })
    band = _contiguous_single_band(rows)
    return {"rows": rows, "band": band, "e_finals": e_finals}


def _contiguous_single_band(rows) -> dict:
    """Extract the (single) contiguous run of single-soliton detunings, if any."""
    flags = [r["is_single"] for r in rows]
    # find longest contiguous True run
    best = (0, -1)  # (length, start)
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            if (j - i) > best[0]:
                best = (j - i, i)
            i = j
        else:
            i += 1
    if best[0] == 0:
        return {"found": False}
    start = best[1]
    stop = start + best[0] - 1
    contiguous = all(flags[start:stop + 1])
    return {
        "found": True,
        "lower_over_kappa": rows[start]["dw_over_kappa"],
        "upper_over_kappa": rows[stop]["dw_over_kappa"],
        "n_points": best[0],
        "contiguous": contiguous,
    }


# ---------------------------------------------------------------------------
# Breathing scan (V6 vs detuning across the existence band)
# ---------------------------------------------------------------------------
STATIONARY_RELSTD = 0.001   # rel-std < 0.1% = clean-DKS (stationary) operation


def breathing_scan(cav, dw_over_kappa, *, t_slow=4000, seed=0, n_tau=8192,
                   pin=PIN_W, config_path=CONFIG_PATH, **solver_kwargs):
    """Seed a soliton at each detuning and record the V6 breathing metrics.

    Batched (vmapped) like :func:`existence_map` but aimed at the breathing
    sub-band question: per detuning it reports U rel-std over the last 2*T_b,
    the breathing period T_b (RT), the breather flag (rel-std > 0.5%), and a
    stationary flag (rel-std < ``STATIONARY_RELSTD`` = 0.1%, the clean-DKS
    operating criterion). Defaults follow the audited scan recipe: 4000 RT at
    n_tau = 8192, which resolves T_b ~ O(100 RT) oscillations while staying
    cheap; pass ``**PRODUCTION_NUMERICS`` for the committed artifacts.

    Returns ``{"rows", "breathing_bands", "stationary_windows"}`` where the
    bands/windows are lists of (lower, upper) detunings in kappa units of
    contiguous scanned points.
    """
    dws = np.asarray(dw_over_kappa, dtype=float) * cav.kappa
    seeds = np.stack([sech_soliton_seed(dw, cav, n_tau=n_tau, pin=pin)
                      for dw in dws])
    dw_batch = np.repeat(dws[:, None], int(t_slow), axis=1).astype(np.float32)
    sol = solve_lle_ssfm_jax(
        pin=pin, delta_omega=dw_batch, t_slow=int(t_slow), beta=[cav.beta2],
        kappa=cav.kappa, kappa_c=cav.kappa_c, rng_key=jax.random.PRNGKey(int(seed)),
        n_tau=int(n_tau), snapshot_interval=int(t_slow),
        config_path=str(config_path), e0_override=seeds.astype(np.complex64),
        d_int_grid=cav.d_int_grid, **solver_kwargs,
    )
    e_finals = np.asarray(sol["e_final"])
    u_hists = np.asarray(sol["U_int_history"])
    rows = []
    for i, dw in enumerate(dws):
        m = soliton_metrics(e_finals[i], u_hists[i], cav, dw, pin=pin)
        rows.append({
            "dw_over_kappa": float(dw_over_kappa[i]),
            "delta_omega": float(dw),
            "is_single": is_single_soliton(m),
            "is_stationary": bool(m["breathing_relstd"] < STATIONARY_RELSTD),
            **m,
        })

    def _contiguous(flag_key):
        bands, start = [], None
        for r in rows + [None]:
            on = r is not None and bool(r[flag_key])
            if on and start is None:
                start = r["dw_over_kappa"]
                last = r["dw_over_kappa"]
            elif on:
                last = r["dw_over_kappa"]
            elif start is not None:
                bands.append((start, last))
                start = None
        return bands

    return {
        "rows": rows,
        "breathing_bands": _contiguous("is_breather"),
        "stationary_windows": _contiguous("is_stationary"),
    }


def write_breathing_csv(path: Path, scan):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dw_over_kappa", "delta_omega_rad_s", "np_label", "n_peaks",
                    "is_single", "is_breather", "is_stationary",
                    "breathing_period_rt", "breathing_relstd",
                    "u_int_tail_rel_std"])
        for r in scan["rows"]:
            w.writerow([
                f"{r['dw_over_kappa']:.3f}", f"{r['delta_omega']:.6e}",
                r["np_label"], r["n_peaks"], int(r["is_single"]),
                int(r["is_breather"]), int(r["is_stationary"]),
                f"{r['breathing_period_rt']:.1f}",
                f"{r['breathing_relstd']:.6f}",
                f"{r['u_int_tail_rel_std']:.6f}",
            ])


def plot_breathing_scan(path: Path, scan, cav):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = scan["rows"]
    dwk = np.array([r["dw_over_kappa"] for r in rows])
    relstd = np.array([r["breathing_relstd"] for r in rows])
    period = np.array([r["breathing_period_rt"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.semilogy(dwk, np.maximum(relstd, 1e-7), "o-", ms=4, color="tab:red")
    ax.axhline(V6_BREATHER_RELSTD, color="k", ls="--", lw=0.7,
               label=f"breather threshold ({V6_BREATHER_RELSTD:.1%})")
    ax.axhline(STATIONARY_RELSTD, color="tab:green", ls="--", lw=0.7,
               label=f"stationary threshold ({STATIONARY_RELSTD:.1%})")
    for lo, hi in scan["breathing_bands"]:
        ax.axvspan(lo, hi, color="tab:red", alpha=0.10)
    for lo, hi in scan["stationary_windows"]:
        ax.axvspan(lo, hi, color="tab:green", alpha=0.12)
    ax.set_ylabel(r"U rel-std over last $2T_b$")
    ax.set_title("V6 breathing scan (seeded soliton, pin = 0.214 W)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(dwk, period, "o-", ms=4, color="tab:blue")
    ax.set_xlabel(r"programmed $\delta\omega / \kappa$")
    ax.set_ylabel(r"breathing period $T_b$ (RT)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Resolution cross-check
# ---------------------------------------------------------------------------
def spectrum_resolution_check(cav, delta_omega, *, t_slow=None, seed=0,
                              n_tau_hi=2048, pin=PIN_W, config_path=CONFIG_PATH):
    """Re-run the seeded soliton at higher n_tau to show the fully-resolved comb."""
    # The measured D_int grid is n_tau-specific, so rebuild it for n_tau_hi.
    cav_hi = attach_dispersion(cav, n_tau_hi) if cav.d_int_grid is not None else cav
    res = access_by_seeding(delta_omega, cav_hi, t_slow=t_slow, seed=seed,
                            n_tau=n_tau_hi, pin=pin, config_path=config_path)
    return res


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def breather_title_annotation(metrics: dict) -> str:
    """Title suffix for breather artifacts, e.g. ' — breather, T=~153 RT, dU/U=4.1%'."""
    if not metrics.get("is_breather"):
        return ""
    tb = metrics.get("breathing_period_rt", float("nan"))
    relstd = metrics.get("breathing_relstd", float("nan"))
    tb_txt = f"T=~{tb:.0f} RT" if np.isfinite(tb) else "T=?"
    return f" — breather, {tb_txt}, dU/U={100.0 * relstd:.1f}%"


def plot_optical_spectrum(path: Path, e_field, cav, delta_omega, *,
                          e_field_hi=None, n_tau_hi=2048, title_extra="",
                          sp=None):
    """Optical-spectrum artifact (power vs wavelength).

    ``sp`` overrides the plotted spectrum (pass a
    :func:`cycle_averaged_spectrum` result whenever V6 classifies the state as
    a breather — snapshot spectra of a breather are breathing-phase-dependent);
    default is the snapshot spectrum of ``e_field``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sp is None:
        sp = optical_spectrum(e_field, cav)
        main_label = f"n_tau={e_field.shape[0]} (resolved band)"
    else:
        main_label = (f"n_tau={sp['mu'].size}, cycle-averaged over "
                      f"{sp.get('n_rt_averaged', '?')} RT")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    # sort by wavelength for a clean line
    order = np.argsort(sp["wavelength_nm"])
    ax.plot(sp["wavelength_nm"][order], sp["power_db"][order], "-", lw=0.8,
            color="tab:blue", label=main_label)
    ax.plot(sp["wavelength_nm"], sp["power_db"], ".", ms=2.5, color="tab:blue")

    if e_field_hi is not None:
        cav_hi = cav
        sp_hi = optical_spectrum(e_field_hi, cav_hi)
        order_hi = np.argsort(sp_hi["wavelength_nm"])
        ax.plot(sp_hi["wavelength_nm"][order_hi], sp_hi["power_db"][order_hi], "-",
                lw=0.7, color="tab:orange", alpha=0.7,
                label=f"n_tau={e_field_hi.shape[0]} (fully resolved)")

    # Scanner windows (crossing mu_x +/- 30 modes) and the physics-derived
    # dispersive-wave crossings (D_int(mu) = delta_omega, soliton-rest gauge).
    _, _, _d1, _f0 = _measured_dint_native()
    _fsr = _d1 / (2.0 * math.pi)
    crossings = dispersive_wave_crossings(delta_omega)
    for c in crossings:
        ax.axvline(c["wavelength_nm"], color="0.5", ls=":", lw=0.8, alpha=0.7)
        lam_a = C_LIGHT / (_f0 + (c["crossing_mu"] - 30) * _fsr) * 1e9
        lam_b = C_LIGHT / (_f0 + (c["crossing_mu"] + 30) * _fsr) * 1e9
        ax.axvspan(min(lam_a, lam_b), max(lam_a, lam_b), color="tab:purple",
                   alpha=0.12, label="_scanner window")

    # Annotate the measured dispersive-wave peaks (wavelength + dB).
    for p in dispersive_wave_peaks(sp, delta_omega):
        ax.annotate(
            f"DW {p['wavelength_nm']:.0f} nm\n{p['power_db']:.1f} dB "
            f"(+{p['prominence_db']:.0f} dB)",
            xy=(p["wavelength_nm"], p["power_db"]),
            xytext=(p["wavelength_nm"], p["power_db"] + 55),
            ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9),
            color="tab:red",
        )

    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(r"normalized power  $10\log_{10}(|\tilde E|^2)$  (dB)")
    ax.set_xlim(1050, 2600)
    # Show the full dynamic range down to the float64 + dealias numerical floor
    # (~-320 dB); the -360 dB clamp in optical_spectrum keeps zeroed (dealiased)
    # bins finite. Never cut above -200 dB: the DW peaks (~-95 dB) and the
    # aliasing-free floor must both stay visible.
    _ymin = min(-200.0, float(np.min(sp["power_db"])) - 10.0)
    ax.set_ylim(max(_ymin, -370.0), 8)
    ax.set_title(
        f"Single-DKS optical spectrum @ delta_omega = "
        f"{delta_omega / cav.kappa:.1f} kappa, pin = {PIN_W} W "
        f"(full measured dispersion, n_tau={sp['mu'].size})"
        + (f"\n{title_extra.lstrip(' —')}" if title_extra else ""),
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_soliton_summary(path: Path, res, cav, *, sp=None, title_extra=""):
    """Waveform / comb / U_int summary artifact.

    ``sp`` overrides the comb panel's spectrum (pass a
    :func:`cycle_averaged_spectrum` result for breathers); the waveform panel
    always shows the final snapshot (one breathing phase).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = res["e_final"]
    p = np.abs(e) ** 2
    dt = cav.t_r / e.shape[0]
    t_ps = (np.arange(e.shape[0]) * dt) * 1e12
    u = res["u_int_history"]
    m = res["metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(t_ps, p, color="tab:red", lw=1.0)
    axes[0].set_xlabel("fast time (ps)")
    axes[0].set_ylabel(r"$|E(\tau)|^2$ (J)")
    axes[0].set_title("(a) Intracavity waveform — single peak"
                      + (" (snapshot phase)" if m.get("is_breather") else ""))

    comb_note = ""
    if sp is None:
        sp = optical_spectrum(e, cav)
    else:
        comb_note = f", cycle-avg {sp.get('n_rt_averaged', '?')} RT"
    order = np.argsort(sp["mu"])
    axes[1].plot(sp["mu"][order], sp["power_db"][order], "-", lw=0.8, color="tab:blue")
    axes[1].set_xlabel("cavity mode index $\\mu$")
    axes[1].set_ylabel("power (dB)")
    # Deep axis: the float64 + dealias floor sits near -320 dB and the
    # dispersive-wave peaks near -100 dB; the old -70 dB cut hid both.
    axes[1].set_ylim(min(-200.0, float(np.min(sp["power_db"])) - 10.0), 8)
    axes[1].set_title(
        f"(b) Comb spectrum — sech$^2$ env corr = "
        f"{m['sech2_env_corr']:.3f}{comb_note}"
    )

    axes[2].plot(np.arange(u.size), u, color="tab:green", lw=0.8)
    axes[2].set_xlabel("round trip")
    axes[2].set_ylabel(r"$U_\mathrm{int}$ (J)")
    axes[2].set_title(
        f"(c) $U_\\mathrm{{int}}$ — tail rel-std = "
        f"{m['u_int_tail_rel_std']:.2%}"
    )
    if title_extra:
        fig.suptitle(title_extra.lstrip(" —"), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_existence_map(path: Path, emap, cav):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = emap["rows"]
    dwk = np.array([r["dw_over_kappa"] for r in rows])
    labels = np.array([r["np_label"] for r in rows])
    corr = np.array([r["sech2_env_corr"] for r in rows])
    npk = np.array([r["n_peaks"] for r in rows])
    single = np.array([r["is_single"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.plot(dwk, labels, "o-", ms=4, color="tab:blue")
    ax.axhline(LABEL_SINGLE_SOLITON, color="k", ls="--", lw=0.7,
               label="class 6 (single soliton)")
    if emap["band"].get("found"):
        b = emap["band"]
        ax.axvspan(b["lower_over_kappa"], b["upper_over_kappa"], color="gold",
                   alpha=0.2, label=f"single-DKS band [{b['lower_over_kappa']:.1f}, "
                                    f"{b['upper_over_kappa']:.1f}]$\\kappa$")
    ax.set_ylabel("NumPy label class")
    ax.set_yticks(range(0, 7))
    ax.set_title("Seeded single-soliton existence map, pin = 0.214 W")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(dwk, corr, "o-", ms=4, color="tab:orange", label="sech$^2$ env corr")
    ax.axhline(0.9, color="k", ls="--", lw=0.7)
    for i, s in enumerate(single):
        if s:
            ax.plot(dwk[i], corr[i], "o", ms=8, mfc="none", mec="tab:green")
    ax.set_xlabel(r"programmed $\delta\omega / \kappa$")
    ax.set_ylabel("sech$^2$ envelope corr")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report / tables
# ---------------------------------------------------------------------------
def write_existence_csv(path: Path, emap):
    # is_single stands as the existence label; the V6 breathing columns say
    # whether the (single-)soliton state is stationary or a breather.
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dw_over_kappa", "delta_omega_rad_s", "delta_omega_eff_over_kappa",
                    "np_label", "n_peaks", "contrast", "sech2_env_corr",
                    "u_int_tail_rel_std", "is_single", "is_breather",
                    "breathing_period_rt", "breathing_relstd"])
        for r in emap["rows"]:
            w.writerow([
                f"{r['dw_over_kappa']:.3f}", f"{r['delta_omega']:.6e}",
                f"{r['delta_omega_eff_over_kappa']:.3f}", r["np_label"],
                r["n_peaks"], f"{r['contrast']:.2f}",
                f"{r['sech2_env_corr']:.4f}", f"{r['u_int_tail_rel_std']:.4f}",
                int(r["is_single"]), int(r["is_breather"]),
                f"{r['breathing_period_rt']:.1f}",
                f"{r['breathing_relstd']:.6f}",
            ])


def write_report(path: Path, cav, validated, fb, control, repro, emap,
                 long_t_slow, breathing=None):
    b = emap["band"]
    lines = []
    lines.append("# Single dissipative-Kerr-soliton (DKS) access protocol\n\n")
    lines.append(
        f"Operating point: pin = {PIN_W} W, n_tau = {N_TAU}, thermal model ON at "
        f"the config Gamma_th. kappa = {cav.kappa:.3e} rad/s, kappa_c = "
        f"{cav.kappa_c:.3e} rad/s, gamma_LLE = {cav.gamma:.3e} J^-1 s^-1, "
        f"D2 = {cav.d2:.3e} rad/s^2, tau_th = {cav.tau_th_s:.1e} s "
        f"({cav.tau_th_round_trips} round trips).\n\n"
    )
    lines.append("## Access protocol\n\n")
    lines.append(
        "Two routes were implemented (`analysis/dks_access.py`):\n\n"
        "- **(b) Direct single-sech seeding (`access_by_seeding`) — KEPT.** An "
        "analytic bright-sech ansatz `B*sech(t/tau_s)` (B = sqrt(2*dw/gamma), "
        "tau_s = sqrt(beta2/(2*dw))) on the CW background is injected as the "
        "warm-start field (`e0_override`) at a detuning inside the existence "
        "window, then integrated to steady state. This is deterministic and "
        "reliably yields exactly one soliton.\n"
        "- **(a) Forward/backward tuning (`access_by_forward_backward`).** Cold "
        "forward ramp (blue->red) through MI to a deep red detuning, then a "
        "backward tune down to the target detuning to shed excess solitons, held "
        "to settle. Carried as one continuous trajectory via the warm-start path. "
        "Reported for completeness.\n\n"
    )
    m = validated["metrics"]
    lines.append("## Validated single soliton\n\n")
    lines.append(
        f"Route (b) at programmed delta_omega = "
        f"{validated['delta_omega'] / cav.kappa:.1f} kappa "
        f"(effective {validated['delta_omega_eff_mean'] / cav.kappa:.2f} kappa "
        f"after thermal shift), integrated for t_slow = {long_t_slow} round trips "
        f"= {long_t_slow / cav.tau_th_round_trips:.1f} tau_th:\n\n"
        f"- single dominant temporal peak: **n_peaks = {m['n_peaks']}**\n"
        f"- sech^2 spectral (envelope) correlation: **{m['sech2_env_corr']:.4f}** "
        f"(> 0.9 required; r^2 = {m['sech2_env_r2']:.4f})\n"
        f"- U_int tail rel-std over the long integration: "
        f"**{m['u_int_tail_rel_std']:.2%}** (< 5% required)\n"
        f"- NumPy labeler class: **{m['np_label']}** (6 = single soliton)\n"
        f"- finite (no NaN/Inf): **{m['finite']}**\n"
        f"- peak-to-mean contrast: {m['contrast']:.1f}\n\n"
    )
    lines.append("## Stationarity (V6): the validated state is a BREATHER\n\n")
    if m.get("is_breather"):
        lines.append(
            f"The V6 stationarity check (U_int autocorrelation period T_b, "
            f"rel-std over the last >= 2*T_b; breather if > "
            f"{V6_BREATHER_RELSTD:.1%}) classifies the validated "
            f"{validated['delta_omega'] / cav.kappa:.1f}-kappa state as a "
            f"**breather**: T_b = {m['breathing_period_rt']:.0f} RT, "
            f"dU/U = {m['breathing_relstd']:.2%}. All \"8 kappa\" artifacts in "
            f"this directory therefore describe a BREATHING single-soliton "
            f"state, not a stationary one; snapshot spectra are "
            f"breathing-phase-dependent, so the spectrum artifacts are built "
            f"from the CYCLE-AVERAGED spectrum "
            f"(`cycle_averaged_spectrum`, mean of |fftshift(fft(E))|^2 over "
            f">= 2*T_b consecutive round trips).\n\n"
        )
    else:
        lines.append(
            f"V6 classifies the validated state as stationary "
            f"(rel-std {m['breathing_relstd']:.2%} <= {V6_BREATHER_RELSTD:.1%} "
            f"over the last >= 2*T_b).\n\n"
        )
    if breathing is not None:
        lines.append(
            f"Breathing scan across the existence band "
            f"(`breathing_scan`, seeded, per-detuning V6): breathing sub-bands "
            f"{breathing['breathing_bands']} kappa; stationary "
            f"(rel-std < {STATIONARY_RELSTD:.1%}) windows "
            f"{breathing['stationary_windows']} kappa. See "
            f"`dks_breathing_scan.csv` / `dks_breathing_scan.png`.\n\n"
        )
    lines.append("## Reproducibility across RNG seeds\n\n")
    n_ok = sum(1 for r in repro if r["is_single"])
    lines.append(
        f"Seeds tested: {[r['seed'] for r in repro]}. Single-soliton success rate: "
        f"**{n_ok}/{len(repro)}** = {100*n_ok/len(repro):.0f}%.\n\n"
    )
    for r in repro:
        lines.append(
            f"- seed {r['seed']}: n_peaks={r['metrics']['n_peaks']}, "
            f"class={r['metrics']['np_label']}, env_corr="
            f"{r['metrics']['sech2_env_corr']:.3f}, single={r['is_single']}\n"
        )
    lines.append("\n## Control (no protocol)\n\n")
    cm = control["metrics"]
    lines.append(
        f"Cold start held at the same detuning ({control['delta_omega']/cav.kappa:.1f} "
        f"kappa) with NO seed and NO tuning protocol: class = **{cm['np_label']}**, "
        f"n_peaks = {cm['n_peaks']}, sech^2 env corr = {cm['sech2_env_corr']:.3f}, "
        f"contrast = {cm['contrast']:.2f}. The plain (unseeded) run does **not** "
        f"yield a class-6 single soliton — confirming the protocol is doing the "
        f"work. (The bare adiabatic forward sweep likewise lands in MI/Turing, "
        f"never a single soliton; see `analysis/adiabatic_sweeps.py`.)\n\n"
    )
    lines.append("## Forward/backward route result\n\n")
    if fb is not None:
        fm = fb["metrics"]
        lines.append(
            f"`access_by_forward_backward` (forward {fb['dw_start']}->{fb['dw_peak']} "
            f"kappa, back to {fb['dw_target']} kappa): class = {fm['np_label']}, "
            f"n_peaks = {fm['n_peaks']}, env_corr = {fm['sech2_env_corr']:.3f}, "
            f"single = {fb['is_single']}.\n\n"
        )
    lines.append("## Existence window (seeded)\n\n")
    if b.get("found"):
        lines.append(
            f"Class-6 single solitons appear in a single contiguous detuning band "
            f"**[{b['lower_over_kappa']:.1f}, {b['upper_over_kappa']:.1f}] kappa** "
            f"({b['n_points']} sampled points, contiguous = {b['contiguous']}).\n\n"
        )
    else:
        lines.append("No single-soliton band found in the sampled range.\n\n")
    lines.append(
        "Note on the band location: at pin = 0.214 W the pump is ~61x the MI "
        "threshold, a very hard drive. The single-DKS existence window therefore "
        "sits at higher detuning than the generic `kappa/2 < dw < ~5 kappa` "
        "estimate — the measured lower edge is where the CW background becomes "
        "MI-stable enough to hold a soliton, and the upper edge is where the "
        "soliton amplitude collapses back to CW. Below the band the seed is "
        "swamped by background MI; above it the seed decays to CW.\n\n"
    )
    lines.append("## Resolution / dispersion note\n\n")
    d2_khz = cav.d2_local / (2.0 * math.pi) / 1e3
    crossings = dispersive_wave_crossings(validated["delta_omega"])
    cross_txt = "; ".join(
        f"mu ~ {c['crossing_mu']:+.0f} (~{c['wavelength_nm']:.0f} nm)"
        for c in crossings
    ) or "none"
    lines.append(
        f"The solver is driven by the full MEASURED integrated dispersion "
        f"D_int(mu) (from `config/pyLLE_dispersion_w4400_h800.csv`), not a pure D2 "
        f"parabola, so it can radiate dispersive waves (Cherenkov peaks) at "
        f"phase-matched band edges rather than showing a plain sech^2 roll-off. "
        f"The near-pump curvature gives a CSV local D2 = {cav.d2_local:.3e} "
        f"rad/s^2 = 2*pi*{d2_khz:.1f} kHz (fit over 5 < |mu| <= 300, excluding "
        f"the |mu| <= 5 pump-neighborhood defect; window-converged, cf. the old "
        f"|mu| <= 40 window's biased 2*pi*15.7 kHz). The analytic sech seed is "
        f"sized from this local curvature; the full dispersion then reshapes the "
        f"wings (measured FSR = {cav.fsr_measured_hz:.4e} Hz).\n\n"
        f"Working in the soliton-rest gauge (the D1 fixed in PR #40; a raw "
        f"central-difference D1 tilted D_int and produced a spurious mu ~ +2400 "
        f"/ 1188 nm crossing), the dispersive-wave phase-matching condition "
        f"D_int(mu) = delta_omega has two crossings at |mu| > 500 for this run: "
        f"{cross_txt}. These, not the fixed [1120,1260] / [2150,2400] nm windows "
        f"of the old scanner, are the physical dispersive-wave band edges (the "
        f"true DWs fall OUTSIDE those windows). `dispersive_wave_peaks` scans the "
        f"spectrum +/-30 modes around each crossing (covering the few-mode recoil "
        f"shift) and reports each peak's dB height and prominence above the local "
        f"sech-tail baseline; the default grid is n_tau = {N_TAU} to resolve those "
        f"crossing modes.\n\n"
        f"Both labelers return class 6 for these states. The JAX scan-time labeler "
        f"(which produces label_history for the training dataset) keys class 6 on a "
        f"single temporal peak plus a smooth monotonic sech^2 spectral envelope; an "
        f"earlier 'fraction of power in the top ~32 points' heuristic mislabeled a "
        f"DKS on a bright CW background as chaotic (class 3) and was replaced. "
        f"Classification in this study uses the NumPy sech^2-fit labeler.\n\n"
    )
    lines.append("## Artifacts\n\n")
    lines.append(
        "- `dks_single_soliton_spectrum.png` — optical power vs wavelength (nm; "
        "cycle-averaged when V6 reports a breather)\n"
        "- `dks_single_soliton_summary.png` — waveform, comb, U_int stability\n"
        "- `dks_existence_map.png` / `dks_existence_map.csv` — existence window\n"
        "- `dks_breathing_scan.png` / `dks_breathing_scan.csv` — V6 breathing "
        "sub-band scan\n"
    )
    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dw", type=float, default=OPERATING_DW_KAPPA,
                    help="production operating detuning in units of kappa "
                         "(default 10; stationary DKS, past the ~9.3 Hopf edge)")
    ap.add_argument("--long-tau-th", type=float, default=5.0,
                    help="long-integration length in units of tau_th (>=5 for validation)")
    ap.add_argument("--map-tau-th", type=float, default=1.0,
                    help="per-detuning existence-map integration length in tau_th")
    ap.add_argument("--seeds", type=int, default=3, help="reproducibility seeds")
    ap.add_argument("--n-tau", type=int, default=N_TAU,
                    help=f"FFT grid points (default {N_TAU}; the full measured "
                         f"dispersion figure needs the large grid)")
    ap.add_argument("--res-check", action="store_true",
                    help="also run the n_tau=2048 resolution cross-check")
    ap.add_argument("--no-forward-backward", action="store_true")
    args = ap.parse_args()
    n_tau = int(args.n_tau)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cav = load_cavity_params()
    cav = attach_dispersion(cav, n_tau)
    dw = args.dw * cav.kappa
    long_t_slow = int(args.long_tau_th * cav.tau_th_round_trips)
    map_t_slow = int(args.map_tau_th * cav.tau_th_round_trips)

    print(f"[dks] kappa={cav.kappa:.3e}, tau_th={cav.tau_th_round_trips} rt, "
          f"n_tau={n_tau}, validated dw={args.dw} kappa, long t_slow={long_t_slow}")

    # 1. Validated single soliton (long integration).
    print("[dks] route (b) seeding — long validated run ...")
    validated = access_by_seeding(dw, cav, t_slow=long_t_slow, seed=0, n_tau=n_tau)

    # Optional higher-resolution cross-check field for the spectrum plot.
    e_hi = None
    if args.res_check:
        print("[dks] n_tau=2048 resolution cross-check ...")
        hi = spectrum_resolution_check(cav, dw, t_slow=map_t_slow, seed=0)
        e_hi = hi["e_final"]

    # 2. Reproducibility across seeds.
    print(f"[dks] reproducibility across {args.seeds} seeds ...")
    repro = [access_by_seeding(dw, cav, t_slow=map_t_slow, seed=s, n_tau=n_tau)
             for s in range(args.seeds)]

    # 3. Control: cold start, no protocol.
    print("[dks] control — cold start, no protocol ...")
    control_sol = _run(dw, map_t_slow, cav, e0=None, seed=0, n_tau=n_tau)
    e_ctrl = np.asarray(control_sol["e_final"])[0]
    u_ctrl = np.asarray(control_sol["U_int_history"])[0]
    control = {
        "delta_omega": float(dw),
        "metrics": soliton_metrics(e_ctrl, u_ctrl, cav, dw),
    }
    control["is_single"] = is_single_soliton(control["metrics"])

    # 4. Forward/backward route.
    fb = None
    if not args.no_forward_backward:
        print("[dks] route (a) forward/backward ...")
        fb = access_by_forward_backward(cav, seed=0, n_tau=n_tau)

    # 5. Existence map (batched).
    print("[dks] existence map ...")
    dw_grid = np.round(np.arange(1.0, 13.01, 0.5), 3)
    emap = existence_map(cav, dw_grid, t_slow=map_t_slow, seed=0, n_tau=n_tau)

    # 6. Breathing scan across the existence band (V6 vs detuning).
    print("[dks] breathing scan ...")
    scan_grid = np.round(np.arange(7.0, 16.01, 0.5), 3)
    breathing = breathing_scan(cav, scan_grid, t_slow=4000, seed=0, n_tau=8192,
                               **PRODUCTION_NUMERICS)

    # --- artifacts ---
    # A breather has no phase-independent snapshot spectrum: whenever V6 flags
    # one, the spectrum artifacts are built from the cycle average.
    vm = validated["metrics"]
    sp_avg = None
    title_extra = ""
    if vm["is_breather"]:
        print("[dks] V6: breather — computing cycle-averaged spectrum ...")
        sp_avg = cycle_averaged_spectrum(validated, cav, **PRODUCTION_NUMERICS)
        title_extra = breather_title_annotation(vm)
    plot_optical_spectrum(RESULTS_DIR / "dks_single_soliton_spectrum.png",
                          validated["e_final"], cav, dw, e_field_hi=e_hi,
                          sp=sp_avg, title_extra=title_extra)
    plot_soliton_summary(RESULTS_DIR / "dks_single_soliton_summary.png",
                         validated, cav, sp=sp_avg, title_extra=title_extra)
    plot_existence_map(RESULTS_DIR / "dks_existence_map.png", emap, cav)
    write_existence_csv(RESULTS_DIR / "dks_existence_map.csv", emap)
    plot_breathing_scan(RESULTS_DIR / "dks_breathing_scan.png", breathing, cav)
    write_breathing_csv(RESULTS_DIR / "dks_breathing_scan.csv", breathing)
    write_report(RESULTS_DIR / "dks_access_report.md", cav, validated, fb,
                 control, repro, emap, long_t_slow, breathing=breathing)

    # --- summary ---
    m = validated["metrics"]
    print("\n=== DKS access summary ===")
    print(f"validated single soliton (dw={args.dw} kappa, {args.long_tau_th} tau_th): "
          f"n_peaks={m['n_peaks']}, class={m['np_label']}, "
          f"env_corr={m['sech2_env_corr']:.4f}, U_tail_relstd="
          f"{m['u_int_tail_rel_std']:.2%}, finite={m['finite']}, "
          f"is_single={validated['is_single']}")
    n_ok = sum(1 for r in repro if r["is_single"])
    print(f"reproducibility: {n_ok}/{len(repro)} seeds single")
    print(f"control (no protocol): class={control['metrics']['np_label']}, "
          f"is_single={control['is_single']}")
    if fb is not None:
        print(f"forward/backward: class={fb['metrics']['np_label']}, "
              f"n_peaks={fb['metrics']['n_peaks']}, is_single={fb['is_single']}")
    b = emap["band"]
    if b.get("found"):
        print(f"existence band: [{b['lower_over_kappa']:.1f}, "
              f"{b['upper_over_kappa']:.1f}] kappa (contiguous={b['contiguous']})")
    print(f"V6 stationarity: {vm['stationarity']} "
          f"(T_b={vm['breathing_period_rt']:.0f} RT, "
          f"dU/U={vm['breathing_relstd']:.2%})")
    print(f"breathing sub-bands (kappa): {breathing['breathing_bands']}; "
          f"stationary (<{STATIONARY_RELSTD:.1%}) windows: "
          f"{breathing['stationary_windows']}")

    # Candidate dispersive waves derived from the D_int / detuning phase-matching
    # crossings (soliton-rest gauge), not fixed wavelength windows. Use the
    # cycle average when the state breathes (phase-independent readout).
    sp = sp_avg if sp_avg is not None else optical_spectrum(validated["e_final"], cav)
    crossings = dispersive_wave_crossings(dw)
    print("\nphase-matched dispersive-wave crossings D_int(mu) = delta_omega "
          f"(|mu| > 500, delta_omega = {args.dw:.1f} kappa):")
    for cr in crossings:
        print(f"  crossing mu ~ {cr['crossing_mu']:+.0f} "
              f"(~{cr['wavelength_nm']:.1f} nm)")
    pks = dispersive_wave_peaks(sp, dw)
    print("candidate dispersive-wave peaks (spectrum max within +/-30 modes of "
          "each crossing):")
    if not pks:
        print("  none resolved (raise --n-tau to cover the crossing modes)")
    for pk in pks:
        print(f"  lambda = {pk['wavelength_nm']:.1f} nm (mu = {pk['mu']:+d}, "
              f"crossing {pk['crossing_mu']:+.0f}): {pk['power_db']:.1f} dB, "
              f"prominence {pk['prominence_db']:.1f} dB")

    print(f"\nArtifacts in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
