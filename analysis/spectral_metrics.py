"""Cycle-averaged spectral metrics for the validated single-DKS comb.

This module operates on the CYCLE-AVERAGED optical power spectrum of the
dissipative-Kerr-soliton (DKS) state produced by ``analysis.dks_access``.  At
the validated operating point (delta_omega = 8 kappa) the attractor is a
deterministic BREATHER, so a single-snapshot spectrum is breathing-phase
dependent and is NOT a reproducible observable.  The physically meaningful
input for every metric here is therefore the spectrum obtained by averaging
``|a_mu|**2`` (LINEAR power, one value per resonator mode ``mu``) over an
integer number of breathing periods -- e.g. the mean produced by
``analysis.dks_access.cycle_averaged_spectrum``.  All public metrics take such
a precomputed averaged spectrum as their primary input; the sole convenience
wrapper that starts from a solver trajectory
(:func:`average_power_spectrum`) performs its averaging in linear power, never
in dB and never on complex fields.

Feature 1 -- 3 dB spectral span
-------------------------------
The 3 dB spectral span is the full width (at the half-power / -3 dB level) of
the soliton comb envelope, reported both in relative mode number
``Delta_mu`` and in Hz (``Delta_mu * FSR``).  The envelope is extracted from
the discrete comb-line powers with the pump line excluded (it sits on a strong
CW background well above the sech^2 soliton envelope), a numerical-floor mask
applied, and light dB-domain median smoothing followed by monotone-cubic
(PCHIP) interpolation.  See :func:`spectral_envelope_db` and
:func:`three_db_span` for the full method and its justification.

Conventions
-----------
* Analysis-layer arrays are NumPy float64 (JAX arrays are converted with
  ``np.asarray``); nothing here imports the solver.
* ``mu`` is the relative cavity-mode number with ``mu = 0`` the pump line,
  matching the FFT-bin indexing used by ``analysis.dks_access`` (one bin per
  FSR).
* Powers are LINEAR (``|a_mu|**2``); the repo normalises the committed spectra
  so the pump line has unit power, but every metric here is invariant to that
  overall scale (it works from relative dB).
"""

from __future__ import annotations

import math
import warnings

import numpy as np

# -3 dB (half-power) level.  "3 dB span" and "FWHM" denote the same thing here:
# the full width where the envelope power falls to half its maximum, i.e. a
# 10*log10(2) = 3.0103 dB drop.  ``level_db`` defaults to exactly 3.0 dB (the
# literal "-3 dB" of the spec); the two differ by 0.04 %, far below the metric's
# few-percent robustness budget.
HALF_POWER_DB = 10.0 * math.log10(2.0)          # 3.0103 dB

DEFAULT_LEVEL_DB = 3.0
DEFAULT_FLOOR_DB = 60.0        # discard lines > this far below the strongest
                               # non-pump line before smoothing
DEFAULT_SMOOTH_MODES = 5       # light median window (single-line-spur removal)
DEFAULT_INTERP_FACTOR = 10     # PCHIP fine-grid density (>= 10x mode density)
AUTO_SMOOTH_W_MIN = 5          # smallest odd median window the auto-search tries
AUTO_SMOOTH_W_MAX = 101        # cap on the auto-search median window

# arccosh(sqrt(2)): sech^2(x) = 1/2 at x = arccosh(sqrt(2)) = 0.881373587...
_ARCCOSH_SQRT2 = math.acosh(math.sqrt(2.0))


# ---------------------------------------------------------------------------
# 1. Input normalisation
# ---------------------------------------------------------------------------
def comb_line_powers(spectrum, mode_index=None, *, pump_mu: int = 0):
    """Normalise a spectrum to per-comb-line powers ``(mu, P_mu)``.

    ``spectrum`` is either an already per-resonator-mode power array or a dense
    (fftshifted) FFT-grid power spectrum -- in this repo these are the same
    object, one FFT bin per cavity mode (one per FSR), so this is a validating
    pass-through.  ``mode_index`` is the matching relative mode number ``mu``
    (``mu = pump_mu`` is the pump line); when omitted the standard fftshifted
    FFT-grid indexing ``mu = arange(n) - n // 2`` is assumed.

    The pump index is taken from the mode indexing (``mu == pump_mu``), NEVER
    inferred as "the strongest line": for a DKS the pump sits on a bright CW
    background and IS the strongest line, but that coincidence must not define
    the pump -- other operating points (or a pump-suppressed comb) would break
    it.  Argmax is used only as a validated fallback, with a warning, when the
    supplied ``mode_index`` does not contain ``pump_mu``.

    Returns ``(mu, P_mu)`` sorted by ascending ``mu``, with ``mu`` int64 and
    ``P_mu`` float64.  Validates that powers are finite and non-negative.
    """
    P = np.asarray(spectrum, dtype=np.float64).ravel()
    n = P.size
    if mode_index is None:
        mu = np.arange(n, dtype=np.int64) - n // 2
    else:
        mu = np.asarray(mode_index).ravel().astype(np.int64)
        if mu.size != n:
            raise ValueError(
                f"mode_index length {mu.size} != spectrum length {n}"
            )
    if not np.all(np.isfinite(P)):
        raise ValueError("spectrum contains non-finite (NaN/Inf) powers")
    if np.any(P < 0.0):
        raise ValueError("spectrum contains negative powers (expected |a_mu|^2)")

    order = np.argsort(mu, kind="stable")
    mu, P = mu[order], P[order]

    pump_hits = np.nonzero(mu == pump_mu)[0]
    if pump_hits.size == 0:
        i = int(np.argmax(P))
        warnings.warn(
            f"pump mode mu={pump_mu} absent from mode_index; falling back to "
            f"argmax at mu={int(mu[i])} (validated fallback only)",
            RuntimeWarning,
            stacklevel=2,
        )
    return mu, P


def average_power_spectrum(snapshots, *, is_power: bool = False,
                           mode_index=None):
    """Cycle/slow-time-average |a_mu|^2 over a stack of snapshots (LINEAR power).

    Convenience wrapper for going from a solver trajectory to an averaged
    spectrum (mirrors ``analysis.dks_access.cycle_averaged_spectrum``, which is
    the production path).  ``snapshots`` has the slow-time / round-trip axis
    first:

    * ``is_power=False`` (default): ``snapshots`` are complex intracavity fast-
      time fields, shape ``(n_avg, n_tau)``.  The per-snapshot mode power is
      ``|fftshift(fft(field))|**2`` and these are averaged.
    * ``is_power=True``: ``snapshots`` are already per-mode LINEAR power spectra,
      shape ``(n_avg, n_modes)`` (fftshifted), and are averaged directly.

    Averaging is done in LINEAR power -- never on dB values (which would bias
    the mean toward the peak) and never on the complex fields (breathing phase
    drift would cancel real power).  Returns ``(mu, P_mu)`` via
    :func:`comb_line_powers`.
    """
    arr = np.asarray(snapshots)
    if arr.ndim != 2:
        raise ValueError(f"snapshots must be 2D (n_avg, n); got shape {arr.shape}")
    if is_power:
        power = np.asarray(arr, dtype=np.float64)
        if np.iscomplexobj(arr):
            raise ValueError("is_power=True but snapshots are complex")
    else:
        fld = np.asarray(arr, dtype=np.complex128)
        power = np.abs(np.fft.fftshift(np.fft.fft(fld, axis=-1), axes=-1)) ** 2
    mean_power = np.mean(power, axis=0)
    return comb_line_powers(mean_power, mode_index)


# ---------------------------------------------------------------------------
# 2. Envelope extraction
# ---------------------------------------------------------------------------
def _build_envelope(mu, P_mu, *, exclude_pump=True,
                    smooth_modes=DEFAULT_SMOOTH_MODES, floor_db=DEFAULT_FLOOR_DB,
                    interp_factor=DEFAULT_INTERP_FACTOR, pump_mu=0):
    """Shared envelope construction; returns every intermediate as a dict.

    Steps (see :func:`spectral_envelope_db` for the physical justification):
    exclude pump -> numerical-floor mask -> dB relative to the strongest
    retained (non-pump) line -> odd-window median smoothing -> PCHIP onto a
    fine mu grid.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import medfilt

    mu = np.asarray(mu).astype(np.int64)
    P = np.asarray(P_mu, dtype=np.float64)
    if mu.shape != P.shape:
        raise ValueError("mu and P_mu must have the same shape")

    keep = np.isfinite(P) & (P > 0.0)
    if exclude_pump:
        keep &= mu != pump_mu
    m, p = mu[keep], P[keep]
    if m.size < 4:
        raise ValueError(
            f"too few usable comb lines ({m.size}) to build an envelope"
        )

    ref_lin = float(p.max())                      # strongest non-pump line
    floor_lin = ref_lin * 10.0 ** (-floor_db / 10.0)
    above = p >= floor_lin
    m, p = m[above], p[above]

    order = np.argsort(m, kind="stable")
    m, p = m[order], p[order]
    s_db_raw = 10.0 * np.log10(p / ref_lin)       # <= 0 dB, peak at 0

    w = int(smooth_modes)
    if w > 1:
        if w % 2 == 0:                            # medfilt requires odd windows
            w += 1
        s_db_sm = medfilt(s_db_raw, kernel_size=min(w, _odd_le(m.size)))
    else:
        s_db_sm = s_db_raw.copy()

    step = 1.0 / float(interp_factor)
    mu_env = np.arange(float(m.min()), float(m.max()) + 0.5 * step, step)
    s_db_env = PchipInterpolator(m.astype(np.float64), s_db_sm)(mu_env)

    return {
        "mu_ret": m,
        "P_ret": p,
        "s_db_raw": s_db_raw,
        "s_db_sm": s_db_sm,
        "mu_env": mu_env,
        "s_db_env": s_db_env,
        "ref_lin": ref_lin,
        "floor_lin": floor_lin,
        "smooth_modes": w,
        "floor_db": float(floor_db),
        "interp_factor": int(interp_factor),
    }


def _odd_le(n: int) -> int:
    """Largest odd integer <= n (>= 1)."""
    n = int(n)
    return n if n % 2 == 1 else max(n - 1, 1)


def spectral_envelope_db(mu, P_mu, *, exclude_pump=True,
                         smooth_modes=DEFAULT_SMOOTH_MODES,
                         floor_db=DEFAULT_FLOOR_DB,
                         interp_factor=DEFAULT_INTERP_FACTOR, pump_mu=0):
    """Comb-envelope in dB on a fine ``mu`` grid: ``(mu_env, S_db)``.

    Method and justification (implemented exactly as below):

    * **Comb-line peak powers only.**  The discrete ``P_mu`` ARE the comb
      lines; there is no inter-line noise floor to reject at the analysis
      layer, only a numerical floor (below).

    * **Exclude the pump line (mu = 0).**  For a DKS the pump line sits on a
      strong intracavity CW background component that lies well above the
      sech^2 soliton envelope, so including it biases both the envelope peak
      and the half-max level.

    * **Numerical-floor mask.**  Lines more than ``floor_db`` (default 60 dB)
      below the strongest non-pump line are discarded before smoothing, so the
      far wings sitting at the float64/dealias numerical floor cannot distort a
      smoother or fit.

    * **dB-domain smoothing.**  ``S_db = 10*log10(P_mu / max(P_mu, non-pump))``
      and all smoothing/interpolation happens in dB.  The sech^2 envelope is
      near-parabolic in dB over its core, so dB-domain smoothing preserves the
      -3 dB crossing shape with minimal bias; linear-domain smoothing would
      over-weight the peak and bias the width low.  (This is orthogonal to the
      upstream breathing-phase averaging, which is done in LINEAR power.)

    * **Smoothing = median + PCHIP.**  A light odd-window median filter
      (``smooth_modes``, default 5) suppresses single-line outliers
      (mode-crossing / dispersive-wave spurs), followed by monotone-cubic
      (PCHIP) interpolation onto a mu grid ``interp_factor`` times denser than
      the modes, for sub-mode crossing resolution.  A global sech^2 model is
      deliberately NOT fit here -- the lab-relevant spectra contain dispersive-
      wave features a global fit cannot represent (see :func:`sech2_core_fwhm`
      for the optional core cross-check).

    ``S_db`` is referenced to the strongest retained non-pump line (its peak is
    ~0 dB).  Returns float64 ``(mu_env, S_db)``.
    """
    env = _build_envelope(mu, P_mu, exclude_pump=exclude_pump,
                          smooth_modes=smooth_modes, floor_db=floor_db,
                          interp_factor=interp_factor, pump_mu=pump_mu)
    return env["mu_env"], env["s_db_env"]


# ---------------------------------------------------------------------------
# 3. 3 dB span
# ---------------------------------------------------------------------------
def _level_crossings(x, y, level):
    """Interpolated ``x`` values where ``y`` crosses ``level`` (linear interp)."""
    d = np.asarray(y, dtype=np.float64) - float(level)
    x = np.asarray(x, dtype=np.float64)
    out = []
    for i in range(d.size - 1):
        a, b = d[i], d[i + 1]
        if a == 0.0:
            out.append(x[i])
            continue
        if (a < 0.0) != (b < 0.0):
            t = a / (a - b)
            out.append(x[i] + t * (x[i + 1] - x[i]))
    return np.asarray(out, dtype=np.float64)


def _span_from_envelope(mu_env, s_db_env, level_db):
    """Outermost -level_db crossings of a prepared dB envelope.

    Returns ``(left, right, peak_mu, reference_db, n_crossings)`` with ``left``/
    ``right`` NaN where the retained band ends before a crossing.
    """
    reference_db = float(np.max(s_db_env))
    peak_mu = float(mu_env[int(np.argmax(s_db_env))])
    level = reference_db - level_db
    crossings = _level_crossings(mu_env, s_db_env, level)
    left_side = crossings[crossings <= peak_mu]
    right_side = crossings[crossings >= peak_mu]
    left = float(left_side.min()) if left_side.size else math.nan     # outermost
    right = float(right_side.max()) if right_side.size else math.nan  # outermost
    return left, right, peak_mu, reference_db, int(crossings.size)


def _auto_smooth_window(mu, P_mu, *, exclude_pump, floor_db, interp_factor,
                        level_db, pump_mu, w_min=AUTO_SMOOTH_W_MIN,
                        w_max=AUTO_SMOOTH_W_MAX):
    """Data-derived median window: smallest odd ``w >= w_min`` giving a clean,
    single-lobe -level_db crossing on each side.

    The deterministic cycle-averaged breather spectrum carries a strong quasi-
    periodic interference modulation (soliton/CW beating and breathing
    sidebands) whose depth exceeds ``level_db``; a light median cannot remove
    it, so the -3 dB crossings would be modulation-dominated and unstable.
    Rather than hardcode a window for one dataset, grow the odd median window
    until the smoothed envelope descends monotonically to exactly ONE crossing
    per side (``n_crossings == 2`` with both sides bracketed) and stays that way
    at the next window (hysteresis, so a fragile single-window dip is not
    picked).  For a clean sech^2 comb this returns ``w_min`` immediately.

    Returns ``(window, converged)``; ``converged`` is False if no window up to
    ``w_max`` produced a clean pair of crossings (then ``w_max`` is returned).
    """
    w = w_min if w_min % 2 == 1 else w_min + 1
    prev_clean = False
    prev_w = None
    while w <= w_max:
        mu_env, s_db_env = spectral_envelope_db(
            mu, P_mu, exclude_pump=exclude_pump, smooth_modes=w,
            floor_db=floor_db, interp_factor=interp_factor, pump_mu=pump_mu)
        left, right, _, _, ncross = _span_from_envelope(mu_env, s_db_env, level_db)
        clean = ncross == 2 and math.isfinite(left) and math.isfinite(right)
        if clean and prev_clean:
            return prev_w, True
        prev_clean, prev_w = clean, w
        w += 2
    # fell through: either never clean, or only became clean at the last window
    if prev_clean:
        return prev_w, True
    return _odd_le(w_max), False


def three_db_span(mu, P_mu, metadata, *, exclude_pump=True, smooth_modes="auto",
                  floor_db=DEFAULT_FLOOR_DB, interp_factor=DEFAULT_INTERP_FACTOR,
                  level_db=DEFAULT_LEVEL_DB, pump_mu=0,
                  auto_w_min=AUTO_SMOOTH_W_MIN, auto_w_max=AUTO_SMOOTH_W_MAX):
    """3 dB spectral span (FWHM) of the soliton comb envelope.

    Builds the envelope with :func:`spectral_envelope_db` and measures its full
    width at ``level_db`` (default -3 dB) below the envelope maximum.

    * **Reference level** is the MAXIMUM of the smoothed envelope over the
      retained (non-pump, above-floor) lines -- not the pump power and not any
      single raw line.  Localised dispersive-wave peaks are suppressed by the
      median step and, at the default ``floor_db``, fall below the floor mask
      entirely; if a DW nonetheless dominates the smoothed envelope (the -3 dB
      band fails to straddle the comb centre ``mu = 0``) a warning is emitted
      naming the peak location, because the "3 dB span" of a DW-dominated
      envelope is not the sech^2 bandwidth.

    * **Outermost crossings.**  On each side the span uses the OUTERMOST
      crossing of ``reference - level_db`` (the last down-crossing before the
      retained-data boundary), which makes the metric robust to small ripple
      near the peak.

    * **Smoothing window.**  ``smooth_modes='auto'`` (default) selects the
      window from the data via :func:`_auto_smooth_window` -- the smallest odd
      window whose envelope has a single clean crossing per side -- so no
      dataset-specific window is hardcoded; pass an int to force a fixed window
      (used by the synthetic unit tests).

    Edge cases return NaN (never raise): if the envelope never rises
    ``level_db`` above its wings, both crossings are NaN with a warning; if only
    one side is bracketed before the retained band ends, that side is NaN and
    ``one_sided`` is set.

    ``metadata`` supplies the mode spacing for the Hz conversion via key
    ``fsr_hz`` (pulled from config/run metadata, never hardcoded); ``pump_mu``
    may also be supplied there.  Returns a dict with the span in mode-number and
    Hz units, crossing positions, reference level, every parameter used, flags,
    warnings, units, and the metric definition string.
    """
    metadata = dict(metadata or {})
    pump_mu = int(metadata.get("pump_mu", pump_mu))
    fsr_hz = metadata.get("fsr_hz", None)

    # Normalise/validate inputs and guarantee ascending-mu ordering.
    mu_arr, P_arr = comb_line_powers(P_mu, mu, pump_mu=pump_mu)

    warnings_list: list[str] = []

    if smooth_modes == "auto":
        window, converged = _auto_smooth_window(
            mu_arr, P_arr, exclude_pump=exclude_pump, floor_db=floor_db,
            interp_factor=interp_factor, level_db=level_db, pump_mu=pump_mu,
            w_min=auto_w_min, w_max=auto_w_max)
        if not converged:
            warnings_list.append(
                f"auto smoothing did not converge to a single-lobe envelope "
                f"below window {auto_w_max}; the comb-core interference "
                f"modulation is deeper than {level_db} dB -- the 3 dB span is "
                f"fringe-sensitive; reporting the window-{window} result")
    else:
        window = int(smooth_modes)

    env = _build_envelope(mu_arr, P_arr, exclude_pump=exclude_pump,
                          smooth_modes=window, floor_db=floor_db,
                          interp_factor=interp_factor, pump_mu=pump_mu)
    mu_env, s_db_env = env["mu_env"], env["s_db_env"]

    left, right, peak_mu, reference_db, ncross = _span_from_envelope(
        mu_env, s_db_env, level_db)

    one_sided = False
    span_modes = math.nan
    dynamic_range = float(np.max(s_db_env) - np.min(s_db_env))
    if dynamic_range < level_db:
        warnings_list.append(
            f"envelope dynamic range {dynamic_range:.2f} dB < {level_db} dB: "
            f"it never falls {level_db} dB below its maximum; span is undefined")
    elif math.isnan(left) != math.isnan(right):
        one_sided = True
        missing = "left" if math.isnan(left) else "right"
        warnings_list.append(
            f"3 dB crossing not bracketed on the {missing} side (retained band "
            f"ends first); reporting a one-sided span flag, span = NaN")
    elif math.isnan(left) and math.isnan(right):
        warnings_list.append(
            "no 3 dB crossing on either side within the retained band")
    else:
        span_modes = right - left
        # DW-dominated check: a soliton comb is centred on the pump, so its
        # 3 dB band must straddle mu = 0.  If it does not, the envelope peak is
        # a dispersive wave / far shoulder, not the sech^2 core.
        if not (left <= pump_mu <= right):
            warnings_list.append(
                f"3 dB band [{left:.1f}, {right:.1f}] does not straddle the "
                f"comb centre mu={pump_mu}: the smoothed-envelope peak at "
                f"mu={peak_mu:.1f} is a dispersive-wave/shoulder feature, so "
                f"this span is NOT the sech^2 comb bandwidth")

    span_hz = span_modes * float(fsr_hz) if fsr_hz is not None else math.nan
    if fsr_hz is None:
        warnings_list.append(
            "metadata has no 'fsr_hz'; span_hz/THz/GHz reported as NaN")

    for w in warnings_list:
        warnings.warn(w, RuntimeWarning, stacklevel=2)

    return {
        "metric": "three_db_span",
        "metric_definition": (
            "Full width, in relative cavity-mode number Delta_mu, of the "
            "pump-excluded, numerical-floor-masked, dB-domain median-smoothed, "
            "PCHIP-interpolated comb-line power envelope, measured between the "
            "OUTERMOST points where the envelope falls level_db below its "
            f"maximum (level_db = {level_db} dB, i.e. the -3 dB / half-power "
            "level). Converted to frequency via Delta_mu * FSR."),
        "span_modes": float(span_modes),
        "span_hz": float(span_hz),
        "span_thz": float(span_hz / 1e12) if math.isfinite(span_hz) else math.nan,
        "span_ghz": float(span_hz / 1e9) if math.isfinite(span_hz) else math.nan,
        "left_crossing_mu": float(left),
        "right_crossing_mu": float(right),
        "envelope_peak_mu": float(peak_mu),
        "reference_level_db": float(reference_db),
        "n_crossings": int(ncross),
        "one_sided": bool(one_sided),
        "fsr_hz": float(fsr_hz) if fsr_hz is not None else math.nan,
        "params": {
            "exclude_pump": bool(exclude_pump),
            "smooth_modes_requested": smooth_modes,
            "smooth_modes_used": int(env["smooth_modes"]),
            "floor_db": float(floor_db),
            "interp_factor": int(interp_factor),
            "level_db": float(level_db),
            "pump_mu": int(pump_mu),
        },
        "warnings": warnings_list,
        "units": {
            "span_modes": "cavity modes (relative mode number Delta_mu)",
            "span_hz": "Hz",
            "span_thz": "THz",
            "span_ghz": "GHz",
            "left_crossing_mu": "cavity mode number (mu)",
            "right_crossing_mu": "cavity mode number (mu)",
            "envelope_peak_mu": "cavity mode number (mu)",
            "reference_level_db": "dB, relative to the strongest non-pump line",
            "fsr_hz": "Hz",
            "floor_db": "dB",
            "level_db": "dB",
        },
    }


# ---------------------------------------------------------------------------
# Optional cross-check: dB-domain sech^2 core fit (NOT the primary method)
# ---------------------------------------------------------------------------
def sech2_core_fwhm(mu, P_mu, *, core_mu=200, exclude_pump=True, pump_mu=0,
                    level_db=DEFAULT_LEVEL_DB, fsr_hz=None):
    """Optional cross-check: FWHM from a dB-domain sech^2 fit of the comb CORE.

    Fits ``10*log10(sech^2((mu - mu_c)/w)) + c`` to the retained comb lines with
    ``|mu| <= core_mu`` and reports the closed-form ``level_db`` full width,
    ``FWHM = 2 * w * arccosh(sqrt(10**(level_db/10) - 1 + ... ))`` (for the
    -3 dB / half-power level this is ``2 * w * arccosh(sqrt(2))``).  This is a
    CROSS-CHECK only -- the primary metric (:func:`three_db_span`) never fits a
    global sech^2 because the wings carry dispersive-wave features a single
    sech^2 cannot represent -- but over the core the sech^2 fit is faithful and
    provides an independent width estimate.  Returns a dict (NaN width on fit
    failure, never raises).
    """
    from scipy.optimize import curve_fit

    mu = np.asarray(mu).astype(np.float64)
    P = np.asarray(P_mu, dtype=np.float64)
    keep = np.isfinite(P) & (P > 0.0) & (np.abs(mu) <= core_mu)
    if exclude_pump:
        keep &= mu != pump_mu
    x, p = mu[keep], P[keep]
    if x.size < 5:
        return {"fwhm_modes": math.nan, "reason": "too few core lines"}
    ref = float(p.max())
    y = 10.0 * np.log10(p / ref)

    def sech2_db(m, w, c, mc):
        a = np.abs(m - mc) / w
        log_cosh = a + np.log1p(np.exp(-2.0 * a)) - math.log(2.0)
        return c - 20.0 / math.log(10.0) * log_cosh

    try:
        popt, _ = curve_fit(sech2_db, x, y, p0=[max(x.max() / 3.0, 5.0), 0.0, 0.0],
                            maxfev=40000)
    except Exception as exc:  # pragma: no cover - fit robustness guard
        return {"fwhm_modes": math.nan, "reason": f"fit failed: {exc}"}
    w = abs(float(popt[0]))
    ratio = 10.0 ** (level_db / 10.0)             # power drop factor (2 at 3 dB)
    half = w * math.acosh(math.sqrt(ratio))
    fwhm = 2.0 * half
    rms = float(np.sqrt(np.mean((sech2_db(x, *popt) - y) ** 2)))
    span_hz = fwhm * float(fsr_hz) if fsr_hz is not None else math.nan
    return {
        "fwhm_modes": float(fwhm),
        "fwhm_hz": float(span_hz),
        "fwhm_thz": float(span_hz / 1e12) if math.isfinite(span_hz) else math.nan,
        "width_w_modes": float(w),
        "center_mu": float(popt[2]),
        "peak_db": float(popt[1]),
        "fit_rms_db": rms,
        "core_mu": int(core_mu),
        "note": "cross-check only (dB-domain sech^2 core fit); not the primary "
                "3 dB span metric",
    }


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------
def plot_spectrum_with_span(mu, P_mu, span_result, path, *, metadata=None,
                            title_extra="", dpi=300, also_pdf=True):
    """Annotated comb-spectrum plot showing the 3 dB span overlay.

    Draws the comb lines (dB relative to the strongest non-pump line), the
    smoothed envelope curve, the horizontal ``reference - level_db`` line, the
    shaded 3 dB span region, and a text annotation of the span in both GHz/THz
    and mode count.  Matches the repo's matplotlib conventions and saves both a
    300-dpi PNG and a PDF (per shared analysis conventions).  ``path`` is the
    PNG path; the PDF sits beside it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    metadata = dict(metadata or {})
    p = span_result["params"]
    env = _build_envelope(mu, P_mu, exclude_pump=p["exclude_pump"],
                          smooth_modes=p["smooth_modes_used"],
                          floor_db=p["floor_db"], interp_factor=p["interp_factor"],
                          pump_mu=p["pump_mu"])
    mu_env, s_db_env = env["mu_env"], env["s_db_env"]
    m_ret, s_ret = env["mu_ret"], env["s_db_raw"]

    ref = span_result["reference_level_db"]
    level_db = p["level_db"]
    left = span_result["left_crossing_mu"]
    right = span_result["right_crossing_mu"]
    span_modes = span_result["span_modes"]
    span_thz = span_result["span_thz"]
    span_ghz = span_result["span_ghz"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    # retained comb lines (stems + markers), pump excluded from the envelope
    ax.vlines(m_ret, np.minimum(s_ret, ref - 1.5 * max(level_db, 1.0) - 40),
              s_ret, color="tab:blue", lw=0.4, alpha=0.35)
    ax.plot(m_ret, s_ret, ".", ms=2.2, color="tab:blue",
            label=f"comb lines (pump excluded), n_tau={np.asarray(mu).size}")
    # smoothed envelope
    ax.plot(mu_env, s_db_env, "-", lw=1.6, color="tab:orange",
            label=f"envelope (median w={p['smooth_modes_used']} + PCHIP, dB)")

    if math.isfinite(span_modes):
        ax.axhline(ref - level_db, color="tab:red", ls="--", lw=1.0,
                   label=f"reference $-${level_db:g} dB")
        ax.axvspan(left, right, color="tab:red", alpha=0.12,
                   label="3 dB span")
        for xc in (left, right):
            ax.axvline(xc, color="tab:red", ls=":", lw=0.9, alpha=0.8)
        ax.annotate(
            f"3 dB span = {span_modes:.1f} modes\n"
            f"= {span_ghz:.1f} GHz ({span_thz:.2f} THz)\n"
            f"[{left:.1f}, {right:.1f}] $\\mu$",
            xy=(0.5 * (left + right), ref - level_db),
            xytext=(0.02, 0.06), textcoords="axes fraction",
            ha="left", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="tab:red",
                      alpha=0.9))
        pad = 0.9 * span_modes
        ax.set_xlim(left - pad, right + pad)
    else:
        ax.set_xlim(m_ret.min(), m_ret.max())

    ax.axhline(ref, color="0.5", ls="-", lw=0.6, alpha=0.6)
    ax.set_xlabel(r"cavity mode index $\mu$  ($\mu=0$ = pump)")
    ax.set_ylabel(r"power (dB, rel. strongest non-pump line)")
    dw_txt = metadata.get("delta_omega_over_kappa", None)
    dw_str = f" @ $\\delta\\omega$ = {dw_txt:.1f}$\\kappa$" if dw_txt else ""
    ttl = (f"Single-DKS 3 dB spectral span{dw_str} "
           f"(cycle-averaged, {metadata.get('n_rt_averaged', '?')} RT)")
    if title_extra:
        ttl += f"\n{title_extra.lstrip(' -')}"
    ax.set_title(ttl, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = Path(path)
    fig.savefig(path, dpi=dpi)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path
