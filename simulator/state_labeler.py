"""Soliton state classification utilities."""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Physically-scaled OFF threshold
# ---------------------------------------------------------------------------
# Intracavity fields are stored as physical energies: |E|² is in Joules, with
# mean|E|² ~ 1e-11 … 1e-9 J across a detuning sweep and an empty-cavity numerical
# floor ~1e-16 J. A field is "OFF" when essentially no coherent CW power has
# built up. Rather than a magic constant, we tie the floor to the smallest CW
# energy the cavity can support over the sweep:
#
#     U_cw(δω) = κ_c · pin / ((κ/2)² + δω²)          [Joules]
#
# U_cw is monotonically decreasing in |δω|, so its minimum over the sweep is at
# the largest |δω|:
#
#     U_cw,min = κ_c · pin / ((κ/2)² + δω_max²)
#
# OFF is then declared when mean(|E|²) < f · U_cw,min, i.e. the field sits a
# factor f below even the dimmest CW state. f ≈ 1e-3 … 1e-2 leaves a wide margin
# above the ~1e-16 J empty-cavity floor while staying far below any real CW field.

def physical_off_floor(
    kappa: float,
    kappa_c: float,
    pin: float,
    delta_omega_max: float,
    off_fraction: float = 1e-3,
) -> float:
    """Smallest CW intracavity energy over the sweep, scaled by ``off_fraction``.

    Parameters are read from the simulation config (all in SI / rad·s⁻¹):
      kappa            total cavity loss rate κ
      kappa_c          coupling rate κ_c
      pin              pump power (W)
      delta_omega_max  largest |δω| in the detuning sweep (rad/s)
      off_fraction     f ∈ (0, 1); fraction of U_cw,min below which a field is OFF

    Returns the OFF power floor in Joules.
    """
    k = float(kappa)
    u_cw_min = float(kappa_c) * float(pin) / ((k / 2.0) ** 2 + float(delta_omega_max) ** 2)
    return float(off_fraction) * u_cw_min


def make_threshold_params(
    kappa: float,
    kappa_c: float,
    pin: float,
    delta_omega_max: float,
    *,
    off_fraction: float = 1e-3,
    overrides: dict | None = None,
) -> dict:
    """Build the shared threshold dict from physical config — single source of truth.

    The OFF floor is derived from (κ, κ_c, pin, δω_max) via ``physical_off_floor``;
    all other (geometric / spectral) thresholds come from ``_DEFAULT_THRESHOLD_PARAMS``.
    The SAME dict feeds both the JAX labeler (``make_state_labeler``) and the NumPy
    labeler (``label_soliton_state``), so the two stay byte-for-byte consistent.
    """
    params = dict(_DEFAULT_THRESHOLD_PARAMS)
    params["power_floor"] = physical_off_floor(
        kappa, kappa_c, pin, delta_omega_max, off_fraction
    )
    if overrides:
        params.update(overrides)
    return params


def make_state_labeler(threshold_params: dict | None = None):
    """Return a JAX-traceable 7-class state labeler for use inside jax.lax.scan.

    The labeler bakes every threshold into Python-float constants at build time,
    so the returned ``state_labeler`` contains no Python branching on traced
    values and is fully ``jax.lax.scan``-traceable. Pass the dict produced by
    ``make_threshold_params`` (or any subset of ``_DEFAULT_THRESHOLD_PARAMS``) so
    the JAX path uses the identical reductions and thresholds as the NumPy path.

    Classes
    -------
    0  Off / below threshold    — mean|E|² below the physical CW floor
    1  CW                       — flat field, low contrast
    2  Modulation instability   — periodic structure, moderate contrast
    3  Chaotic                  — high contrast, high spectral entropy
    4  Multi-soliton            — high contrast, low entropy, >1 peak
    5  Soliton Crystal          - high contrast, low entropy, evenly spaced peaks (highly ordered)
    6  Single soliton           — high contrast, low entropy, sech² comb
    """

    # --- bake all thresholds into float constants (no traced Python branching) ---
    _params = {**_DEFAULT_THRESHOLD_PARAMS, **(threshold_params or {})}
    POWER_FLOOR        = float(_params["power_floor"])
    CONTRAST_CW        = float(_params["contrast_cw"])
    CONTRAST_HIGH      = float(_params["contrast_high"])
    ENTROPY_CHAOTIC    = float(_params["entropy_chaotic"])
    CRYSTAL_CV         = float(_params["crystal_cv"])
    PEAK_AMP_THRESHOLD = float(_params["peak_prominence"])  # fraction of p_max
    SECH2_ENV_MONO_MIN = float(_params["sech2_env_mono_min"])  # single-DKS envelope test
    SECH2_ENV_TOL      = float(_params["sech2_env_tol"])       # per-step log tolerance
    COMB_MIN_DB        = float(_params["comb_structure_min_db"])  # comb-vs-flat-floor gate

    def state_labeler(e_t: jnp.ndarray) -> jnp.int32:
        n_tau = e_t.shape[0]
        p = jnp.abs(e_t) ** 2

        # --- spectral features ---
        spec = jnp.abs(jnp.fft.fft(e_t)) ** 2
        spec_norm = spec / jnp.maximum(jnp.sum(spec), 1e-20)
        # spectral entropy: low = ordered comb, high = chaotic
        entropy = -jnp.sum(spec_norm * jnp.log(jnp.maximum(spec_norm, 1e-20)))
        entropy_max = jnp.log(jnp.array(e_t.shape[0], dtype=jnp.float32))
        norm_entropy = entropy / entropy_max   # in [0, 1]

        # --- temporal features ---
        p_mean = jnp.mean(p)
        p_max  = jnp.max(p)
        contrast = p_max / jnp.maximum(p_mean, 1e-20)

        # number of peaks: count points above 50% of max with positive->negative
        # zero-crossings of the gradient (proxy for peak count, JAX-traceable)
        # Only count peaks whose amplitude exceeds PEAK_AMP_THRESHOLD·p_max (sourced
        # from params["peak_prominence"], identical to the NumPy labeler's prominence).
        # Without this threshold, dispersive-wave tails and FFT ringing on the 512-point
        # grid produce O(10–50) spurious peaks in single-soliton states, making
        # sign_changes >> 1 and systematically mislabeling single solitons as multi-soliton.
        grad = jnp.diff(p, append=p[:1])          # circular gradient, length n_tau
        _is_local_max = (grad > 0) & (jnp.roll(grad, -1) <= 0)
        peak_mask = _is_local_max & (p > PEAK_AMP_THRESHOLD * p_max)
        sign_changes = jnp.sum(peak_mask).astype(jnp.float32)

        # Smooth-sech²-envelope test (single-DKS vs chaos), JAX-traceable analogue
        # of the NumPy path's temporal sech² goodness-of-fit.
        #
        # A single dissipative Kerr soliton is a strong pump (DC) line plus a comb of
        # sidebands whose power envelope is sech² — i.e. it decreases MONOTONICALLY
        # outward from the pump on BOTH sides. Chaos/MI spectra are jagged and
        # non-monotonic. We measure the fraction of outward mode-steps whose (log)
        # power does not increase (within SECH2_ENV_TOL). This REPLACES the old
        # top-N-points "sharpness" proxy, which silently mislabels a genuine single
        # DKS as chaotic: the soliton sits on a bright CW background that carries most
        # of the total energy, so the fraction of power in the top ~32 points is only
        # ~0.2 (< the 0.75 sharpness gate), and the state fell through to class 3.
        # The envelope monotonicity is ~1.0 for a single DKS at any resolution
        # (the comb may be broad, but it is smooth), and ~0.5–0.75 for MI/chaos.
        spec_shift = jnp.fft.fftshift(spec)
        log_env = jnp.log10(
            jnp.maximum(spec_shift / jnp.maximum(jnp.max(spec_shift), 1e-30), 1e-12)
        )
        c_idx = n_tau // 2                                   # DC / pump line index
        right_steps = log_env[c_idx + 1:] - log_env[c_idx:-1]   # outward, i > center
        left_steps = log_env[:c_idx] - log_env[1:c_idx + 1]     # outward, i < center
        mono_frac = 0.5 * (
            jnp.mean((right_steps <= SECH2_ENV_TOL).astype(jnp.float32))
            + jnp.mean((left_steps <= SECH2_ENV_TOL).astype(jnp.float32))
        )

        # Comb-structure ("central bulge") test: a real soliton comb concentrates
        # sideband power NEAR the pump (inner half-band) and decays outward, so the
        # inner-band mean is well above the outer-band mean. A CW field carrying a
        # single-sample numerical spike has a FLAT sideband floor (the spike spreads
        # equally over all modes), giving inner ≈ outer ≈ 0 dB. mono_frac alone
        # cannot separate these — a flat floor is trivially "monotonic" — so we also
        # require a minimum inner/outer sideband ratio. This is the JAX analogue of
        # "the spectrum is a comb, not a pump line on a flat floor".
        q_idx = n_tau // 4
        inner_band = jnp.concatenate(
            [spec_shift[c_idx + 1:c_idx + q_idx], spec_shift[c_idx - q_idx + 1:c_idx]]
        )
        outer_band = jnp.concatenate(
            [spec_shift[c_idx + q_idx:], spec_shift[:c_idx - q_idx + 1]]
        )
        inner_outer_db = 10.0 * jnp.log10(
            jnp.mean(inner_band) / jnp.maximum(jnp.mean(outer_band), 1e-30)
        )


        # --- decision tree (all jnp.where for JAX traceability) ---
        # Physical fields are in Joules (mean|E|² ~ 1e-11–1e-9). Use mean(|E|²)
        # (matches the NumPy labeler) compared against the physically-scaled OFF
        # floor (POWER_FLOOR = f·U_cw,min, baked from config at build time).
        is_off     = p_mean < POWER_FLOOR
        is_cw      = contrast < CONTRAST_CW
        is_mi      = (contrast >= CONTRAST_CW) & (contrast < CONTRAST_HIGH)
        is_chaotic = (contrast >= CONTRAST_HIGH) & (norm_entropy > ENTROPY_CHAOTIC)


        # ---- crystal detection: peak spacing coefficient of variation ----
        # Extract peak positions as a sorted array of length n_tau,
        # with non-peak slots filled by n_tau (a sentinel beyond all valid indices).
        # After sorting, the first sign_changes entries are the true peak positions.
        sentinel = jnp.float32(n_tau)
        peak_locs = jnp.where(peak_mask, jnp.arange(n_tau, dtype=jnp.float32), sentinel)
        peak_locs_sorted = jnp.sort(peak_locs)          # real peaks first, sentinels at end
        
        # Spacings between consecutive real peaks.
        # diff of sorted locs: entry i = peak_locs_sorted[i] - peak_locs_sorted[i-1]
        # The last entry wraps to peak_locs_sorted[0]+n_tau-peak_locs_sorted[-1] (circular),
        # but we only use the first (sign_changes - 1) entries, so the wrap-around
        # and sentinel-to-sentinel diffs don't matter if we mask them.
        locs_shifted = jnp.roll(peak_locs_sorted, 1)
        raw_spacings = peak_locs_sorted - locs_shifted   # (n_tau,); first entry is garbage
        
        # Valid entries: indices 1 .. sign_changes-1 (between real peaks)
        # Build a validity mask: entry i is valid if i >= 1 and i < sign_changes
        valid_idx = jnp.arange(n_tau, dtype=jnp.float32)
        spacing_valid = (valid_idx >= 1.0) & (valid_idx < sign_changes)
        
        n_valid = jnp.maximum(sign_changes - 1.0, 1.0)
        sp_mean = jnp.sum(jnp.where(spacing_valid, raw_spacings, 0.0)) / n_valid
        sp_sq   = jnp.sum(jnp.where(spacing_valid, (raw_spacings - sp_mean)**2, 0.0)) / n_valid
        spacing_cv = jnp.sqrt(sp_sq) / jnp.maximum(sp_mean, 1.0)
        
        is_crystal = (
            (contrast >= CONTRAST_HIGH) & (norm_entropy <= ENTROPY_CHAOTIC)
            & (sign_changes > 2.5)                       # ← require ≥ 3 peaks, not ≥ 2
            & (spacing_cv < CRYSTAL_CV)
        )
        is_multi = (
            (contrast >= CONTRAST_HIGH) & (norm_entropy <= ENTROPY_CHAOTIC)
            & (sign_changes > 1.5)
            & ~is_crystal                                 # anything multi that isn't crystal
        )

        # CW-dominated with a single-sample numerical spike: high contrast and ONE
        # peak, but the sideband spectrum is FLAT (no comb — inner ≈ outer). This is
        # physically a CW state whose lone hot sample fakes a high peak-to-mean; it
        # must NOT be read as a soliton. Route it to CW (1). (A genuine soliton has a
        # comb: inner_outer_db well above COMB_MIN_DB.)
        is_flat_spike = (
            (contrast >= CONTRAST_HIGH)
            & (sign_changes <= 1.5)
            & (inner_outer_db < COMB_MIN_DB)
        )

        # single soliton: high contrast, ordered spectrum, ONE temporal peak, a smooth
        # (monotonic) sech² spectral envelope, AND real comb structure (sidebands
        # concentrated near the pump, not a flat floor). Keying on single-peak +
        # smooth sech² comb (the features the NumPy sech²-fit path uses) makes this
        # robust to a soliton on a bright CW background, while chaos (jagged, low
        # mono_frac), multi/MI (multiple peaks), and CW+spike (flat, is_flat_spike)
        # are all excluded.
        is_single = (
            (contrast >= CONTRAST_HIGH) & (norm_entropy <= ENTROPY_CHAOTIC)
            & (sign_changes <= 1.5)
            & (mono_frac >= SECH2_ENV_MONO_MIN)
            & (inner_outer_db >= COMB_MIN_DB)
        )

        label = jnp.where(is_off,        0,
                jnp.where(is_cw,         1,
                jnp.where(is_mi,         2,
                jnp.where(is_chaotic,    3,
                jnp.where(is_flat_spike, 1,   # CW + single-sample spike -> CW
                jnp.where(is_multi,      4,
                jnp.where(is_crystal,    5,
                jnp.where(is_single,     6,
                                         3))))))))

        return label.astype(jnp.int32)

    return state_labeler

_DEFAULT_THRESHOLD_PARAMS: dict = {
    # power_floor is a conservative fallback only. The canonical OFF floor is
    # derived from physical config via make_threshold_params() / physical_off_floor();
    # callers generating real (Joule-scale) data should ALWAYS pass that floor in.
    "power_floor": 1e-13,
    "contrast_cw": 2.0,
    "contrast_high": 8.0,
    "entropy_chaotic": 0.5,
    "crystal_cv": 0.1,
    "sech2_r2": 0.95,        # NumPy single-soliton sech² goodness-of-fit (class 6)
    # JAX single-soliton test: fraction of outward spectral-envelope steps that are
    # monotonically non-increasing (within sech2_env_tol, in log10 units). A single
    # DKS comb is ~1.0; MI/chaos are ~0.5-0.75. Replaces the old "sharpness_min"
    # top-N-power-fraction proxy, which mislabeled a DKS on a bright CW background.
    "sech2_env_mono_min": 0.9,
    "sech2_env_tol": 0.05,
    # Minimum inner/outer sideband power ratio (dB) for a real comb. A single DKS is
    # >= ~2.5 dB (sidebands bunch near the pump); a CW field with a single-sample
    # numerical spike has a FLAT sideband floor (~0 dB) and is routed to CW. Used by
    # both is_single (require a comb) and is_flat_spike (flat floor -> CW).
    "comb_structure_min_db": 1.5,
    "sharpness_min": 0.75,   # DEPRECATED: no longer used by the JAX labeler
    "peak_prominence": 0.3,
    "peak_width": 2.0,
}

def label_soliton_state(E_tau, threshold_params) -> int:
    """Label one intracavity-field snapshot using a 7-class soliton scheme."""
    
    params = {**_DEFAULT_THRESHOLD_PARAMS, **(threshold_params or {})}

    p = np.abs(E_tau) ** 2
    p_mean = float(np.mean(p))
    if p_mean < params["power_floor"]:
        return 0

    p_max = float(np.max(p))
    contrast = p_max / p_mean
    if contrast < params["contrast_cw"]:
        return 1

    n_tau = E_tau.shape[0]
    spec = np.abs(np.fft.fft(E_tau)) ** 2
    spec_norm = spec / max(float(np.sum(spec)), 1e-20)
    entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-20))
    norm_entropy = float(entropy / np.log(n_tau))

    peaks, _ = find_peaks(
        p,
        prominence=params["peak_prominence"] * p_max,
        width=params["peak_width"],
    )
    n_peaks = int(peaks.size)

    # Comb-structure ("central bulge") metric: inner/outer sideband power ratio (dB).
    # A real soliton comb bunches sideband power near the pump (inner >> outer); a CW
    # field carrying a single-sample numerical spike has a FLAT sideband floor
    # (inner ≈ outer ≈ 0 dB). Mirrors the JAX labeler's inner_outer_db.
    spec_shift = np.fft.fftshift(spec)
    c_idx = n_tau // 2
    q_idx = n_tau // 4
    inner_band = np.concatenate(
        [spec_shift[c_idx + 1:c_idx + q_idx], spec_shift[c_idx - q_idx + 1:c_idx]]
    )
    outer_band = np.concatenate(
        [spec_shift[c_idx + q_idx:], spec_shift[:c_idx - q_idx + 1]]
    )
    inner_outer_db = 10.0 * np.log10(
        float(np.mean(inner_band)) / max(float(np.mean(outer_band)), 1e-300)
    )

    if contrast >= params["contrast_high"]:
        if norm_entropy > params["entropy_chaotic"]:
            return 3
        if n_peaks >= 3:
            spacings = np.diff(np.sort(peaks))
            spacing_cv = float(spacings.std() / max(float(spacings.mean()), 1.0))
            if spacing_cv < params["crystal_cv"]:
                return 5
            return 4
        if n_peaks == 2:
            return 4
        if n_peaks <= 1:
            # CW + single-sample numerical spike: high contrast, <=1 wide peak, but a
            # FLAT sideband spectrum (no comb). Physically a CW state; route to CW (1)
            # so it is never read as a soliton. A genuine soliton has real comb
            # structure (inner_outer_db well above the flat floor).
            if inner_outer_db < params["comb_structure_min_db"]:
                return 1

            x = np.arange(n_tau, dtype=float)

            def sech2_model(x_vals, A, x0, w, B):
                return A / np.cosh((x_vals - x0) / w) ** 2 + B

            p0 = [p_max, float(np.argmax(p)), n_tau / 20.0, float(np.min(p))]
            try:
                popt, _ = curve_fit(sech2_model, x, p, p0=p0, maxfev=10000)
            except Exception:
                return 3

            p_fit = sech2_model(x, *popt)
            ss_res = float(np.sum((p - p_fit) ** 2))
            ss_tot = float(np.sum((p - p_mean) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-20)
            if r2 >= params["sech2_r2"]:
                return 6
            return 3

    if contrast < params["contrast_high"] and contrast >= params["contrast_cw"]:
        return 2

    return 0


def label_trajectory(E_history, threshold_params=None) -> np.ndarray:
    """Label all snapshots in a trajectory with the 7-class soliton scheme."""
    params = {**_DEFAULT_THRESHOLD_PARAMS, **(threshold_params or {})}

    n_snapshots = E_history.shape[0]
    labels = np.zeros((n_snapshots,), dtype=np.int32)
    for i in range(n_snapshots):
        labels[i] = label_soliton_state(E_history[i], params)
    return labels


def sech2_envelope_correlation(e_field: np.ndarray) -> tuple[float, float, float]:
    """sech^2 correlation of the comb envelope (dB), excluding the pump line.

    This is the quantitative, fit-based counterpart of the single-soliton
    discriminator the labelers use (single temporal peak + smooth sech^2 comb):
    a dissipative Kerr soliton spectrum is a strong pump (DC) line plus a sech^2
    comb of sidebands (|FT of sech|^2 = sech^2). We fit a width-matched sech^2 to
    the (fftshifted) sideband envelope in log/dB space — where the comb spans many
    decades — with the pump line itself excluded (it is not part of the envelope).

    Lives in the simulator layer alongside the labeler so ``analysis`` code can
    import it from here; nothing in ``simulator`` imports from ``analysis``.

    Returns (pearson_corr, r2, fitted_mode_width). On fit failure returns NaNs.
    A single DKS scores > 0.99; MI/chaos combs score near 0 or negative.
    """
    n = e_field.shape[0]
    spec = np.abs(np.fft.fftshift(np.fft.fft(e_field))) ** 2
    spec_n = spec / max(spec.max(), 1e-300)
    mu = np.arange(n) - n // 2
    y = np.log10(np.maximum(spec_n, 1e-12))

    mask = np.ones(n, dtype=bool)
    mask[n // 2] = False  # drop the pump (DC) line

    def model(m, log_a, mode_w, log_floor):
        return np.log10(10.0 ** log_a / np.cosh(m / mode_w) ** 2 + 10.0 ** log_floor)

    try:
        popt, _ = curve_fit(
            model, mu[mask], y[mask], p0=[0.0, 60.0, -4.0], maxfev=40000
        )
        fit = model(mu, *popt)
        corr = float(np.corrcoef(y[mask], fit[mask])[0, 1])
        ss_res = float(np.sum((y[mask] - fit[mask]) ** 2))
        ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        return corr, r2, abs(float(popt[1]))
    except Exception:
        return float("nan"), float("nan"), float("nan")


def assert_labelers_consistent(
    e_field: np.ndarray,
    atol: float = 0.0,
    threshold_params: dict | None = None,
) -> None:
    """Verify JAX and NumPy labelers agree on a test field.

    Both labelers are driven by the *same* threshold dict so any disagreement is
    a genuine reduction/mechanism drift rather than a config mismatch. Run this
    during dataset generation to catch labeler drift early.
    Raises AssertionError if the two labelers disagree.
    """
    params = {**_DEFAULT_THRESHOLD_PARAMS, **(threshold_params or {})}
    jax_labeler = make_state_labeler(params)
    jax_label = int(jax_labeler(jnp.array(e_field, dtype=jnp.complex64)))
    np_label = int(label_soliton_state(e_field, threshold_params=params))
    assert jax_label == np_label, (
        f"Labeler inconsistency: JAX={jax_label}, NumPy={np_label} for field with "
        f"max_power={float(np.max(np.abs(e_field)**2)):.3e}, "
        f"contrast={float(np.max(np.abs(e_field)**2)/np.mean(np.abs(e_field)**2)):.1f}"
    )
