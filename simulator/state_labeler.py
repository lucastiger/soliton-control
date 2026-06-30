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
    IS_SHARP_THRESHOLD = float(_params["sharpness_min"])

    def state_labeler(e_t: jnp.ndarray) -> jnp.int32:
        n_tau = e_t.shape[0]
        p = jnp.abs(e_t) ** 2
        total_power = jnp.sum(p)

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

        # Sharpness proxy: fraction of total power contained in the top-N_peak points.
        # A true sech² soliton concentrates ~80% of power in ~N_tau/16 ≈ 32 points.
        # A broad chaotic hump spreads power across many points → low sharpness.
        # This replaces the scipy sech²-fit that cannot run inside jax.lax.scan.
        N_peak_pts = n_tau // 16                            # 32 for n_tau=512
        p_sorted = jnp.sort(p)[::-1]                        # descending
        power_in_top = jnp.sum(p_sorted[:N_peak_pts])
        sharpness = power_in_top / jnp.maximum(total_power, 1e-20)


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

        # single soliton: high contrast, ordered spectrum, single peak
        # Update is_single to require sharpness (avoids labeling broad chaotic humps as solitons)
        is_single = (
            (contrast >= CONTRAST_HIGH) & (norm_entropy <= ENTROPY_CHAOTIC)
            & (sign_changes <= 1.5)
            & (sharpness >= IS_SHARP_THRESHOLD)
        )

        label = jnp.where(is_off,     0,
                jnp.where(is_cw,      1,
                jnp.where(is_mi,      2,
                jnp.where(is_chaotic, 3,
                jnp.where(is_multi,   4,
                jnp.where(is_crystal, 5,
                jnp.where(is_single,  6,
                                      3)))))))

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
    "sharpness_min": 0.75,   # JAX single-soliton top-N power fraction (class 6)
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
