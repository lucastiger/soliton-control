"""Soliton state classification utilities."""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def make_state_labeler():
    """Return a JAX-traceable 7-class state labeler for use inside jax.lax.scan.

    Classes
    -------
    0  Off / below threshold    — total power near zero
    1  CW                       — flat field, low contrast
    2  Modulation instability   — periodic structure, moderate contrast
    3  Chaotic                  — high contrast, high spectral entropy
    4  Multi-soliton            — high contrast, low entropy, >1 peak
    5  Soliton Crystal          - high contrast, low entropy, evenly spaced peaks (highly ordered)
    6  Single soliton           — high contrast, low entropy, sech² comb
    """

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
        grad = jnp.diff(p, append=p[:1])          # length n_tau
        peak_mask = (grad > 0) & (jnp.roll(grad, -1) <= 0)
        sign_changes = jnp.sum(
            (grad > 0) & (jnp.roll(grad, -1) <= 0)
        ).astype(jnp.float32)

        # --- decision tree (all jnp.where for JAX traceability) ---
        is_off     = total_power < 1e-6
        is_cw      = contrast < 2.0
        is_mi      = (contrast >= 2.0) & (contrast < 8.0)
        is_chaotic = (contrast >= 8.0) & (norm_entropy > 0.5)


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
        
        CRYSTAL_CV_THRESHOLD = 0.1
        is_crystal = (
            (contrast >= 8.0) & (norm_entropy <= 0.5)
            & (sign_changes > 2.5)                       # ← require ≥ 3 peaks, not ≥ 2
            & (spacing_cv < CRYSTAL_CV_THRESHOLD)
        )
        is_multi = (
            (contrast >= 8.0) & (norm_entropy <= 0.5)
            & (sign_changes > 1.5)
            & ~is_crystal                                 # anything multi that isn't crystal
        )
        
        # single soliton: high contrast, ordered spectrum, single peak
        is_single  = (contrast >= 8.0) & (norm_entropy <= 0.5) & (sign_changes <= 1.5)

        label = jnp.where(is_off,     0,
                jnp.where(is_cw,      1,
                jnp.where(is_mi,      2,
                jnp.where(is_chaotic, 3,
                jnp.where(is_multi,   4,
                jnp.where(is_crystal, 5,
                jnp.where(is_single,  6,
                                      6)))))))

        return label.astype(jnp.int32)

    return state_labeler


def label_soliton_state(E_tau, threshold_params) -> int:
    """Label one intracavity-field snapshot using a 7-class soliton scheme."""
    defaults = {
        "power_floor": 1e-6,
        "contrast_cw": 2.0,
        "contrast_high": 8.0,
        "entropy_chaotic": 0.5,
        "crystal_cv": 0.1,
        "sech2_r2": 0.95,
        "peak_prominence": 0.3,
        "peak_width": 2.0,
    }
    params = defaults.copy()
    params.update(threshold_params or {})

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
            spacing_cv = float(spacings.std() / spacings.mean())
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
    defaults = {
        "power_floor": 1e-6,
        "contrast_cw": 2.0,
        "contrast_high": 8.0,
        "entropy_chaotic": 0.5,
        "crystal_cv": 0.1,
        "sech2_r2": 0.95,
        "peak_prominence": 0.3,
        "peak_width": 2.0,
    }
    params = defaults.copy()
    params.update(threshold_params or {})

    n_snapshots = E_history.shape[0]
    labels = np.empty((n_snapshots,), dtype=np.int32)
    for i in range(n_snapshots):
        labels[i] = label_soliton_state(E_history[i], params)
    return labels
