"""Soliton state classification utilities."""

from __future__ import annotations
import jax.numpy as jnp


def make_state_labeler():
    """Return a JAX-traceable 6-class state labeler for use inside jax.lax.scan.

    Classes
    -------
    0  Off / below threshold   — total power near zero
    1  CW                      — flat field, low contrast
    2  Modulation instability   — periodic structure, moderate contrast
    3  Chaotic                  — high contrast, high spectral entropy
    4  Multi-soliton            — high contrast, low entropy, >1 peak
    5  Single soliton           — high contrast, low entropy, sech² comb
    """

    def state_labeler(e_t: jnp.ndarray) -> jnp.int32:
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
        grad = jnp.diff(p, append=p[:1])
        sign_changes = jnp.sum(
            (grad[:-1] > 0) & (grad[1:] <= 0)
        ).astype(jnp.float32)

        # --- decision tree (all jnp.where for JAX traceability) ---
        is_off     = total_power < 1e-6
        is_cw      = contrast < 2.0
        is_mi      = (contrast >= 2.0) & (contrast < 8.0)
        is_chaotic = (contrast >= 8.0) & (norm_entropy > 0.5)
        is_multi   = (contrast >= 8.0) & (norm_entropy <= 0.5) & (sign_changes > 1.5)
        # single soliton: high contrast, ordered spectrum, single peak
        is_single  = (contrast >= 8.0) & (norm_entropy <= 0.5) & (sign_changes <= 1.5)

        label = jnp.where(is_off,     0,
                jnp.where(is_cw,      1,
                jnp.where(is_mi,      2,
                jnp.where(is_chaotic, 3,
                jnp.where(is_multi,   4,
                                      5)))))
        return label.astype(jnp.int32)

    return state_labeler
