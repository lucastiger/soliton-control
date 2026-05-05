import numpy as np
import pytest
from simulator.state_labeler import label_soliton_state, label_trajectory

N_TAU = 512
DEFAULT_PARAMS = {
    'power_floor': 1e-6, 'contrast_cw': 2.0, 'contrast_high': 8.0,
    'entropy_chaotic': 0.5, 'crystal_cv': 0.1, 'sech2_r2': 0.95,
    'peak_prominence': 0.3, 'peak_width': 2.0,
}

def make_sech2_field(n_tau=N_TAU, center=None, width=20.0, amplitude=1.0):
    """Analytical single sech² soliton field."""
    if center is None:
        center = n_tau // 2
    x = np.arange(n_tau, dtype=float)
    envelope = amplitude / np.cosh((x - center) / width)
    return envelope.astype(np.complex64)

def test_single_soliton_labeled_correctly():
    E = make_sech2_field(amplitude=100.0, width=15.0)
    label = label_soliton_state(E, DEFAULT_PARAMS)
    assert label == 6, f"Expected 6 (single soliton), got {label}"

def test_flat_field_labeled_subthreshold():
    E = np.ones(N_TAU, dtype=np.complex64) * 1e-4
    label = label_soliton_state(E, DEFAULT_PARAMS)
    assert label == 0, f"Expected 0 (sub-threshold), got {label}"

def test_two_soliton_labeled_multi():
    E = (make_sech2_field(center=N_TAU//4, amplitude=100.0, width=15.0) +
         make_sech2_field(center=3*N_TAU//4, amplitude=100.0, width=15.0))
    label = label_soliton_state(E, DEFAULT_PARAMS)
    assert label == 4, f"Expected 4 (multi-soliton), got {label}"

def test_label_trajectory_shape():
    E_hist = np.stack([make_sech2_field(amplitude=100.0) for _ in range(20)])
    labels = label_trajectory(E_hist)
    assert labels.shape == (20,)
    assert labels.dtype == np.int32
    assert np.all(labels == 6)
