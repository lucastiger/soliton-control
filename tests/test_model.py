"""Fast CPU smoke tests for the PI-RNN training pipeline (model/train.py).

These exercise the trainer integration points against the LOCKED model/loss/dataloader
contracts using a tiny ModelConfig and synthetic tensors — no .h5 dataset is required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import pytest
import torch
from torch.nn.utils import clip_grad_norm_

from model.loss import LossConfig, LossScaler, PhysicsInformedLoss
from model.pi_rnn import SINGLE_SOLITON_IDX, ModelConfig, PIRNNObserver
from model.train import (
    ClassMetricAccumulator,
    Trainer,
    assemble_observer_targets,
    build_optimizer,
    build_scaler,
    build_scheduler,
    collapse_diagnostics,
    load_yaml,
    resolve_horizon_dt,
    to_namespace,
    train_controller,
)

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        W=10, H=4, n_context=4, n_states=7,
        gru_hidden=16, gru_layers=2, decoder_hidden=8, dropout=0.0,
        logvar_min=-10.0, logvar_max=5.0,
    )


def tiny_loss_config() -> LossConfig:
    return LossConfig(Gamma_th=0.1, tau_th=5e-6, horizon_dt=1e-7, thermal_norm="empirical")


def make_observer_batch(mcfg: ModelConfig, batch_size: int = 4) -> dict[str, torch.Tensor]:
    B = batch_size
    return {
        "x": torch.randn(B, mcfg.W, 1),
        "context": torch.randn(B, mcfg.n_context),
        "label": torch.randint(0, mcfg.n_states, (B,)),
        "future_detuning": torch.randn(B, mcfg.H),
        "future_P_trans": torch.randn(B, mcfg.H),
        "future_U_int": torch.randn(B, mcfg.H),
        "future_DeltaT": torch.randn(B, mcfg.H),
    }


# --------------------------------------------------------------------------- #
# 1. Target assembly / key rename
# --------------------------------------------------------------------------- #
def test_target_assembly_label_to_state_label():
    mcfg = tiny_model_config()
    batch = make_observer_batch(mcfg)
    targets = assemble_observer_targets(batch, DEVICE)

    assert "state_label" in targets
    assert "label" not in targets
    assert targets["state_label"].dtype == torch.long
    torch.testing.assert_close(targets["state_label"], batch["label"])

    model = PIRNNObserver(mcfg)
    out = model(batch["x"], batch["context"])
    loss = PhysicsInformedLoss(tiny_loss_config())(out, targets, mode="observer")
    assert loss["total"].ndim == 0
    assert torch.isfinite(loss["total"])


# --------------------------------------------------------------------------- #
# 2. One-step overfit: total loss decreases
# --------------------------------------------------------------------------- #
def test_one_step_overfit_decreases():
    torch.manual_seed(0)
    mcfg = tiny_model_config()
    model = PIRNNObserver(mcfg)
    loss_fn = PhysicsInformedLoss(tiny_loss_config())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    batches = [make_observer_batch(mcfg) for _ in range(2)]
    losses = []
    for step in range(20):
        batch = batches[step % 2]
        out = model(batch["x"], batch["context"])
        targets = assemble_observer_targets(batch, DEVICE)
        total = loss_fn(out, targets, mode="observer")["total"]
        opt.zero_grad(set_to_none=True)
        total.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(total.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# --------------------------------------------------------------------------- #
# 3. Grad-clip path: finite grads and finite clip norm
# --------------------------------------------------------------------------- #
def test_grad_clip_finite():
    mcfg = tiny_model_config()
    model = PIRNNObserver(mcfg)
    loss_fn = PhysicsInformedLoss(tiny_loss_config())
    batch = make_observer_batch(mcfg)

    out = model(batch["x"], batch["context"])
    targets = assemble_observer_targets(batch, DEVICE)
    loss_fn(out, targets, mode="observer")["total"].backward()

    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    total_norm = clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(total_norm)


# --------------------------------------------------------------------------- #
# 4. Classification metrics
# --------------------------------------------------------------------------- #
def test_class_metrics_shapes_and_bounds():
    acc = ClassMetricAccumulator(n_states=7, focus_idx=SINGLE_SOLITON_IDX)
    logits = torch.randn(40, 7)
    labels = torch.randint(0, 7, (40,))
    acc.update(logits, labels)
    m = acc.compute()

    assert len(m["per_class_recall"]) == 7
    assert 0.0 <= m["accuracy"] <= 1.0
    ss = m["single_soliton_recall"]
    assert ss is None or 0.0 <= ss <= 1.0


# --------------------------------------------------------------------------- #
# 5. Collapse diagnostic is a cross-batch (not within-trajectory) variance
# --------------------------------------------------------------------------- #
def test_collapse_diagnostic_batch_dim_variance():
    mcfg = tiny_model_config()
    model = PIRNNObserver(mcfg).eval()
    batch = make_observer_batch(mcfg, batch_size=8)
    with torch.no_grad():
        out = model(batch["x"], batch["context"])

    diag = collapse_diagnostics(out, logvar_max=mcfg.logvar_max)
    var = diag["final_step_detuning_var"]
    assert var >= 0.0 and torch.isfinite(torch.tensor(var))

    # It must be the variance ACROSS the batch dim at the final horizon step.
    expected = out["pred_detuning"][:, -1].var(unbiased=False).item()
    assert abs(var - expected) < 1e-6
    assert 0.0 <= diag["frac_logvar_pinned"] <= 1.0


# --------------------------------------------------------------------------- #
# 6. LossScaler: reference and regularizer lambdas are fixed; balanced ones move
# --------------------------------------------------------------------------- #
def test_loss_scaler_invariants():
    cfg = tiny_loss_config()
    before_class = cfg.lambda_class_obs
    before_thermal = cfg.lambda_thermal_obs

    scaler = LossScaler(
        cfg, reference_key="class_obs",
        balanced_map={"detune_obs": "lambda_detune_obs", "pred_obs": "lambda_pred_obs"},
        period=1,
    )
    for _ in range(3):
        scaler.update({
            "class_obs": torch.tensor(1.0),
            "detune_obs": torch.tensor(4.0),
            "pred_obs": torch.tensor(2.0),
        })
    scaler.maybe_rescale(epoch=1)

    assert cfg.lambda_class_obs == before_class       # reference fixed (FN-1)
    assert cfg.lambda_thermal_obs == before_thermal   # regularizer never balanced
    assert cfg.lambda_detune_obs == pytest.approx(0.25)   # 1 * mean(ref)/mean(detune) = 1/4
    assert cfg.lambda_pred_obs == pytest.approx(0.5)      # 1 * 1/2


# --------------------------------------------------------------------------- #
# 6b. build_scaler yields a FLOAT clamp that survives a real rescale
# --------------------------------------------------------------------------- #
def test_build_scaler_clamp_is_float_and_rescales():
    # PyYAML parses an unsigned-exponent literal (e.g. 1.0e3) as a string; build_scaler
    # must coerce so LossScaler.maybe_rescale's min/max clamp does not raise TypeError.
    cfg = to_namespace(load_yaml("config/training_config.yaml"))
    loss_cfg = tiny_loss_config()
    scaler = build_scaler(cfg, loss_cfg)
    assert scaler is not None
    assert all(isinstance(c, float) for c in scaler.clamp)

    # A rescale that would clamp high must not raise.
    for _ in range(3):
        scaler.update({
            "class_obs": torch.tensor(1.0),
            "detune_obs": torch.tensor(1e-6),   # huge ref/comp ratio -> wants to clamp to hi
            "pred_obs": torch.tensor(1.0),
        })
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        scaler.maybe_rescale(epoch=scaler.period)
    assert loss_cfg.lambda_detune_obs <= scaler.clamp[1]


# --------------------------------------------------------------------------- #
# 7. Phase-2 guard
# --------------------------------------------------------------------------- #
def test_phase2_controller_guard_raises():
    cfg = SimpleNamespace(phase2_controller=SimpleNamespace(switching_h5_path=None))
    with pytest.raises(NotImplementedError):
        train_controller(cfg, DEVICE)


# --------------------------------------------------------------------------- #
# 8. Trainer smoke test (full epoch loop, checkpoint + jsonl logging)
# --------------------------------------------------------------------------- #
def _tiny_train_cfg(tmp_path) -> SimpleNamespace:
    return to_namespace({
        "experiment": {"name": "smoke", "seed": 0, "out_dir": str(tmp_path)},
        "optim": {"lr": 1e-3, "weight_decay": 0.0, "epochs": 2, "warmup_epochs": 0, "grad_clip": 1.0},
        "train": {
            "early_stop_metric": "val_total_loss", "early_stop_patience": 10,
            "checkpoint_metric": "val_accuracy", "checkpoint_every": 5,
            "amp": False, "device": "cpu",
        },
        "model": {"logvar_max": 5.0},
    })


def test_trainer_smoke(tmp_path):
    torch.manual_seed(0)
    mcfg = tiny_model_config()
    cfg = _tiny_train_cfg(tmp_path)

    model = PIRNNObserver(mcfg)
    loss_fn = PhysicsInformedLoss(tiny_loss_config())
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)

    train_loader = [make_observer_batch(mcfg) for _ in range(2)]
    val_loader = [make_observer_batch(mcfg)]

    trainer = Trainer(
        model, opt, sched, loss_fn, train_loader, val_loader, cfg, DEVICE,
        mode="observer", observer_frozen=False, class_weights=None, scaler=None,
    )
    best = trainer.fit()

    run_dir = tmp_path / "smoke"
    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(lines) == 2
    assert "val" in lines[0] and "accuracy" in lines[0]["val"]
    assert len(lines[0]["val"]["per_class_recall"]) == 7

    assert (run_dir / "best_model.pt").exists()
    assert (run_dir / "config.snapshot.yaml").exists()
    assert "val_accuracy" in best


# --------------------------------------------------------------------------- #
# 9. resolve_horizon_dt: explicit override / derived / fallback precedence
# --------------------------------------------------------------------------- #
def test_resolve_horizon_dt_explicit():
    # Explicit non-null number is returned verbatim, even if the h5 path is bogus.
    dt, prov = resolve_horizon_dt(2.5e-7, "/does/not/exist.h5", fsr_hz=2e11)
    assert dt == 2.5e-7
    assert prov == "explicit_config"


def test_resolve_horizon_dt_derived(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as f:
        md = f.create_group("metadata")
        md.attrs["snapshot_interval"] = 10
    dt, prov = resolve_horizon_dt(None, str(path), fsr_hz=2e11)
    assert dt == pytest.approx(5e-11)           # 10 / 2e11
    assert prov.startswith("derived")
    assert "snapshot_interval=10" in prov


def test_resolve_horizon_dt_fallback_missing_path():
    with pytest.warns(RuntimeWarning):
        dt, prov = resolve_horizon_dt(None, "/does/not/exist.h5", fsr_hz=2e11)
    assert dt == pytest.approx(5e-11)           # 10 / 2e11, NOT the old 1e-7
    assert dt != 1e-7
    assert prov.startswith("fallback")


def test_resolve_horizon_dt_fallback_no_attr(tmp_path):
    # h5 exists but has no metadata['snapshot_interval'] (older dataset) -> warn + fallback.
    path = tmp_path / "no_meta.h5"
    with h5py.File(path, "w") as f:
        f.create_group("sim_0")
    with pytest.warns(RuntimeWarning):
        dt, prov = resolve_horizon_dt(None, str(path), fsr_hz=2e11)
    assert dt == pytest.approx(5e-11)
    assert prov.startswith("fallback")


# --------------------------------------------------------------------------- #
# 10. frac_logvar_pinned: pinned tail -> 1.0; normal init -> in [0, 1]
# --------------------------------------------------------------------------- #
def test_frac_logvar_pinned_saturated_and_normal():
    mcfg = tiny_model_config()

    # Force the detuning logvar head to saturate: zero the weight and set the logvar bias
    # (output index 1) far above logvar_max, so every horizon step clamps to logvar_max.
    pinned = PIRNNObserver(mcfg).eval()
    with torch.no_grad():
        pinned.detuning_out.weight.zero_()
        pinned.detuning_out.bias[1] = 1e6
    batch = make_observer_batch(mcfg, batch_size=8)
    with torch.no_grad():
        out = pinned(batch["x"], batch["context"])
    diag = collapse_diagnostics(out, logvar_max=mcfg.logvar_max)
    assert diag["frac_logvar_pinned"] == 1.0

    # Normal init: fraction is a valid probability.
    torch.manual_seed(0)
    normal = PIRNNObserver(mcfg).eval()
    with torch.no_grad():
        out2 = normal(batch["x"], batch["context"])
    d2 = collapse_diagnostics(out2, logvar_max=mcfg.logvar_max)
    assert 0.0 <= d2["frac_logvar_pinned"] <= 1.0
