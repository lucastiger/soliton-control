"""Physics-informed RNN for TFLN soliton state control.

Two-phase training strategy:
  Phase 1 — PIRNNObserver: trained on synthetic detuning sweep trajectories to classify
    the current soliton state and forecast dynamics under no intervention. Deployed in
    the Phase 1 MPC loop for single-soliton access and stabilization from chaotic starts.
  Phase 2 — PIRNNController: wraps a pretrained PIRNNObserver (optionally frozen) and
    adds action-conditioned heads trained on supplementary switching trajectories. Extends
    the MPC to navigate toward arbitrary target soliton states via controlled annihilation.
    Valid only for downward transitions (N → N-1); upward nucleation requires auxiliary
    actuators and is outside scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

STATE_INDICES: dict[str, int] = {
    "off": 0,
    "CW": 1,
    "MI": 2,
    "chaotic": 3,
    "multi_soliton": 4,
    "soliton_crystal": 5,
    "single_soliton": 6,
}
SINGLE_SOLITON_IDX: int = STATE_INDICES["single_soliton"]


@dataclass
class ModelConfig:
    W: int = 200
    H: int = 50
    n_context: int = 4
    n_states: int = 7
    context_proj_dim: int = 32
    phys_state_dim: int = 4
    gru_hidden: int = 256
    gru_layers: int = 3
    decoder_hidden: int = 128
    action_embed_dim: int = 32
    target_embed_dim: int = 16
    dropout: float = 0.1


class PIRNNObserver(nn.Module):
    """Observes the current soliton state from a P_trans window and forecasts dynamics under no intervention. Phase 1 model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.obs_fusion_dim = config.gru_hidden + config.context_proj_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(config.n_context, config.context_proj_dim),
            nn.LayerNorm(config.context_proj_dim),
            nn.GELU(),
            nn.Linear(config.context_proj_dim, config.context_proj_dim),
        )
        self.context_encoder.__doc__ = (
            "Projects physical operating point scalars into a fixed-dim embedding shared by encoder and decoders."
        )

        self.physics_estimator = nn.Sequential(
            nn.Linear(config.n_context, 64),
            nn.GELU(),
            nn.Linear(64, config.phys_state_dim),
        )
        self.physics_estimator.__doc__ = (
            "Estimates latent physical state from context scalars to ground the GRU initial hidden state in physics-derived priors."
        )

        self.h0_projector = nn.Linear(config.phys_state_dim, config.gru_hidden)
        self.h0_projector.__doc__ = (
            "Projects physics state estimate into GRU layer-0 initial hidden state; upper layers cold-started at zero."
        )

        self.gru_encoder = nn.GRU(
            input_size=1 + config.context_proj_dim,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            dropout=config.dropout,
            batch_first=True,
        )
        self.gru_encoder.__doc__ = (
            "Processes P_trans window sequentially with physical context concatenated at each step."
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.obs_fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.n_states),
        )
        self.classifier.__doc__ = "Classifies current soliton state from fused encoder representation."

        self.decoder_seed = nn.Linear(self.obs_fusion_dim, config.decoder_hidden)
        self.decoder_seed.__doc__ = "Seeds both prediction decoder hidden states from the observer fusion vector."

        self.detuning_gru_cell = nn.GRUCell(input_size=1, hidden_size=config.decoder_hidden)
        self.detuning_out = nn.Linear(config.decoder_hidden, 1)
        self.detuning_gru_cell.__doc__ = (
            "Forecasts detuning trajectory under no intervention; used for thermal drift detection and Phase 1 MPC planning."
        )

        self.ptrans_gru_cell = nn.GRUCell(input_size=1, hidden_size=config.decoder_hidden)
        self.ptrans_out = nn.Linear(config.decoder_hidden, 1)
        self.ptrans_gru_cell.__doc__ = (
            "Forecasts P_trans trajectory under no intervention; auxiliary reconstruction head for physics consistency loss in losses.py."
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        assert x.ndim == 3 and x.size(1) == self.config.W and x.size(2) == 1
        assert context.ndim == 2 and context.size(1) == self.config.n_context

        ctx = self.context_encoder(context)
        phys_state = self.physics_estimator(context)

        h0_layer0 = self.h0_projector(phys_state).unsqueeze(0)
        h0_zeros = torch.zeros(
            self.config.gru_layers - 1,
            x.size(0),
            self.config.gru_hidden,
            device=h0_layer0.device,
            dtype=h0_layer0.dtype,
        )
        h0 = torch.cat([h0_layer0, h0_zeros], dim=0)

        ctx_expanded = ctx.unsqueeze(1).expand(-1, self.config.W, -1)
        gru_input = torch.cat([x, ctx_expanded], dim=-1)
        gru_out, _ = self.gru_encoder(gru_input, h0)
        h_final = gru_out[:, -1, :]

        fused_obs = torch.cat([h_final, ctx], dim=-1)
        logits = self.classifier(fused_obs)

        decoder_h0 = torch.tanh(self.decoder_seed(fused_obs))

        h = decoder_h0
        inp = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        preds = []
        for _ in range(self.config.H):
            h = self.detuning_gru_cell(inp, h)
            pred = self.detuning_out(h)
            preds.append(pred)
            inp = pred.detach()
        pred_detuning = torch.cat(preds, dim=1)

        h = decoder_h0
        inp = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        preds = []
        for _ in range(self.config.H):
            h = self.ptrans_gru_cell(inp, h)
            pred = self.ptrans_out(h)
            preds.append(pred)
            inp = pred.detach()
        pred_p_trans = torch.cat(preds, dim=1)

        return {
            "logits": logits,
            "pred_detuning": pred_detuning,
            "pred_P_trans": pred_p_trans,
            "phys_state": phys_state,
            "h_final": h_final,
            "ctx": ctx,
        }

    def predict_proba(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x, context)["logits"], dim=-1)

    def count_parameters(self, verbose: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if verbose:
            print("submodule_name | param_count")
            print("-" * 32)
            for name, module in self.named_children():
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"{name:<16} | {count}")
            print("-" * 32)
            print(f"{'TOTAL':<16} | {total}")
        return int(total)


class PIRNNController(nn.Module):
    """Wraps a pretrained PIRNNObserver and adds action-conditioned heads for target-state navigation. Phase 2 model. Valid for downward soliton transitions only."""

    def __init__(self, config: ModelConfig, observer: PIRNNObserver, freeze_observer: bool = True):
        super().__init__()
        self.config = config
        self.observer = observer
        self._observer_frozen = False
        self.act_fusion_dim = (
            config.gru_hidden
            + config.context_proj_dim
            + config.action_embed_dim
            + config.target_embed_dim
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(1, config.action_embed_dim),
            nn.LayerNorm(config.action_embed_dim),
            nn.GELU(),
            nn.Linear(config.action_embed_dim, config.action_embed_dim),
        )
        self.action_encoder.__doc__ = (
            "Encodes proposed detuning correction; sole gradient path from delta_cmd to MPC objective."
        )

        self.target_state_embed = nn.Embedding(config.n_states, config.target_embed_dim)
        self.target_state_embed.__doc__ = (
            "Encodes operator-specified target soliton state, specializing predictions toward that attractor basin."
        )

        self.act_decoder_seed = nn.Linear(self.act_fusion_dim, config.decoder_hidden)
        self.act_decoder_seed.__doc__ = "Seeds action-conditioned decoder; kept on autograd graph per Rule G5."

        self.act_detuning_gru_cell = nn.GRUCell(1, config.decoder_hidden)
        self.act_detuning_out = nn.Linear(config.decoder_hidden, 1)
        self.act_detuning_gru_cell.__doc__ = (
            "Forecasts detuning trajectory under proposed action; supports MPC planning over the prediction horizon."
        )

        self.act_classifier = nn.Sequential(
            nn.Linear(self.act_fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.n_states),
        )
        self.act_classifier.__doc__ = (
            "Predicts soliton state at horizon H under proposed action; primary MPC optimization target. Downward transitions only."
        )

        self.set_observer_frozen(freeze_observer)

    def set_observer_frozen(self, frozen: bool) -> None:
        for p in self.observer.parameters():
            p.requires_grad_(not frozen)
        self.observer.train(not frozen)
        self._observer_frozen = frozen

  def train(self, mode: bool = True) -> "PIRNNController":
    super().train(mode)
    if self._observer_frozen:
        self.observer.eval()
    return self

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        delta_cmd: torch.Tensor,
        target_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x is None or context is None or delta_cmd is None or target_state is None:
            raise ValueError("x, context, delta_cmd, and target_state are all required")

        assert delta_cmd.ndim == 2 and delta_cmd.size(1) == 1
        assert target_state.ndim == 1 and target_state.dtype == torch.long, (
            f"target_state must be 1D torch.long, got ndim={target_state.ndim}, dtype={target_state.dtype}"
        )
        assert target_state.ge(0).all() and target_state.lt(self.config.n_states).all(), (
            f"target_state values must be in [0, {self.config.n_states}), got min={target_state.min()}, max={target_state.max()}"
        )
        assert delta_cmd.size(0) == target_state.size(0) == x.size(0)

        if self._observer_frozen:
            with torch.no_grad():
                observer_out = self.observer(x, context)
        else:
            observer_out = self.observer(x, context)

        h_final = observer_out["h_final"]
        ctx = observer_out["ctx"]

        action_emb = self.action_encoder(delta_cmd)
        target_emb = self.target_state_embed(target_state)

        fused_act = torch.cat([h_final, ctx, action_emb, target_emb], dim=-1)
        decoder_h0_act = torch.tanh(self.act_decoder_seed(fused_act))

        h = decoder_h0_act
        inp = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        preds = []
        for _ in range(self.config.H):
            h = self.act_detuning_gru_cell(inp, h)
            pred = self.act_detuning_out(h)
            preds.append(pred)
            inp = pred.detach()
        act_pred_detuning = torch.cat(preds, dim=1)

        act_logits = self.act_classifier(fused_act)

        return {
            "logits": observer_out["logits"],
            "pred_detuning": observer_out["pred_detuning"],
            "pred_P_trans": observer_out["pred_P_trans"],
            "phys_state": observer_out["phys_state"],
            "act_pred_detuning": act_pred_detuning,
            "act_logits": act_logits,
        }

    def predict_action_proba(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        delta_cmd: torch.Tensor,
        target_state: torch.Tensor,
    ) -> torch.Tensor:
        return torch.softmax(self.forward(x, context, delta_cmd, target_state)["act_logits"], dim=-1)

    def count_parameters(self, verbose: bool = True) -> int:
        observer_total = sum(p.numel() for p in self.observer.parameters())
        controller_total = sum(
            p.numel() for n, p in self.named_parameters() if not n.startswith("observer.") and p.requires_grad
        )
        total = observer_total + controller_total

        if verbose:
            status = "frozen" if self._observer_frozen else "trainable"
            print(f"Observer parameters ({status})")
            print("submodule_name | param_count")
            print("-" * 32)
            for name, module in self.observer.named_children():
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"{name:<16} | {count}")
            print(f"{'OBSERVER TOTAL':<16} | {observer_total}")
            print()
            print("Controller-head parameters")
            print("submodule_name | param_count")
            print("-" * 32)
            for name, module in self.named_children():
                if name == "observer":
                    continue
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"{name:<16} | {count}")
            print(f"{'HEAD TOTAL':<16} | {controller_total}")
            print("-" * 32)
            print(f"{'COMBINED TOTAL':<16} | {total}")
        return int(total)


if __name__ == "__main__":
    config = ModelConfig()
    observer = PIRNNObserver(config)

    x = torch.randn(8, config.W, 1)
    context = torch.randn(8, config.n_context)

    out_obs = observer(x, context)
    for k, v in out_obs.items():
        print(k, tuple(v.shape))

    assert out_obs["logits"].shape == (8, config.n_states)
    assert out_obs["pred_detuning"].shape == (8, config.H)
    assert out_obs["pred_P_trans"].shape == (8, config.H)
    assert out_obs["phys_state"].shape == (8, config.phys_state_dim)
    assert out_obs["h_final"].shape == (8, config.gru_hidden)
    assert out_obs["ctx"].shape == (8, config.context_proj_dim)

    controller = PIRNNController(config, observer, freeze_observer=True)

    delta_cmd = torch.randn(8, 1, requires_grad=True)
    target_state = torch.randint(0, config.n_states, (8,))
    out_ctrl = controller(x, context, delta_cmd=delta_cmd, target_state=target_state)
    for k, v in out_ctrl.items():
        print(k, tuple(v.shape))

    assert out_ctrl["logits"].shape == (8, config.n_states)
    assert out_ctrl["pred_detuning"].shape == (8, config.H)
    assert out_ctrl["pred_P_trans"].shape == (8, config.H)
    assert out_ctrl["phys_state"].shape == (8, config.phys_state_dim)
    assert out_ctrl["act_pred_detuning"].shape == (8, config.H)
    assert out_ctrl["act_logits"].shape == (8, config.n_states)

    out_ctrl["act_logits"].mean().backward()
    assert delta_cmd.grad is not None, "delta_cmd.grad is None: gradient not reaching action input"
    grad_norm = delta_cmd.grad.norm().item()
    assert grad_norm > 0.0, f"delta_cmd.grad norm is zero ({grad_norm}): action gradient path is broken"
    print("delta_cmd.grad.norm:", grad_norm)

    assert all(not p.requires_grad for p in controller.observer.parameters())

    observer.count_parameters(verbose=True)
    controller.count_parameters(verbose=True)
