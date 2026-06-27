"""Physics-informed RNN for TFLN soliton state control.

Two-phase training strategy:
  Phase 1 — PIRNNObserver: trained on synthetic detuning sweep trajectories to classify
    the current soliton state and forecast dynamics (mean + uncertainty) under no intervention.
    The observer has NO action input and does NOT itself perform MPC. It is the sensor in a
    deterministic predictive-triggering / proportional-feedback loop implemented in mpc/, which
    handles single-soliton access and drift holding from chaotic starts.
  Phase 2 — PIRNNController: wraps a pretrained PIRNNObserver (optionally frozen) and adds
    action-conditioned heads trained on supplementary switching trajectories. Provides the
    learned action->state map for gradient-based MPC over the prediction horizon. Valid ONLY
    for downward transitions (N -> N-1) via controlled annihilation; it does NOT navigate to
    arbitrary states and does NOT perform learned drift-holding (holding is a deterministic
    mpc/ law on the observer forecast). Upward nucleation requires auxiliary actuators and is
    out of scope.
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
"""Canonical index for single-soliton state. Imported by mpc/ and data/ modules; do not remove."""


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
    delta_cmd_max: float = 1.0  # TODO(Lucas): set to the REAL |Δ_max| actuator limit, in delta_cmd's native (normalized) units, before any MPC run. The tanh saturation wall is placed here; a wrong value either lets gradient ascent extrapolate past trained actions or clips effective ones.
    logvar_min: float = -10.0   # clamp range for heteroscedastic forecast log-variance (numerical stability)
    logvar_max: float = 5.0


class PIRNNObserver(nn.Module):
    """Observes the current soliton state from a P_trans window and forecasts dynamics under no intervention. Phase 1 model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Fuse encoder state with the physics POSTERIOR (phys_state_refined, dim = phys_state_dim),
        # not ctx. Using ctx (context_proj_dim) here dilutes physics-state influence over 200 GRU steps;
        # ctx information is already present in h_final (concatenated at every encoder step) and in the
        # context-only phys_state prior, so it is not lost.
        self.obs_fusion_dim = config.gru_hidden + config.phys_state_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(config.n_context, config.context_proj_dim),
            nn.LayerNorm(config.context_proj_dim),
            nn.GELU(),
            nn.Linear(config.context_proj_dim, config.context_proj_dim),
        )
        self.context_encoder.__doc__ = (
            "Projects physical operating point scalars into a fixed-dim embedding shared by encoder and decoders."
        )

        # phys_state semantic convention (FIXED HERE; consumed by losses.py thermal term):
        #   index 0 -> delta_omega_est  (signed, normalized; bounded by tanh)
        #   index 1 -> N_soliton_est     (>= 0; softplus)
        #   index 2 -> Delta_T_est       (>= 0; softplus)
        #   index 3 -> U_int_est         (>= 0; softplus)  <-- used by thermal-rate consistency loss
        # These are NORMALIZED latent estimates; physical units are applied in losses.py.
        self.physics_estimator = nn.Sequential(
            nn.Linear(config.n_context, 64),
            nn.GELU(),
            nn.Linear(64, config.phys_state_dim),
        )
        self.physics_estimator.__doc__ = (
            "Estimates a latent physical-state PRIOR from context scalars only (no observation of P_trans). "
            "Static across the forecast horizon. Grounds the GRU layer-0 initial hidden state; refined into a "
            "posterior by phys_state_refiner."
        )

        self.phys_state_refiner = nn.Sequential(
            nn.Linear(config.phys_state_dim + config.gru_hidden, 64),
            nn.GELU(),
            nn.Linear(64, config.phys_state_dim),
        )
        nn.init.zeros_(self.phys_state_refiner[-1].weight)
        nn.init.zeros_(self.phys_state_refiner[-1].bias)
        self.phys_state_refiner.__doc__ = (
            "Refines the context-only phys_state prior into a POSTERIOR using h_final (i.e. using the observed "
            "P_trans window), so U_int_est reflects the actual current dynamical state rather than only the nominal "
            "operating point (which is identical at every point in a sweep). Zero-initialized residual: "
            "phys_state_refined == phys_state at init. Does NOT feed h0 (would be circular)."
        )

        self.h0_projector = nn.Linear(config.phys_state_dim, config.gru_hidden)
        self.h0_projector.__doc__ = (
            "Projects the physics-state PRIOR into GRU layer-0 initial hidden state; upper layers cold-started at zero."
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
        self.detuning_out = nn.Linear(config.decoder_hidden, 2)  # [mean, logvar]
        self.detuning_gru_cell.__doc__ = (
            "Forecasts the no-intervention detuning trajectory (mean + heteroscedastic logvar). Mean is fed "
            "autoregressively (detached); logvar drives the deterministic mpc/ hold-law gain schedule and the "
            "low-confidence fallback. Used for thermal-drift detection. NOT an MPC actuator — observer has no action input."
        )

        self.ptrans_gru_cell = nn.GRUCell(input_size=1, hidden_size=config.decoder_hidden)
        self.ptrans_out = nn.Linear(config.decoder_hidden, 2)  # [mean, logvar]
        self.ptrans_gru_cell.__doc__ = (
            "Forecasts the no-intervention P_trans trajectory (mean + heteroscedastic logvar); auxiliary "
            "reconstruction head consumed by losses.py (NLL)."
        )

    @staticmethod
    def _bound_phys_state(raw: torch.Tensor) -> torch.Tensor:
        # idx0 signed (tanh); idx1..3 physically non-negative (softplus). See convention above.
        delta_omega = torch.tanh(raw[:, 0:1])
        nonneg = nn.functional.softplus(raw[:, 1:4])
        return torch.cat([delta_omega, nonneg], dim=-1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        assert x.ndim == 3 and x.size(1) == self.config.W and x.size(2) == 1
        assert context.ndim == 2 and context.size(1) == self.config.n_context
        assert x.size(0) == context.size(0), (
            f"batch mismatch: x={x.size(0)} context={context.size(0)}"
        )

        ctx = self.context_encoder(context)
        phys_state_raw = self.physics_estimator(context)
        phys_state = self._bound_phys_state(phys_state_raw)

        h0_layer0 = torch.tanh(self.h0_projector(phys_state)).unsqueeze(0)
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

        # Posterior physics state: bound(prior_raw + zero-init residual(prior, h_final)).
        # At init the residual is zero, so phys_state_refined == phys_state.
        refiner_in = torch.cat([phys_state, h_final], dim=-1)
        phys_state_refined = self._bound_phys_state(phys_state_raw + self.phys_state_refiner(refiner_in))

        fused_obs = torch.cat([h_final, phys_state_refined], dim=-1)
        logits = self.classifier(fused_obs)

        decoder_h0 = torch.tanh(self.decoder_seed(fused_obs))

        lvmin, lvmax = self.config.logvar_min, self.config.logvar_max

        h = decoder_h0
        inp = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        means, logvars = [], []
        for _ in range(self.config.H):
            h = self.detuning_gru_cell(inp, h)
            out = self.detuning_out(h)
            mean = out[:, 0:1]
            logvar = out[:, 1:2].clamp(lvmin, lvmax)
            means.append(mean)
            logvars.append(logvar)
            inp = mean.detach()
        pred_detuning = torch.cat(means, dim=1)
        pred_detuning_logvar = torch.cat(logvars, dim=1)

        h = decoder_h0
        inp = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        means, logvars = [], []
        for _ in range(self.config.H):
            h = self.ptrans_gru_cell(inp, h)
            out = self.ptrans_out(h)
            mean = out[:, 0:1]
            logvar = out[:, 1:2].clamp(lvmin, lvmax)
            means.append(mean)
            logvars.append(logvar)
            inp = mean.detach()
        pred_p_trans = torch.cat(means, dim=1)
        pred_p_trans_logvar = torch.cat(logvars, dim=1)

        return {
            "logits": logits,
            "pred_detuning": pred_detuning,
            "pred_detuning_logvar": pred_detuning_logvar,
            "pred_P_trans": pred_p_trans,
            "pred_P_trans_logvar": pred_p_trans_logvar,
            "phys_state": phys_state,
            "phys_state_refined": phys_state_refined,
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
        # NOT zero-initialized: a zero-init output layer makes the correction identically zero and severs the
        # gradient delta_cmd -> act_pred_detuning (required nonzero). Under the sole-entry-point invariant
        # (delta_cmd may enter only via action_encoder -> fused_act -> decoder seed), a residual that is exactly
        # zero at init is mutually exclusive with that gradient, so default (small) init is used instead and the
        # correction is learned down toward the switching dataset by the loss.
        self.act_detuning_gru_cell.__doc__ = (
            "Forecasts the action-conditioned CORRECTION to the observer's no-intervention detuning forecast, "
            "i.e. act_pred_detuning = observer pred_detuning.detach() + correction. Default-initialized so the "
            "gradient delta_cmd -> correction stays nonzero; the magnitude is regularized by the switching-dataset loss."
        )

        # Classifier reads the decoder's final hidden state so the state prediction is coupled to the actual
        # unrolled action-conditioned trajectory, not read independently of it.
        self.act_classifier = nn.Sequential(
            nn.Linear(self.act_fusion_dim + config.decoder_hidden, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.n_states),
        )
        self.act_classifier.__doc__ = (
            "Predicts soliton state at horizon H under proposed action, conditioned on BOTH fused_act and the "
            "detuning decoder's final hidden state. Primary MPC optimization target. Downward transitions only."
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
        assert delta_cmd.size(0) == target_state.size(0) == x.size(0) == context.size(0), (
            "batch mismatch among x, context, delta_cmd, target_state"
        )

        if self._observer_frozen:
            with torch.no_grad():
                observer_out = self.observer(x, context)
        else:
            observer_out = self.observer(x, context)

        h_final = observer_out["h_final"]
        ctx = observer_out["ctx"]
        # Baseline is ALWAYS detached: the residual must never become a backdoor gradient path into observer
        # params. Observer terms are supervised directly on observer_out in losses.py. Holds frozen or unfrozen.
        pred_detuning_baseline = observer_out["pred_detuning"].detach()

        # Saturate the proposed action to the physical actuator range BEFORE encoding, so MPC gradient ascent
        # cannot walk delta_cmd into action regions absent from training. Gradient still flows (tanh is smooth).
        dmax = self.config.delta_cmd_max
        delta_cmd_bounded = dmax * torch.tanh(delta_cmd / dmax)

        action_emb = self.action_encoder(delta_cmd_bounded)
        target_emb = self.target_state_embed(target_state)

        fused_act = torch.cat([h_final, ctx, action_emb, target_emb], dim=-1)
        decoder_h0_act = torch.tanh(self.act_decoder_seed(fused_act))

        h = decoder_h0_act
        inp = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        deltas = []
        for _ in range(self.config.H):
            h = self.act_detuning_gru_cell(inp, h)
            delta = self.act_detuning_out(h)
            deltas.append(delta)
            inp = delta.detach()
        detuning_correction = torch.cat(deltas, dim=1)
        act_pred_detuning = pred_detuning_baseline + detuning_correction

        # h is the decoder's final hidden state after H steps; couples act_logits to the unrolled trajectory.
        act_logits = self.act_classifier(torch.cat([fused_act, h], dim=-1))

        return {
            "logits": observer_out["logits"],
            "pred_detuning": observer_out["pred_detuning"],
            "pred_detuning_logvar": observer_out["pred_detuning_logvar"],
            "pred_P_trans": observer_out["pred_P_trans"],
            "pred_P_trans_logvar": observer_out["pred_P_trans_logvar"],
            "phys_state": observer_out["phys_state"],
            "phys_state_refined": observer_out["phys_state_refined"],
            "act_pred_detuning": act_pred_detuning,
            "act_detuning_correction": detuning_correction,
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
                count = sum(p.numel() for p in module.parameters())
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
    torch.manual_seed(0)
    config = ModelConfig()
    observer = PIRNNObserver(config)
    observer.eval()  # deterministic reference forecast (no GRU dropout); matches the frozen-eval observer the controller uses

    B = 8
    x = torch.randn(B, config.W, 1)
    context = torch.randn(B, config.n_context)

    out_obs = observer(x, context)
    for k, v in out_obs.items():
        print(k, tuple(v.shape))

    assert out_obs["logits"].shape == (B, config.n_states)
    assert out_obs["pred_detuning"].shape == (B, config.H)
    assert out_obs["pred_detuning_logvar"].shape == (B, config.H)
    assert out_obs["pred_P_trans"].shape == (B, config.H)
    assert out_obs["pred_P_trans_logvar"].shape == (B, config.H)
    assert out_obs["phys_state"].shape == (B, config.phys_state_dim)
    assert out_obs["phys_state_refined"].shape == (B, config.phys_state_dim)
    assert out_obs["h_final"].shape == (B, config.gru_hidden)
    assert out_obs["ctx"].shape == (B, config.context_proj_dim)

    # phys_state non-negativity convention (idx 1..3) and bounded delta_omega (idx 0)
    assert (out_obs["phys_state"][:, 1:] >= 0).all()
    assert out_obs["phys_state"][:, 0].abs().le(1.0 + 1e-5).all()

    # At init the refiner residual is zero -> posterior equals prior.
    assert torch.allclose(out_obs["phys_state_refined"], out_obs["phys_state"], atol=1e-5), \
        "phys_state_refined must equal phys_state at init (zero-init refiner)"

    controller = PIRNNController(config, observer, freeze_observer=True)
    target_state = torch.randint(0, config.n_states, (B,))

    # --- residual decomposition: act_pred_detuning == observer baseline + learned correction ---
    # (The correction head is intentionally NOT zero-initialized; a zero-init residual would sever the
    #  required delta_cmd -> act_pred_detuning gradient under the sole-entry-point invariant. So instead of
    #  asserting correction==0 at init, verify the decomposition identity, which is the real invariant.)
    with torch.no_grad():
        out_zero = controller(x, context, delta_cmd=torch.zeros(B, 1), target_state=target_state)
        assert torch.allclose(
            out_zero["act_pred_detuning"] - out_zero["act_detuning_correction"],
            out_obs["pred_detuning"], atol=1e-5,
        ), "act_pred_detuning must equal observer pred_detuning + correction (residual decomposition)"

    # --- shapes ---
    delta_cmd = torch.randn(B, 1, requires_grad=True)
    out_ctrl = controller(x, context, delta_cmd=delta_cmd, target_state=target_state)
    for k, v in out_ctrl.items():
        print(k, tuple(v.shape))
    assert out_ctrl["act_pred_detuning"].shape == (B, config.H)
    assert out_ctrl["act_detuning_correction"].shape == (B, config.H)
    assert out_ctrl["act_logits"].shape == (B, config.n_states)

    # --- gradient path: delta_cmd -> act_logits ---
    delta_a = torch.randn(B, 1, requires_grad=True)
    o = controller(x, context, delta_cmd=delta_a, target_state=target_state)
    o["act_logits"].mean().backward()
    assert delta_a.grad is not None and delta_a.grad.norm().item() > 0.0, \
        "gradient path delta_cmd -> act_logits is broken"
    print("act_logits  delta_cmd.grad.norm:", delta_a.grad.norm().item())

    # --- gradient path: delta_cmd -> act_pred_detuning ---
    delta_b = torch.randn(B, 1, requires_grad=True)
    o = controller(x, context, delta_cmd=delta_b, target_state=target_state)
    o["act_pred_detuning"].sum().backward()
    assert delta_b.grad is not None and delta_b.grad.norm().item() > 0.0, \
        "gradient path delta_cmd -> act_pred_detuning is broken"
    print("act_detune  delta_cmd.grad.norm:", delta_b.grad.norm().item())

    # --- frozen observer receives no gradient ---
    assert all(not p.requires_grad for p in controller.observer.parameters())

    # --- saturation: huge delta_cmd does not explode the correction ---
    with torch.no_grad():
        big = controller(x, context, delta_cmd=torch.full((B, 1), 1e3), target_state=target_state)
        assert torch.isfinite(big["act_pred_detuning"]).all(), "saturation failed: non-finite forecast"

    observer.count_parameters(verbose=True)
    controller.count_parameters(verbose=True)
    print("ALL CHECKS PASSED")
