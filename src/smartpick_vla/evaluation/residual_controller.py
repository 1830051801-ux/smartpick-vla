"""Deterministic benchmark adapter for a frozen VLA plus residual SAC."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smartpick_vla.evaluation.controller import ActionDecision, LearnedPolicyController
from smartpick_vla.training.checkpoint import load_checkpoint_payload
from smartpick_vla.training.imitation import load_trained_policy
from smartpick_vla.training.residual_sac import ResidualSAC, ResidualSACConfig


class ResidualPolicyController:
    """Compose a deterministic bounded residual with a frozen base action."""

    def __init__(
        self,
        base_checkpoint: str | Path,
        residual_checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        base_replan_interval: int = 4,
    ) -> None:
        self.device = torch.device(device)
        base_model, _ = load_trained_policy(base_checkpoint, device=self.device)
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        self.base_controller = LearnedPolicyController(
            base_model,
            device=self.device,
            replan_interval=base_replan_interval,
        )
        payload = load_checkpoint_payload(residual_checkpoint, map_location=self.device)
        config_payload = dict(payload.get("config", {}))
        if "hidden_dims" in config_payload:
            config_payload["hidden_dims"] = tuple(config_payload["hidden_dims"])
        if "residual_scale" in config_payload and isinstance(
            config_payload["residual_scale"], list
        ):
            config_payload["residual_scale"] = tuple(config_payload["residual_scale"])
        self.agent = ResidualSAC(ResidualSACConfig(**config_payload)).to(self.device)
        self.agent.load_state_dict(payload["model_state"])
        self.agent.eval()

    def reset(self) -> None:
        self.base_controller.reset()

    @torch.no_grad()
    def act(self, observation: dict[str, Any]) -> ActionDecision:
        state = torch.from_numpy(observation["robot_state"]).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        base_decision = self.base_controller.act(observation)
        base = torch.from_numpy(base_decision.action).unsqueeze(0).to(self.device)
        encoded = torch.cat((state, base), dim=-1)
        final = self.agent.select_action(encoded, base, deterministic=True)
        latency_ms = (time.perf_counter() - started) * 1000.0
        base_array = base[0].cpu().numpy().astype(np.float32)
        final_array = final[0].cpu().numpy().astype(np.float32)
        return ActionDecision(
            action=final_array,
            inference_latency_ms=latency_ms,
            replanned=True,
            base_action=base_array,
            residual_action=final_array - base_array,
        )
