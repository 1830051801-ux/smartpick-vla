"""Receding-horizon controllers for learned policy evaluation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: np.ndarray
    inference_latency_ms: float
    replanned: bool
    base_action: np.ndarray | None = None
    residual_action: np.ndarray | None = None


class LearnedPolicyController:
    """Execute single actions or cached VLA chunks with regular replanning."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device = "cpu",
        replan_interval: int = 1,
    ) -> None:
        if replan_interval < 1:
            raise ValueError("replan_interval must be positive")
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.replan_interval = replan_interval
        self._cached: deque[np.ndarray] = deque()
        self._steps_since_plan = 0

    def reset(self) -> None:
        self._cached.clear()
        self._steps_since_plan = 0

    @torch.no_grad()
    def act(self, observation: dict[str, Any]) -> ActionDecision:
        replan = not self._cached or self._steps_since_plan >= self.replan_interval
        latency_ms = 0.0
        if replan:
            rgb = torch.from_numpy(observation["rgb"]).permute(2, 0, 1).unsqueeze(0).to(self.device)
            state = torch.from_numpy(observation["robot_state"]).unsqueeze(0).to(self.device)
            started = time.perf_counter()
            output = self.model(rgb, [observation["instruction"]], state)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if output.ndim == 2:
                chunk = output[0].detach().cpu().numpy()[None, :]
            elif output.ndim == 3:
                chunk = output[0].detach().cpu().numpy()
            else:
                raise ValueError("policy output must have shape [B,A] or [B,H,A]")
            self._cached = deque(np.asarray(chunk, dtype=np.float32))
            self._steps_since_plan = 0
        action = self._cached.popleft()
        self._steps_since_plan += 1
        return ActionDecision(np.clip(action, -1.0, 1.0), latency_ms, replan)
