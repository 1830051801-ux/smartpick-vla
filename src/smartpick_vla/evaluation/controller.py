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


class TemporalPolicyController:
    """Receding-horizon controller that preserves episode-safe observation history."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device = "cpu",
        replan_interval: int = 1,
    ) -> None:
        if replan_interval < 1:
            raise ValueError("replan_interval must be positive")
        config = getattr(model, "config", None)
        observation_horizon = getattr(config, "observation_horizon", None)
        if not isinstance(observation_horizon, int) or observation_horizon < 1:
            raise TypeError("temporal policy must expose a positive observation_horizon")
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.replan_interval = replan_interval
        self.observation_horizon = observation_horizon
        self._cached: deque[np.ndarray] = deque()
        self._rgb_history: deque[np.ndarray] = deque(maxlen=observation_horizon)
        self._state_history: deque[np.ndarray] = deque(maxlen=observation_horizon)
        self._steps_since_plan = 0

    def reset(self) -> None:
        self._cached.clear()
        self._rgb_history.clear()
        self._state_history.clear()
        self._steps_since_plan = 0

    def _append_observation(self, observation: dict[str, Any]) -> None:
        rgb = np.asarray(observation["rgb"], dtype=np.uint8)
        state = np.asarray(observation["robot_state"], dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("temporal controller requires RGB observations shaped [H,W,3]")
        if state.ndim != 1:
            raise ValueError("temporal controller requires a one-dimensional robot state")
        self._rgb_history.append(rgb.copy())
        self._state_history.append(state.copy())

    def _history_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._rgb_history or not self._state_history:
            raise RuntimeError("temporal controller needs an observation before inference")
        latest_rgb = self._rgb_history[-1]
        latest_state = self._state_history[-1]
        history_rgb = np.zeros((self.observation_horizon, *latest_rgb.shape), dtype=np.uint8)
        history_state = np.zeros(
            (self.observation_horizon, latest_state.shape[0]), dtype=np.float32
        )
        history_mask = np.zeros(self.observation_horizon, dtype=bool)
        first_valid = self.observation_horizon - len(self._rgb_history)
        for index, (rgb, state) in enumerate(
            zip(self._rgb_history, self._state_history, strict=True), start=first_valid
        ):
            history_rgb[index] = rgb
            history_state[index] = state
            history_mask[index] = True
        return (
            torch.from_numpy(history_rgb).permute(0, 3, 1, 2).unsqueeze(0).to(self.device),
            torch.from_numpy(history_state).unsqueeze(0).to(self.device),
            torch.from_numpy(history_mask).unsqueeze(0).to(self.device),
        )

    @torch.no_grad()
    def act(self, observation: dict[str, Any]) -> ActionDecision:
        self._append_observation(observation)
        replan = not self._cached or self._steps_since_plan >= self.replan_interval
        latency_ms = 0.0
        if replan:
            rgb_history, state_history, history_mask = self._history_tensors()
            started = time.perf_counter()
            output = self.model(
                rgb_history=rgb_history,
                instruction=[observation["instruction"]],
                robot_state_history=state_history,
                history_mask=history_mask,
            )
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
