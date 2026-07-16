"""Online residual-SAC collection around a permanently frozen VLA policy."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.envs.smartpick_env import SmartPickEnv
from smartpick_vla.evaluation.controller import LearnedPolicyController
from smartpick_vla.training.checkpoint import save_checkpoint
from smartpick_vla.training.imitation import load_trained_policy
from smartpick_vla.training.residual_sac import ResidualSAC, ResidualSACConfig
from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import sha256_file
from smartpick_vla.utils.seed import seed_everything


@dataclass(frozen=True, slots=True)
class ResidualOnlineConfig:
    seed: int = 3701
    environment_steps: int = 1800
    warmup_steps: int = 200
    updates_per_step: int = 1
    batch_size: int = 64
    replay_capacity: int = 12000
    observation_dim: int = 29
    base_replan_interval: int = 4
    hidden_dims: tuple[int, ...] = (128, 128)
    residual_scale: tuple[float, ...] = (0.20, 0.20, 0.16, 0.10, 0.08)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    initial_temperature: float = 0.08
    gamma: float = 0.99
    tau: float = 0.005
    image_size: int = 64
    max_episode_steps: int = 180
    residual_penalty: float = 0.03
    device: str = "auto"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResidualOnlineConfig:
        normalized = dict(payload)
        if "hidden_dims" in normalized:
            normalized["hidden_dims"] = tuple(normalized["hidden_dims"])
        if "residual_scale" in normalized:
            normalized["residual_scale"] = tuple(normalized["residual_scale"])
        normalized.pop("domain_randomization", None)
        return cls(**normalized)

    def __post_init__(self) -> None:
        if self.environment_steps < 1 or self.warmup_steps < 0:
            raise ValueError("invalid environment step budget")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must cover one batch")
        if self.observation_dim != 29:
            raise ValueError("residual observation is robot_state(24) + base_action(5) = 29")
        if self.base_replan_interval < 1:
            raise ValueError("base_replan_interval must be positive")
        if len(self.residual_scale) != 5 or any(value < 0 for value in self.residual_scale):
            raise ValueError("residual_scale must contain five non-negative values")


class ResidualReplayBuffer:
    """Fixed-size NumPy replay for encoded state; RGB is never duplicated."""

    def __init__(self, capacity: int, observation_dim: int, action_dim: int = 5) -> None:
        self.capacity = capacity
        self.observation = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.base_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observation = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_base_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        base_action: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        next_base_action: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observation[index] = observation
        self.base_action[index] = base_action
        self.action[index] = action
        self.reward[index, 0] = reward
        self.next_observation[index] = next_observation
        self.next_base_action[index] = next_base_action
        self.done[index, 0] = float(done)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("not enough replay samples")
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "observation": torch.from_numpy(self.observation[indices]),
            "base_action": torch.from_numpy(self.base_action[indices]),
            "action": torch.from_numpy(self.action[indices]),
            "reward": torch.from_numpy(self.reward[indices]),
            "next_observation": torch.from_numpy(self.next_observation[indices]),
            "next_base_action": torch.from_numpy(self.next_base_action[indices]),
            "done": torch.from_numpy(self.done[indices]),
        }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return selected


@torch.no_grad()
def _base_action(
    controller: LearnedPolicyController,
    observation: dict[str, Any],
) -> np.ndarray:
    return controller.act(observation).action.copy()


def _encoded_observation(observation: dict[str, Any], base_action: np.ndarray) -> np.ndarray:
    return np.concatenate((observation["robot_state"], base_action)).astype(np.float32)


def train_residual_online(
    base_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    config: ResidualOnlineConfig,
    domain_randomization: DomainRandomizationConfig,
) -> dict[str, Any]:
    """Train SAC corrections without ever updating the base VLA weights."""

    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    device = _device(config.device)
    base_model, base_metadata = load_trained_policy(base_checkpoint, device=device)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
    base_controller = LearnedPolicyController(
        base_model,
        device=device,
        replan_interval=config.base_replan_interval,
    )
    sac_config = ResidualSACConfig(
        observation_dim=config.observation_dim,
        action_dim=5,
        hidden_dims=config.hidden_dims,
        residual_scale=config.residual_scale,
        actor_learning_rate=config.actor_learning_rate,
        critic_learning_rate=config.critic_learning_rate,
        temperature_learning_rate=config.temperature_learning_rate,
        initial_temperature=config.initial_temperature,
        gamma=config.gamma,
        tau=config.tau,
    )
    agent = ResidualSAC(sac_config).to(device)
    replay = ResidualReplayBuffer(config.replay_capacity, config.observation_dim)
    env = SmartPickEnv(
        image_size=config.image_size,
        max_episode_steps=config.max_episode_steps,
        domain_randomization=domain_randomization,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    update_rows: list[dict[str, float | int]] = []
    episode_rows: list[dict[str, Any]] = []
    episode_index = 0
    observation, _ = env.reset(seed=config.seed)
    base_controller.reset()
    base_action = _base_action(base_controller, observation)
    encoded = _encoded_observation(observation, base_action)
    episode_return = 0.0
    episode_steps = 0

    try:
        for environment_step in range(1, config.environment_steps + 1):
            if environment_step <= config.warmup_steps:
                normalized_residual = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
                residual = np.asarray(config.residual_scale, dtype=np.float32) * normalized_residual
                action = np.clip(base_action + residual, -1.0, 1.0)
            else:
                with torch.no_grad():
                    action_tensor = agent.select_action(
                        torch.from_numpy(encoded).unsqueeze(0).to(device),
                        torch.from_numpy(base_action).unsqueeze(0).to(device),
                        deterministic=False,
                    )
                action = action_tensor[0].cpu().numpy().astype(np.float32)
                residual = action - base_action

            next_observation, reward, terminated, truncated, info = env.step(action)
            penalized_reward = float(reward - config.residual_penalty * np.linalg.norm(residual))
            next_base_action = _base_action(base_controller, next_observation)
            next_encoded = _encoded_observation(next_observation, next_base_action)
            done = bool(terminated or truncated)
            replay.add(
                encoded,
                base_action,
                action,
                penalized_reward,
                next_encoded,
                next_base_action,
                done,
            )
            episode_return += penalized_reward
            episode_steps += 1

            if replay.size >= config.batch_size and environment_step > config.warmup_steps:
                for _ in range(config.updates_per_step):
                    metrics = agent.update(replay.sample(config.batch_size, rng))
                    update_rows.append({"environment_step": environment_step, **metrics})

            if done:
                episode_rows.append(
                    {
                        "episode": episode_index,
                        "seed": config.seed + episode_index,
                        "environment_step": environment_step,
                        "steps": episode_steps,
                        "return": episode_return,
                        "success": bool(info["success"]),
                        "collision_steps": int(info["collision_steps"]),
                        "wrong_pick": bool(info["wrong_pick"]),
                        "wrong_bin": bool(info["wrong_bin"]),
                    }
                )
                episode_index += 1
                observation, _ = env.reset(seed=config.seed + episode_index)
                base_controller.reset()
                base_action = _base_action(base_controller, observation)
                encoded = _encoded_observation(observation, base_action)
                episode_return = 0.0
                episode_steps = 0
            else:
                observation = next_observation
                base_action = next_base_action
                encoded = next_encoded
    finally:
        env.close()

    save_checkpoint(
        destination / "last.pt",
        agent,
        optimizers=agent.optimizers,
        step=config.environment_steps,
        config=sac_config,
        extra={
            "kind": "bounded_residual_sac",
            "base_checkpoint": str(Path(base_checkpoint)),
            "base_policy_kind": base_metadata.get("policy_kind"),
            "online_config": asdict(config),
            "domain_randomization": asdict(domain_randomization),
        },
    )
    _write_rows(destination / "updates.csv", update_rows)
    _write_rows(destination / "episodes.csv", episode_rows)
    manifest = {
        "schema_version": "smartpick-residual-training/v1",
        "base_checkpoint": str(Path(base_checkpoint)),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "base_policy_frozen": all(
            not parameter.requires_grad for parameter in base_model.parameters()
        ),
        "online_config": asdict(config),
        "sac_config": asdict(sac_config),
        "domain_randomization": asdict(domain_randomization),
        "device": str(device),
        "environment_steps": config.environment_steps,
        "gradient_updates": int(agent.update_count.item()),
        "episodes_completed": len(episode_rows),
        "training_successes": sum(int(row["success"]) for row in episode_rows),
        "checkpoint": {
            "path": "last.pt",
            "sha256": sha256_file(destination / "last.pt"),
        },
        "disclosure": "online simulation training; no physical-robot trials",
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
