"""Expert demonstration collection with attempt-level accounting."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from smartpick_vla.data.dataset import save_trajectory_npz, validate_trajectory_arrays
from smartpick_vla.data.expert import IKWaypointExpert
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.envs.smartpick_env import SmartPickEnv
from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import sha256_file


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    episodes: int = 30
    seed: int = 17
    image_size: int = 96
    max_episode_steps: int = 180
    domain_randomization: bool = False
    noisy_expert_detection: bool = False
    grasp_assist: bool = True
    six_axis: bool = False
    mission_length: int = 1


def generate_expert_dataset(
    output_path: str | Path,
    *,
    config: GenerationConfig,
    domain_config: DomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    """Collect every attempted transition and write a truthful manifest."""

    if config.episodes < 1:
        raise ValueError("episodes must be positive")
    randomization = domain_config or DomainRandomizationConfig(enabled=config.domain_randomization)
    scene_path = Path(__file__).resolve().parents[1] / "envs" / "assets" / "smartpick_scene.xml"
    env = SmartPickEnv(
        image_size=config.image_size,
        max_episode_steps=config.max_episode_steps,
        domain_randomization=randomization,
        grasp_assist=config.grasp_assist,
        six_axis=config.six_axis,
        mission_length=config.mission_length,
    )
    records: dict[str, list[Any]] = {
        "rgb": [],
        "robot_state": [],
        "action": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
        "collision": [],
        "wrong_pick": [],
        "wrong_bin": [],
        "episode_id": [],
        "step_index": [],
        "instruction": [],
        "task_class": [],
        "episode_success": [],
    }
    episode_buffers: list[dict[str, list[Any]]] = []
    episode_summaries: list[dict[str, Any]] = []
    try:
        for episode_id in range(config.episodes):
            episode_seed = config.seed + episode_id
            observation, _ = env.reset(seed=episode_seed)
            expert = IKWaypointExpert(env, use_noisy_detection=config.noisy_expert_detection)
            expert.reset()
            buffer: dict[str, list[Any]] = {key: [] for key in records}
            episode_return = 0.0
            info: dict[str, Any] = {}
            for step_index in range(config.max_episode_steps):
                action, _ = expert.act()
                buffer["rgb"].append(observation["rgb"])
                buffer["robot_state"].append(observation["robot_state"])
                buffer["action"].append(action)
                # Terminal labels are filled from the post-action info below.
                # Keeping them aligned with the action makes the archive usable
                # for one-step dynamics/world-model training without replaying
                # the simulator during dataset loading.
                buffer["reward"].append(0.0)
                buffer["terminated"].append(False)
                buffer["truncated"].append(False)
                buffer["collision"].append(False)
                buffer["wrong_pick"].append(False)
                buffer["wrong_bin"].append(False)
                buffer["episode_id"].append(episode_id)
                buffer["step_index"].append(step_index)
                buffer["instruction"].append(observation["instruction"])
                buffer["task_class"].append(env.task.target_class)
                buffer["episode_success"].append(False)  # filled after terminal state
                observation, reward, terminated, truncated, info = env.step(action)
                buffer["reward"][-1] = float(reward)
                buffer["terminated"][-1] = bool(terminated)
                buffer["truncated"][-1] = bool(truncated)
                buffer["collision"][-1] = bool(info.get("collision", False))
                buffer["wrong_pick"][-1] = bool(info.get("wrong_pick", False))
                buffer["wrong_bin"][-1] = bool(info.get("wrong_bin", False))
                episode_return += reward
                if terminated or truncated:
                    break
            success = bool(info.get("success", False))
            buffer["episode_success"] = [success] * len(buffer["action"])
            episode_buffers.append(buffer)
            episode_summaries.append(
                {
                    "episode_id": episode_id,
                    "seed": episode_seed,
                    "task_class": env.task.target_class,
                    "instruction_template_id": env.task.template_id,
                    "success": success,
                    "steps": len(buffer["action"]),
                    "return": episode_return,
                    "collision_steps": int(info.get("collision_steps", 0)),
                    "wrong_pick": bool(info.get("wrong_pick", False)),
                    "wrong_bin": bool(info.get("wrong_bin", False)),
                    "randomization": info.get("randomization", {}),
                }
            )
    finally:
        env.close()

    for buffer in episode_buffers:
        for key in records:
            records[key].extend(buffer[key])
    arrays: dict[str, np.ndarray] = {
        "rgb": np.asarray(records["rgb"], dtype=np.uint8),
        "robot_state": np.asarray(records["robot_state"], dtype=np.float32),
        "action": np.asarray(records["action"], dtype=np.float32),
        "reward": np.asarray(records["reward"], dtype=np.float32),
        "terminated": np.asarray(records["terminated"], dtype=np.bool_),
        "truncated": np.asarray(records["truncated"], dtype=np.bool_),
        "collision": np.asarray(records["collision"], dtype=np.bool_),
        "wrong_pick": np.asarray(records["wrong_pick"], dtype=np.bool_),
        "wrong_bin": np.asarray(records["wrong_bin"], dtype=np.bool_),
        "episode_id": np.asarray(records["episode_id"], dtype=np.int32),
        "step_index": np.asarray(records["step_index"], dtype=np.int32),
        "instruction": np.asarray(records["instruction"], dtype=np.str_),
        "task_class": np.asarray(records["task_class"], dtype=np.str_),
        "episode_success": np.asarray(records["episode_success"], dtype=np.bool_),
    }
    statistics = validate_trajectory_arrays(arrays)
    destination = save_trajectory_npz(output_path, arrays)
    manifest = {
        "schema_version": "smartpick-demonstrations/v1",
        "dataset_file": destination.name,
        "dataset_sha256": sha256_file(destination),
        "scene_file": "smartpick_scene.xml",
        "scene_sha256": sha256_file(scene_path),
        "generation": asdict(config),
        "statistics": asdict(statistics),
        "attempted_episodes": config.episodes,
        "successful_episodes": statistics.successful_episodes,
        "failed_episodes_retained": config.episodes - statistics.successful_episodes,
        "episodes": episode_summaries,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
        },
        "disclosure": {
            "privileged_expert": True,
            "grasp_assist": config.grasp_assist,
            "physical_robot_data": False,
            "arm_variant": env.arm_variant,
        },
    }
    atomic_write_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest
