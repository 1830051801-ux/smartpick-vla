"""Fixed-seed benchmark orchestration tests without expensive physics rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
from torch import nn

import smartpick_vla.evaluation.benchmark as benchmark_module
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.evaluation import load_episode_results
from smartpick_vla.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkSuite,
    evaluate_method,
    run_benchmark,
)


class _FakeEnvironment:
    constructed_randomization: ClassVar[list[bool]] = []
    reset_calls: ClassVar[list[tuple[int, dict[str, Any], bool]]] = []

    def __init__(
        self,
        *,
        image_size: int,
        max_episode_steps: int,
        domain_randomization: DomainRandomizationConfig,
        grasp_assist: bool,
    ) -> None:
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.randomized = domain_randomization.enabled
        self.grasp_assist = grasp_assist
        self.step_index = 0
        self.options: dict[str, Any] = {}
        self.seed = 0
        self.task_class = "accepted"
        self.closed = False
        self.constructed_randomization.append(self.randomized)

    def _observation(self) -> dict[str, Any]:
        split = str(self.options.get("instruction_split", "train"))
        return {
            "rgb": np.full(
                (self.image_size, self.image_size, 3),
                min(255, self.step_index * 40),
                dtype=np.uint8,
            ),
            "robot_state": np.zeros(24, dtype=np.float32),
            "instruction": f"{split} instruction for {self.task_class}",
        }

    def _info(self, *, success: bool, collision: bool) -> dict[str, Any]:
        return {
            "success": success,
            "collision": collision,
            "collision_steps": int(self.randomized and self.step_index > 0),
            "contact_count": self.step_index,
            "wrong_pick": False,
            "wrong_bin": False,
            "cycle_time_s": self.step_index * 0.04,
            "task_class": self.task_class,
            "instruction_template_id": (
                f"{self.options.get('instruction_split', 'train')}/{self.task_class}/0"
            ),
            "instruction_split": self.options.get("instruction_split", "train"),
            "grasp_assist": self.grasp_assist,
            "randomization": {
                "enabled": self.randomized,
                "mass_scales": [1.2, 0.8, 1.1] if self.randomized else [1.0, 1.0, 1.0],
            },
            "control_dt": 0.04,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert seed is not None
        self.seed = seed
        self.options = dict(options or {})
        self.step_index = 0
        self.task_class = ("accepted", "scratch", "unknown")[seed % 3]
        self.reset_calls.append((seed, dict(self.options), self.randomized))
        return self._observation(), self._info(success=False, collision=False)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert action.shape == (5,)
        self.step_index += 1
        terminated = self.step_index >= 2
        collision = self.randomized and self.step_index == 1
        return (
            self._observation(),
            1.0 if terminated else -0.1,
            terminated,
            False,
            self._info(success=terminated, collision=collision),
        )

    def close(self) -> None:
        self.closed = True


class _FakeExpert:
    def __init__(
        self,
        environment: _FakeEnvironment,
        *,
        use_noisy_detection: bool = False,
    ) -> None:
        self.environment = environment
        self.use_noisy_detection = use_noisy_detection

    def reset(self) -> None:
        return None

    def act(self) -> tuple[np.ndarray, None]:
        return np.zeros(5, dtype=np.float32), None


class _TinyPolicy(nn.Module):
    def forward(
        self,
        rgb: torch.Tensor,
        instruction: list[str],
        robot_state: torch.Tensor,
    ) -> torch.Tensor:
        assert rgb.ndim == 4
        assert len(instruction) == rgb.shape[0]
        assert robot_state.shape[1] == 24
        return torch.zeros((rgb.shape[0], 5), device=rgb.device)


class _ArrayController:
    def __init__(self) -> None:
        self.reset_count = 0
        self.action_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        assert observation["robot_state"].shape == (24,)
        self.action_count += 1
        return np.zeros(5, dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_fake_logs() -> None:
    _FakeEnvironment.constructed_randomization.clear()
    _FakeEnvironment.reset_calls.clear()


def test_four_suites_share_seed_manifest_and_write_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark_module, "SmartPickEnv", _FakeEnvironment)
    monkeypatch.setattr(benchmark_module, "IKWaypointExpert", _FakeExpert)
    config = BenchmarkConfig(
        seeds=(5, 9),
        image_size=32,
        max_episode_steps=2,
        gif_seeds=(5,),
        gif_fps=5.0,
    )
    run = run_benchmark(
        tmp_path / "benchmark",
        {"ik_expert": None, "learned": _TinyPolicy()},
        config=config,
    )

    assert run.seed_manifest == tuple(
        (suite, seed) for suite in ("id", "paraphrase", "ood", "physics") for seed in (5, 9)
    )
    assert len(run.results) == 16
    assert len(load_episode_results(run.artifacts.episode_csv)) == 16
    budgets = {
        method: {(row.suite, row.seed) for row in run.results if row.method == method}
        for method in ("ik_expert", "learned")
    }
    assert budgets["ik_expert"] == budgets["learned"] == set(run.seed_manifest)
    paired_ids = {
        (row.suite, row.seed): {
            candidate.episode_id
            for candidate in run.results
            if candidate.suite == row.suite and candidate.seed == row.seed
        }
        for row in run.results
    }
    assert all(len(episode_ids) == 1 for episode_ids in paired_ids.values())

    assert _FakeEnvironment.constructed_randomization == [False, False, False, True] * 2
    for row in run.results:
        options = row.metadata["reset_options"]
        if row.suite == "paraphrase":
            assert options == {"instruction_split": "paraphrase", "ood_layout": False}
        elif row.suite == "ood":
            assert options == {"instruction_split": "train", "ood_layout": True}
        else:
            assert options == {"instruction_split": "train", "ood_layout": False}
        assert row.metadata["randomization"]["enabled"] is (row.suite == "physics")
        assert row.success

    summary = json.loads(run.artifacts.summary_json.read_text(encoding="utf-8"))
    assert len(summary["groups"]) == 8
    assert {group["episode_count"] for group in summary["groups"]} == {2}
    assert len(run.media) == 8
    assert all(item.seed == 5 for item in run.media)
    assert all(item.gif_path.exists() and item.sidecar_path.exists() for item in run.media)


def test_single_controller_entrypoint_and_invalid_method_contracts(tmp_path: Path) -> None:
    controller = _ArrayController()

    def factory(suite: BenchmarkSuite, config: BenchmarkConfig) -> _FakeEnvironment:
        randomization = (
            config.physics_randomization
            if suite.physics_randomization
            else DomainRandomizationConfig(enabled=False)
        )
        return _FakeEnvironment(
            image_size=config.image_size,
            max_episode_steps=config.max_episode_steps,
            domain_randomization=randomization,
            grasp_assist=config.grasp_assist,
        )

    config = BenchmarkConfig(
        seeds=(2, 3),
        suites=("id",),
        image_size=32,
        max_episode_steps=2,
    )
    run = evaluate_method(
        tmp_path / "controller",
        method="custom_controller",
        controller=controller,
        config=config,
        environment_factory=factory,
    )
    assert len(run.results) == 2
    assert controller.reset_count == 2
    assert controller.action_count == 4
    assert all(row.inference_latency_ms >= 0.0 for row in run.results)

    with pytest.raises(ValueError, match="either model or controller"):
        evaluate_method(
            tmp_path / "invalid",
            method="learned",
            model=_TinyPolicy(),
            controller=controller,
            config=config,
            environment_factory=factory,
        )
    with pytest.raises(ValueError, match="takes no source"):
        run_benchmark(
            tmp_path / "invalid-expert",
            {"ik_expert": _TinyPolicy()},
            config=config,
            environment_factory=factory,
        )
    with pytest.raises(ValueError, match="filesystem-safe"):
        run_benchmark(
            tmp_path / "invalid-name",
            {"../learned": controller},
            config=config,
            environment_factory=factory,
        )
    with pytest.raises(ValueError, match="seeds must be unique"):
        BenchmarkConfig(seeds=(1, 1), suites=("id",))
    with pytest.raises(ValueError, match="requires enabled domain randomization"):
        BenchmarkConfig(
            seeds=(1,),
            suites=("physics",),
            physics_randomization=DomainRandomizationConfig(enabled=False),
        )
