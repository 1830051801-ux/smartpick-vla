"""Regression coverage for six-axis, multiview, mission, and safety upgrades."""

from __future__ import annotations

import numpy as np
import pytest

from smartpick_vla.data.expert import IKWaypointExpert
from smartpick_vla.data.perception import (
    PerceptionGenerationConfig,
    generate_synthetic_perception_dataset,
)
from smartpick_vla.envs import SmartPickEnv
from smartpick_vla.evaluation.safety_filter import (
    PredictiveSafetyFilter,
    PredictiveSafetyFilterConfig,
)


@pytest.mark.mujoco
def test_six_axis_contract_and_multiview_labels() -> None:
    env = SmartPickEnv(image_size=32, six_axis=True)
    observation, info = env.reset(seed=21)
    assert observation["robot_state"].shape == (29,)
    assert env.action_space.shape == (6,)
    assert info["arm_variant"] == "six_axis"
    assert info["action_order"] == ("dx", "dy", "dz", "dyaw", "droll", "gripper")

    perception = env.render_perception("top")
    assert perception["rgb"].shape == (32, 32, 3)
    assert perception["depth_m"].shape == (32, 32)
    assert perception["instance_mask"].shape == (32, 32)
    assert {instance["quality_class"] for instance in perception["instances"]} == {
        "accepted",
        "scratch",
        "unknown",
    }
    assert all(instance["visible_pixels"] > 0 for instance in perception["instances"])
    assert env.render_camera("wrist").shape == (32, 32, 3)
    env.close()


@pytest.mark.mujoco
def test_mission_expert_completes_three_ordered_subtasks() -> None:
    env = SmartPickEnv(image_size=32, six_axis=True, max_episode_steps=320, mission_length=3)
    env.reset(seed=123, options={"task_sequence": ["scratch", "accepted", "unknown"]})
    expert = IKWaypointExpert(env)
    expert.reset()
    info = {}
    for _ in range(320):
        action, _ = expert.act()
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["success"], info
    assert info["mission_complete"], info
    assert info["completed_task_classes"] == ("scratch", "accepted", "unknown")
    env.close()


@pytest.mark.mujoco
def test_predictive_safety_filter_scales_a_future_table_collision() -> None:
    env = SmartPickEnv(image_size=32)
    env.reset(seed=1)
    downward = np.array([0.0, 0.0, -1.0, 0.0, 1.0], dtype=np.float32)
    for _ in range(12):
        env.step(downward)
    qpos_before = env.data.qpos.copy()
    shield = PredictiveSafetyFilter(
        env,
        config=PredictiveSafetyFilterConfig(
            lookahead_control_steps=4,
            allow_finger_table_contact=False,
        ),
    )
    decision = shield.filter(downward)
    assert decision.predicted_collision
    assert decision.intervened
    assert 0.0 <= decision.motion_scale < 1.0
    np.testing.assert_allclose(env.data.qpos, qpos_before)
    env.close()


@pytest.mark.mujoco
def test_synthetic_perception_dataset_has_multiview_supervision(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "perception.npz"
    manifest = generate_synthetic_perception_dataset(
        output,
        config=PerceptionGenerationConfig(
            episodes=1,
            frames_per_episode=2,
            capture_stride=8,
            image_size=32,
            cameras=("top", "oblique"),
            six_axis=True,
        ),
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["rgb"].shape == (2, 2, 32, 32, 3)
        assert archive["depth_m"].shape == (2, 2, 32, 32)
        assert archive["instance_mask"].shape == (2, 2, 32, 32)
        assert archive["bbox_xyxy"].shape == (2, 2, 3, 4)
        assert archive["action"].shape == (2, 6)
        assert archive["robot_state"].shape == (2, 29)
    assert manifest["samples"] == 2
    assert manifest["disclosure"]["synthetic_data"]
