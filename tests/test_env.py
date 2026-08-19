"""MuJoCo environment and expert integration tests."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from smartpick_vla.data.expert import IKWaypointExpert
from smartpick_vla.envs import SmartPickEnv
from smartpick_vla.envs.randomization import DomainRandomizationConfig


@pytest.mark.mujoco
def test_environment_contract_and_checker() -> None:
    env = SmartPickEnv(image_size=32, max_episode_steps=5)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=5)

    assert env.observation_space.contains(observation)
    assert observation["rgb"].shape == (32, 32, 3)
    assert observation["rgb"].dtype == np.uint8
    assert observation["robot_state"].shape == (24,)
    assert info["action_frame"] == "base_link"
    result = env.step(np.zeros(5, dtype=np.float32))
    assert len(result) == 5
    env.close()


@pytest.mark.mujoco
def test_seeded_reset_is_reproducible() -> None:
    env = SmartPickEnv(image_size=32)
    observation_one, info_one = env.reset(seed=77)
    positions_one = np.stack([env.object_position(index) for index in range(3)])
    observation_two, info_two = env.reset(seed=77)
    positions_two = np.stack([env.object_position(index) for index in range(3)])

    np.testing.assert_allclose(positions_one, positions_two)
    np.testing.assert_array_equal(observation_one["rgb"], observation_two["rgb"])
    assert info_one["instruction_template_id"] == info_two["instruction_template_id"]
    env.close()


@pytest.mark.mujoco
def test_domain_randomization_changes_declared_parameters() -> None:
    config = DomainRandomizationConfig(
        enabled=True,
        object_mass_scale=(0.7, 1.3),
        friction_scale=(0.7, 1.3),
        camera_position_std_m=0.01,
        camera_fovy_delta_deg=2.0,
        control_delay_steps=(1, 2),
    )
    env = SmartPickEnv(image_size=32, domain_randomization=config)
    _, info = env.reset(seed=12)
    randomized = info["randomization"]

    assert randomized["enabled"]
    assert randomized["control_delay_steps"] in (1, 2)
    assert not np.allclose(randomized["mass_scales"], 1.0)
    assert not np.allclose(randomized["camera_offset_m"], 0.0)
    assert info["scene_initialization"]["layout_attempts"] >= 1
    assert info["scene_initialization"]["settle_steps"] == 80
    env.close()


@pytest.mark.mujoco
def test_ood_layout_does_not_spawn_inside_destination_bins() -> None:
    env = SmartPickEnv(image_size=32, six_axis=True)
    _, info = env.reset(seed=914, options={"task_class": "unknown", "ood_layout": True})
    bin_geometries = {
        geom_id for geom_id in env._obstacle_geom_ids if env.model.geom(geom_id).name != "table"
    }
    object_geometries = set(env._object_geom_ids)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        assert not (
            (int(contact.geom1) in object_geometries and int(contact.geom2) in bin_geometries)
            or (int(contact.geom2) in object_geometries and int(contact.geom1) in bin_geometries)
        )
    assert info["scene_initialization"]["settle_duration_s"] > 0.0
    env.close()


@pytest.mark.mujoco
def test_perception_randomization_exposes_noise_occlusion_and_latency() -> None:
    perception_config = DomainRandomizationConfig(
        enabled=True,
        object_mass_scale=(1.0, 1.0),
        friction_scale=(1.0, 1.0),
        camera_position_std_m=0.0,
        camera_fovy_delta_deg=0.0,
        light_intensity_scale=(1.0, 1.0),
        object_color_jitter=0.0,
        robot_state_noise_std=0.0,
        detection_noise_std_m=0.0,
        control_delay_steps=(0, 0),
        image_noise_std_px=12.0,
        image_occlusion_probability=1.0,
        image_occlusion_max_fraction=0.20,
        vision_latency_frames=(2, 2),
    )
    clean_env = SmartPickEnv(image_size=32)
    stressed_env = SmartPickEnv(image_size=32, domain_randomization=perception_config)
    clean_observation, _ = clean_env.reset(seed=97)
    stressed_observation, stressed_info = stressed_env.reset(seed=97)

    randomized = stressed_info["randomization"]
    assert randomized["vision_latency_frames"] == 2
    assert randomized["image_noise_std_px"] == 12.0
    assert randomized["image_occlusion_probability"] == 1.0
    assert not np.array_equal(clean_observation["rgb"], stressed_observation["rgb"])

    stressed_env.step(np.zeros(5, dtype=np.float32))
    clean_env.close()
    stressed_env.close()


@pytest.mark.mujoco
@pytest.mark.parametrize("category", ["accepted", "scratch", "unknown"])
def test_ik_expert_completes_each_quality_class(category: str) -> None:
    env = SmartPickEnv(image_size=32, max_episode_steps=180)
    env.reset(seed=123, options={"task_class": category})
    expert = IKWaypointExpert(env)
    expert.reset()
    info = {}
    for _ in range(180):
        action, _ = expert.act()
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert info["success"], info
    assert not info["wrong_pick"]
    assert not info["wrong_bin"]
    env.close()


@pytest.mark.mujoco
def test_action_rejects_nonfinite_values() -> None:
    env = SmartPickEnv(image_size=32)
    env.reset(seed=1)
    with pytest.raises(ValueError, match="NaN or infinity"):
        env.step(np.array([0.0, 0.0, np.nan, 0.0, 1.0], dtype=np.float32))
    env.close()
