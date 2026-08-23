"""Unit coverage for the resumable industrial experiment contract."""

from smartpick_vla.evaluation.industrial import (
    IndustrialExperimentConfig,
    IndustrialTask,
    build_industrial_manifest,
)


def test_industrial_manifest_is_task_suite_seed_complete() -> None:
    config = IndustrialExperimentConfig(
        tasks=(IndustrialTask("single", 1), IndustrialTask("pair", 2)),
        suites=("id", "physics"),
        seed_start=50,
        episodes_per_task=2,
        workers=2,
    )
    manifest = build_industrial_manifest(config)
    assert len(manifest) == 8
    assert {(item.task_id, item.suite, item.seed) for item in manifest} == {
        (task, suite, seed)
        for task in ("single", "pair")
        for suite in ("id", "physics")
        for seed in (50, 51)
    }
    assert {item.shard for item in manifest} == {0, 1}


def test_industrial_config_from_mapping_converts_nested_yaml_values() -> None:
    config = IndustrialExperimentConfig.from_mapping(
        {
            "tasks": [{"task_id": "pair", "mission_length": 2}],
            "suites": ["id"],
            "seed_start": 9,
            "episodes_per_task": 1,
            "workers": 1,
            "physics_randomization": {
                "enabled": True,
                "control_delay_steps": [1, 2],
            },
            "perception_randomization": {
                "enabled": True,
                "vision_latency_frames": [0, 1],
            },
        }
    )
    assert config.tasks == (IndustrialTask("pair", 2),)
    assert config.suites == ("id",)
    assert config.physics_randomization.control_delay_steps == (1, 2)
    assert config.perception_randomization.vision_latency_frames == (0, 1)
