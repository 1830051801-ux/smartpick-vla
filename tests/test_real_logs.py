"""Tests for strict real-log import, calibration, and deterministic replay."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from smartpick_vla.real import (
    RealLogReplay,
    RealLogValidationError,
    RigidTransform,
    estimate_rigid_transform,
    load_calibration,
    load_real_config,
    load_real_log,
    resample_episode,
    save_calibration,
    save_real_log,
)

PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLE_LOG = PROJECT_ROOT / "examples" / "real_logs" / "example_episode.jsonl"


def test_example_jsonl_is_explicitly_synthetic_and_replayable() -> None:
    (episode,) = load_real_log(EXAMPLE_LOG)

    assert episode.episode_id == "real_demo_0001"
    assert len(episode.steps) == 3
    assert episode.duration_s == pytest.approx(0.1)
    assert episode.success is True
    assert all(
        step.metadata["source"] == "synthetic_example_not_hardware" for step in episode.steps
    )
    assert RealLogReplay(episode).run(lambda _frame: None) == 3


def test_flat_csv_import_supports_declared_contract(tmp_path: Path) -> None:
    path = tmp_path / "episode.csv"
    fields = [
        "schema_version",
        "episode_id",
        "step_index",
        "timestamp_s",
        "instruction",
        "task_class",
        "frame_id",
        "translation_unit",
        "rotation_unit",
        "time_unit",
        "action_order",
        "state_stamp_s",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
        "tcp_yaw_rad",
        "joint_positions_rad",
        "state_gripper",
        "dx_m",
        "dy_m",
        "dz_m",
        "dyaw_rad",
        "action_gripper",
        "image_file",
        "success",
        "metadata_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(2):
            timestamp = 20.0 + index * 0.05
            writer.writerow(
                {
                    "schema_version": "1.0",
                    "episode_id": "csv-episode",
                    "step_index": index,
                    "timestamp_s": timestamp,
                    "instruction": "send the scratched part to inspection",
                    "task_class": "scratch",
                    "frame_id": "base_link",
                    "translation_unit": "m",
                    "rotation_unit": "rad",
                    "time_unit": "s",
                    "action_order": "dx_m;dy_m;dz_m;dyaw_rad;gripper",
                    "state_stamp_s": timestamp,
                    "tcp_x_m": 0.4,
                    "tcp_y_m": 0.0,
                    "tcp_z_m": 0.2,
                    "tcp_yaw_rad": 0.0,
                    "joint_positions_rad": "0.0;0.2;-0.3",
                    "state_gripper": 1.0,
                    "dx_m": 0.002,
                    "dy_m": 0.0,
                    "dz_m": 0.0,
                    "dyaw_rad": 0.0,
                    "action_gripper": -1.0,
                    "image_file": "",
                    "success": "true" if index == 1 else "",
                    "metadata_json": json.dumps({"source": "unit_test"}),
                }
            )

    (episode,) = load_real_log(path)
    assert len(episode.steps) == 2
    assert episode.steps[1].robot_state.joint_positions_rad == (0.0, 0.2, -0.3)
    assert episode.success is True


@pytest.mark.parametrize(
    "missing",
    ["schema_version", "translation_unit", "rotation_unit", "time_unit"],
)
def test_missing_unit_declaration_fails_closed(tmp_path: Path, missing: str) -> None:
    (episode,) = load_real_log(EXAMPLE_LOG)
    row = episode.steps[0].to_dict()
    del row[missing]
    path = tmp_path / "missing.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RealLogValidationError, match=missing):
        load_real_log(path)


def test_missing_frame_and_malformed_json_have_line_context(tmp_path: Path) -> None:
    (episode,) = load_real_log(EXAMPLE_LOG)
    row = episode.steps[0].to_dict()
    del row["frame_id"]
    del row["robot_state"]["frame_id"]
    missing_frame = tmp_path / "missing_frame.jsonl"
    missing_frame.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RealLogValidationError, match="missing required frame_id"):
        load_real_log(missing_frame)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"episode_id":\n', encoding="utf-8")
    with pytest.raises(RealLogValidationError, match=r"malformed.jsonl:1: invalid JSON"):
        load_real_log(malformed)


def test_camera_frame_log_requires_and_applies_calibration(tmp_path: Path) -> None:
    (episode,) = load_real_log(EXAMPLE_LOG)
    row = episode.steps[0].to_dict()
    row["frame_id"] = "camera_link"
    row["robot_state"]["frame_id"] = "camera_link"
    row["robot_state"]["tcp_position_m"] = [0.2, 0.0, 0.1]
    row["robot_state"]["tcp_yaw_rad"] = 0.1
    row["action"] = {
        "dx_m": 0.01,
        "dy_m": 0.0,
        "dz_m": 0.0,
        "dyaw_rad": 0.02,
        "gripper": 0.0,
    }
    path = tmp_path / "camera.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RealLogValidationError, match="requires explicit calibration"):
        load_real_log(path)

    transform = RigidTransform(
        source_frame="camera_link",
        target_frame="base_link",
        rotation=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        translation_m=(0.1, 0.2, 0.3),
    )
    (normalized,) = load_real_log(path, calibration=transform)
    step = normalized.steps[0]
    assert step.robot_state.tcp_position_m == pytest.approx((0.1, 0.4, 0.4))
    assert step.robot_state.tcp_yaw_rad == pytest.approx(0.1 + np.pi / 2)
    assert step.action.dx_m == pytest.approx(0.0)
    assert step.action.dy_m == pytest.approx(0.01)
    assert step.metadata["coordinate_transform"]["target_frame"] == "base_link"


def test_kabsch_calibration_round_trip(tmp_path: Path) -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.1, 0.2, 0.2]])
    expected_rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_translation = np.asarray([0.4, -0.2, 0.1])
    target = (expected_rotation @ source.T).T + expected_translation
    result = estimate_rigid_transform(source, target)

    assert result.rmse_m < 1e-12
    np.testing.assert_allclose(result.transform.rotation_matrix, expected_rotation, atol=1e-12)
    np.testing.assert_allclose(
        result.transform.translation_vector, expected_translation, atol=1e-12
    )

    path = tmp_path / "calibration.yaml"
    save_calibration(result, path)
    restored = load_calibration(path)
    np.testing.assert_allclose(restored.transform.rotation_matrix, expected_rotation, atol=1e-12)
    assert restored.sample_count == 4


def test_resampling_uses_zoh_for_actions_and_discrete_gripper() -> None:
    (episode,) = load_real_log(EXAMPLE_LOG)
    resampled = resample_episode(episode, 0.025)

    assert len(resampled.steps) == 5
    assert resampled.steps[-1].timestamp_s == episode.steps[-1].timestamp_s
    assert resampled.steps[1].action == episode.steps[0].action
    assert resampled.steps[1].robot_state.gripper == episode.steps[0].robot_state.gripper
    assert resampled.steps[3].robot_state.gripper == episode.steps[1].robot_state.gripper
    assert resampled.success is True


def test_jsonl_round_trip_and_real_config_defaults(tmp_path: Path) -> None:
    episodes = load_real_log(EXAMPLE_LOG)
    path = tmp_path / "roundtrip.jsonl"
    save_real_log(path, episodes)
    restored = load_real_log(path)
    assert restored == episodes

    config = load_real_config(PROJECT_ROOT / "configs" / "real" / "default.yaml")
    assert config.execution.dry_run is True
    assert config.execution.hardware_enabled is False
    assert config.camera_to_base is None
    assert config.audit_jsonl == "results/real/dry_run_actions.jsonl"


def test_non_monotonic_state_time_is_rejected(tmp_path: Path) -> None:
    (episode,) = load_real_log(EXAMPLE_LOG)
    first = episode.steps[0]
    second = replace(
        episode.steps[1],
        robot_state=replace(episode.steps[1].robot_state, stamp_s=first.robot_state.stamp_s),
    )
    path = tmp_path / "non_monotonic.jsonl"
    path.write_text(
        json.dumps(first.to_dict()) + "\n" + json.dumps(second.to_dict()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RealLogValidationError, match="non-increasing robot-state stamp_s"):
        load_real_log(path)
