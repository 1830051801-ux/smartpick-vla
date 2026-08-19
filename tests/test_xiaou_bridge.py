"""XiaoU camera-to-six-axis planning-preview tests."""

import json
from pathlib import Path

import pytest
import yaml

from smartpick_vla.real import (
    XIAOU_HARDWARE_PROFILE_SCHEMA,
    XIAOU_JOINT_NAMES,
    XiaoUDetection,
    XiaoUGraspProfile,
    XiaoUHomography,
    build_xiaou_plan_preview,
    load_xiaou_grasp_profiles,
    load_xiaou_hardware_profile,
    load_xiaou_homography,
    save_xiaou_plan_preview,
)


def _homography() -> XiaoUHomography:
    return XiaoUHomography(
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        mean_error_mm=0.4,
        max_error_mm=1.1,
    )


def _profiles() -> dict[str, XiaoUGraspProfile]:
    return {
        "cola": XiaoUGraspProfile(
            label="cola",
            grasp_height_m=0.04,
            approach_height_m=0.11,
            lift_height_m=0.16,
            yaw_rad=0.2,
        )
    }


def test_xiaou_preview_projects_pixel_detection_to_ordered_targets(tmp_path: Path) -> None:
    preview = build_xiaou_plan_preview(
        XiaoUDetection(label="cola", u_px=250.0, v_px=-120.0, confidence=0.88, stamp_s=2.5),
        homography=_homography(),
        profiles=_profiles(),
        task_id="unit-preview",
    )
    payload = preview.to_dict()

    assert [target["phase"] for target in payload["targets"]] == ["pregrasp", "grasp", "lift"]
    assert payload["planning_only"] is True
    assert payload["real_motion_authorized"] is False
    assert payload["targets"][0]["position_m"] == [0.25, -0.12, 0.11]
    assert payload["targets"][1]["position_m"] == [0.25, -0.12, 0.04]
    assert payload["targets"][0]["ros2_pose_stamped_preview"]["header"]["frame_id"] == "base_link"

    output = save_xiaou_plan_preview(preview, tmp_path / "preview.json")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["task_id"] == "unit-preview"


def test_xiaou_preview_rejects_low_confidence_bad_calibration_and_missing_profile() -> None:
    detection = XiaoUDetection(label="cola", u_px=0.0, v_px=0.0, confidence=0.40, stamp_s=1.0)
    with pytest.raises(ValueError, match="confidence"):
        build_xiaou_plan_preview(
            detection,
            homography=_homography(),
            profiles=_profiles(),
            task_id="low-confidence",
        )

    high_error = XiaoUHomography(
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        mean_error_mm=1.0,
        max_error_mm=4.0,
    )
    with pytest.raises(ValueError, match="max error"):
        build_xiaou_plan_preview(
            XiaoUDetection(label="cola", u_px=0.0, v_px=0.0, confidence=0.90, stamp_s=1.0),
            homography=high_error,
            profiles=_profiles(),
            task_id="bad-calibration",
        )
    with pytest.raises(ValueError, match="no complete"):
        build_xiaou_plan_preview(
            XiaoUDetection(label="cup", u_px=0.0, v_px=0.0, confidence=0.90, stamp_s=1.0),
            homography=_homography(),
            profiles=_profiles(),
            task_id="missing-profile",
        )


def test_xiaou_loaders_accept_simulated_assets_and_reject_unmeasured_profiles(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    homography = load_xiaou_homography(root / "configs" / "real" / "xiaou_demo_homography.yaml")
    profiles = load_xiaou_grasp_profiles(
        root / "configs" / "real" / "xiaou_simulated_profiles.yaml"
    )
    assert homography.project_pixel(1060.0, 500.0) == pytest.approx((0.1, -0.04))
    assert {"cup", "cola", "pen"}.issubset(profiles)

    incomplete = tmp_path / "unmeasured.json"
    incomplete.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "classes": {
                    "cola": {"grasp_height_m": None, "approach_height_m": None},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_xiaou_grasp_profiles(incomplete)


def test_xiaou_hardware_profile_is_six_axis_and_motion_disabled(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    profile_path = root / "configs" / "real" / "xiaou_hardware_profile.yaml"
    profile = load_xiaou_hardware_profile(profile_path)

    assert profile.schema_version == XIAOU_HARDWARE_PROFILE_SCHEMA
    assert profile.joint_count == 6
    assert tuple(profile.joint_limit_map) == XIAOU_JOINT_NAMES
    assert profile.link_lengths_mm == (156.0, 180.0, 180.0, 93.0, 106.0)
    assert profile.trajectory_payload_bytes == 26
    assert profile.trajectory_interpolation_period_ms == 10
    assert profile.uart_baud == 115200
    assert profile.can_bitrate_bps == 1000000
    assert profile.real_motion_ready is False
    assert profile.hardware_execution_enabled is False
    assert "raw_can_uart_bytes" in profile.forbidden_low_level_outputs
    assert profile.to_dict()["execution"]["real_motion_ready"] is False

    invalid_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    assert isinstance(invalid_payload, dict)
    invalid_payload["execution"]["hardware_execution_enabled"] = True
    invalid_path = tmp_path / "invalid_hardware_profile.yaml"
    invalid_path.write_text(yaml.safe_dump(invalid_payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be enabled"):
        load_xiaou_hardware_profile(invalid_path)

    invalid_payload["execution"]["hardware_execution_enabled"] = False
    invalid_payload["trajectory"]["payload_bytes"] = 25
    invalid_path.write_text(yaml.safe_dump(invalid_payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="payload"):
        load_xiaou_hardware_profile(invalid_path)

    invalid_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    assert isinstance(invalid_payload, dict)
    invalid_payload["interfaces"]["pi_f407_uart"]["baud"] = 57600
    invalid_path.write_text(yaml.safe_dump(invalid_payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="UART baseline"):
        load_xiaou_hardware_profile(invalid_path)

    invalid_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    assert isinstance(invalid_payload, dict)
    invalid_payload["trajectory"]["angle_unit"] = "rad"
    invalid_path.write_text(yaml.safe_dump(invalid_payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory encoding"):
        load_xiaou_hardware_profile(invalid_path)
