"""XiaoU camera-to-six-axis planning-preview tests."""

import json
from pathlib import Path

import pytest

from smartpick_vla.real import (
    XiaoUDetection,
    XiaoUGraspProfile,
    XiaoUHomography,
    build_xiaou_plan_preview,
    load_xiaou_grasp_profiles,
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
