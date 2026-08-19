"""Write a simulation-only XiaoU six-axis planning preview JSON artifact."""

from pathlib import Path

from smartpick_vla.real import (
    XiaoUDetection,
    build_xiaou_plan_preview,
    load_xiaou_grasp_profiles,
    load_xiaou_homography,
    save_xiaou_plan_preview,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    homography = load_xiaou_homography(ROOT / "configs" / "real" / "xiaou_demo_homography.yaml")
    profiles = load_xiaou_grasp_profiles(
        ROOT / "configs" / "real" / "xiaou_simulated_profiles.yaml"
    )
    preview = build_xiaou_plan_preview(
        XiaoUDetection(label="cola", u_px=1030.0, v_px=490.0, confidence=0.92, stamp_s=1.0),
        homography=homography,
        profiles=profiles,
        task_id="xiaou-simulation-preview",
    )
    path = save_xiaou_plan_preview(
        preview,
        ROOT / "results" / "examples" / "xiaou_plan_preview.json",
    )
    print(path)


if __name__ == "__main__":
    main()
