"""Command-line entry points for data, training, evaluation, and replay."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium
import imageio.v3 as iio
import mujoco
import numpy as np
import torch

from smartpick_vla import __version__
from smartpick_vla.data.expert import IKWaypointExpert
from smartpick_vla.data.generate import GenerationConfig, generate_expert_dataset
from smartpick_vla.data.perception import (
    PerceptionGenerationConfig,
    generate_synthetic_perception_dataset,
)
from smartpick_vla.envs import SmartPickEnv
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.evaluation.benchmark import BenchmarkConfig, run_benchmark
from smartpick_vla.evaluation.plots import plot_grouped_evaluation, plot_learning_curves
from smartpick_vla.evaluation.residual_controller import ResidualPolicyController
from smartpick_vla.evaluation.safety_filter import PredictiveSafetyFilterConfig
from smartpick_vla.evaluation.vision_controller import (
    VisionEvaluationConfig,
    build_simulated_planar_calibration,
    run_vision_guided_evaluation,
)
from smartpick_vla.real import (
    RealLogReplay,
    XiaoUDetection,
    build_xiaou_plan_preview,
    load_real_config,
    load_real_log,
    load_xiaou_grasp_profiles,
    load_xiaou_hardware_profile,
    load_xiaou_homography,
    save_xiaou_plan_preview,
)
from smartpick_vla.training.imitation import (
    ImitationTrainingConfig,
    load_trained_policy,
    train_imitation,
)
from smartpick_vla.training.online_residual import ResidualOnlineConfig, train_residual_online
from smartpick_vla.training.vision import VisionTrainingConfig, train_vision_localizer
from smartpick_vla.utils.config import load_config


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _program_name() -> str:
    invoked_as = Path(sys.argv[0]).stem.lower()
    return invoked_as if invoked_as in {"picksort", "smartpick"} else "picksort"


def _command_doctor(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "smartpick_vla": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    if args.skip_env:
        report["environment_smoke"] = "skipped"
    else:
        env = SmartPickEnv(image_size=32, max_episode_steps=2)
        try:
            observation, info = env.reset(seed=7)
            _, reward, _, _, step_info = env.step(np.zeros(5, dtype=np.float32))
            report["environment_smoke"] = {
                "ok": True,
                "rgb_shape": list(observation["rgb"].shape),
                "state_shape": list(observation["robot_state"].shape),
                "task_class": info["task_class"],
                "first_reward": reward,
                "control_dt": step_info["control_dt"],
            }
        finally:
            env.close()
    _json_print(report)
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    payload = load_config(args.config)
    generation = GenerationConfig(**payload.get("generation", {}))
    randomization = DomainRandomizationConfig.from_dict(payload.get("domain_randomization"))
    manifest = generate_expert_dataset(
        args.output,
        config=generation,
        domain_config=randomization,
    )
    _json_print(
        {
            "dataset": str(Path(args.output)),
            "attempted_episodes": manifest["attempted_episodes"],
            "successful_episodes": manifest["successful_episodes"],
            "transitions": manifest["statistics"]["transitions"],
        }
    )
    return 0


def _command_generate_perception(args: argparse.Namespace) -> int:
    payload = load_config(args.config)
    generation = PerceptionGenerationConfig(**payload.get("generation", {}))
    randomization = DomainRandomizationConfig.from_dict(payload.get("domain_randomization"))
    manifest = generate_synthetic_perception_dataset(
        args.output,
        config=generation,
        domain_config=randomization,
    )
    _json_print(
        {
            "dataset": str(Path(args.output)),
            "samples": manifest["samples"],
            "views": manifest["views"],
            "robot_state_dim": manifest["robot_state_dim"],
            "action_dim": manifest["action_dim"],
            "synthetic_data": True,
        }
    )
    return 0


def _command_vision_calibrate(args: argparse.Namespace) -> int:
    """Generate a persisted nine-point top-camera calibration fixture."""

    environment = SmartPickEnv(image_size=args.image_size, six_axis=True)
    try:
        environment.reset(seed=args.seed)
        calibration = build_simulated_planar_calibration(
            environment,
            pixel_noise_std=args.reference_pixel_noise_std,
        )
        destination = calibration.save(args.output)
        _json_print(
            {
                "output": str(destination),
                "camera": calibration.camera_name,
                "arm_variant": "six_axis",
                "reference_point_count": int(calibration.reference_pixels.shape[0]),
                "fit_rmse_mm": calibration.fit_rmse_mm,
            }
        )
    finally:
        environment.close()
    return 0


def _command_train_vision(args: argparse.Namespace) -> int:
    payload = load_config(args.config)
    training = VisionTrainingConfig(**payload["training"])
    manifest = train_vision_localizer(
        args.dataset,
        args.output,
        training_config=training,
        model_options=payload.get("model"),
    )
    _json_print(
        {
            "output": str(Path(args.output)),
            "best_validation_mean_pixel_error": manifest["best_validation_mean_pixel_error"],
            "best_epoch": manifest["best_epoch"],
            "total_parameters": manifest["total_parameters"],
            "training_episodes": manifest["training_episodes"],
            "validation_episodes": manifest["validation_episodes"],
        }
    )
    return 0


def _command_train(args: argparse.Namespace) -> int:
    payload = load_config(args.config)
    training = ImitationTrainingConfig(**payload["training"])
    manifest = train_imitation(
        args.dataset,
        args.output,
        training_config=training,
        model_options=payload.get("model"),
        initial_checkpoint=args.init_checkpoint,
    )
    _json_print(
        {
            "output": str(Path(args.output)),
            "policy": manifest["policy_kind"],
            "device": manifest["device"],
            "total_parameters": manifest["total_parameters"],
            "trainable_parameters": manifest["trainable_parameters"],
            "global_steps": manifest["global_steps"],
            "best_validation_loss": manifest["best_validation_loss"],
        }
    )
    return 0


def _command_demo_expert(args: argparse.Namespace) -> int:
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = SmartPickEnv(image_size=args.image_size, max_episode_steps=args.max_steps)
    frames: list[np.ndarray] = []
    try:
        observation, _ = env.reset(seed=args.seed, options={"task_class": args.task_class})
        expert = IKWaypointExpert(env)
        expert.reset()
        frames.append(observation["rgb"])
        info: dict[str, Any] = {}
        for step in range(args.max_steps):
            action, _ = expert.act()
            observation, _, terminated, truncated, info = env.step(action)
            if step % args.frame_stride == 0:
                frames.append(observation["rgb"])
            if terminated or truncated:
                break
    finally:
        env.close()
    iio.imwrite(destination, frames, duration=args.frame_duration_ms, loop=0)
    _json_print(
        {
            "output": str(destination),
            "frames": len(frames),
            "success": bool(info.get("success", False)),
            "steps": step + 1,
            "seed": args.seed,
            "task_class": args.task_class,
            "grasp_assist": True,
        }
    )
    return 0 if info.get("success", False) else 2


def _command_train_residual(args: argparse.Namespace) -> int:
    payload = load_config(args.config)
    config = ResidualOnlineConfig.from_dict(payload)
    randomization = DomainRandomizationConfig.from_dict(payload.get("domain_randomization"))
    manifest = train_residual_online(
        args.base_checkpoint,
        args.output,
        config=config,
        domain_randomization=randomization,
    )
    _json_print(manifest)
    return 0


def _parse_method_checkpoints(values: list[str]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--method must use NAME=CHECKPOINT syntax")
        name, checkpoint = value.split("=", 1)
        if not name or not checkpoint:
            raise ValueError("--method must use non-empty NAME=CHECKPOINT values")
        if name in methods or name == "ik_expert":
            raise ValueError(f"duplicate or reserved method name: {name}")
        model, _ = load_trained_policy(checkpoint, device="cpu")
        methods[name] = model
    return methods


def _command_evaluate(args: argparse.Namespace) -> int:
    payload = load_config(args.config)
    seed_start = int(payload["seed_start"])
    episode_count = int(payload["episodes_per_suite"])
    seeds = tuple(range(seed_start, seed_start + episode_count))
    methods = _parse_method_checkpoints(args.method)
    for value in args.residual:
        if "=" not in value or "," not in value:
            raise ValueError("--residual must use NAME=BASE_CHECKPOINT,RESIDUAL_CHECKPOINT syntax")
        name, checkpoint_pair = value.split("=", 1)
        base_checkpoint, residual_checkpoint = checkpoint_pair.split(",", 1)
        if not name or not base_checkpoint or not residual_checkpoint or name in methods:
            raise ValueError("invalid or duplicate --residual specification")
        methods[name] = ResidualPolicyController(
            base_checkpoint, residual_checkpoint, device=args.device
        )
    if args.include_expert:
        methods["ik_expert"] = None
    if not methods:
        raise ValueError("provide at least one --method or --include-expert")
    randomization = DomainRandomizationConfig.from_dict(
        payload.get("physics_randomization", {"enabled": True})
    )
    perception_randomization = DomainRandomizationConfig.from_dict(
        payload.get(
            "perception_randomization",
            {
                "enabled": True,
                "object_mass_scale": [1.0, 1.0],
                "friction_scale": [1.0, 1.0],
                "camera_position_std_m": 0.0,
                "camera_fovy_delta_deg": 0.0,
                "light_intensity_scale": [1.0, 1.0],
                "object_color_jitter": 0.0,
                "robot_state_noise_std": 0.0,
                "detection_noise_std_m": 0.0,
                "control_delay_steps": [0, 0],
                "image_noise_std_px": 14.0,
                "image_occlusion_probability": 0.55,
                "image_occlusion_max_fraction": 0.20,
                "vision_latency_frames": [1, 3],
            },
        )
    )
    config = BenchmarkConfig(
        seeds=seeds,
        suites=tuple(payload["suites"]),
        experiment_tier="smoke" if episode_count < 20 else "local-benchmark",
        image_size=int(payload.get("image_size", 96)),
        max_episode_steps=int(payload.get("max_episode_steps", 180)),
        device=args.device,
        replan_interval=int(payload.get("replan_interval", 1)),
        six_axis=bool(payload.get("six_axis", False)),
        mission_length=int(payload.get("mission_length", 1)),
        safety_filter=(
            None
            if payload.get("safety_filter") is None
            else PredictiveSafetyFilterConfig(**payload["safety_filter"])
        ),
        physics_randomization=randomization,
        perception_randomization=perception_randomization,
        gif_seeds=(seeds[0],) if args.gif else (),
        gif_frame_stride=int(payload.get("gif_frame_stride", 2)),
    )
    run = run_benchmark(args.output, methods, config=config)
    successes = {
        method: sum(result.success for result in run.results if result.method == method)
        for method in methods
    }
    _json_print(
        {
            "output": str(Path(args.output)),
            "episodes_per_method": len(run.seed_manifest),
            "methods": list(methods),
            "success_counts": successes,
            "episodes_csv": str(run.artifacts.episode_csv),
            "summary_json": str(run.artifacts.summary_json),
            "media": [str(item.gif_path) for item in run.media],
        }
    )
    return 0


def _command_evaluate_vision(args: argparse.Namespace) -> int:
    payload = load_config(args.config)
    evaluation_payload = payload.get("evaluation", payload)
    config = VisionEvaluationConfig.from_dict(evaluation_payload)
    summary = run_vision_guided_evaluation(
        args.checkpoint,
        args.output,
        config=config,
    )
    _json_print(
        {
            "output": str(Path(args.output)),
            "success_rate": summary["aggregate"]["success_rate"],
            "mean_localization_error_mm": summary["aggregate"]["mean_localization_error_mm"],
            "task_selection_accuracy": summary["aggregate"]["task_selection_accuracy"],
            "mean_perception_to_action_latency_ms": summary["aggregate"][
                "mean_perception_to_action_latency_ms"
            ],
            "media": summary["artifacts"]["gifs"],
        }
    )
    return 0


def _command_real_validate(args: argparse.Namespace) -> int:
    runtime = load_real_config(args.config)
    episodes = load_real_log(
        args.log,
        calibration=runtime.camera_to_base,
        require_images=args.require_images,
    )
    frame_count = sum(len(episode.steps) for episode in episodes)
    _json_print(
        {
            "log": str(Path(args.log)),
            "episodes": len(episodes),
            "frames": frame_count,
            "episode_ids": [episode.episode_id for episode in episodes],
            "execution_config": asdict(runtime.execution),
            "physical_success_rate_claimed": False,
        }
    )
    return 0


def _command_real_replay(args: argparse.Namespace) -> int:
    runtime = load_real_config(args.config)
    episodes = load_real_log(args.log, calibration=runtime.camera_to_base)
    summaries = []
    for episode in episodes:
        replay = RealLogReplay(
            episode,
            speed=runtime.replay.speed,
            sample_period_s=runtime.replay.sample_period_s if args.resample else None,
        )
        frames = list(replay.frames())
        summaries.append(
            {
                "episode_id": episode.episode_id,
                "source_steps": len(episode.steps),
                "replay_frames": len(frames),
                "replay_duration_s": frames[-1].relative_time_s if frames else 0.0,
                "logged_success": episode.success,
            }
        )
    _json_print(
        {
            "scope": "real-log replay / Sim2Real preparation",
            "realtime": False,
            "physical_success_rate_claimed": False,
            "episodes": summaries,
        }
    )
    return 0


def _command_xiaou_preview(args: argparse.Namespace) -> int:
    """Create a planning-only XiaoU six-axis target preview."""

    homography = load_xiaou_homography(args.homography)
    profiles = load_xiaou_grasp_profiles(args.profiles)
    detection = XiaoUDetection(
        label=args.label,
        u_px=args.u_px,
        v_px=args.v_px,
        confidence=args.confidence,
        stamp_s=args.stamp_s,
    )
    preview = build_xiaou_plan_preview(
        detection,
        homography=homography,
        profiles=profiles,
        task_id=args.task_id,
        minimum_confidence=args.minimum_confidence,
        maximum_calibration_error_mm=args.maximum_calibration_error_mm,
    )
    payload = preview.to_dict()
    if args.output is not None:
        payload["output"] = str(save_xiaou_plan_preview(preview, args.output))
    _json_print(payload)
    return 0


def _command_xiaou_hardware_profile(args: argparse.Namespace) -> int:
    """Inspect the versioned XiaoU baseline without opening a transport."""

    payload = load_xiaou_hardware_profile(args.profile).to_dict()
    payload["scope"] = "hardware baseline inspection only"
    payload["hardware_transport_opened"] = False
    _json_print(payload)
    return 0


def _command_report(args: argparse.Namespace) -> int:
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for value in args.training:
        if "=" not in value:
            raise ValueError("--training must use NAME=HISTORY_CSV syntax")
        name, history_csv = value.split("=", 1)
        output = plot_learning_curves(
            history_csv,
            destination / f"learning_{name}.png",
            x_column="epoch",
            metric_columns=("train_loss", "validation_loss"),
            title=f"{name} supervised learning curves",
        )
        generated.append(str(output))
    if args.residual_updates:
        output = plot_learning_curves(
            args.residual_updates,
            destination / "learning_residual_sac.png",
            x_column="environment_step",
            metric_columns=("critic_loss", "actor_loss", "mean_absolute_residual"),
            rolling_window=25,
            title="Bounded residual SAC learning curves",
        )
        generated.append(str(output))
    if args.evaluation_csv:
        for metric in ("success_rate", "collision_rate", "timeout_rate"):
            output = plot_grouped_evaluation(
                args.evaluation_csv,
                destination / f"evaluation_{metric}.png",
                metric=metric,
                title=f"Smoke benchmark {metric.replace('_', ' ')}",
            )
            generated.append(str(output))
    _json_print({"output": str(destination), "generated": generated})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_program_name(),
        description="PickSort-VLA simulation, training, and Real2Sim2Real tools",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="report runtime and run a MuJoCo smoke check")
    doctor.add_argument("--skip-env", action="store_true")
    doctor.set_defaults(handler=_command_doctor)

    generate = subparsers.add_parser("generate", help="collect privileged IK demonstrations")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.set_defaults(handler=_command_generate)

    perception = subparsers.add_parser(
        "generate-perception",
        help="collect synthetic multi-view RGB-D and instance labels from MuJoCo",
    )
    perception.add_argument("--config", required=True)
    perception.add_argument("--output", required=True)
    perception.set_defaults(handler=_command_generate_perception)

    calibration = subparsers.add_parser(
        "vision-calibrate",
        help="save a nine-point top-camera calibration artifact for six-axis vision control",
    )
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--seed", type=int, default=721)
    calibration.add_argument("--image-size", type=int, default=64)
    calibration.add_argument("--reference-pixel-noise-std", type=float, default=0.0)
    calibration.set_defaults(handler=_command_vision_calibrate)

    train_vision = subparsers.add_parser(
        "train-vision",
        help="train a class-conditioned RGB target localizer from synthetic labels",
    )
    train_vision.add_argument("--config", required=True)
    train_vision.add_argument("--dataset", required=True)
    train_vision.add_argument("--output", required=True)
    train_vision.set_defaults(handler=_command_train_vision)

    train = subparsers.add_parser(
        "train", help="train BC, Compact VLA, or Temporal VLA from demonstrations"
    )
    train.add_argument("--config", required=True)
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--init-checkpoint")
    train.set_defaults(handler=_command_train)

    residual = subparsers.add_parser(
        "train-residual", help="train bounded residual SAC around a frozen VLA"
    )
    residual.add_argument("--config", required=True)
    residual.add_argument("--base-checkpoint", required=True)
    residual.add_argument("--output", required=True)
    residual.set_defaults(handler=_command_train_residual)

    evaluate = subparsers.add_parser(
        "evaluate", help="run paired ID/OOD/paraphrase/physics/perception evaluation"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--method", action="append", default=[], metavar="NAME=CHECKPOINT")
    evaluate.add_argument(
        "--residual",
        action="append",
        default=[],
        metavar="NAME=BASE_CHECKPOINT,RESIDUAL_CHECKPOINT",
    )
    evaluate.add_argument("--include-expert", action="store_true")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--gif", action="store_true")
    evaluate.set_defaults(handler=_command_evaluate)

    evaluate_vision = subparsers.add_parser(
        "evaluate-vision",
        help="evaluate RGB + calibration six-axis visual closed-loop sorting",
    )
    evaluate_vision.add_argument("--config", required=True)
    evaluate_vision.add_argument("--checkpoint", required=True)
    evaluate_vision.add_argument("--output", required=True)
    evaluate_vision.set_defaults(handler=_command_evaluate_vision)

    demo = subparsers.add_parser("demo-expert", help="render a fixed-seed IK expert GIF")
    demo.add_argument("--output", required=True)
    demo.add_argument("--seed", type=int, default=11)
    demo.add_argument("--task-class", choices=("accepted", "scratch", "unknown"), default="scratch")
    demo.add_argument("--image-size", type=int, default=128)
    demo.add_argument("--max-steps", type=int, default=180)
    demo.add_argument("--frame-stride", type=int, default=2)
    demo.add_argument("--frame-duration-ms", type=int, default=80)
    demo.set_defaults(handler=_command_demo_expert)

    validate = subparsers.add_parser("real-validate", help="validate a real-log contract")
    validate.add_argument("--config", default="configs/real/default.yaml")
    validate.add_argument("--log", required=True)
    validate.add_argument("--require-images", action="store_true")
    validate.set_defaults(handler=_command_real_validate)

    replay = subparsers.add_parser("real-replay", help="deterministically replay a real log")
    replay.add_argument("--config", default="configs/real/default.yaml")
    replay.add_argument("--log", required=True)
    replay.add_argument("--resample", action="store_true")
    replay.set_defaults(handler=_command_real_replay)

    xiaou_preview = subparsers.add_parser(
        "xiaou-preview",
        help="create a planning-only six-axis target preview from a XiaoU camera detection",
    )
    xiaou_preview.add_argument("--homography", required=True)
    xiaou_preview.add_argument("--profiles", required=True)
    xiaou_preview.add_argument("--label", required=True)
    xiaou_preview.add_argument("--u-px", type=float, required=True)
    xiaou_preview.add_argument("--v-px", type=float, required=True)
    xiaou_preview.add_argument("--confidence", type=float, default=0.90)
    xiaou_preview.add_argument("--stamp-s", type=float, default=0.0)
    xiaou_preview.add_argument("--task-id", default="xiaou-plan-preview")
    xiaou_preview.add_argument("--minimum-confidence", type=float, default=0.55)
    xiaou_preview.add_argument("--maximum-calibration-error-mm", type=float, default=2.0)
    xiaou_preview.add_argument("--output")
    xiaou_preview.set_defaults(handler=_command_xiaou_preview)

    xiaou_hardware_profile = subparsers.add_parser(
        "xiaou-hardware-profile",
        help="inspect the XiaoU six-axis baseline without opening a hardware transport",
    )
    xiaou_hardware_profile.add_argument(
        "--profile", default="configs/real/xiaou_hardware_profile.yaml"
    )
    xiaou_hardware_profile.set_defaults(handler=_command_xiaou_hardware_profile)

    report = subparsers.add_parser("report", help="regenerate plots from raw CSV files")
    report.add_argument("--training", action="append", default=[], metavar="NAME=HISTORY_CSV")
    report.add_argument("--residual-updates")
    report.add_argument("--evaluation-csv")
    report.add_argument("--output", required=True)
    report.set_defaults(handler=_command_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
