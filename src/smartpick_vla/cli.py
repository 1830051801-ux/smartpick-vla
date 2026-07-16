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
from smartpick_vla.envs import SmartPickEnv
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.evaluation.benchmark import BenchmarkConfig, run_benchmark
from smartpick_vla.evaluation.plots import plot_grouped_evaluation, plot_learning_curves
from smartpick_vla.evaluation.residual_controller import ResidualPolicyController
from smartpick_vla.real import RealLogReplay, load_real_config, load_real_log
from smartpick_vla.training.imitation import (
    ImitationTrainingConfig,
    load_trained_policy,
    train_imitation,
)
from smartpick_vla.training.online_residual import ResidualOnlineConfig, train_residual_online
from smartpick_vla.utils.config import load_config


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


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
    config = BenchmarkConfig(
        seeds=seeds,
        suites=tuple(payload["suites"]),
        experiment_tier="smoke" if episode_count < 20 else "local-benchmark",
        image_size=int(payload.get("image_size", 96)),
        max_episode_steps=int(payload.get("max_episode_steps", 180)),
        device=args.device,
        replan_interval=int(payload.get("replan_interval", 1)),
        physics_randomization=randomization,
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
        prog="smartpick",
        description="SmartPick-VLA simulation, training, and Real2Sim2Real tools",
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

    train = subparsers.add_parser("train", help="train BC or Compact VLA from demonstrations")
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
        "evaluate", help="run paired ID/OOD/paraphrase/physics evaluation"
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
