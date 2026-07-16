"""Freeze hashes, runtime versions, and exact configs for one completed smoke run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import runtime_snapshot, sha256_file


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_commit(root: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        candidate = Path("C:/Program Files/Git/cmd/git.exe")
        git = str(candidate) if candidate.is_file() else None
    if git is None:
        return None
    result = subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()
    root = args.root.resolve()
    results = (root / args.results).resolve()
    results.mkdir(parents=True, exist_ok=True)

    files = [
        root / "src/smartpick_vla/envs/assets/smartpick_scene.xml",
        root / "datasets/generated/smoke/expert_nominal.npz",
        root / "datasets/generated/smoke/expert_nominal.manifest.json",
        root / "datasets/generated/smoke/expert_dr.npz",
        root / "datasets/generated/smoke/expert_dr.manifest.json",
        root / "checkpoints/smoke/bc/best.pt",
        root / "checkpoints/smoke/bc/history.csv",
        root / "checkpoints/smoke/bc/manifest.json",
        root / "checkpoints/smoke/vla/best.pt",
        root / "checkpoints/smoke/vla/history.csv",
        root / "checkpoints/smoke/vla/manifest.json",
        root / "checkpoints/smoke/vla_dr/best.pt",
        root / "checkpoints/smoke/vla_dr/history.csv",
        root / "checkpoints/smoke/vla_dr/manifest.json",
        root / "checkpoints/smoke/residual/last.pt",
        root / "checkpoints/smoke/residual/updates.csv",
        root / "checkpoints/smoke/residual/episodes.csv",
        root / "checkpoints/smoke/residual/manifest.json",
        results / "eval_episodes.csv",
        results / "summary.json",
    ]
    files.extend(sorted((results / "plots").glob("*.png")))
    files.extend(sorted((results / "media").glob("**/*.gif")))
    files.extend(sorted((results / "media").glob("**/*.gif.json")))
    config_paths = [
        root / "configs/train/data_smoke.yaml",
        root / "configs/train/data_dr_smoke.yaml",
        root / "configs/train/bc_smoke.yaml",
        root / "configs/train/vla_smoke.yaml",
        root / "configs/train/vla_dr_smoke.yaml",
        root / "configs/train/residual_smoke.yaml",
        root / "configs/eval/smoke.yaml",
    ]
    config_destination = results / "configs"
    config_destination.mkdir(parents=True, exist_ok=True)
    for path in config_paths:
        shutil.copy2(path, config_destination / path.name)

    packages = {}
    for name in (
        "gymnasium",
        "imageio",
        "matplotlib",
        "mujoco",
        "numpy",
        "pandas",
        "pillow",
        "pyyaml",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "smartpick-smoke-run/v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_tier": "smoke",
        "claim_scope": "simulation pipeline verification; not statistical or physical-robot evidence",
        "physical_robot_trials": 0,
        "physical_robot_success_rate_claimed": False,
        "git_commit": _git_commit(root),
        "runtime": runtime_snapshot(),
        "packages": packages,
        "artifacts": [
            _artifact(path).copy() | {"path": path.relative_to(root).as_posix()} for path in files
        ],
        "configs": [
            _artifact(path).copy() | {"path": path.relative_to(root).as_posix()}
            for path in config_paths
        ],
    }
    atomic_write_json(results / "run_manifest.json", manifest)
    (results / "environment.json").write_text(
        json.dumps({"runtime": manifest["runtime"], "packages": packages}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
