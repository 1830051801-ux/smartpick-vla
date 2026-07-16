#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python3 -m venv .venv
fi

"$python_bin" -m pip install -e '.[dev]'
"$python_bin" -m smartpick_vla doctor
"$python_bin" -m pytest -q
"$python_bin" -m smartpick_vla generate --config configs/train/data_smoke.yaml --output datasets/generated/smoke/expert_nominal.npz
"$python_bin" -m smartpick_vla generate --config configs/train/data_dr_smoke.yaml --output datasets/generated/smoke/expert_dr.npz
"$python_bin" -m smartpick_vla train --config configs/train/bc_smoke.yaml --dataset datasets/generated/smoke/expert_nominal.npz --output checkpoints/smoke/bc
"$python_bin" -m smartpick_vla train --config configs/train/vla_smoke.yaml --dataset datasets/generated/smoke/expert_nominal.npz --output checkpoints/smoke/vla
"$python_bin" -m smartpick_vla train --config configs/train/vla_dr_smoke.yaml --dataset datasets/generated/smoke/expert_dr.npz --output checkpoints/smoke/vla_dr --init-checkpoint checkpoints/smoke/vla/best.pt
"$python_bin" -m smartpick_vla train-residual --config configs/train/residual_smoke.yaml --base-checkpoint checkpoints/smoke/vla_dr/best.pt --output checkpoints/smoke/residual
"$python_bin" -m smartpick_vla evaluate --config configs/eval/smoke.yaml --include-expert --method bc=checkpoints/smoke/bc/best.pt --method compact_vla=checkpoints/smoke/vla/best.pt --method vla_dr=checkpoints/smoke/vla_dr/best.pt --residual vla_dr_residual=checkpoints/smoke/vla_dr/best.pt,checkpoints/smoke/residual/last.pt --output results/smoke --gif
"$python_bin" -m smartpick_vla report --training bc=checkpoints/smoke/bc/history.csv --training compact_vla=checkpoints/smoke/vla/history.csv --training vla_dr=checkpoints/smoke/vla_dr/history.csv --residual-updates checkpoints/smoke/residual/updates.csv --evaluation-csv results/smoke/eval_episodes.csv --output results/smoke/plots
"$python_bin" scripts/check_release.py
