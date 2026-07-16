param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    & py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv with Python 3.13 (exit code ${LASTEXITCODE})"
    }
}

Invoke-Python -m pip install -e ".[dev]"
Invoke-Python -m smartpick_vla doctor
Invoke-Python -m pytest -q
Invoke-Python -m smartpick_vla generate --config configs/train/data_smoke.yaml --output datasets/generated/smoke/expert_nominal.npz
Invoke-Python -m smartpick_vla generate --config configs/train/data_dr_smoke.yaml --output datasets/generated/smoke/expert_dr.npz
Invoke-Python -m smartpick_vla train --config configs/train/bc_smoke.yaml --dataset datasets/generated/smoke/expert_nominal.npz --output checkpoints/smoke/bc
Invoke-Python -m smartpick_vla train --config configs/train/vla_smoke.yaml --dataset datasets/generated/smoke/expert_nominal.npz --output checkpoints/smoke/vla
Invoke-Python -m smartpick_vla train --config configs/train/vla_dr_smoke.yaml --dataset datasets/generated/smoke/expert_dr.npz --output checkpoints/smoke/vla_dr --init-checkpoint checkpoints/smoke/vla/best.pt
Invoke-Python -m smartpick_vla train-residual --config configs/train/residual_smoke.yaml --base-checkpoint checkpoints/smoke/vla_dr/best.pt --output checkpoints/smoke/residual
Invoke-Python -m smartpick_vla evaluate --config configs/eval/smoke.yaml --include-expert --method bc=checkpoints/smoke/bc/best.pt --method compact_vla=checkpoints/smoke/vla/best.pt --method vla_dr=checkpoints/smoke/vla_dr/best.pt --residual vla_dr_residual=checkpoints/smoke/vla_dr/best.pt,checkpoints/smoke/residual/last.pt --output results/smoke --gif
Invoke-Python -m smartpick_vla report --training bc=checkpoints/smoke/bc/history.csv --training compact_vla=checkpoints/smoke/vla/history.csv --training vla_dr=checkpoints/smoke/vla_dr/history.csv --residual-updates checkpoints/smoke/residual/updates.csv --evaluation-csv results/smoke/eval_episodes.csv --output results/smoke/plots
Invoke-Python scripts/check_release.py
