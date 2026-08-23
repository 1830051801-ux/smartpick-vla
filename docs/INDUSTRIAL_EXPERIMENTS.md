# Industrial simulation experiments

PickSort-VLA treats the simulator as a data and model evaluation system, not
as a collection of screenshots. The experiment runner combines:

1. a six-axis MuJoCo scene with task-specific mission lengths;
2. language, layout, physics, and perception evaluation suites;
3. fixed seed manifests shared by every method;
4. resumable worker outputs and episode-level CSV evidence; and
5. summary statistics with Wilson 95% intervals and explicit simulation-only
   disclosure.

## Run

Use the small configuration first:

```powershell
.venv\Scripts\python.exe -m smartpick_vla industrial-evaluate `
  --config configs/eval/industrial_smoke.yaml `
  --include-expert `
  --method vla=checkpoints/smoke/vla/best.pt `
  --output results/industrial_smoke
```

The full local configuration is
`configs/eval/industrial_two_task.yaml`. It is intentionally a configuration
rather than a pre-filled result: a release report must contain the actual
machine, seed, checkpoint, and runtime evidence from the run that produced it.

## Evidence contract

`industrial_manifest.json` records the task/suite/seed matrix, config hash,
Git revision when available, expected methods, and worker failures.
`industrial_episodes.csv` records one row per method and episode. The sidecar
summary groups by task, suite, and method and includes success, collision,
timeout, wrong-pick, latency, and smoothness metrics inherited from the paired
benchmark.

The runner never converts an expert or simulated policy result into a physical
robot result. A real arm requires a separate calibrated log, dry-run bridge,
and hardware authorization path documented in `docs/REAL2SIM2REAL.md`.

## Why this matters for a robot-data role

This layout makes the iteration loop inspectable: change the policy or data
generator, rerun the same seed manifest, compare task and failure groups, then
promote only the checkpoint and configuration that have reproducible evidence.
The perception checkpoint can additionally be exported with
`export-vision-onnx`; its manifest compares ONNX Runtime outputs to PyTorch on
the same input before it is treated as a deployment artifact.
