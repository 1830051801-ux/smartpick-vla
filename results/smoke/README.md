# Smoke result bundle

This directory contains the measured 2026-07-14 local smoke run. It is a
simulation pipeline check with one training seed and six paired evaluation
episodes per suite; it is not a statistically powered benchmark.

Primary evidence:

- `eval_episodes.csv`: all 120 episode rows (5 methods × 4 suites × 6 seeds).
- `summary.json`: aggregates derived from the CSV with Wilson 95% intervals.
- `plots/`: plots regenerated only from raw training/evaluation CSV files.
- `media/`: fixed-seed GIFs plus provenance sidecars, including failed policies.
- `run_manifest.json`: runtime, configs, SHA-256 hashes, and claim boundary.
- `environment.json`: exact local dependency/runtime summary.

Overall successes were IK expert 24/24, BC 0/24, Compact VLA 1/24, VLA+DR
2/24, and VLA+DR+residual SAC 2/24. Residual SAC did not improve the overall
count and increased collision episodes. These values are intentionally retained
as a negative/neutral result.

All episodes use the disclosed proximity-gated grasp assist. No physical-robot
trial or success rate is included.
