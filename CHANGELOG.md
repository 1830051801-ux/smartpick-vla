# Changelog

All notable changes to PickSort-VLA are documented here. The project follows
the structure of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
intends to use semantic versioning after its first tagged release.

## [Unreleased]

### Added

- Public repository scope document and ignore rules for separating reproducible
  source from local long-run datasets, checkpoints, logs and media.
- Episode-safe multimodal world-model Transformer v2 with residual next-state
  dynamics, learned state-uncertainty diagnostics, action-conditioned imagined
  rollouts, and a documented six-axis training/evaluation contract.
- ROS 2 `PredictiveRisk` fields for wrong-pick risk, state uncertainty, model
  identity, and machine-readable risk reasons; the monitor now propagates the
  complete action plan and can fail closed when predictive blocking is enabled.
- Camera-clear six-axis home pose selected by fixed-seed visibility/collision
  checks, with a seed-909 top-camera visibility regression test.
- Versioned `vision_six_axis_release_v2` synthetic perception dataset,
  spatial-heatmap RGB localizer checkpoint, and recorded 15-seed visual
  closed-loop simulation evidence.
- Optional six-axis simulator control with a sixth tool-roll joint, a 29D
  state/6D action contract, and preserved legacy five-axis compatibility.
- Ordered one-to-three-object missions with language instruction handoff after
  each completed sort, plus mission-aware privileged IK demonstrations.
- Synthetic multi-view MuJoCo perception export with RGB, depth, segmentation
  masks, visible-pixel counts, boxes, actions, language, hashes, and explicit
  synthetic-data provenance.
- Copied-state predictive simulator safety filter with geometric motion scaling
  and per-episode intervention reports in benchmark metadata.
- Six-axis mission/safety/perception configs, regression coverage, and recorded
  upgrade smoke artifacts with a visual mission showcase.
- History-aware Temporal VLA policy, episode-safe observation-history dataset
  windows, temporal evaluation controller, and a reproducible smoke config.
- Camera-stress domain randomization with image noise, partial occlusion, and
  bounded visual latency, plus an optional paired `perception` suite.
- XiaoU pixel-homography to six-axis planning-preview adapter with complete
  grasp-profile validation, ROS 2-compatible pose previews, and no hardware
  transport path.
- Source-traced XiaoU six-axis hardware profile with mechanical dimensions,
  joint limits, Pi-F407 UART/CAN contracts, 26-byte synchronized trajectory
  schema, software output boundary, read-only inspection CLI, and explicit
  hardware-readiness gates.

### Safety

- XiaoU profiles with unknown heights are rejected rather than given default
  numeric grasp values; all XiaoU output remains `planning_only=true`.
- XiaoU hardware profile loading keeps `real_motion_ready=false` and
  `hardware_execution_enabled=false`; it never opens serial, CAN, ROS, or a
  motor transport.

## [0.1.1] - 2026-07-17

### Changed

- Renamed the public project and GitHub repository from SmartPick-VLA to
  PickSort-VLA to make the grasp-and-sort scope explicit.
- Added `picksort` as the primary command while retaining `smartpick` as a
  compatibility alias.
- Kept the `smartpick-vla` distribution, `smartpick_vla` Python import,
  `smartpick_vla_ros2` package, ROS topic names, checkpoint schemas, and Gym
  environment ID stable so existing scripts and artifacts continue to work.

## [0.1.0] - 2026-07-16

### Added

- Primitive-only MuJoCo sorting cell with accepted, scratch, and unknown tasks.
- Gymnasium environment contracts, seeded instruction splits, domain
  randomization, and five-dimensional Cartesian action schema.
- Privileged waypoint/IK expert and demonstration-data interfaces.
- RGB-language-state behavior cloning and compact ACT-style action-chunk policy.
- Bounded residual action composition and residual SAC training boundary.
- ID, OOD, paraphrase, and physics/sensing evaluation protocol with
  machine-readable artifact requirements.
- Real-log validation/replay modules and ROS 2 dry-run interface design.
- Model card, data card, architecture, limitations, safety, contribution, and
  third-party documentation.
- Windows/Ubuntu CI coverage across Python 3.11, 3.12, and 3.13 plus
  release-readiness checks.
- GitHub issue forms, pull-request checklist, dependency update policy, and a
  tag-triggered draft-release workflow with clean-wheel verification.
- Bandit source scanning and installed-dependency vulnerability auditing in CI
  and tag-triggered release checks.
- CI separates headless Linux rendering tests from the Windows platform-safe
  suite because GitHub-hosted Windows runners do not provide an OpenGL context.
- Reproducibility and release guides covering provenance, seed/artifact
  contracts, resource-conscious local runs, claim audits, and rollback policy.
- Source distributions include the committed smoke checkpoints and complete
  evaluation evidence bundle instead of shipping code-only results claims.

### Safety

- ROS 2 execution defaults to `dry_run=true` and `hardware_enabled=false`.
- Canonical action shape, units, frame, finite-value, freshness, and bounds
  checks are documented as mandatory external-interface gates.
- Version 2 checkpoints use restricted PyTorch weights-only loading and reject
  legacy payloads that require unrestricted pickle deserialization.

### Notes

- No real-robot performance is claimed for this release.
- Smoke experiments are pipeline validation only and must remain labeled as
  such in generated results.
