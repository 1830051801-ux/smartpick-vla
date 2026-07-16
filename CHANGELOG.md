# Changelog

All notable changes to PickSort-VLA are documented here. The project follows
the structure of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
intends to use semantic versioning after its first tagged release.

## [Unreleased]

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
