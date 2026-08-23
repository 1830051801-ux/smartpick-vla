# PickSort-VLA

[![CI](https://github.com/1830051801-ux/picksort-vla/actions/workflows/ci.yml/badge.svg)](https://github.com/1830051801-ux/picksort-vla/actions/workflows/ci.yml)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-3776AB)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PickSort-VLA is a six-axis tabletop manipulation and evaluation platform built
around MuJoCo, Gymnasium and PyTorch. It connects camera-space localization,
language-conditioned task selection, Cartesian action chunks, simulation
evaluation, ONNX export and a ROS 2 dry-run boundary in one reproducible code
base.

The project is intended to show how a vision-to-action pipeline is engineered
and evaluated. The learned policy is a compact VLA, not a large pretrained
foundation model. The repository includes real-log replay and dry-run
interfaces, but it is not a calibrated industrial robot controller or evidence
of physical-robot success.

## At A Glance

~~~
RGB observation + instruction + robot state
        -> target localization and task parsing
        -> six-axis Cartesian action chunk
        -> copied-state predictive simulation check
        -> MuJoCo rollout or ROS 2 dry-run preview
~~~

The six-axis path is the primary current interface. The older five-axis action
contract remains available for compatibility with earlier checkpoints and
examples. The Python distribution is still named smartpick-vla and the import
package is smartpick_vla; picksort is the preferred CLI and smartpick remains
an alias.

## Demonstration

![IK expert demonstration](results/smoke/media/ik_expert/id-seed-9100.gif)

The GIF is one recorded simulator episode from the tracked smoke evidence. It
is a visual demonstration, not a benchmark by itself. The six-axis release
path, manifests and limitations are documented in
[Embodied simulation upgrade](docs/EMBODIED_SIMULATION_UPGRADE.md).

## What Is In The Repository

| Area | Implementation | Evidence or entry point |
| --- | --- | --- |
| Environment | MuJoCo tabletop cell, movable parts, trays, collisions, camera views and deterministic seeds | src/smartpick_vla/envs/ |
| Perception | Synthetic RGB-D/mask export, spatial heatmap localizer, calibration fixture and detector-noise controls | src/smartpick_vla/data/ and src/smartpick_vla/deployment/ |
| Policies | IK demonstrations, behavior cloning, ACT-style action chunks, temporal VLA and bounded residual SAC | src/smartpick_vla/models/ and src/smartpick_vla/training/ |
| Data and evaluation | Episode-safe splits, ID/paraphrase/OOD/physics suites, industrial matrix, resumable manifests and Wilson intervals | src/smartpick_vla/evaluation/ |
| World model | Compact action-conditioned state dynamics, event-risk heads, uncertainty diagnostics and imagined rollouts | src/smartpick_vla/models/world_model.py |
| Deployment | PyTorch-to-ONNX export, graph validation, runtime parity and SHA-256 manifests | src/smartpick_vla/deployment/ |
| ROS 2 | Typed action preview and optional predictive-risk monitor with fail-closed preview behavior | ros2_ws/ |
| XiaoU bridge | Source-traced six-axis profile and camera-homography planning preview | docs/XIAOU_BRIDGE.md |

## Verified Local Evidence

The numbers below are from recorded local simulator runs. They are kept next
to their configuration and machine-readable sidecars; no number below is a
claim about a physical arm or an industrial inspection line.

| Check | Recorded result | Interpretation |
| --- | ---: | --- |
| Six-axis RGB-guided evaluation | 15/15 episode success, 0/15 collision episodes | Simulation-only release run; grasp_assist=true |
| Localization | 12.21 mm mean, 18.22 mm P95 | Episode-disjoint simulated validation/evaluation split |
| Perception-to-action latency | 4.63 ms mean, 5.52 ms P95 | Local PyTorch run; not a hardware timing guarantee |
| ONNX parity | Max absolute error below 1.2e-6 | PyTorch vs ONNX Runtime on the exported localizer |
| World-model smoke | 569 transitions, episode-disjoint split | Small dynamics/risk diagnostic, not calibrated uncertainty |
| Learned industrial smoke policy | Did not succeed at the checked budget | Negative result retained in the experiment record |

The detailed run record is in [RESULTS_20260822.md](docs/RESULTS_20260822.md).
The claim policy, data provenance and known limitations are in the
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md),
[data card](docs/DATA_CARD.md), and [model card](docs/MODEL_CARD.md).

## Quick Start

Python 3.11-3.13 is supported. The commands below stay in the simulator and
do not open a serial port, CAN socket, vendor SDK, or robot transport.

### Windows PowerShell

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m smartpick_vla doctor
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m smartpick_vla demo-expert --seed 11 --task-class scratch --output results/quickstart/expert.gif
~~~

For ONNX export and parity checks, install the optional deployment extra:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,onnx]"
.\.venv\Scripts\python.exe -m smartpick_vla export-vision-onnx --checkpoint checkpoints/vision_six_axis_release_v2/best.pt --output outputs/vision_localizer.onnx
~~~

The full reproducible smoke workflow is kept in
[scripts/run_smoke.ps1](scripts/run_smoke.ps1). It performs training and is
therefore intentionally separate from CI.

## Six-Axis Vision Loop

The current release path is deliberately staged so each boundary can be
inspected independently:

1. Generate RGB-D, masks, boxes, language labels and expert actions from the
   MuJoCo scene.
2. Fit the simulated nine-point camera-to-table homography and train the RGB
   spatial heatmap localizer.
3. Parse the active instruction, confirm the target across frames and convert
   the localization into a bounded Cartesian waypoint sequence.
4. Run a copied-state predictive safety check before each simulator step.
5. Record episode rows, collision events, latency, calibration parameters and
   artifact hashes.

The explicit commands and the exact scope of the 15-seed run are in
[Embodied simulation upgrade](docs/EMBODIED_SIMULATION_UPGRADE.md). The
three-object mission and the world-model path have separate evidence and are
not silently combined with the single-object visual score.

## ROS 2 Boundary

The ROS package is a typed preview interface. It validates action shape,
finite values, bounds, freshness and optional predictive risk, but it does not
send motor commands. The default real-runtime configuration is
dry_run=true, hardware_enabled=false, and calibrated=false.

See [ROS 2 dry-run](docs/ROS2_DRY_RUN.md) and
[Real2Sim2Real](docs/REAL2SIM2REAL.md) before adapting the interfaces to a
hardware system. A physical deployment would still require measured
calibration, a versioned robot state interface, an independent watchdog and a
separate bring-up record.

## Repository Layout

~~~
src/smartpick_vla/                 environment, models, data and evaluation
configs/                           versioned smoke and experiment contracts
ros2_ws/src/                       ROS 2 dry-run package and typed messages
examples/real_logs/                sanitized replay-schema examples
docs/                              architecture, claims, results and release rules
scripts/                           CLI wrappers, smoke runs and release checks
tests/                             unit and contract tests
results/smoke/                     small tracked evidence used by CI and README
~~~

Generated datasets, long-run checkpoints, local logs and large media are not
part of the default source snapshot. Read
[PUBLICATION.md](docs/PUBLICATION.md) before adding experiment output to a
public commit.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Embodied simulation upgrade](docs/EMBODIED_SIMULATION_UPGRADE.md)
- [World-model Transformer](docs/WORLD_MODEL.md)
- [Industrial experiment matrix](docs/INDUSTRIAL_EXPERIMENTS.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Real2Sim2Real](docs/REAL2SIM2REAL.md)
- [ROS 2 dry-run](docs/ROS2_DRY_RUN.md)
- [Data card](docs/DATA_CARD.md)
- [Model card](docs/MODEL_CARD.md)
- [Limitations](docs/LIMITATIONS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Release process](docs/RELEASE.md)
- [Public repository scope](docs/PUBLICATION.md)

## License

Released under the MIT License. See [LICENSE](LICENSE). Third-party runtime
and media obligations are summarized in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
