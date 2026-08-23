# Model card

## Model family

PickSort-VLA defines three learned policy families:

| Variant | Output | Role |
| --- | --- | --- |
| Behavior cloning | one normalized 5D action | single-step imitation baseline |
| Compact VLA | normalized action chunk `[H,5]` | RGB-language-state chunk policy |
| Residual SAC | bounded 5D correction | correction to a frozen Compact VLA base |

The repository also contains a separate predictive model:

| Variant | Output | Role |
| --- | --- | --- |
| World-model Transformer v2 | next 29D state, reward, event risks, state std | offline dynamics and ROS 2 risk preview |

This card describes the architecture and reporting contract. It does not claim
that a particular checkpoint has been trained or achieved a particular score.
Every distributed checkpoint requires its own metadata and evaluation summary.

### RGB spatial localizer

The six-axis visual closed-loop path uses a separate class-conditioned RGB
localizer rather than an action policy. `spatial_heatmap_v1` retains a spatial
feature grid, appends coordinate channels, predicts one heatmap and visibility
logit per quality class, and uses soft-argmax for the grasp-keypoint estimate.
The controller combines that prediction with a persisted camera homography and
the parsed language task; it does not pass MuJoCo object poses to the model.

The current checkpoint is
`checkpoints/vision_six_axis_release_v2/best.pt` (305,702 parameters). Its
episode-disjoint synthetic validation error is 1.606 px, with a separately
recorded 15-seed closed-loop simulation evaluation in
`results/vision_guided_six_axis_release_v2`. These are simulated results, not
real-camera or real-robot measurements.

## Intended use

- Research and education on language-conditioned simulated manipulation.
- Local comparison of single-step BC, action chunking, domain randomization,
  and bounded residual RL.
- Replaying compatible logs in simulation.
- Producing ROS 2 action previews in dry-run mode.

## Out-of-scope use

- Direct control of a physical robot without an independent safety layer and
  hardware validation.
- Industrial quality inspection or safety-critical classification.
- Human-robot collaboration, medical, automotive-safety, or autonomous factory
  deployment.
- General-purpose language understanding or open-world manipulation.
- Claims of equivalence to large pretrained VLA foundation models.

## Inputs

- RGB image from the configured fixed camera.
- Natural-language sorting instruction.
- Raw or explicitly preprocessed robot-state vector matching checkpoint
  metadata. The default environment emits a 24-value `float32` vector and does
  not normalize it automatically.

The learned policy must not receive privileged object poses used by the IK
expert or reward unless the checkpoint is explicitly labeled as an oracle
ablation.

## Outputs

The canonical order is:

```text
[dx, dy, dz, dyaw, gripper]
```

Policy outputs are normalized to `[-1,1]`. Environment, real-log, and ROS 2
adapters convert to or store physical metres/radians with `frame_id=base_link`.
Compact VLA predicts a configurable horizon, currently eight by default. A
residual checkpoint is incomplete without the exact base checkpoint and
per-dimension residual-scale vector.

### World-model Transformer

The six-axis world model consumes RGB, language, a 29D robot-state history, a
6D action history, and an optional planned 6D action. It predicts a residual
from the last valid state, five event-risk logits, reward, and a positive
per-state standard-deviation diagnostic. Its imagined rollout can evaluate a
candidate action sequence, but holds the latest image because it has no learned
pixel renderer. See [`WORLD_MODEL.md`](WORLD_MODEL.md) for the exact contract,
smoke evidence, and ROS 2 adapter limits.

## Architecture

### Behavior cloning

A compact CoordConv encoder produces visual tokens, a learned byte-level
Transformer encodes the instruction, and an MLP encodes robot state. Pooled
features pass through a small action head.

### Compact VLA

Visual grid tokens, byte-language tokens, and a robot-state token form memory
for a Transformer decoder. Learned action queries decode a complete action
chunk in parallel. The default uses width 128, four heads, two decoder layers,
and horizon eight. This is an ACT-style deterministic chunk decoder, not the
full ACT conditional variational model.

### Residual SAC

The base VLA is frozen. A stochastic actor predicts a `tanh`-bounded residual;
the system multiplies it by a per-dimension bound, adds it to the base action,
clips the normalized action, and applies safety projection. Twin critics and
target networks follow the SAC objective implemented by the release.

## Training data

The expected primary data source is MuJoCo demonstration episodes collected by
the privileged waypoint/IK expert. Optional imported real logs remain labeled
with their original source. See the [data card](DATA_CARD.md).

Model metadata must record:

- dataset manifest and checksum;
- episode counts by split, class, and source;
- whether failed expert attempts were excluded;
- domain-randomization settings;
- image and state preprocessing;
- action horizon and scales;
- seed, optimizer, learning-rate schedule, batch size, and update budget;
- total/trainable parameter counts and dependency versions.

## Evaluation

Use the ID, OOD, instruction-paraphrase, and physics/sensing suites defined in
the [experiment protocol](EXPERIMENT_PROTOCOL.md). Required metrics include
success, collision, wrong pick/bin, timeout, successful-cycle time, and sample
count. Residual policies must be paired with an evaluation of their frozen base
on identical episodes.

### Published results

No metric is asserted by this model card. Smoke and benchmark summaries belong
in generated JSON/CSV artifacts and may be linked only after they exist. A
checkpoint without episode-level evaluation artifacts is unevaluated.

## Limitations and risks

- The language encoder is trained from project data and has limited semantic
  coverage.
- Synthetic visual cues and small scenes enable shortcut learning.
- BC can fail after drifting from expert states.
- Residual RL can reduce performance despite its magnitude bound.
- Simulation dynamics and grasp assist do not establish real-world behavior.
- Normalized actions are unsafe if interpreted with the wrong scale, order, or
  coordinate frame.

See [limitations](LIMITATIONS.md) for the full discussion.

## Checkpoint loading and trust

PyTorch checkpoints may use pickle-backed serialization. Load only artifacts
from a trusted source and verify their checksum. A loader should reject a
checkpoint whose schema version, state dimension, action order, horizon, or
normalization metadata is incompatible with the runtime.

## Hardware safety

Model output is a proposal, not an authorization to move hardware. The ROS 2
reference path is dry-run by default and publishes preview/status data. Any
future hardware executor must independently enforce workspace, velocity,
acceleration, joint, gripper, timeout, heartbeat, and emergency-stop constraints.

## Environmental considerations

The models are intentionally small to reduce compute and make experiments
repeatable on consumer hardware. Reports should include wall-clock time and
hardware so training cost is visible.
