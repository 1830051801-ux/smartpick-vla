# World-model Transformer

## Scope

The world model is a compact, multimodal latent-dynamics model for six-axis
MuJoCo manipulation. It is used for offline transition prediction, imagined
rollout diagnostics, and optional ROS 2 predictive-risk preview. It is not a
pixel-generative foundation model, a VLA replacement, or a physical safety
certification.

## Contract

The model consumes an episode-safe observation window:

```text
rgb_history:          [B,T,3,H,W] uint8
instruction:          B language strings
robot_state_history:  [B,T,29] float32
action_history:       [B,T,6] float32
history_mask:         [B,T] bool, left-padded and contiguous
planned_action:       [B,6] float32, optional
```

The six-axis action order is `[dx, dy, dz, dyaw, droll, gripper]`. The model
predicts the next 29-value proprioceptive state, reward, terminal/truncation
and task-risk logits for collision, wrong pick, and wrong bin. It also emits a
positive per-state standard-deviation diagnostic.

## Architecture

1. A compact spatial vision encoder produces a pooled RGB token per timestep.
2. A byte-level language Transformer encodes the instruction without claiming
   unrestricted language understanding.
3. State and historical action encoders are fused with learned time embeddings.
4. A temporal Transformer models the masked observation/action context.
5. The planned action is fused at the prediction query, making the imagined
   rollout action-conditioned instead of repeating an implicit last action.
6. The dynamics head predicts a state residual from the last valid observation;
   the residual identity path improves short-horizon stability.
7. Event heads and the uncertainty head provide diagnostics for risk preview.

The rollout keeps the latest RGB frame because this repository does not contain
a learned pixel renderer. It updates the predicted state and action history at
each imagined step and can accept a candidate action sequence. A rollout is
therefore a latent/proprioceptive forecast, not a visual simulator.

## Train and evaluate

The checked-in six-axis smoke configuration is:

```powershell
.venv\Scripts\python.exe -m smartpick_vla train-world-model `
  --config configs/train/world_model_six_axis_smoke.yaml `
  --dataset datasets/generated/upgrade_20260822/world_model_six_axis_smoke.npz `
  --output checkpoints/upgrade_20260822/world_model_six_axis_smoke_v2

.venv\Scripts\python.exe -m smartpick_vla evaluate-world-model `
  --checkpoint checkpoints/upgrade_20260822/world_model_six_axis_smoke_v2/best.pt `
  --dataset datasets/generated/upgrade_20260822/world_model_six_axis_smoke.npz `
  --output results/upgrade_20260822/world_model_evaluation_v2.json
```

The training split is episode-disjoint. The manifest records the dataset hash,
model configuration, seed, split episodes, parameter count, loss history, and
the fact that physical execution is disabled.

The local v2 smoke artifact used 569 transition rows and two expert episodes.
Its full-archive evaluation reported:

| Metric | Result |
| --- | ---: |
| Next-state RMSE | 0.1831 |
| Event-head accuracy | 0.9937 |
| Mean predicted state std | 0.5183 |
| Elementwise 2-sigma coverage | 0.9958 |

These values are small-sample simulation diagnostics. They are not held-out
real-camera results, robot success rate, or industrial reliability evidence.

## ROS 2 integration

The optional `smartpick_vla_ros2` package loads a trusted checkpoint through
`PredictiveRiskMonitor`. The monitor maintains a bounded history of RGB frames,
feedback states, and actions. A ROS five-channel action chunk is mapped to the
six-axis model by inserting `droll=0` before `gripper`; the full chunk is passed
as the imagined action plan and is padded with its last action if needed.

The monitor publishes `PredictiveRisk` on `/smartpick/predictive_risk` with:

- collision, termination, wrong-pick, and wrong-bin probabilities;
- maximum predicted state standard deviation;
- model hash identifier and risk reasons;
- horizon, availability, and blocked status.

Risk preview is advisory by default. Setting `predictive_risk_blocking=true`
also fails closed when the checkpoint, camera observation, or prediction is
unavailable. This block is still only a software preview gate; the separate
dry-run/hardware interlocks remain mandatory.

Important parameters include:

```text
world_model_checkpoint
world_model_device
world_model_horizon
world_model_collision_threshold
world_model_wrong_pick_threshold
world_model_wrong_bin_threshold
world_model_termination_threshold
world_model_uncertainty_threshold
predictive_risk_blocking
```

The current `RobotState` message does not carry joint velocities or wrist-roll
feedback. The preview adapter zero-fills those missing fields and labels the
path as preview-only. A hardware deployment would need a versioned state
interface, time synchronization, measured calibration, collision checking, and
independent controller/watchdog validation.

## Reproducibility and limits

- Use only checkpoints whose manifest and SHA-256 are available and trusted.
- Keep training/validation splits episode-disjoint; random transition splits
  leak near-identical frames across the boundary.
- A high event accuracy on a mostly nominal smoke archive can hide class
  imbalance; inspect per-event precision/recall on a larger stress suite.
- The uncertainty head is a learned error diagnostic, not a calibrated
  probability or a substitute for deterministic collision limits.
- The ROS node does not open serial, CAN, vendor SDK, or motor transports.
- No real-robot performance claim is made by this model or its smoke artifact.
