# Experiment protocol

## Purpose

This protocol defines the minimum evidence required to compare PickSort-VLA
methods. It prevents smoke checks, privileged experts, cherry-picked videos,
and unverified real-world claims from being mixed into one result table.

The checked-in configuration used by a run is authoritative. If this document
and a configuration disagree, fix the inconsistency before publishing results.

## Experiment tiers

| Tier | Purpose | Permitted claim |
| --- | --- | --- |
| Unit/integration | Verify shapes, bounds, determinism contracts, and file I/O | Component passes the named test |
| Smoke | Prove data generation, training, checkpoint loading, evaluation, and media export execute end to end | Pipeline executed on the recorded machine and seed |
| Local benchmark | Compare methods on a fixed multi-seed suite | Simulation result for the exact suite and configuration |
| Long training | Study scaling with more demonstrations or RL steps | Simulation result for the exact long-run budget |
| Real-log replay | Reapply recorded real actions/states in simulation | Replay or model-mismatch result, never robot task success |
| Real robot | Hardware experiment with safety review and raw logs | Only results actually collected on identified hardware |

The repository defines evidence contracts for the first five tiers. A tier is
implemented only when its named command/configuration and required artifacts
exist in that release; this protocol alone is not evidence that a run occurred.
Unless a later release adds traceable hardware logs, the real-robot tier is
unverified.

## Required methods

Evaluate these methods without silently changing their inputs or environment:

1. **IK expert**: privileged waypoint controller. Report as an expert ceiling,
   not a vision-policy baseline.
2. **BC**: single-step RGB-language-state behavior cloning.
3. **Compact VLA**: ACT-style action-chunk decoder without domain randomization.
4. **Compact VLA + DR**: the same architecture trained with the declared domain
   randomization distribution.
5. **Compact VLA + DR + residual SAC**: the frozen DR base policy plus bounded
   residual correction.

For a valid DR ablation, demonstration count, optimization steps, model shape,
and evaluation seeds stay fixed. For a valid residual ablation, the exact base
checkpoint is evaluated both alone and with the residual checkpoint.

## Data generation and splits

- Assign `train`, `validation`, and evaluation partitions at the episode level
  before constructing action chunks.
- Record attempted, successful, rejected, and retained expert episodes.
- Do not place adjacent frames or overlapping chunks from one episode in
  different splits.
- Keep evaluation layout seeds out of demonstration generation.
- Keep paraphrase and OOD instruction templates out of training, including data
  augmentation.
- Do not tune on the final test suite. Select checkpoints on validation data.
- Store the generator configuration and per-episode randomization parameters.

If failed demonstrations are excluded from supervised training, retain their
count and failure reasons in generation metadata.

## Evaluation suites

All methods are evaluated on the same ordered episode manifest.

### ID

Uses nominal or in-distribution physics, training-range object layouts, and
held-out episodes using the training-language family. It measures basic task
learning without replaying training episodes.

### OOD layout and appearance

Uses held-out pose regions or object/appearance combinations. The precise
shift must be recorded; the label `OOD` alone is insufficient.

### Instruction paraphrase

Uses only templates from the held-out paraphrase set. Report class-stratified
results because a policy that defaults to one tray can otherwise appear
partially competent.

### Physics and sensing perturbation

Applies a declared mass, friction, camera, illumination, state-noise,
detection-noise, and/or delay shift. Record sampled parameters for every
episode. If several perturbations are combined, also retain single-factor
results when the budget permits.

## Seed policy

Every run records:

- one top-level run seed;
- environment reset seeds;
- data split and sampler seeds;
- PyTorch/NumPy/Python seeds;
- the ordered evaluation episode IDs;
- model initialization seed;
- the MuJoCo, Python, PyTorch, CUDA, and driver versions.

A smoke profile may use one seed and a small evaluation manifest, but its result
must contain `experiment_tier: smoke`. Comparative simulation claims should use
multiple independent training seeds and enough evaluation episodes to report
uncertainty. The long profile is a budget definition, not evidence that the run
has been completed.

## Training protocol

### Imitation learning

- Normalize robot state and actions with training-set statistics or fixed
  environment scales; save those values in the checkpoint.
- Use future-action masks at episode boundaries.
- Select checkpoints with a validation loss or rollout metric declared before
  examining the final test set.
- Record trainable and total parameter counts from the instantiated model.
- Record examples/second, peak GPU memory when available, and wall-clock time.

### Residual SAC

- Load the declared frozen base checkpoint and place it in evaluation mode.
- Start the residual actor near zero and enforce per-dimension residual bounds.
- Store only policy-accessible observations/features in replay; the reward may
  use privileged simulation state, but the actor may not.
- Record replay warm-up, environment steps, gradient steps, batch size,
  discount, target update rate, entropy settings, reward weights, and bounds.
- Evaluate fixed checkpoints during training; do not choose a visually pleasing
  episode as the checkpoint criterion.
- Preserve a zero-residual rollout as a composition sanity check.

## Metric definitions

Let `N` be the number of evaluated episodes.

| Metric | Definition |
| --- | --- |
| Success rate | `successful episodes / N`; success requires correct target, correct tray, release, and the configured stability window |
| Collision rate | fraction of episodes with at least one disallowed contact; allowed finger-target contacts are excluded by the evaluator's named contact policy |
| Wrong-pick rate | fraction of episodes in which a non-target object is grasped or transported |
| Wrong-bin rate | fraction of episodes ending with the target in an incorrect tray |
| Timeout rate | truncated episodes divided by `N` |
| Mean cycle time | simulated seconds to success, averaged over successful episodes only; always publish success count and timeout rate beside it |
| Return | undiscounted evaluation reward under the exact reward configuration |
| Action smoothness | mean norm of consecutive physical-action differences, with units and dimensions reported |
| Residual magnitude | mean and high quantile of the physical correction, per action dimension |

Collision and cycle-time definitions must not change between methods in one
table. Report the number of episodes and a confidence interval for rates in
local-benchmark and long tiers. Bootstrap intervals must resample whole
episodes, not frames.

## Artifact contract

Each run directory should be immutable after completion and contain at least:

```text
run_id/
  config.resolved.yaml
  manifest.json
  environment.json
  checkpoints/
  train_metrics.csv
  eval_episodes.csv
  summary.json
  figures/
  media/
```

`manifest.json` identifies the experiment tier, method, dataset manifest,
checkpoint hashes, seeds, command, start/end timestamps, package versions, and
working-tree revision when available. `eval_episodes.csv` is the evidence for
aggregated `summary.json`; never keep only the aggregate.

Learning curves and bar charts must be regenerated from stored CSV/JSON. Demo
GIFs carry episode ID, method, seed, suite, and success status in adjacent
metadata. A video is qualitative evidence for that episode only.

## Reporting template

Every published result table states:

- experiment tier and suite;
- method and exact base/residual checkpoints;
- training demonstration count and optimizer/environment-step budget;
- number of training seeds and evaluation episodes;
- success, collision, wrong-pick/wrong-bin, timeout, and cycle-time metrics;
- uncertainty for benchmark rates;
- whether grasp assist and domain randomization were enabled;
- a link or path to raw machine-readable artifacts.

An empty or failed run stays visible in the run index. Failed residual training
or a method scoring below the base policy is a valid result, not a reason to
replace the seed.

## Claims policy

Do not publish:

- handwritten metrics not backed by episode-level artifacts;
- smoke metrics as a statistically meaningful comparison;
- an expert result as a learned-policy result;
- simulation or real-log replay as real-robot performance;
- “state of the art”, “production ready”, or safety-certified claims;
- an ID/OOD label without defining the distribution shift;
- a real-robot success rate without raw hardware logs and experiment metadata.

The README may summarize completed experiments, but this protocol and the raw
artifacts determine what those numbers mean.
