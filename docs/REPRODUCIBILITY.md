# Reproducibility guide

> Windows note: recent PyTorch wheels include deeply nested third-party license
> paths. A long checkout plus an in-tree virtual environment can exceed the
> legacy path limit (`WinError 206`). Use a short virtual-environment path or
> enable Windows long paths; keep the project and data paths unchanged.

## Reproducibility target

SmartPick-VLA targets repeatable experiment construction, not bit-for-bit
identity across every GPU, driver, operating system, or MuJoCo renderer. A run
is reproducible when another operator can recover its code, dependency set,
resolved configuration, data split, seeds, checkpoint, episode-level metrics,
and aggregation procedure, then execute the same protocol without guessing.

A fixed seed is necessary but insufficient. Physics-library versions, renderer
backend, training data order, CUDA kernels, and checkpoint selection can all
change an outcome. Reports therefore preserve the environment and raw episode
records instead of treating the seed as a complete provenance record.

## Supported environment envelope

The package metadata supports Python 3.11 through 3.13. Continuous integration
exercises the fast suite on both Ubuntu and Windows with Python 3.11 and 3.13.
Python 3.12 remains inside the declared range; release candidates should also be
checked there when dependency compatibility changes.

CI installs the official CPU-only PyTorch wheel to keep the four-platform
matrix bounded. It validates CPU execution, imports, and contracts; it does not
claim CUDA-kernel or GPU-performance coverage. Local GPU runs must record their
CUDA, driver, and PyTorch build separately.

The intended local resource envelope is a Windows laptop with 16 GB RAM and an
RTX 3050-class 4 GB GPU. The model is compact. OpenVLA-scale model
loading or full fine-tuning is not part of this project.

### Windows setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -m "not slow"
```

Use `num_workers=0` for the first data-loading smoke run. It avoids Windows
spawn overhead and makes failures easier to attribute. Increase it only after a
measured throughput test. If the default MuJoCo backend is unavailable, record
the backend override with the run; do not silently switch renderers between
methods.

### Ubuntu setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
MUJOCO_GL=egl python -m pytest -m "not slow"
```

Headless hosts need compatible EGL or OSMesa system libraries. CI installs
those libraries explicitly. A rendered-pixel checksum is not a cross-platform
acceptance criterion; observation shape, type, finite model output, task state,
and stored metrics are.

## Source and environment capture

Every non-trivial run records:

- source revision and whether tracked files were modified;
- operating system and architecture;
- Python, MuJoCo, Gymnasium, NumPy, PyTorch, CUDA, driver, and renderer versions;
- CPU, GPU, and available memory;
- the exact command and working directory;
- a resolved configuration with all command-line overrides applied;
- start/end time, wall-clock duration, and termination status.

Capture an environment snapshot beside the run without treating it as a
portable lockfile:

```powershell
python -VV | Out-File -Encoding utf8 environment-python.txt
python -m pip freeze --all | Out-File -Encoding utf8 environment-pip.txt
Get-FileHash -Algorithm SHA256 checkpoints\model.pt
```

```bash
python -VV > environment-python.txt
python -m pip freeze --all > environment-pip.txt
sha256sum checkpoints/model.pt
```

`pip freeze` describes the executed environment. A published release should
also retain the package metadata and any future lock/constraints file used to
create it. Do not replace the declared version ranges in `pyproject.toml` with a
single developer machine's platform-specific freeze.

## Seed contract

Use one recorded run seed to derive, rather than reuse implicitly, seeds for:

1. Python process-level randomness;
2. NumPy generators;
3. PyTorch model initialization and sampling;
4. demonstration generation and episode splitting;
5. Gymnasium environment reset;
6. data-loader order;
7. residual replay sampling;
8. evaluation episode manifests.

Evaluation manifests store the ordered episode IDs and reset seeds. Methods in
one comparison consume the same manifest. Training seeds and evaluation seeds
remain disjoint, and held-out paraphrase/OOD templates never enter the training
augmentation pool.

When deterministic PyTorch algorithms are enabled, record that setting and any
operation that had to fall back to a nondeterministic implementation. Do not
drop a run because its seed is unfavorable. Failed and aborted seeds remain in
the run index with a reason.

## Run identity and immutable artifacts

A run ID should be unique and filesystem-safe. Once the run is complete, its
evidence directory is immutable. Corrections generate a new run or derived
report that points to the original.

Minimum layout:

```text
run_id/
  config.resolved.yaml
  manifest.json
  environment.json
  environment-python.txt
  environment-pip.txt
  checkpoints/
  train_metrics.csv
  eval_episodes.csv
  summary.json
  figures/
  media/
```

`manifest.json` ties together the method, experiment tier, source revision,
dataset manifest, checkpoint hashes, seeds, command, and artifact checksums.
`summary.json` is derived from `eval_episodes.csv`; the aggregate is never the
only retained evidence. Plots and media are also derived artifacts and must not
be edited to change the underlying values.

## Dataset and checkpoint identity

Split complete episodes before producing overlapping action windows. A dataset
manifest records every episode ID, source, class, instruction template, split,
generation configuration, and checksum. It also records attempted, rejected,
successful, and retained expert trajectories so filtering is visible.

A checkpoint is not just a tensor file. Its metadata includes:

- model type and constructor configuration;
- action order, state dimension, image preprocessing, language encoding, and
  action horizon;
- normalization statistics;
- dataset and split-manifest hashes;
- optimizer/update state when training can resume;
- total and trainable parameter counts;
- base-checkpoint hash and residual scale for residual SAC;
- source revision, dependency versions, and seed.

Loading incompatible schema, state dimension, action order, horizon, or
normalization metadata must fail rather than adapt silently. Only load trusted
PyTorch checkpoints; pickle-backed serialization can execute code.

## Resource-conscious local profile

For a 4 GB GPU, establish correctness before increasing the budget:

- start with 64-96 px RGB observations;
- use a small batch and measure peak allocation before increasing it;
- keep action chunks and compact features in memory instead of entire SAC image
  histories;
- use automatic mixed precision only after finite-loss and checkpoint-resume
  checks pass;
- avoid loading a complete image dataset into RAM;
- evaluate and export media in a separate process if training memory is tight.

These are operational starting points, not hidden benchmark settings. The
resolved run configuration, rather than this guide, determines a published
result. An out-of-memory run is retained as a failed run and may justify a new,
versioned resource profile.

## Verification ladder

Run checks in increasing cost order:

1. `python scripts/check_release.py` for structure, metadata, links, and safe
   defaults;
2. `python -m ruff check .` and `python -m ruff format --check .`;
3. `python -m mypy src/smartpick_vla`;
4. `python -m pytest -m "not slow"`;
5. a fixed-seed environment and model smoke run;
6. one complete smoke experiment with JSON/CSV/checkpoint/media artifacts;
7. multi-seed local benchmark suites;
8. long training only after the earlier artifacts are auditable.

Passing step 6 proves that the pipeline executed. It does not establish an
ordering between methods. Simulation, real-log replay, and ROS 2 dry-run do not
establish physical-robot performance.

## Comparing two reproductions

Compare the following before comparing aggregate scores:

1. source, resolved configuration, and dataset/checkpoint hashes;
2. episode manifests and task/instruction distributions;
3. dependency and hardware/renderer environment;
4. retained/rejected episode counts and termination reasons;
5. per-episode outputs;
6. aggregation code and confidence-interval method.

If rendered observations differ but physics outcomes agree, report the renderer
difference. If per-episode outcomes differ, narrow the first divergent state,
action, randomization sample, or physics event. Do not average over an
unexplained divergence and call the reproduction exact.
