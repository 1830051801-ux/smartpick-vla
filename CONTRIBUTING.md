# Contributing to PickSort-VLA

Keep PickSort-VLA small, inspectable, and reproducible. Claims must stay tied
to committed evidence.

## Before opening a change

- Search existing issues and pull requests.
- For a large environment, data-schema, action-contract, or ROS interface
  change, open a design issue first.
- Do not commit private robot logs, credentials, facility images, or third-party
  assets without documented permission and license.
- Keep the canonical action order and units backward compatible, or provide a
  schema version and migration path.

## Development setup

Python 3.11-3.13 is supported. Python 3.11 is the conservative choice for a
Windows training environment.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Bash:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the local checks before submitting:

```bash
ruff check .
ruff format --check .
mypy src/smartpick_vla
python -m bandit -r src scripts ros2_ws -q -ll -ii
python -m pip_audit --local --skip-editable
pytest -m "not slow"
python scripts/check_release.py
python -m build
```

Before a tag, also follow [docs/RELEASE.md](docs/RELEASE.md) and run the strict
gate with the intended version, for example
`python scripts/check_release.py --strict --tag v0.1.0`.

MuJoCo rendering on Linux CI uses a software/off-screen backend. GitHub's
Windows runners execute the platform-safe suite because they do not expose a
working OpenGL context; use a Windows machine with OpenGL for renderer changes.

## Code expectations

- Add type annotations to public and non-trivial internal functions.
- Keep modules small and preserve environment/model/data/real-robot boundaries.
- Validate shapes, units, frames, schema versions, and finite values at external
  interfaces.
- Pass random-number generators or seeds explicitly; avoid hidden global
  randomness.
- Add tests for normal behavior and failure behavior.
- Do not weaken safety checks to make a demo pass.
- Avoid a heavy dependency when a small, maintained dependency or standard
  library implementation is sufficient.

## Tests

Changes should add the smallest useful test at the appropriate layer:

- unit tests for tokenization, model shapes, metrics, log validation, action
  composition, and safety bounds;
- Gymnasium/MuJoCo integration tests for reset/step, seeded task selection,
  controller bounds, and expert phases;
- short training smoke tests only where they are stable and appropriately
  marked;
- ROS-independent fixtures for ROS message semantics where possible.

Never make a stochastic performance threshold part of the fast CI suite unless
the confidence and failure behavior are understood. A deterministic shape or
finite-loss check is usually better for CI.

## Experiment contributions

A result contribution includes:

- resolved configuration and exact command;
- run tier (`smoke`, `local_benchmark`, `long`, or `real_log_replay`);
- dataset/evaluation manifests and checksums;
- seeds, dependency versions, hardware, and source revision;
- episode-level CSV, aggregate JSON, and checkpoint metadata;
- failed/aborted run information where it affects selection;
- plots generated from the committed machine-readable metrics.

Do not enter metrics manually in documentation. Smoke results must be labeled
as pipeline checks. Simulation and real-log replay must never be described as
real-robot performance. See
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Data contributions

Confirm provenance, consent, license, and privacy before sharing logs. Synthetic
example logs must say they are synthetic. Keep raw generated datasets and large
checkpoints out of Git history; publish them as versioned assets with checksums
when appropriate.

## Documentation

- Explain what is implemented, what was actually run, and what remains
  unverified.
- Use “Compact VLA” or “ACT-style action chunking”; do not imply the model is an
  LLM-scale VLA or the full ACT CVAE.
- Link result statements to JSON/CSV evidence.
- Keep Windows and Bash commands where behavior differs.
- Update the model card, data card, limitations, and third-party notices when a
  change affects them.
- Preserve the source/environment/seed/artifact contract in
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Pull requests

Keep each pull request focused. The description should include motivation,
changed contracts, test commands and outcomes, reproducibility impact, safety
impact, and any result artifacts. Reviewers may request a migration for schema
changes or reject untraceable result claims.

By contributing, you agree that your code contribution is licensed under this
repository's MIT License and that you have the right to provide it.
