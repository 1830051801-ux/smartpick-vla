## Purpose

Explain the problem and the smallest coherent change that solves it.

## Changed contracts

- Observation/action/schema/checkpoint/ROS changes:
- Backward compatibility or migration:

## Verification

List exact commands and outcomes. Attach machine-readable artifacts for result claims.

```text
python -m ruff check .
python -m ruff format --check .
python -m mypy src/smartpick_vla
python -m pytest -m "not slow"
python scripts/check_release.py
```

## Reproducibility and evidence

- Seeds and resolved configuration:
- Hardware and dependency versions:
- Dataset/checkpoint manifest and hashes:
- Failed or aborted runs relevant to selection:

## Safety and data review

- [ ] External actions still validate shape, units, frame, freshness, and finite values.
- [ ] ROS 2 remains dry-run by default; no vendor controller path was added implicitly.
- [ ] Simulation, smoke, and real-log replay are not presented as real-robot performance.
- [ ] Private logs, credentials, personal images, and unlicensed assets are excluded.
- [ ] Documentation and third-party notices were updated where needed.

## Checklist

- [ ] The change is scoped and type annotated.
- [ ] Deterministic normal and failure-path tests were added or the omission is explained.
- [ ] Public interfaces and generated artifacts are documented.
- [ ] No generated caches, large checkpoints, or local absolute paths are committed.
