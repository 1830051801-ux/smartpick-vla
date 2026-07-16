# Release process

## Release invariants

A PickSort-VLA release is a traceable source and Python-package snapshot. A
tag is not enough. Before a release candidate can become public:

- package, citation, ROS package, tag, and changelog versions agree;
- the console entry point exists in the built wheel and responds to `--help`;
- fast tests pass on Ubuntu and Windows with Python 3.11 and 3.13;
- lint, formatting, typing, source-distribution, and wheel checks pass;
- local documentation links resolve;
- repository URLs and maintainer contacts contain no placeholders;
- generated caches, secrets, private logs, and oversized local artifacts are
  not tracked;
- every reported metric links to machine-readable episode evidence;
- simulation, smoke, real-log replay, and ROS 2 dry-run remain clearly
  separated from unverified real-robot performance.

The release workflow builds a draft GitHub release. It does not publish to PyPI
and does not convert the draft into a public release. Those remain deliberate
maintainer actions after inspecting the candidate.

## 1. Prepare metadata

Choose a semantic version `X.Y.Z` and update:

1. `[project].version` in `pyproject.toml`;
2. `__version__` in `src/smartpick_vla/__init__.py`;
3. `version` in `CITATION.cff`;
4. the ROS package version in `ros2_ws/src/smartpick_vla_ros2/package.xml`;
5. the changelog, moving completed entries from `[Unreleased]` to
   `[X.Y.Z] - YYYY-MM-DD` while retaining a new empty `[Unreleased]` section.

Replace repository-owner and maintainer placeholders before the first public
tag. Do not invent an organization, DOI, support SLA, or hardware validation
record. Add a DOI only after an archive has actually issued it.

Run the strict metadata gate:

```bash
python scripts/check_release.py --strict --tag vX.Y.Z
```

Strict mode promotes unresolved repository/contact placeholders and a
non-initialized Git repository to errors. With `--tag`, it also requires the
tag version to match package/citation metadata and have a changelog section.

## 2. Audit claims and evidence

Review README, documentation, changelog, figures, captions, model card, and
release notes together. For every number, identify:

- experiment tier and method;
- resolved configuration and source revision;
- dataset, base checkpoint, and residual checkpoint hashes;
- training seeds, evaluation episode count, and suite;
- episode-level CSV and aggregate JSON;
- grasp-assist and domain-randomization settings;
- hardware and dependency environment;
- uncertainty where the protocol requires it.

Remove a result if its evidence is missing. Keep unfavorable or failed runs in
the run index. A demo GIF is evidence for one recorded episode, not a success
rate. A long-training configuration is a budget definition until the run
artifacts exist.

The initial real-system scope is real-log validation/replay, calibration,
software bounds, and ROS 2 dry-run. Unless separately collected hardware logs
exist and pass the experiment protocol, release notes must say that real-robot
task success is unverified.

## 3. Run the local verification stack

From a clean environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" twine
python -m ruff check .
python -m ruff format --check .
python -m mypy src/smartpick_vla
python -m pytest -m "not slow"
python scripts/check_release.py --strict --tag vX.Y.Z
python -m build
python -m twine check dist/*
```

On Windows PowerShell, quote the editable extra in the same way and inspect the
wheel path rather than relying on shell glob expansion for commands that do not
support it. Record these command outcomes in the release issue or checklist.

Do not skip a failing platform because another matrix cell passes. Platform-
specific rendering differences may be documented, but import, environment,
model-shape, finite-output, safety, and artifact tests are release contracts.

## 4. Test the built wheel

Editable installs can hide missing package data or entry-point errors. Create a
fresh environment and install the wheel itself:

```bash
python -m venv wheel-test
wheel-test/bin/python -m pip install --upgrade pip
wheel-test/bin/python -m pip install dist/smartpick_vla-*.whl
wheel-test/bin/picksort --help
wheel-test/bin/smartpick --help  # compatibility alias
```

PowerShell equivalent:

```powershell
py -3.11 -m venv wheel-test
.\wheel-test\Scripts\python -m pip install --upgrade pip
$wheel = Get-ChildItem dist\smartpick_vla-*.whl | Select-Object -First 1
.\wheel-test\Scripts\python -m pip install $wheel.FullName
.\wheel-test\Scripts\picksort.exe --help
.\wheel-test\Scripts\smartpick.exe --help  # compatibility alias
```

Inspect the source distribution and wheel to confirm required package assets
are present and private datasets, checkpoints, local outputs, caches, and
absolute workstation paths are absent.

## 5. Create the candidate tag

Use an annotated tag after all intended release changes are committed:

```bash
git tag -a vX.Y.Z -m "PickSort-VLA vX.Y.Z"
git push origin vX.Y.Z
```

The tag starts `.github/workflows/release.yml`. That workflow reruns the strict
release check, fast tests, package build, distribution metadata check, and
clean-wheel CLI smoke test. It generates `SHA256SUMS.txt` from those verified
distributions. On success it attaches the distributions and checksums to a
**draft** GitHub release. A manual dispatch performs the verification/build but
does not create a release because it has no immutable release tag.

## 6. Inspect and publish

Before publishing the draft:

- compare workflow artifact hashes with the files attached to the draft;
- verify the tag points to the reviewed commit;
- inspect generated release notes for unsupported claims;
- link only evidence that is immutable and access-controlled appropriately;
- confirm license and provenance for every bundled asset, dataset, checkpoint,
  model weight, and media file;
- verify installation from the attached wheel on a clean machine when the
  release changes native dependencies or package data.

PyPI publication, if added later, should use a trusted-publishing environment,
require the same verified build artifact, and be protected by a manual release
approval. Do not rebuild between GitHub and package-index publication.

## 7. Post-release and rollback

After publication, install from the public artifact and run a minimal smoke
check. Add the archive/DOI link to citation metadata only after it exists.

Never move or overwrite a published tag to repair a release. If packaging or
code is defective, mark the release and artifacts clearly, document the impact
in the changelog, and publish a new patch version. If an artifact contains a
credential, private log, or unsafe executable path, revoke the credential and
follow the security response process; deleting the visible asset alone is not
remediation.

## Maintainer checklist

- [ ] Versions and changelog date agree.
- [ ] Repository URLs and contacts are real and current.
- [ ] CI matrix and strict release check pass.
- [ ] Wheel installs in a clean environment and both `picksort --help` and the
  compatibility alias `smartpick --help` work.
- [ ] Result claims have raw JSON/CSV evidence and honest tier labels.
- [ ] Real-log/dry-run work is not called real-robot success.
- [ ] Data, checkpoint, asset, and FFmpeg obligations were reviewed.
- [ ] No secrets, private logs, caches, or local absolute paths are shipped.
- [ ] GitHub draft assets match the verified workflow artifacts.
- [ ] Public release remains reversible through a new patch, not a moved tag.
