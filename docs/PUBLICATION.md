# Public Repository Scope

This file separates the reproducible source project from local experiment
output. It is part of the release review, not a promise that every file in the
working directory belongs in a GitHub commit.

## Keep In The Source Commit

- Python source, tests, versioned configuration and the ROS 2 dry-run package.
- Sanitized examples under examples/real_logs/.
- The small results/smoke/ evidence bundle already used by CI and the README.
- Documentation that states the configuration, seed, split, artifact hashes and
  simulation or hardware scope for every reported number.
- License, citation, security, contribution and third-party notices.

## Review Before Adding

The following files can be useful for a release, but should be selected rather
than copied wholesale:

- checkpoints/vision_* and checkpoints/upgrade_*;
- datasets/generated/*;
- results/vision_*, results/upgrade_* and results/industrial_*;
- exported ONNX graphs and recorded GIF/MP4 media.

Add a result only when its sidecar manifest, resolved configuration, source
revision, seed, raw episode table and claim boundary are present. A model or
media file larger than 10 MiB belongs in a GitHub Release or Git LFS, not in a
normal source commit. Do not use git add -f to bypass the ignore rules until
the provenance and redistribution rights have been reviewed.

## Never Publish

- Real camera frames, factory screenshots, internal company documents, or
  customer/product identifiers.
- Private robot logs, calibration values, serial/CAN settings, credentials,
  .env files, local absolute paths, or machine-specific checkpoints.
- A simulation metric described as real-robot success, production accuracy,
  industrial cycle time, or safety certification.
- A failed or incomplete training run presented as a finished model result.

## Pre-Push Gate

Run the following from the repository root:

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src/smartpick_vla
python scripts/check_release.py
git diff --check
git ls-files | rg '(^|/)(\.env|.*\.pyc|build|dist|outputs|work)(/|$)'
~~~

For a tagged package release, follow RELEASE.md and run the strict tag check
after the version, changelog and citation metadata have been synchronized. The
release workflow creates a draft artifact; it does not publish unreviewed
output automatically.
