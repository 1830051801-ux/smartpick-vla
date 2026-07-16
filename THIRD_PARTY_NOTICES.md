# Third-party notices

PickSort-VLA source code is distributed under the MIT License. Third-party
software, models, assets, data, and media retain their own licenses. This file
is an attribution aid, not legal advice; the installed lockfile/environment and
the upstream projects are authoritative.

## Direct runtime dependencies

| Project | Typical upstream license | Use |
| --- | --- | --- |
| Gymnasium | MIT | environment API and spaces |
| imageio | BSD-2-Clause | image/GIF/video I/O |
| imageio-ffmpeg | BSD-2-Clause for Python package; bundled FFmpeg binary has separate FFmpeg terms | video backend |
| Matplotlib | PSF-based Matplotlib license | plots |
| MuJoCo | Apache-2.0 | simulation and rendering |
| NumPy | BSD-3-Clause | arrays and numerical operations |
| pandas | BSD-3-Clause | result tables |
| Pillow | HPND | image processing and GIF export |
| PyYAML | MIT | configuration parsing |
| PyTorch | BSD-3-Clause | model training and inference |
| tqdm | MPL-2.0 and MIT | progress reporting |

Consult the installed package metadata and upstream license files before
redistributing a binary environment. Transitive dependencies are not exhaustively
listed here.

## Development dependencies

Hatchling, build, mypy, pytest, pytest-cov, and Ruff are used for packaging,
typing, testing, coverage, and linting. They are not incorporated into the
runtime source distribution; their upstream license files remain authoritative.
Twine is installed only in the release workflow to validate distribution
metadata.

## Continuous integration services and actions

GitHub-hosted workflows use GitHub's checkout, Python setup, and artifact
upload/download actions. Dependabot may propose dependency updates. These
services and workflow actions are development infrastructure and are not
incorporated into the PickSort-VLA Python distribution. Their source and terms
remain with their respective providers and repositories.

## FFmpeg

`imageio-ffmpeg` may download or install a prebuilt FFmpeg executable. FFmpeg's
effective LGPL/GPL terms depend on how that binary was configured. A distributor
of media tooling or bundled executables must inspect the binary's build/license
information and satisfy the applicable FFmpeg obligations. GIF export can be
used where distributing an FFmpeg binary is undesirable.

## MuJoCo scene and assets

The default PickSort scene is constructed from repository-authored MJCF
primitives and built-in MuJoCo texture generators. No MuJoCo Menagerie robot
mesh is required by the default scene. Added robot descriptions, meshes,
textures, fonts, sounds, or calibration targets must carry their own attribution
and redistribution terms.

## Pretrained models and checkpoints

The default Compact VLA uses repository-defined trainable encoders and does not
require a redistributed third-party pretrained model. Future optional
pretrained encoders and their weights may have terms separate from their Python
library. Add the model name, source revision, weight license, and usage
restrictions here before distribution.

Project-generated checkpoints inherit obligations from their training data and
included third-party components. The MIT code license alone does not determine
a checkpoint's or dataset's license.

## ROS 2

ROS 2 packages and message-generation tools are supplied by the selected ROS 2
distribution and retain their upstream licenses. The `ros2` optional dependency
group intentionally does not redistribute ROS itself through PyPI.

## Data and external logs

Synthetic data generated solely from MIT-licensed project assets may be licensed
separately by its creator. Imported real logs, images, annotations, robot
descriptions, and factory data retain source permissions and privacy
restrictions. Do not assume repository access grants redistribution rights.

## Updating this notice

When adding a dependency or asset:

1. record its name, source, version/revision, and purpose;
2. retain required copyright/license text;
3. confirm redistribution rights for source and binary artifacts;
4. update this notice, the data/model card where relevant, and release assets.
