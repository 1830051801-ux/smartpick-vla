#!/usr/bin/env python3
"""Perform deterministic, network-free release-readiness checks.

The checker intentionally validates repository contracts rather than running the
test suite.  CI invokes both this script and the tests so a release cannot hide a
missing entry point, stale version, broken documentation link, or placeholder
metadata behind otherwise passing unit tests.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
TOP_LEVEL_CFF_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9-]*):\s*(?P<value>.+?)\s*$")
DEPENDENCY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")

REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/INDEX.md",
    "docs/ARCHITECTURE.md",
    "docs/DATA_CARD.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/LIMITATIONS.md",
    "docs/MODEL_CARD.md",
    "docs/REAL2SIM2REAL.md",
    "docs/ROS2_DRY_RUN.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RELEASE.md",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/experiment_report.yml",
    ".github/pull_request_template.md",
    "configs/real/default.yaml",
    "examples/real_logs/example_episode.jsonl",
    "scripts/check_release.py",
    "src/smartpick_vla/__init__.py",
    "ros2_ws/src/smartpick_vla_ros2/package.xml",
    "tests",
)

REQUIRED_DOC_BOUNDARIES = {
    "README.md": ("compact vla", "real-log replay"),
    "docs/ARCHITECTURE.md": ("not an llm", "bounded residual sac"),
    "docs/EXPERIMENT_PROTOCOL.md": ("smoke", "real-log replay"),
    "docs/REAL2SIM2REAL.md": ("no verified real-robot success rate",),
    "docs/ROS2_DRY_RUN.md": ("hardware_enabled=false", "not a certified safety controller"),
}

PLACEHOLDER_PATTERNS = {
    "repository owner placeholder": re.compile(
        r"(?i)github\.com/(?:your[-_ ]?org|owner|username)/"
    ),
    "example email address": re.compile(r"(?i)(?:maintainers?|security)@example\.(?:com|org)"),
    "generic ROS maintainer address": re.compile(r"(?i)smartpick-vla@users\.noreply\.github\.com"),
    "unfinished marker": re.compile(r"(?im)(?:^|\W)(?:TODO|TBD|CHANGEME|REPLACE_ME)(?:$|\W)"),
    "construction-only README wording": re.compile(
        r"(?i)under active construction in this worktree"
    ),
}

FORBIDDEN_TRACKED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    "outputs",
    "work",
}

SECRET_NAME_RE = re.compile(
    r"(?i)(?:^|/)(?:\.env(?:\..+)?|id_rsa|id_ed25519|credentials(?:\.[^/]+)?|secrets?\.ya?ml)$"
)


@dataclass(frozen=True, slots=True)
class Finding:
    level: str
    code: str
    message: str


class ReleaseChecker:
    """Collect release-readiness findings without mutating the repository."""

    def __init__(self, root: Path, *, strict: bool, tag: str | None) -> None:
        self.root = root.resolve()
        self.strict = strict
        self.tag = tag
        self.findings: list[Finding] = []
        self.project: dict[str, Any] = {}

    def error(self, code: str, message: str) -> None:
        self.findings.append(Finding("error", code, message))

    def warn(self, code: str, message: str, *, release_blocking: bool = False) -> None:
        level = "error" if self.strict and release_blocking else "warning"
        self.findings.append(Finding(level, code, message))

    def run(self) -> None:
        self._check_required_paths()
        self._check_pyproject()
        self._check_package_version()
        self._check_citation_and_changelog()
        self._check_ros_package_metadata()
        self._check_console_scripts()
        self._check_markdown_links()
        self._check_claim_boundaries()
        self._check_placeholders()
        self._check_machine_readable_examples()
        self._check_real_runtime_defaults()
        self._check_workflows()
        self._check_git_hygiene()

    def _check_required_paths(self) -> None:
        for relative in REQUIRED_PATHS:
            path = self.root / relative
            if not path.exists():
                self.error("required-path", f"missing required path: {relative}")
            elif path.is_file() and path.stat().st_size == 0:
                self.error("empty-file", f"required file is empty: {relative}")

    def _check_pyproject(self) -> None:
        path = self.root / "pyproject.toml"
        if not path.is_file():
            return
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            self.error("pyproject-parse", f"cannot parse pyproject.toml: {exc}")
            return

        project = payload.get("project")
        if not isinstance(project, dict):
            self.error("project-table", "pyproject.toml has no [project] table")
            return
        self.project = project

        required_fields = (
            "name",
            "version",
            "description",
            "readme",
            "requires-python",
            "license",
            "authors",
            "dependencies",
        )
        for field in required_fields:
            if not project.get(field):
                self.error("project-metadata", f"[project].{field} is missing or empty")

        version = project.get("version")
        if not isinstance(version, str) or not SEMVER_RE.fullmatch(version):
            self.error("version-format", f"project version is not SemVer-compatible: {version!r}")

        python_spec = project.get("requires-python")
        if (
            not isinstance(python_spec, str)
            or ">=3.11" not in python_spec
            or "<3.14" not in python_spec
        ):
            self.error(
                "python-range",
                "requires-python must express the tested >=3.11,<3.14 range",
            )

        classifiers = project.get("classifiers", [])
        if not isinstance(classifiers, list):
            self.error("classifiers", "[project].classifiers must be a list")
        else:
            for minor in ("3.11", "3.12", "3.13"):
                expected = f"Programming Language :: Python :: {minor}"
                if expected not in classifiers:
                    self.error("classifiers", f"missing classifier: {expected}")

        dependencies = project.get("dependencies", [])
        dependency_names = {
            match.group(0).lower().replace("_", "-")
            for item in dependencies
            if isinstance(item, str) and (match := DEPENDENCY_NAME_RE.match(item))
        }
        for required in ("gymnasium", "mujoco", "numpy", "torch", "pyyaml"):
            if required not in dependency_names:
                self.error("runtime-dependency", f"missing declared runtime dependency: {required}")

        optional = project.get("optional-dependencies", {})
        dev = optional.get("dev", []) if isinstance(optional, dict) else []
        dev_names = {
            match.group(0).lower().replace("_", "-")
            for item in dev
            if isinstance(item, str) and (match := DEPENDENCY_NAME_RE.match(item))
        }
        for required in ("bandit", "build", "mypy", "pip-audit", "pytest", "ruff"):
            if required not in dev_names:
                self.error("dev-dependency", f"missing declared dev dependency: {required}")

        scripts = project.get("scripts")
        if not isinstance(scripts, dict) or not scripts:
            self.error("console-script", "[project.scripts] must expose at least one CLI")

        tool = payload.get("tool", {})
        hatch = tool.get("hatch", {}) if isinstance(tool, dict) else {}
        build = hatch.get("build", {}) if isinstance(hatch, dict) else {}
        targets = build.get("targets", {}) if isinstance(build, dict) else {}
        sdist = targets.get("sdist", {}) if isinstance(targets, dict) else {}
        includes = sdist.get("include", []) if isinstance(sdist, dict) else []
        included = (
            [str(item).rstrip("/") for item in includes] if isinstance(includes, list) else []
        )
        for release_file in (
            "CHANGELOG.md",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "THIRD_PARTY_NOTICES.md",
        ):
            if not any(fnmatch.fnmatchcase(release_file, pattern) for pattern in included):
                self.warn(
                    "sdist-metadata",
                    f"Hatch sdist include list omits {release_file}",
                    release_blocking=True,
                )
        for evidence_directory in ("checkpoints/smoke", "results/smoke"):
            if not any(fnmatch.fnmatchcase(evidence_directory, pattern) for pattern in included):
                self.warn(
                    "sdist-evidence",
                    f"Hatch sdist include list omits {evidence_directory}",
                    release_blocking=True,
                )

    def _check_citation_and_changelog(self) -> None:
        cff_path = self.root / "CITATION.cff"
        cff: dict[str, str] = {}
        if cff_path.is_file():
            try:
                for line in cff_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith((" ", "\t")):
                        continue
                    match = TOP_LEVEL_CFF_RE.match(line)
                    if match:
                        cff[match.group("key")] = match.group("value").strip("'\"")
            except (OSError, UnicodeDecodeError) as exc:
                self.error("citation-read", f"cannot read CITATION.cff: {exc}")
        for field in ("cff-version", "title", "type", "authors", "version", "license"):
            if field not in cff and field != "authors":
                self.error("citation-field", f"CITATION.cff is missing top-level {field!r}")
        if cff_path.is_file() and not re.search(
            r"(?m)^authors:\s*$", cff_path.read_text(encoding="utf-8")
        ):
            self.error("citation-field", "CITATION.cff is missing an authors list")

        version = self.project.get("version")
        if isinstance(version, str) and cff.get("version") != version:
            self.error(
                "version-sync",
                f"CITATION.cff version {cff.get('version')!r} does not match project {version!r}",
            )
        repository_code = cff.get("repository-code")
        if not repository_code:
            self.warn(
                "citation-repository",
                "CITATION.cff has no repository-code URL",
                release_blocking=True,
            )
        project_urls = self.project.get("urls", {})
        homepage = project_urls.get("Homepage") if isinstance(project_urls, dict) else None
        if not isinstance(homepage, str) or not homepage.strip():
            self.warn(
                "project-repository",
                "[project.urls].Homepage is missing",
                release_blocking=True,
            )
        if isinstance(homepage, str) and repository_code:
            normalized_home = homepage.rstrip("/").removesuffix(".git")
            normalized_cff = repository_code.rstrip("/").removesuffix(".git")
            if normalized_home != normalized_cff:
                self.error(
                    "citation-repository",
                    "CITATION.cff repository-code does not match [project.urls].Homepage",
                )

        changelog_path = self.root / "CHANGELOG.md"
        changelog = changelog_path.read_text(encoding="utf-8") if changelog_path.is_file() else ""
        if "## [Unreleased]" not in changelog:
            self.error("changelog", "CHANGELOG.md must contain an [Unreleased] section")

        if self.tag is not None:
            normalized_tag = self.tag.removeprefix("v")
            if not isinstance(version, str) or normalized_tag != version:
                self.error(
                    "tag-version",
                    f"release tag {self.tag!r} does not match project version {version!r}",
                )
            if f"## [{normalized_tag}]" not in changelog:
                self.error(
                    "tag-changelog",
                    f"CHANGELOG.md has no release section for {normalized_tag}",
                )
            if "date-released" not in cff:
                self.error("citation-date", "tagged releases require date-released in CITATION.cff")

    def _check_package_version(self) -> None:
        path = self.root / "src/smartpick_vla/__init__.py"
        if not path.is_file():
            return
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            self.error("package-version", f"cannot parse {path.relative_to(self.root)}: {exc}")
            return
        package_version: str | None = None
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == "__version__" for target in targets
            ):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    package_version = value.value
                break
        project_version = self.project.get("version")
        if package_version is None:
            self.error(
                "package-version", "src/smartpick_vla/__init__.py has no literal __version__"
            )
        elif isinstance(project_version, str) and package_version != project_version:
            self.error(
                "version-sync",
                f"package __version__ {package_version!r} does not match project {project_version!r}",
            )

    def _check_console_scripts(self) -> None:
        scripts = self.project.get("scripts", {})
        if not isinstance(scripts, dict):
            return
        for name, target in scripts.items():
            if not isinstance(target, str) or ":" not in target:
                self.error(
                    "console-target", f"console script {name!r} has invalid target {target!r}"
                )
                continue
            module_name, attribute = target.split(":", 1)
            relative_module = Path(*module_name.split("."))
            module_file = self.root / "src" / relative_module.with_suffix(".py")
            package_file = self.root / "src" / relative_module / "__init__.py"
            source_file = module_file if module_file.is_file() else package_file
            if not source_file.is_file():
                self.error(
                    "console-module",
                    f"console script {name!r} points to missing module {module_name!r}",
                )
                continue
            try:
                tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                self.error(
                    "console-parse", f"cannot parse {source_file.relative_to(self.root)}: {exc}"
                )
                continue
            top_level_names = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            if "." not in attribute and attribute not in top_level_names:
                self.error(
                    "console-attribute",
                    f"console target {target!r} is not defined at module top level",
                )

    def _check_ros_package_metadata(self) -> None:
        path = self.root / "ros2_ws/src/smartpick_vla_ros2/package.xml"
        if not path.is_file():
            return
        if path.stat().st_size > 1_000_000:
            self.error("ros-package-size", "ROS package.xml exceeds the 1 MB metadata limit")
            return
        try:
            # package.xml is a size-bounded, repository-controlled release file.
            package = ET.parse(path).getroot()  # nosec B314
        except (OSError, ET.ParseError) as exc:
            self.error("ros-package-parse", f"cannot parse {path.relative_to(self.root)}: {exc}")
            return

        package_version = package.findtext("version")
        project_version = self.project.get("version")
        if isinstance(project_version, str) and package_version != project_version:
            self.error(
                "version-sync",
                f"ROS package version {package_version!r} does not match project {project_version!r}",
            )
        repository_url = package.find("url[@type='repository']")
        if repository_url is None or not (repository_url.text or "").strip():
            self.warn(
                "ros-package-repository",
                "ROS package.xml has no repository URL",
                release_blocking=True,
            )
        if package.findtext("license") != "MIT":
            self.error(
                "ros-package-license", "ROS package license must match the MIT repository license"
            )
        maintainer = package.find("maintainer")
        if maintainer is None or not (maintainer.text or "").strip() or not maintainer.get("email"):
            self.error(
                "ros-package-maintainer", "ROS package must declare a maintainer name and email"
            )

    def _check_markdown_links(self) -> None:
        for path in self._iter_files(("*.md",)):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                self.error("markdown-read", f"cannot read {path.relative_to(self.root)}: {exc}")
                continue
            for match in MARKDOWN_LINK_RE.finditer(text):
                raw_target = match.group(1).strip()
                target = raw_target.split(maxsplit=1)[0].strip("<>")
                if not target or target.startswith(
                    ("#", "http://", "https://", "mailto:", "data:")
                ):
                    continue
                path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not path_part:
                    continue
                resolved = (path.parent / path_part).resolve()
                try:
                    resolved.relative_to(self.root)
                except ValueError:
                    self.error(
                        "markdown-link-escape",
                        f"{path.relative_to(self.root)} links outside the repository: {target}",
                    )
                    continue
                if not resolved.exists():
                    line = text.count("\n", 0, match.start()) + 1
                    self.error(
                        "markdown-link",
                        f"broken local link in {path.relative_to(self.root)}:{line}: {target}",
                    )

    def _check_claim_boundaries(self) -> None:
        for relative, phrases in REQUIRED_DOC_BOUNDARIES.items():
            path = self.root / relative
            if not path.is_file():
                continue
            normalized = " ".join(path.read_text(encoding="utf-8").lower().split())
            for phrase in phrases:
                if phrase not in normalized:
                    self.error(
                        "claim-boundary",
                        f"{relative} must retain the scope boundary phrase {phrase!r}",
                    )

    def _check_placeholders(self) -> None:
        candidates = (
            "README.md",
            "pyproject.toml",
            "CITATION.cff",
            "SECURITY.md",
            "ros2_ws/src/smartpick_vla_ros2/package.xml",
        )
        for relative in candidates:
            path = self.root / relative
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for label, pattern in PLACEHOLDER_PATTERNS.items():
                if pattern.search(text):
                    self.warn(
                        "placeholder",
                        f"{relative} contains {label}",
                        release_blocking=True,
                    )

    def _check_machine_readable_examples(self) -> None:
        examples = self.root / "examples"
        if not examples.is_dir():
            return
        for path in sorted(examples.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            try:
                if path.suffix.lower() == ".json":
                    json.loads(path.read_text(encoding="utf-8"))
                else:
                    for _line_number, line in enumerate(
                        path.read_text(encoding="utf-8").splitlines(), start=1
                    ):
                        if line.strip():
                            json.loads(line)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.error(
                    "example-json",
                    f"invalid machine-readable example {path.relative_to(self.root)}: {exc}",
                )

    def _check_real_runtime_defaults(self) -> None:
        path = self.root / "configs/real/default.yaml"
        if not path.is_file():
            return
        text = path.read_text(encoding="utf-8")
        required_patterns = {
            "five-dimensional action": r"(?m)^\s*dim:\s*5\s*$",
            "canonical base frame": r"(?m)^\s*frame_id:\s*base_link\s*$",
            "dry-run enabled": r"(?m)^\s*dry_run:\s*true\s*$",
            "hardware disabled": r"(?m)^\s*hardware_enabled:\s*false\s*$",
            "uncalibrated default": r"(?m)^\s*calibrated:\s*false\s*$",
        }
        for label, pattern in required_patterns.items():
            if re.search(pattern, text) is None:
                self.error("real-default", f"configs/real/default.yaml must retain {label}")

    def _check_workflows(self) -> None:
        ci_path = self.root / ".github/workflows/ci.yml"
        if ci_path.is_file():
            ci = ci_path.read_text(encoding="utf-8")
            for token in ("ubuntu-latest", "windows-latest", '"3.11"', '"3.13"'):
                if token not in ci:
                    self.error("ci-matrix", f"CI workflow is missing matrix token {token!r}")
            for command in ("pytest", "ruff", "mypy", "scripts/check_release.py"):
                if command not in ci:
                    self.error("ci-command", f"CI workflow does not invoke {command!r}")

        release_path = self.root / ".github/workflows/release.yml"
        if release_path.is_file():
            release = release_path.read_text(encoding="utf-8")
            for token in (
                "python -m build",
                "twine check",
                "--strict",
                "--tag",
                "SHA256SUMS.txt",
            ):
                if token not in release:
                    self.error("release-workflow", f"release workflow is missing {token!r}")
            if "pypi" in release.lower() and "workflow_dispatch" not in release:
                self.warn(
                    "release-publish",
                    "automated package publication has no manual dispatch gate",
                    release_blocking=True,
                )

    def _check_git_hygiene(self) -> None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root), "ls-files", "-z"],
                check=False,
                capture_output=True,
                text=False,
            )
        except OSError as exc:
            self.warn("git-unavailable", f"cannot inspect tracked files: {exc}")
            return
        if completed.returncode != 0:
            self.warn(
                "git-repository",
                "repository is not initialized; tracked-file hygiene was not checked",
                release_blocking=True,
            )
            return

        tracked = [Path(item.decode("utf-8")) for item in completed.stdout.split(b"\0") if item]
        for relative in tracked:
            normalized = relative.as_posix()
            if (
                any(part in FORBIDDEN_TRACKED_PARTS for part in relative.parts)
                or relative.suffix == ".pyc"
            ):
                self.error("tracked-generated", f"generated/cache file is tracked: {normalized}")
            if SECRET_NAME_RE.search(normalized):
                self.error("tracked-secret", f"secret-like file is tracked: {normalized}")
            path = self.root / relative
            if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
                self.warn(
                    "large-file",
                    f"tracked file exceeds 10 MiB; use a release asset or LFS: {normalized}",
                    release_blocking=True,
                )

    def _iter_files(self, patterns: Sequence[str]) -> Iterable[Path]:
        ignored_roots = {".git", ".venv", "venv", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
        seen: set[Path] = set()
        for pattern in patterns:
            for path in self.root.rglob(pattern):
                if not path.is_file() or any(
                    part in ignored_roots for part in path.relative_to(self.root).parts
                ):
                    continue
                if path not in seen:
                    seen.add(path)
                    yield path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of scripts/)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="promote release-blocking warnings such as placeholders to errors",
    )
    parser.add_argument(
        "--tag",
        help="tag being released, for example v0.1.0; also checks the changelog",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    checker = ReleaseChecker(args.root, strict=args.strict, tag=args.tag)
    checker.run()
    errors = sum(finding.level == "error" for finding in checker.findings)
    warnings = sum(finding.level == "warning" for finding in checker.findings)

    if args.json:
        print(
            json.dumps(
                {
                    "root": str(checker.root),
                    "strict": checker.strict,
                    "tag": checker.tag,
                    "errors": errors,
                    "warnings": warnings,
                    "findings": [asdict(finding) for finding in checker.findings],
                },
                indent=2,
            )
        )
    else:
        for finding in checker.findings:
            print(f"[{finding.level.upper()}] {finding.code}: {finding.message}")
        print(f"release check: {errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
