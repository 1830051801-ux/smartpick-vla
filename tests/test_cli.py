"""Public CLI and lightweight configuration contract tests."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

from smartpick_vla import __version__
from smartpick_vla.cli import main
from smartpick_vla.utils.config import deep_update, load_config

PROJECT_ROOT = Path(__file__).parents[1]


def _captured_json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_version_and_required_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert __version__ in capsys.readouterr().out

    with pytest.raises(SystemExit) as missing_exit:
        main([])
    assert missing_exit.value.code == 2
    assert "required" in capsys.readouterr().err


def test_primary_and_compatibility_console_scripts() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = metadata["project"]["scripts"]
    assert scripts["picksort"] == "smartpick_vla.cli:main"
    assert scripts["smartpick"] == scripts["picksort"]


@pytest.mark.parametrize("command", ["picksort", "smartpick"])
def test_console_alias_reports_invoked_name(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [f"{command}.exe"])
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() == f"{command} {__version__}"


def test_doctor_skip_environment_is_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["doctor", "--skip-env"]) == 0
    report = _captured_json(capsys)
    assert report["smartpick_vla"] == __version__
    assert report["environment_smoke"] == "skipped"
    assert isinstance(report["torch_cuda_available"], bool)


def test_synthetic_real_log_validate_and_replay(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = PROJECT_ROOT / "configs" / "real" / "default.yaml"
    log = PROJECT_ROOT / "examples" / "real_logs" / "example_episode.jsonl"

    assert main(["real-validate", "--config", str(config), "--log", str(log)]) == 0
    validated = _captured_json(capsys)
    assert validated["episodes"] == 1
    assert validated["frames"] == 3
    assert validated["physical_success_rate_claimed"] is False

    assert (
        main(
            [
                "real-replay",
                "--config",
                str(config),
                "--log",
                str(log),
                "--resample",
            ]
        )
        == 0
    )
    replayed = _captured_json(capsys)
    assert replayed["scope"] == "real-log replay / Sim2Real preparation"
    assert replayed["physical_success_rate_claimed"] is False
    assert len(replayed["episodes"]) == 1  # type: ignore[arg-type]


def test_xiaou_hardware_profile_command_is_read_only(capsys: pytest.CaptureFixture[str]) -> None:
    profile = PROJECT_ROOT / "configs" / "real" / "xiaou_hardware_profile.yaml"
    assert main(["xiaou-hardware-profile", "--profile", str(profile)]) == 0
    report = _captured_json(capsys)
    assert report["scope"] == "hardware baseline inspection only"
    assert report["hardware_transport_opened"] is False
    assert report["mechanical"]["joint_count"] == 6  # type: ignore[index]
    assert report["execution"]["hardware_execution_enabled"] is False  # type: ignore[index]


def test_yaml_loader_and_recursive_merge_are_strict(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  width: 32\nseed: 7\n", encoding="utf-8")
    scalar_path = tmp_path / "scalar.yaml"
    scalar_path.write_text("not-a-mapping\n", encoding="utf-8")

    base = load_config(config_path)
    override = {"model": {"depth": 2}, "seed": 11}
    merged = deep_update(base, override)

    assert merged == {"model": {"width": 32, "depth": 2}, "seed": 11}
    assert base == {"model": {"width": 32}, "seed": 7}
    assert override == {"model": {"depth": 2}, "seed": 11}
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(scalar_path)
