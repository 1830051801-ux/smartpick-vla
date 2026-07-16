"""Tests for episode evidence, aggregates, plots, and qualitative media."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from smartpick_vla.evaluation import (
    EpisodeAccumulator,
    EpisodeResult,
    aggregate_episode_results,
    group_episode_results,
    load_episode_results,
    plot_grouped_evaluation,
    plot_learning_curves,
    save_episode_gif,
    save_episode_results,
    wilson_interval,
    write_evaluation_artifacts,
)


def _episode_results() -> tuple[EpisodeResult, ...]:
    return (
        EpisodeResult(
            episode_id="id-accepted-000",
            suite="id",
            method="compact_vla",
            seed=101,
            task_class="accepted",
            success=True,
            collision=False,
            cycle_time_s=2.4,
            episode_return=11.5,
            inference_latency_ms=3.2,
            inference_latency_p95_ms=4.1,
            step_count=48,
            action_smoothness=0.08,
            instruction="put the accepted part in its tray",
            instruction_template_id="train/accepted/0",
            metadata={"friction": 0.82, "camera_variant": "nominal"},
        ),
        EpisodeResult(
            episode_id="id-scratch-001",
            suite="id",
            method="compact_vla",
            seed=102,
            task_class="scratch",
            success=False,
            collision=True,
            cycle_time_s=7.0,
            episode_return=-1.25,
            inference_latency_ms=3.6,
            inference_latency_p95_ms=4.8,
            step_count=140,
            timeout=True,
            instruction="sort the scratched part",
            instruction_template_id="train/scratch/0",
            metadata={"friction": 0.77, "control_delay_steps": 1},
        ),
        EpisodeResult(
            episode_id="physics-unknown-002",
            suite="physics",
            method="compact_vla_dr",
            seed=201,
            task_class="unknown",
            success=True,
            collision=False,
            cycle_time_s=3.1,
            episode_return=10.0,
            inference_latency_ms=3.4,
            inference_latency_p95_ms=4.4,
            step_count=62,
            mean_residual_magnitude=0.0,
            p95_residual_magnitude=0.0,
            instruction="send the unknown part for inspection",
            instruction_template_id="paraphrase/unknown/0",
            metadata={"mass_scale": 1.2, "detection_noise_std_m": 0.004},
        ),
    )


def test_accumulator_wilson_interval_and_success_only_cycle_time() -> None:
    accumulator = EpisodeAccumulator(
        episode_id="episode-0",
        suite="id",
        method="bc",
        seed=7,
        task_class="accepted",
    )
    accumulator.record_step(reward=1.0, inference_latency_ms=2.0)
    accumulator.record_step(reward=-0.25, inference_latency_ms=4.0, collision=True)
    successful = accumulator.finalize(success=True, cycle_time_s=0.4)
    failed = EpisodeResult(
        episode_id="episode-1",
        suite="id",
        method="bc",
        seed=8,
        task_class="scratch",
        success=False,
        collision=False,
        cycle_time_s=9.0,
        episode_return=-2.0,
        inference_latency_ms=5.0,
        step_count=50,
        timeout=True,
    )

    assert successful.episode_return == pytest.approx(0.75)
    assert successful.inference_latency_ms == pytest.approx(3.0)
    assert successful.inference_latency_p95_ms == pytest.approx(3.9)
    assert successful.collision
    aggregate = aggregate_episode_results((successful, failed), group={"method": "bc"})
    assert aggregate.success.rate == pytest.approx(0.5)
    assert aggregate.success.wilson95_low < 0.5 < aggregate.success.wilson95_high
    assert aggregate.successful_cycle_count == 1
    assert aggregate.mean_success_cycle_time_s == pytest.approx(0.4)
    assert aggregate.mean_episode_return == pytest.approx(-0.625)

    lower, upper = wilson_interval(0, 10)
    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(0.2775328, abs=1e-6)
    with pytest.raises(RuntimeError, match="already finalized"):
        accumulator.finalize(success=True, cycle_time_s=0.4)


def test_csv_round_trip_and_json_summary_remain_episode_backed(tmp_path: Path) -> None:
    expected = _episode_results()
    artifacts = write_evaluation_artifacts(
        tmp_path / "run",
        expected,
        group_by=("suite", "method", "task_class"),
    )

    assert load_episode_results(artifacts.episode_csv) == expected
    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert summary["source_episode_csv"] == "eval_episodes.csv"
    assert summary["episode_count"] == 3
    assert summary["group_by"] == ["suite", "method", "task_class"]
    assert summary["confidence_interval"].startswith("Wilson")
    assert sum(group["episode_count"] for group in summary["groups"]) == 3
    assert all("success_rate_wilson95_low" in group for group in summary["groups"])

    with pytest.raises(ValueError, match="duplicate evaluation episode identity"):
        save_episode_results(tmp_path / "duplicate.csv", (expected[0], expected[0]))


def test_grouping_and_plots_are_regenerated_from_raw_csv(tmp_path: Path) -> None:
    results = _episode_results()
    episode_csv = save_episode_results(tmp_path / "eval_episodes.csv", results)
    groups = group_episode_results(results, group_by=("suite", "method"))
    assert [dict(group.group) for group in groups] == [
        {"suite": "id", "method": "compact_vla"},
        {"suite": "physics", "method": "compact_vla_dr"},
    ]

    train_csv = tmp_path / "train_metrics.csv"
    pd.DataFrame(
        {
            "step": [0, 1, 2, 0, 1, 2],
            "seed": [1, 1, 1, 2, 2, 2],
            "loss": [1.0, 0.7, 0.5, 1.1, 0.8, 0.6],
            "eval_success_rate": [0.0, 0.25, 0.5, 0.0, 0.2, 0.4],
        }
    ).to_csv(train_csv, index=False)
    learning_plot = plot_learning_curves(
        train_csv,
        tmp_path / "figures" / "learning.png",
        metric_columns=("loss", "eval_success_rate"),
        group_column="seed",
        rolling_window=2,
    )
    grouped_plot = plot_grouped_evaluation(
        episode_csv,
        tmp_path / "figures" / "success_by_suite.png",
        group_by=("suite", "method"),
    )

    for plot in (learning_plot, grouped_plot):
        assert plot.stat().st_size > 1_000
        with Image.open(plot) as image:
            image.verify()


def test_gif_has_episode_provenance_sidecar(tmp_path: Path) -> None:
    result = _episode_results()[0]
    frames = [np.full((20, 24, 3), (index / 2.0), dtype=np.float32) for index in range(3)]
    gif_path, sidecar_path = save_episode_gif(
        frames,
        tmp_path / "media" / "episode.gif",
        result,
        fps=5.0,
        extra_metadata={"checkpoint": "compact-vla-smoke.pt"},
    )

    with Image.open(gif_path) as image:
        assert image.n_frames == 3
        assert image.size == (24, 20)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["episode_id"] == result.episode_id
    assert sidecar["method"] == result.method
    assert sidecar["suite"] == result.suite
    assert sidecar["success"] is result.success
    assert sidecar["frame_count"] == 3
    assert sidecar["checkpoint"] == "compact-vla-smoke.pt"
