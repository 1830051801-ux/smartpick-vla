"""Headless plots regenerated only from machine-readable CSV artifacts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smartpick_vla.evaluation.artifacts import load_episode_results
from smartpick_vla.evaluation.metrics import (
    AggregateMetrics,
    RateEstimate,
    group_episode_results,
)


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("plot output must use .png, .pdf, or .svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_learning_curves(
    train_metrics_csv: str | Path,
    output_path: str | Path,
    *,
    x_column: str = "step",
    metric_columns: Sequence[str],
    group_column: str | None = None,
    rolling_window: int = 1,
    title: str = "Training metrics",
) -> Path:
    """Plot one or more learning curves from the unmodified training CSV.

    A rolling window is optional and explicit. The underlying points always
    come from ``train_metrics_csv``; this helper never synthesizes missing
    runs or interpolates across seeds.
    """

    if not metric_columns:
        raise ValueError("metric_columns must contain at least one column")
    if rolling_window < 1:
        raise ValueError("rolling_window must be positive")
    source = Path(train_metrics_csv)
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("training metrics CSV contains no rows")
    required = {x_column, *metric_columns}
    if group_column is not None:
        required.add(group_column)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"training metrics CSV is missing columns: {sorted(missing)}")
    numeric = frame[[x_column, *metric_columns]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("learning-curve columns contain NaN or infinity")
    frame = frame.copy()
    frame[[x_column, *metric_columns]] = numeric

    figure, axes_value = plt.subplots(
        len(metric_columns),
        1,
        figsize=(8.0, max(3.2, 3.0 * len(metric_columns))),
        squeeze=False,
        sharex=True,
    )
    axes = axes_value[:, 0]
    groups: list[tuple[str, pd.DataFrame]]
    if group_column is None:
        groups = [("", frame)]
    else:
        groups = [
            (str(group), subset)
            for group, subset in frame.groupby(group_column, sort=True, dropna=False)
        ]
    for axis, metric in zip(axes, metric_columns, strict=True):
        for group_name, subset in groups:
            ordered = subset.sort_values(x_column, kind="stable")
            values = ordered[metric]
            if rolling_window > 1:
                values = values.rolling(rolling_window, min_periods=1).mean()
            label = group_name if group_column is not None else metric
            axis.plot(ordered[x_column], values, linewidth=1.8, label=label)
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        if group_column is not None:
            axis.legend(title=group_column, frameon=False)
    axes[-1].set_xlabel(x_column)
    figure.suptitle(title)
    figure.tight_layout()
    output = _prepare_output(output_path)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


_RATE_METRICS = {
    "success_rate": "success",
    "collision_rate": "collision",
    "wrong_pick_rate": "wrong_pick",
    "wrong_bin_rate": "wrong_bin",
    "timeout_rate": "timeout",
}


def _rate_for_metric(aggregate: AggregateMetrics, metric: str) -> RateEstimate:
    try:
        attribute = _RATE_METRICS[metric]
    except KeyError as error:
        raise ValueError(f"unsupported grouped rate metric: {metric}") from error
    rate = getattr(aggregate, attribute)
    if not isinstance(rate, RateEstimate):
        raise TypeError("aggregate rate has an invalid type")
    return rate


def plot_grouped_evaluation(
    episode_csv: str | Path,
    output_path: str | Path,
    *,
    group_by: Sequence[str] = ("suite", "method"),
    metric: str = "success_rate",
    title: str | None = None,
) -> Path:
    """Create an episode-grouped rate chart with Wilson 95% error bars."""

    results = load_episode_results(episode_csv)
    aggregates = group_episode_results(results, group_by=group_by)
    estimates = [_rate_for_metric(aggregate, metric) for aggregate in aggregates]
    values = np.asarray([estimate.rate for estimate in estimates], dtype=np.float64)
    errors = np.asarray(
        [
            [estimate.rate - estimate.wilson95_low for estimate in estimates],
            [estimate.wilson95_high - estimate.rate for estimate in estimates],
        ],
        dtype=np.float64,
    )
    labels = [
        "\n".join(f"{name}={aggregate.group[name]}" for name in group_by)
        if group_by
        else "all episodes"
        for aggregate in aggregates
    ]

    if tuple(group_by) == ("suite", "method"):
        preferred_suites = ("id", "paraphrase", "ood", "physics")
        available_suites = {str(item.group["suite"]) for item in aggregates}
        suites = [name for name in preferred_suites if name in available_suites]
        suites.extend(sorted(available_suites.difference(suites)))
        methods = sorted({str(item.group["method"]) for item in aggregates})
        lookup = {
            (str(item.group["suite"]), str(item.group["method"])): item for item in aggregates
        }
        figure, axis = plt.subplots(figsize=(10.5, 5.6))
        centers = np.arange(len(suites), dtype=np.float64)
        group_width = 0.82
        bar_width = group_width / max(1, len(methods))
        palette = plt.get_cmap("tab10")
        for method_index, method in enumerate(methods):
            method_aggregates = [lookup.get((suite, method)) for suite in suites]
            method_estimates = [
                None if item is None else _rate_for_metric(item, metric)
                for item in method_aggregates
            ]
            method_values = np.asarray(
                [0.0 if item is None else item.rate for item in method_estimates]
            )
            method_errors = np.asarray(
                [
                    [
                        0.0 if item is None else item.rate - item.wilson95_low
                        for item in method_estimates
                    ],
                    [
                        0.0 if item is None else item.wilson95_high - item.rate
                        for item in method_estimates
                    ],
                ]
            )
            offset = (method_index - (len(methods) - 1) / 2.0) * bar_width
            bars = axis.bar(
                centers + offset,
                method_values,
                width=bar_width * 0.92,
                yerr=method_errors,
                capsize=3,
                label=method,
                color=palette(method_index),
                alpha=0.9,
            )
            for bar, aggregate in zip(bars, method_aggregates, strict=True):
                if aggregate is None:
                    label = "n=0"
                else:
                    estimate = _rate_for_metric(aggregate, metric)
                    label = f"{estimate.positive_count}/{aggregate.episode_count}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    min(0.97, bar.get_height() + 0.025),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        axis.set_xticks(centers, suites)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_xlabel("evaluation suite")
        axis.set_title(title or f"{metric.replace('_', ' ').title()} by suite and method")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="method", frameon=False, ncol=min(3, len(methods)))
        figure.tight_layout()
        output = _prepare_output(output_path)
        figure.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return output

    width = max(6.5, 1.25 * len(aggregates))
    figure, axis = plt.subplots(figsize=(width, 4.8))
    positions = np.arange(len(aggregates))
    bars = axis.bar(positions, values, yerr=errors, capsize=4, color="#3978b5")
    axis.set_xticks(positions, labels)
    if len(labels) > 6:
        axis.tick_params(axis="x", labelrotation=35)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(metric.replace("_", " "))
    axis.set_title(title or f"{metric.replace('_', ' ').title()} by episode group")
    axis.grid(axis="y", alpha=0.25)
    for bar, aggregate in zip(bars, aggregates, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(0.98, bar.get_height() + 0.03),
            f"n={aggregate.episode_count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    output = _prepare_output(output_path)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    if not output.exists() or not math.isfinite(float(values.sum())):
        raise RuntimeError("grouped evaluation plot was not created")
    return output
