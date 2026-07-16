"""Deterministic real-log replay and optional fixed-rate resampling."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .logs import RealLogEpisode, RealLogStep
from .types import RobotState


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    relative_time_s: float
    step: RealLogStep


class RealLogReplay:
    """Replay an episode without sleeping unless explicitly requested."""

    def __init__(
        self,
        episode: RealLogEpisode,
        *,
        speed: float = 1.0,
        sample_period_s: float | None = None,
    ) -> None:
        if speed <= 0.0 or not np.isfinite(speed):
            raise ValueError("replay speed must be finite and positive")
        if sample_period_s is not None and (
            sample_period_s <= 0.0 or not np.isfinite(sample_period_s)
        ):
            raise ValueError("sample_period_s must be finite and positive")
        self.episode = (
            resample_episode(episode, sample_period_s) if sample_period_s is not None else episode
        )
        self.speed = float(speed)

    def frames(self) -> Iterator[ReplayFrame]:
        start_s = self.episode.steps[0].timestamp_s
        for step in self.episode.steps:
            yield ReplayFrame((step.timestamp_s - start_s) / self.speed, step)

    def run(
        self,
        callback: Callable[[ReplayFrame], None],
        *,
        realtime: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> int:
        start = monotonic()
        count = 0
        for frame in self.frames():
            if realtime:
                remaining = start + frame.relative_time_s - monotonic()
                if remaining > 0.0:
                    sleeper(remaining)
            callback(frame)
            count += 1
        return count


def resample_episode(episode: RealLogEpisode, sample_period_s: float) -> RealLogEpisode:
    """Resample continuous feedback linearly and discrete fields with zero-order hold.

    Zero-order hold preserves the command actually active in the source log;
    it avoids inventing intermediate policy commands.  Robot feedback is
    linearly interpolated for simulator playback and visualization.  Gripper
    feedback remains zero-order held because it represents a discrete command.
    """

    if sample_period_s <= 0.0 or not np.isfinite(sample_period_s):
        raise ValueError("sample_period_s must be finite and positive")
    source = episode.steps
    if len(source) == 1:
        return episode

    start_s = source[0].timestamp_s
    end_s = source[-1].timestamp_s
    timestamps = [
        float(timestamp)
        for timestamp in np.arange(start_s, end_s, sample_period_s, dtype=np.float64)
    ]
    if timestamps and np.isclose(timestamps[-1], end_s):
        timestamps[-1] = end_s
    else:
        timestamps.append(end_s)

    output: list[RealLogStep] = []
    source_times = np.asarray([step.timestamp_s for step in source], dtype=np.float64)
    for new_index, timestamp in enumerate(timestamps):
        right = int(np.searchsorted(source_times, timestamp, side="right"))
        left = max(0, min(len(source) - 1, right - 1))
        upper = min(len(source) - 1, left + 1)
        left_step = source[left]
        upper_step = source[upper]
        span = upper_step.timestamp_s - left_step.timestamp_s
        alpha = 0.0 if span <= 0.0 else float((timestamp - left_step.timestamp_s) / span)
        alpha = min(1.0, max(0.0, alpha))

        left_tcp = np.asarray(left_step.robot_state.tcp_position_m, dtype=np.float64)
        upper_tcp = np.asarray(upper_step.robot_state.tcp_position_m, dtype=np.float64)
        tcp = left_tcp + alpha * (upper_tcp - left_tcp)
        yaw_delta = float(
            np.arctan2(
                np.sin(upper_step.robot_state.tcp_yaw_rad - left_step.robot_state.tcp_yaw_rad),
                np.cos(upper_step.robot_state.tcp_yaw_rad - left_step.robot_state.tcp_yaw_rad),
            )
        )
        yaw = left_step.robot_state.tcp_yaw_rad + alpha * yaw_delta

        left_joints = np.asarray(left_step.robot_state.joint_positions_rad, dtype=np.float64)
        upper_joints = np.asarray(upper_step.robot_state.joint_positions_rad, dtype=np.float64)
        if left_joints.shape != upper_joints.shape:
            raise ValueError("joint-state dimensions must remain constant within an episode")
        joints = left_joints + alpha * (upper_joints - left_joints)
        # Gripper state is a discrete controller signal; never invent a partial
        # open/close transition while aligning streams.
        gripper = left_step.robot_state.gripper
        state = RobotState(
            stamp_s=float(timestamp),
            tcp_position_m=tuple(float(value) for value in tcp),  # type: ignore[arg-type]
            tcp_yaw_rad=float(yaw),
            joint_positions_rad=tuple(float(value) for value in joints),
            gripper=float(gripper),
            frame_id=left_step.robot_state.frame_id,
        )
        metadata = dict(left_step.metadata)
        metadata["resampled"] = {
            "sample_period_s": sample_period_s,
            "source_step_index": left_step.step_index,
        }
        output.append(
            RealLogStep(
                schema_version=left_step.schema_version,
                episode_id=left_step.episode_id,
                step_index=new_index,
                timestamp_s=float(timestamp),
                instruction=left_step.instruction,
                task_class=left_step.task_class,
                robot_state=state,
                action=left_step.action,
                image_file=left_step.image_file,
                success=left_step.success if timestamp < end_s else source[-1].success,
                metadata=metadata,
            )
        )
    return RealLogEpisode(episode.schema_version, episode.episode_id, tuple(output))
