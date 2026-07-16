"""Real2Sim2Real interfaces and dry-run safety boundary."""

from .calibration import (
    CalibrationResult,
    RigidTransform,
    estimate_rigid_transform,
    load_calibration,
    save_calibration,
)
from .config import RealRuntimeConfig, ReplayConfig, load_real_config
from .execution import (
    ActionAuditLogger,
    ExecutionGate,
    ExecutionMode,
    ExecutionSignals,
    GateConfig,
    PipelineResult,
    SafeCommandPipeline,
)
from .logs import (
    REAL_LOG_SCHEMA_VERSION,
    RealLogEpisode,
    RealLogStep,
    RealLogValidationError,
    load_real_log,
    save_real_log,
    transform_step_to_base,
)
from .replay import RealLogReplay, ReplayFrame, resample_episode
from .safety import (
    SafetyCode,
    SafetyIssue,
    SafetyLimits,
    SafetyReport,
    SafetySupervisor,
    SafetyViolation,
)
from .system_parameters import SystemParameters
from .types import (
    ACTION_DIM,
    ACTION_ORDER,
    BASE_FRAME,
    ActionChunk,
    CartesianDelta,
    ComposedActionChunk,
    RobotState,
    zero_chunk_like,
)

__all__ = [
    "ACTION_DIM",
    "ACTION_ORDER",
    "BASE_FRAME",
    "REAL_LOG_SCHEMA_VERSION",
    "ActionAuditLogger",
    "ActionChunk",
    "CalibrationResult",
    "CartesianDelta",
    "ComposedActionChunk",
    "ExecutionGate",
    "ExecutionMode",
    "ExecutionSignals",
    "GateConfig",
    "PipelineResult",
    "RealLogEpisode",
    "RealLogReplay",
    "RealLogStep",
    "RealLogValidationError",
    "RealRuntimeConfig",
    "ReplayConfig",
    "ReplayFrame",
    "RigidTransform",
    "RobotState",
    "SafeCommandPipeline",
    "SafetyCode",
    "SafetyIssue",
    "SafetyLimits",
    "SafetyReport",
    "SafetySupervisor",
    "SafetyViolation",
    "SystemParameters",
    "estimate_rigid_transform",
    "load_calibration",
    "load_real_config",
    "load_real_log",
    "resample_episode",
    "save_calibration",
    "save_real_log",
    "transform_step_to_base",
    "zero_chunk_like",
]
