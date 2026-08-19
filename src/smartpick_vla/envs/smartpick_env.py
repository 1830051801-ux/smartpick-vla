"""MuJoCo/Gymnasium language-conditioned quality sorting environment."""

from __future__ import annotations

import string
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from smartpick_vla.data.schema import QualityClass
from smartpick_vla.envs.randomization import DomainRandomizationConfig, DomainRandomizer
from smartpick_vla.envs.tasks import InstructionSplit, SortTask, sample_task

CameraName = Literal["top", "oblique", "wrist"]


@dataclass(frozen=True, slots=True)
class ControlTargets:
    """Simulator-only joint targets derived from one normalized action.

    The object is deliberately a control preview: it contains no hardware
    transport, serial, CAN, or ROS execution behavior. The predictive safety
    filter uses it to roll a candidate action forward on a copied MuJoCo state.
    """

    arm_qpos: np.ndarray
    finger_position: float
    desired_eef_position: np.ndarray
    desired_yaw: float
    desired_roll: float | None


class SmartPickEnv(gym.Env[dict[str, Any], np.ndarray]):
    """Sort quality-labelled objects using RGB, language, and robot state.

    Legacy-compatible five-axis control accepts ``[dx, dy, dz, dyaw, gripper]``.
    The optional six-axis mode exposes
    ``[dx, dy, dz, dyaw, droll, gripper]`` while retaining the same scene and
    task logic. Translation and tool orientation deltas are expressed in
    ``base_link``. A configurable proximity weld provides a deterministic
    contact latch for data generation; it is surfaced in every reset's metadata
    and is not represented as a pure-contact grasp.
    """

    metadata: ClassVar[dict[str, Any]] = {  # type: ignore[misc]
        "render_modes": ["rgb_array"],
        "render_fps": 25,
    }

    QUALITY_CLASSES: tuple[QualityClass, ...] = ("accepted", "scratch", "unknown")
    ACTION_DIM = 5
    ROBOT_STATE_DIM = 24
    SIX_AXIS_ACTION_DIM = 6
    SIX_AXIS_ROBOT_STATE_DIM = 29

    def __init__(
        self,
        *,
        image_size: int = 96,
        max_episode_steps: int = 140,
        frame_skip: int = 20,
        max_translation_m: float = 0.025,
        max_yaw_rad: float = 0.10,
        instruction_split: InstructionSplit = "train",
        domain_randomization: DomainRandomizationConfig | dict[str, Any] | None = None,
        grasp_assist: bool = True,
        six_axis: bool = False,
        mission_length: int = 1,
        reset_settle_steps: int = 80,
        render_mode: str | None = "rgb_array",
    ) -> None:
        super().__init__()
        if image_size < 32 or image_size > 192:
            raise ValueError("image_size must be in [32, 192]")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.max_translation_m = max_translation_m
        self.max_yaw_rad = max_yaw_rad
        self.instruction_split = instruction_split
        self.grasp_assist = grasp_assist
        self.six_axis = six_axis
        if mission_length < 1 or mission_length > len(self.QUALITY_CLASSES):
            raise ValueError("mission_length must be in [1,3]")
        if reset_settle_steps < 0:
            raise ValueError("reset_settle_steps must be non-negative")
        self.mission_length = mission_length
        self.reset_settle_steps = reset_settle_steps
        self.action_dim = self.SIX_AXIS_ACTION_DIM if six_axis else self.ACTION_DIM
        self.robot_state_dim = self.SIX_AXIS_ROBOT_STATE_DIM if six_axis else self.ROBOT_STATE_DIM

        asset_path = Path(__file__).resolve().parent / "assets" / "smartpick_scene.xml"
        # MuJoCo's Windows path bridge can mis-decode non-ASCII user profiles.
        # Loading this self-contained MJCF as text keeps installation paths fully
        # Unicode-safe; the model deliberately has no relative mesh includes.
        self.model = mujoco.MjModel.from_xml_string(asset_path.read_text(encoding="utf-8"))
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        dr_config = (
            domain_randomization
            if isinstance(domain_randomization, DomainRandomizationConfig)
            else DomainRandomizationConfig.from_dict(domain_randomization)
        )
        self.domain_randomizer = DomainRandomizer(self.model, dr_config)
        self._dr_config = dr_config
        self._renderer: mujoco.Renderer | None = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
                "robot_state": spaces.Box(
                    -100.0, 100.0, shape=(self.robot_state_dim,), dtype=np.float32
                ),
                "instruction": spaces.Text(
                    max_length=180,
                    min_length=1,
                    charset=string.ascii_letters + string.digits + " -",
                ),
            }
        )

        self._arm_joint_names = (
            "joint_base_yaw",
            "joint_shoulder",
            "joint_elbow",
            "joint_wrist_pitch",
            "joint_wrist_yaw",
        ) + (("joint_wrist_roll",) if self.six_axis else ())
        self._finger_joint_names = ("left_finger_joint", "right_finger_joint")
        self._arm_joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self._arm_joint_names
            ],
            dtype=np.int32,
        )
        self._finger_joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self._finger_joint_names
            ],
            dtype=np.int32,
        )
        self._arm_qpos_adr = self.model.jnt_qposadr[self._arm_joint_ids]
        self._finger_qpos_adr = self.model.jnt_qposadr[self._finger_joint_ids]
        self._arm_dof_adr = self.model.jnt_dofadr[self._arm_joint_ids]
        self._finger_dof_adr = self.model.jnt_dofadr[self._finger_joint_ids]
        self._arm_actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name,
                )
                for name in (
                    "base_yaw_act",
                    "shoulder_act",
                    "elbow_act",
                    "wrist_pitch_act",
                    "wrist_yaw_act",
                )
                + (("wrist_roll_act",) if self.six_axis else ())
            ],
            dtype=np.int32,
        )
        self._finger_actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ("left_finger_act", "right_finger_act")
            ],
            dtype=np.int32,
        )
        self._grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._camera_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("top", "oblique", "wrist")
        }
        self._top_camera_id = self._camera_ids["top"]
        self._object_body_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(3)],
            dtype=np.int32,
        )
        self._object_joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
                for i in range(3)
            ],
            dtype=np.int32,
        )
        self._object_qpos_adr = self.model.jnt_qposadr[self._object_joint_ids]
        self._bin_site_ids = {
            category: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"{category}_bin_site"
            )
            for category in self.QUALITY_CLASSES
        }
        self._weld_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_obj{i}")
                for i in range(3)
            ],
            dtype=np.int32,
        )
        self._robot_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            and (
                "arm" in name
                or "wrist" in name
                or "finger" in name
                or "palm" in name
                or "shoulder" in name
                or "elbow" in name
            )
        }
        self._obstacle_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            and (name == "table" or "_bin_" in name)
        }
        self._object_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obj{i}_geom"): i
            for i in range(3)
        }
        self._object_visual_geom_ids = {
            object_index: np.asarray(
                [
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        f"obj{object_index}_geom",
                    ),
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        f"obj{object_index}_mark",
                    ),
                ],
                dtype=np.int32,
            )
            for object_index in range(3)
        }

        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._workspace_low = np.array([-0.39, -0.18, 0.055], dtype=np.float64)
        self._workspace_high = np.array([0.39, 0.34, 0.40], dtype=np.float64)

        self._task = SortTask("accepted", "place the accepted part", "bootstrap", "train")
        self._mission_sequence: tuple[QualityClass, ...] = ("accepted",)
        self._mission_index = 0
        self._completed_task_classes: list[QualityClass] = []
        self._step_count = 0
        self._episode_collision_steps = 0
        self._contact_count = 0
        self._wrong_pick = False
        self._wrong_bin = False
        self._stable_success_steps = 0
        self._grasped_object: int | None = None
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._desired_eef_pos = np.zeros(3, dtype=np.float64)
        self._desired_yaw = 0.0
        self._desired_roll: float | None = None
        self._action_queue: deque[np.ndarray] = deque()
        self._randomization: dict[str, Any] = {}
        self._vision_frames: deque[np.ndarray] = deque(maxlen=1)
        self._layout_attempts = 0

    @property
    def task(self) -> SortTask:
        return self._task

    @property
    def target_object_index(self) -> int:
        return self.QUALITY_CLASSES.index(self._task.target_class)

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    @property
    def arm_variant(self) -> str:
        return "six_axis" if self.six_axis else "five_axis_legacy"

    @property
    def action_order(self) -> tuple[str, ...]:
        if self.six_axis:
            return ("dx", "dy", "dz", "dyaw", "droll", "gripper")
        return ("dx", "dy", "dz", "dyaw", "gripper")

    @property
    def mission_index(self) -> int:
        """Zero-based index of the active subtask in the current mission."""

        return self._mission_index

    @property
    def mission_complete(self) -> bool:
        return len(self._completed_task_classes) == len(self._mission_sequence)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        split = options.get("instruction_split", self.instruction_split)
        target_class = options.get("task_class")
        if split not in ("train", "paraphrase", "ood"):
            raise ValueError("instruction_split must be train, paraphrase, or ood")
        self._mission_sequence = self._resolve_mission_sequence(
            target_class=target_class,
            task_sequence=options.get("task_sequence"),
        )
        self._mission_index = 0
        self._completed_task_classes = []
        self._task = sample_task(
            self.np_random,
            split=split,
            target_class=self._mission_sequence[self._mission_index],
        )

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        self.data.eq_active[:] = 0
        self._randomization = self.domain_randomizer.apply(self.np_random)
        vision_latency = int(self._randomization.get("vision_latency_frames", 0))
        self._vision_frames = deque(maxlen=vision_latency + 1)
        self._layout_attempts = self._sample_object_layout(
            ood=bool(options.get("ood_layout", False))
        )
        self._settle_reset_scene()

        self._step_count = 0
        self._episode_collision_steps = 0
        self._contact_count = 0
        self._wrong_pick = False
        self._wrong_bin = False
        self._stable_success_steps = 0
        self._grasped_object = None
        self._last_action.fill(0.0)
        self._desired_eef_pos = self.data.site_xpos[self._grip_site_id].copy()
        self._desired_yaw = float(self.data.qpos[self._arm_qpos_adr[4]])
        self._desired_roll = float(self.data.qpos[self._arm_qpos_adr[5]]) if self.six_axis else None
        delay = int(self._randomization["control_delay_steps"])
        self._action_queue = deque(
            [np.zeros(self.action_dim, dtype=np.float32) for _ in range(delay)]
        )

        observation = self._get_observation()
        info = self._info(success=False, collision=False, reward_terms={})
        info["reset_seed"] = seed
        return observation, info

    def _resolve_mission_sequence(
        self,
        *,
        target_class: Any,
        task_sequence: Any,
    ) -> tuple[QualityClass, ...]:
        """Select an ordered, duplicate-free sequence of object classes."""

        if task_sequence is not None:
            if isinstance(task_sequence, str):
                raise TypeError("task_sequence must be a sequence of quality classes")
            try:
                requested_sequence = tuple(task_sequence)
            except TypeError as error:
                raise TypeError("task_sequence must be a sequence of quality classes") from error
            if not requested_sequence or len(requested_sequence) > len(self.QUALITY_CLASSES):
                raise ValueError("task_sequence length must be in [1,3]")
            if any(item not in self.QUALITY_CLASSES for item in requested_sequence):
                raise ValueError("task_sequence contains an unsupported quality class")
            if len(set(requested_sequence)) != len(requested_sequence):
                raise ValueError("task_sequence must not contain duplicate classes")
            return tuple(cast(QualityClass, item) for item in requested_sequence)

        if target_class is not None and target_class not in self.QUALITY_CLASSES:
            raise ValueError("task_class contains an unsupported quality class")
        if self.mission_length == 1:
            if target_class is None:
                selected = self.QUALITY_CLASSES[int(self.np_random.integers(0, 3))]
            else:
                selected = target_class
            return (selected,)

        available = list(self.QUALITY_CLASSES)
        sequence: list[QualityClass] = []
        if target_class is not None:
            sequence.append(target_class)
            available.remove(target_class)
        shuffled = self.np_random.permutation(available).tolist()
        sequence.extend(shuffled[: self.mission_length - len(sequence)])
        return tuple(sequence)

    def _advance_mission(self) -> None:
        """Switch the language target after a correctly placed subtask object."""

        self._completed_task_classes.append(self._task.target_class)
        self._mission_index += 1
        if self._mission_index >= len(self._mission_sequence):
            return
        self._task = sample_task(
            self.np_random,
            split=self._task.split,
            target_class=self._mission_sequence[self._mission_index],
        )
        self._stable_success_steps = 0

    def _sample_object_layout(self, *, ood: bool) -> int:
        """Sample an object layout that does not begin inside a bin or robot."""

        x_limit = 0.27 if ood else 0.21
        y_low, y_high = (-0.15, 0.085) if ood else (-0.11, 0.045)
        for layout_attempt in range(1, 201):
            positions: list[np.ndarray] = []
            for _ in range(3):
                for _candidate_attempt in range(200):
                    candidate = np.array(
                        [
                            self.np_random.uniform(-x_limit, x_limit),
                            self.np_random.uniform(y_low, y_high),
                            0.026,
                        ],
                        dtype=np.float64,
                    )
                    if all(
                        np.linalg.norm(candidate[:2] - point[:2]) > 0.105 for point in positions
                    ):
                        positions.append(candidate)
                        break
                else:
                    break
            if len(positions) != len(self.QUALITY_CLASSES):
                continue
            for object_index, position in enumerate(positions):
                address = int(self._object_qpos_adr[object_index])
                self.data.qpos[address : address + 3] = position
                yaw = float(self.np_random.uniform(-np.pi, np.pi))
                self.data.qpos[address + 3 : address + 7] = np.array(
                    [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
                )
                dof_address = int(self.model.jnt_dofadr[self._object_joint_ids[object_index]])
                self.data.qvel[dof_address : dof_address + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if not self._layout_has_invalid_contact():
                return layout_attempt
        raise RuntimeError("could not sample a collision-free object layout")

    def _layout_has_invalid_contact(self) -> bool:
        """Reject reset layouts that place a free object inside fixed geometry."""

        object_geoms = set(self._object_geom_ids)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in object_geoms and geom2 not in object_geoms:
                continue
            other = geom2 if geom1 in object_geoms else geom1
            if other in object_geoms or other in self._robot_geom_ids:
                return True
            if other in self._obstacle_geom_ids:
                other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                if other_name != "table":
                    return True
        return False

    def _settle_reset_scene(self) -> None:
        """Let gravity and contact resolution stabilize a valid reset layout."""

        self.data.ctrl[self._arm_actuator_ids] = self.data.qpos[self._arm_qpos_adr]
        self.data.ctrl[self._finger_actuator_ids] = self.data.qpos[self._finger_qpos_adr]
        if self.reset_settle_steps:
            mujoco.mj_step(self.model, self.data, nstep=self.reset_settle_steps)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        normalized = self.normalize_action(action)
        self._last_action = normalized.copy()
        self._action_queue.append(normalized)
        applied = self._action_queue.popleft()

        targets = self.preview_control_targets(applied)
        self._desired_eef_pos = targets.desired_eef_position
        self._desired_yaw = targets.desired_yaw
        self._desired_roll = targets.desired_roll
        self.data.ctrl[self._arm_actuator_ids] = targets.arm_qpos
        self.data.ctrl[self._finger_actuator_ids] = targets.finger_position

        self._update_grasp_latch(gripper_command=float(applied[-1]))
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._step_count += 1

        collision, new_contacts = self._detect_collision()
        self._contact_count += new_contacts
        if collision:
            self._episode_collision_steps += 1

        placed_task_class = self._task.target_class
        subtask_success = self._check_success()
        final_subtask = self._mission_index == len(self._mission_sequence) - 1
        success = bool(subtask_success and final_subtask)
        self._wrong_bin = self._check_wrong_bin()
        target_position = self.object_position(self.target_object_index)
        bin_position = self.bin_position(self._task.target_class)
        gripper_position = self.gripper_position()
        grasp_bonus = 1.0 if self._grasped_object == self.target_object_index else 0.0
        reward_terms = {
            "reach": -0.35 * float(np.linalg.norm(gripper_position - target_position)),
            "transport": -0.20 * float(np.linalg.norm(target_position - bin_position)),
            "grasp": 0.20 * grasp_bonus,
            "collision": -0.35 if collision else 0.0,
            "wrong_pick": -1.0 if self._wrong_pick else 0.0,
            "wrong_bin": -3.0 if self._wrong_bin else 0.0,
            "subtask": 4.0 if subtask_success and not final_subtask else 0.0,
            "success": 12.0 if success else 0.0,
            "time": -0.01,
        }
        reward = float(sum(reward_terms.values()))
        terminated = bool(success or self._wrong_bin)
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        if subtask_success:
            if final_subtask:
                self._completed_task_classes.append(placed_task_class)
            elif not terminated and not truncated:
                self._advance_mission()
        observation = self._get_observation()
        info = self._info(
            success=success,
            collision=collision,
            reward_terms=reward_terms,
            subtask_success=subtask_success,
            placed_task_class=placed_task_class if subtask_success else None,
        )
        return observation, reward, terminated, truncated, info

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and clip an action against the active arm variant."""

        normalized = np.asarray(action, dtype=np.float32)
        if normalized.shape != (self.action_dim,):
            raise ValueError(f"expected action shape {(self.action_dim,)}, got {normalized.shape}")
        if not np.isfinite(normalized).all():
            raise ValueError("action contains NaN or infinity")
        return np.clip(normalized, -1.0, 1.0)

    def preview_control_targets(self, action: np.ndarray) -> ControlTargets:
        """Return a MuJoCo-only joint-target preview for a candidate action.

        The calculation is anchored to the measured simulator state, which
        prevents Cartesian target wind-up near a workspace boundary. It is
        intentionally side-effect free with respect to ``self.data`` so a
        copied state can be used for safety rollouts.
        """

        normalized = self.normalize_action(action)
        desired_position = np.clip(
            self.gripper_position() + normalized[:3] * self.max_translation_m,
            self._workspace_low,
            self._workspace_high,
        )
        measured_yaw = float(
            self.data.qpos[self._arm_qpos_adr[0]] + self.data.qpos[self._arm_qpos_adr[4]]
        )
        desired_yaw = float(np.clip(measured_yaw + normalized[3] * self.max_yaw_rad, -2.8, 2.8))
        desired_roll: float | None = None
        if self.six_axis:
            measured_roll = float(self.data.qpos[self._arm_qpos_adr[5]])
            desired_roll = float(
                np.clip(measured_roll + normalized[4] * self.max_yaw_rad, -2.8, 2.8)
            )
        arm_qpos = self._solve_ik(desired_position, desired_yaw, desired_roll)
        return ControlTargets(
            arm_qpos=arm_qpos,
            finger_position=float((normalized[-1] + 1.0) * 0.5 * 0.032),
            desired_eef_position=desired_position,
            desired_yaw=desired_yaw,
            desired_roll=desired_roll,
        )

    def _solve_ik(
        self,
        target_position: np.ndarray,
        target_yaw: float,
        target_roll: float | None = None,
    ) -> np.ndarray:
        np.copyto(self._ik_data.qpos, self.data.qpos)
        np.copyto(self._ik_data.qvel, self.data.qvel)
        mujoco.mj_forward(self.model, self._ik_data)
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        joint_low = self.model.jnt_range[self._arm_joint_ids, 0]
        joint_high = self.model.jnt_range[self._arm_joint_ids, 1]

        for _ in range(18):
            error = target_position - self._ik_data.site_xpos[self._grip_site_id]
            if float(np.linalg.norm(error)) < 8e-4:
                break
            mujoco.mj_jacSite(
                self.model,
                self._ik_data,
                jacobian_position,
                jacobian_rotation,
                self._grip_site_id,
            )
            jacobian = jacobian_position[:, self._arm_dof_adr[:4]]
            damping = 0.035
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(3), error
            )
            delta = np.clip(delta, -0.18, 0.18)
            self._ik_data.qpos[self._arm_qpos_adr[:4]] += 0.75 * delta
            shoulder = self._ik_data.qpos[self._arm_qpos_adr[1]]
            elbow = self._ik_data.qpos[self._arm_qpos_adr[2]]
            self._ik_data.qpos[self._arm_qpos_adr[3]] = np.clip(
                -np.pi - shoulder - elbow, joint_low[3], joint_high[3]
            )
            self._ik_data.qpos[self._arm_qpos_adr[:4]] = np.clip(
                self._ik_data.qpos[self._arm_qpos_adr[:4]], joint_low[:4], joint_high[:4]
            )
            mujoco.mj_forward(self.model, self._ik_data)

        base_yaw = float(self._ik_data.qpos[self._arm_qpos_adr[0]])
        self._ik_data.qpos[self._arm_qpos_adr[4]] = np.clip(
            target_yaw - base_yaw, joint_low[4], joint_high[4]
        )
        if self.six_axis:
            if target_roll is None:
                raise ValueError("six-axis IK requires a target roll")
            self._ik_data.qpos[self._arm_qpos_adr[5]] = np.clip(
                target_roll,
                joint_low[5],
                joint_high[5],
            )
        return self._ik_data.qpos[self._arm_qpos_adr].copy()

    def _update_grasp_latch(self, *, gripper_command: float) -> None:
        if self._grasped_object is not None and gripper_command > 0.20:
            self.data.eq_active[self._weld_ids[self._grasped_object]] = 0
            self._grasped_object = None
            return
        if not self.grasp_assist or self._grasped_object is not None or gripper_command > -0.20:
            return

        grip_position = self.gripper_position()
        distances = np.asarray(
            [np.linalg.norm(grip_position - self.object_position(i)) for i in range(3)]
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 0.090:
            self.data.eq_active[self._weld_ids[nearest]] = 1
            self._grasped_object = nearest
            if nearest != self.target_object_index:
                self._wrong_pick = True

    def detect_collision(
        self,
        data: mujoco.MjData | None = None,
        *,
        allow_finger_table_contact: bool = False,
    ) -> tuple[bool, int]:
        """Return robot-obstacle/non-target contacts for current or copied state.

        ``allow_finger_table_contact`` is reserved for the predictive simulator
        shield. It compensates for the simplified proximity-grasp geometry,
        where a nominal downward approach can brush the table with a finger.
        Contacts with tray walls, the arm/palm, and non-target objects remain
        unsafe in either mode.
        """

        resolved_data = self.data if data is None else data
        collision = False
        contact_count = 0
        for contact_index in range(resolved_data.ncon):
            contact = resolved_data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self._robot_geom_ids or geom2 in self._robot_geom_ids:
                robot_geom = geom1 if geom1 in self._robot_geom_ids else geom2
                other = geom2 if geom1 in self._robot_geom_ids else geom1
                if other in self._obstacle_geom_ids:
                    robot_name = mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        robot_geom,
                    )
                    other_name = mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        other,
                    )
                    if (
                        allow_finger_table_contact
                        and robot_name is not None
                        and "finger" in robot_name
                        and other_name == "table"
                    ):
                        continue
                    collision = True
                    contact_count += 1
                elif other in self._object_geom_ids:
                    object_index = self._object_geom_ids[other]
                    if object_index != self.target_object_index:
                        collision = True
                        contact_count += 1
        return collision, contact_count

    def _detect_collision(self) -> tuple[bool, int]:
        """Backward-compatible private collision helper for the live state."""

        return self.detect_collision()

    def _check_success(self) -> bool:
        if self._grasped_object is not None:
            self._stable_success_steps = 0
            return False
        target_position = self.object_position(self.target_object_index)
        bin_position = self.bin_position(self._task.target_class)
        inside = bool(
            abs(target_position[0] - bin_position[0]) < 0.075
            and abs(target_position[1] - bin_position[1]) < 0.055
            and target_position[2] < 0.095
        )
        self._stable_success_steps = self._stable_success_steps + 1 if inside else 0
        return self._stable_success_steps >= 2

    def _check_wrong_bin(self) -> bool:
        target_position = self.object_position(self.target_object_index)
        for category in self.QUALITY_CLASSES:
            if category == self._task.target_class:
                continue
            bin_position = self.bin_position(category)
            if (
                abs(target_position[0] - bin_position[0]) < 0.075
                and abs(target_position[1] - bin_position[1]) < 0.055
                and target_position[2] < 0.095
            ):
                return True
        return False

    def gripper_position(self) -> np.ndarray:
        return self.data.site_xpos[self._grip_site_id].copy()

    def object_position(self, object_index: int) -> np.ndarray:
        return self.data.xpos[self._object_body_ids[object_index]].copy()

    def target_position(self, *, noisy: bool = False) -> np.ndarray:
        position = self.object_position(self.target_object_index)
        if noisy and self._randomization.get("detection_noise_std_m", 0.0) > 0.0:
            position = position + self.np_random.normal(
                0.0, self._randomization["detection_noise_std_m"], size=3
            )
        return position

    def bin_position(self, category: QualityClass) -> np.ndarray:
        return self.data.site_xpos[self._bin_site_ids[category]].copy()

    @property
    def calibration_plane_z_m(self) -> float:
        """Return the settled object-origin plane used by planar calibration."""

        plane_z = float(np.median([self.object_position(index)[2] for index in range(3)]))
        if not np.isfinite(plane_z):
            raise RuntimeError("settled calibration plane is not finite")
        return plane_z

    def _robot_state(self) -> np.ndarray:
        qpos = self.data.qpos[np.concatenate((self._arm_qpos_adr, self._finger_qpos_adr))]
        qvel = self.data.qvel[np.concatenate((self._arm_dof_adr, self._finger_dof_adr))]
        wrist_world_yaw = float(qpos[0] + qpos[4])
        orientation = [np.sin(wrist_world_yaw), np.cos(wrist_world_yaw)]
        if self.six_axis:
            wrist_roll = float(qpos[5])
            orientation.extend((np.sin(wrist_roll), np.cos(wrist_roll)))
        state = np.concatenate(
            (
                qpos,
                qvel,
                self.gripper_position(),
                np.asarray(orientation),
                self._last_action,
            )
        ).astype(np.float32)
        noise_std = float(self._randomization.get("robot_state_noise_std", 0.0))
        if noise_std > 0.0:
            state += self.np_random.normal(0.0, noise_std, size=state.shape).astype(np.float32)
        return state

    def _get_observation(self) -> dict[str, Any]:
        return {
            "rgb": self._vision_observation(),
            "robot_state": self._robot_state(),
            "instruction": self._task.instruction,
        }

    def _vision_observation(self) -> np.ndarray:
        """Apply episode-level camera latency and per-frame image corruption."""

        rendered = self.render()
        self._vision_frames.append(rendered)
        buffer_length = self._vision_frames.maxlen
        if buffer_length is None:
            raise RuntimeError("vision history must use a bounded deque")
        while len(self._vision_frames) < buffer_length:
            self._vision_frames.appendleft(rendered.copy())
        image = self._vision_frames[0].copy()
        image_noise_std = float(self._randomization.get("image_noise_std_px", 0.0))
        if image_noise_std > 0.0:
            noise = self.np_random.normal(0.0, image_noise_std, size=image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        occlusion_probability = float(self._randomization.get("image_occlusion_probability", 0.0))
        max_fraction = float(self._randomization.get("image_occlusion_max_fraction", 0.0))
        if max_fraction > 0.0 and self.np_random.random() < occlusion_probability:
            height, width = image.shape[:2]
            min_fraction = min(0.03, max_fraction)
            fraction = float(self.np_random.uniform(min_fraction, max_fraction))
            scale = np.sqrt(fraction)
            occluder_width = max(1, min(width, round(width * scale)))
            occluder_height = max(1, min(height, round(height * scale)))
            left = int(self.np_random.integers(0, width - occluder_width + 1))
            top = int(self.np_random.integers(0, height - occluder_height + 1))
            image[top : top + occluder_height, left : left + occluder_width] = 0
        return image

    def _info(
        self,
        *,
        success: bool,
        collision: bool,
        reward_terms: dict[str, float],
        subtask_success: bool = False,
        placed_task_class: QualityClass | None = None,
    ) -> dict[str, Any]:
        return {
            "success": success,
            "subtask_success": subtask_success,
            "placed_task_class": placed_task_class,
            "collision": collision,
            "collision_steps": self._episode_collision_steps,
            "contact_count": self._contact_count,
            "wrong_pick": self._wrong_pick,
            "wrong_bin": self._wrong_bin,
            "cycle_time_s": self._step_count * self.control_dt,
            "task_class": self._task.target_class,
            "instruction": self._task.instruction,
            "instruction_template_id": self._task.template_id,
            "instruction_split": self._task.split,
            "grasp_assist": self.grasp_assist,
            "grasped_object": self._grasped_object,
            "randomization": self._randomization,
            "reward_terms": reward_terms,
            "control_dt": self.control_dt,
            "action_frame": "base_link",
            "action_order": self.action_order,
            "arm_variant": self.arm_variant,
            "robot_state_dim": self.robot_state_dim,
            "mission_length": len(self._mission_sequence),
            "mission_index": self._mission_index,
            "mission_sequence": self._mission_sequence,
            "completed_task_classes": tuple(self._completed_task_classes),
            "mission_complete": self.mission_complete,
            "scene_initialization": {
                "layout_attempts": self._layout_attempts,
                "settle_steps": self.reset_settle_steps,
                "settle_duration_s": self.reset_settle_steps * self.model.opt.timestep,
            },
        }

    def render(self) -> np.ndarray:
        return self.render_camera("top")

    def render_camera(self, camera: CameraName = "top") -> np.ndarray:
        """Render one named RGB camera without changing the observation contract."""

        if camera not in self._camera_ids:
            raise ValueError(f"unsupported camera {camera!r}")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.image_size, width=self.image_size
            )
        self._renderer.update_scene(self.data, camera=self._camera_ids[camera])
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def project_world_to_pixel(
        self,
        world_position_m: np.ndarray | list[float],
        *,
        camera: CameraName = "top",
    ) -> np.ndarray:
        """Project a simulated 3D point into a named camera image.

        This helper exists to construct and evaluate a *simulated calibration
        fixture*. It must not be called by deployable visual control code: the
        runtime controller uses the persisted homography it receives instead.
        """

        if camera not in self._camera_ids:
            raise ValueError(f"unsupported camera {camera!r}")
        point = np.asarray(world_position_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("world_position_m must be a finite [x,y,z] point")
        camera_id = self._camera_ids[camera]
        # MuJoCo stores camera orientation as a local-to-world rotation. Its
        # optical axis is the negative local z-axis and image y grows down.
        world_from_camera = self.data.cam_xmat[camera_id].reshape(3, 3)
        camera_from_world = world_from_camera.T
        camera_position = self.data.cam_xpos[camera_id]
        camera_point = camera_from_world @ (point - camera_position)
        depth = -float(camera_point[2])
        if depth <= 1e-9:
            raise ValueError("point is behind the selected camera")
        fovy_rad = np.deg2rad(float(self.model.cam_fovy[camera_id]))
        focal_px = 0.5 * self.image_size / np.tan(fovy_rad / 2.0)
        u_px = focal_px * float(camera_point[0]) / depth + 0.5 * self.image_size
        v_px = -focal_px * float(camera_point[1]) / depth + 0.5 * self.image_size
        return np.asarray([u_px, v_px], dtype=np.float64)

    def planar_calibration_reference_points(
        self,
        *,
        camera: CameraName = "top",
        plane_z_m: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a nine-point simulated calibration-board correspondence set.

        The returned arrays are ``[N,2]`` pixel and base-frame XY points. The
        values are intentionally exposed only at calibration creation time;
        downstream vision control receives a saved :class:`PlanarCalibration`
        and never consumes scene poses or object positions.
        """

        resolved_plane_z_m = self.calibration_plane_z_m if plane_z_m is None else plane_z_m
        if not np.isfinite(resolved_plane_z_m):
            raise ValueError("plane_z_m must be finite")
        x_coordinates = (-0.24, 0.0, 0.24)
        y_coordinates = (-0.12, -0.015, 0.09)
        base_xy = np.asarray(
            [
                (x_coordinate, y_coordinate)
                for y_coordinate in y_coordinates
                for x_coordinate in x_coordinates
            ],
            dtype=np.float64,
        )
        pixels = np.stack(
            [
                self.project_world_to_pixel(
                    [base_x, base_y, resolved_plane_z_m],
                    camera=camera,
                )
                for base_x, base_y in base_xy
            ]
        )
        return pixels, base_xy

    def render_perception(
        self,
        camera: CameraName = "top",
        *,
        include_depth: bool = True,
    ) -> dict[str, Any]:
        """Render RGB-D, instance masks, and deterministic synthetic labels.

        ``instance_mask`` uses 0 for background and ``object_index + 1`` for
        the three semantic parts. Labels are generated from MuJoCo's own
        segmentation buffer, so occlusion is reflected in pixel counts rather
        than estimated from hidden simulator coordinates.
        """

        if camera not in self._camera_ids:
            raise ValueError(f"unsupported camera {camera!r}")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.image_size, width=self.image_size
            )
        self._renderer.update_scene(self.data, camera=self._camera_ids[camera])
        rgb = np.asarray(self._renderer.render(), dtype=np.uint8).copy()

        self._renderer.enable_segmentation_rendering()
        try:
            segmentation = np.asarray(self._renderer.render(), dtype=np.int32).copy()
        finally:
            self._renderer.disable_segmentation_rendering()

        depth_m: np.ndarray | None = None
        if include_depth:
            self._renderer.enable_depth_rendering()
            try:
                depth_m = np.asarray(self._renderer.render(), dtype=np.float32).copy()
            finally:
                self._renderer.disable_depth_rendering()

        instance_mask = np.zeros(segmentation.shape[:2], dtype=np.int16)
        instances: list[dict[str, Any]] = []
        for object_index, geom_ids in self._object_visual_geom_ids.items():
            visible = np.isin(segmentation[..., 0], geom_ids)
            instance_mask[visible] = object_index + 1
            ys, xs = np.nonzero(visible)
            if xs.size:
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                bbox: list[int] | None = [x0, y0, x1, y1]
                centroid: list[float] | None = [float(xs.mean()), float(ys.mean())]
                bbox_fill_fraction = float(xs.size / max(1, (x1 - x0) * (y1 - y0)))
            else:
                bbox = None
                centroid = None
                bbox_fill_fraction = 0.0
            instances.append(
                {
                    "instance_id": object_index + 1,
                    "object_index": object_index,
                    "quality_class": self.QUALITY_CLASSES[object_index],
                    "bbox_xyxy": bbox,
                    "visible_pixels": int(xs.size),
                    "bbox_fill_fraction": bbox_fill_fraction,
                    "centroid_xy": centroid,
                    "world_position_m": self.object_position(object_index).round(6).tolist(),
                }
            )
        return {
            "camera": camera,
            "rgb": rgb,
            "depth_m": depth_m,
            "instance_mask": instance_mask,
            "instances": instances,
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(
    id="SmartPickSort-v0",
    entry_point="smartpick_vla.envs.smartpick_env:SmartPickEnv",
)
