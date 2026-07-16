"""MuJoCo/Gymnasium language-conditioned quality sorting environment."""

from __future__ import annotations

import string
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from smartpick_vla.data.schema import QualityClass
from smartpick_vla.envs.randomization import DomainRandomizationConfig, DomainRandomizer
from smartpick_vla.envs.tasks import InstructionSplit, SortTask, sample_task


class SmartPickEnv(gym.Env[dict[str, Any], np.ndarray]):
    """Sort quality-labelled objects using RGB, language, and robot state.

    The normalized action is ``[dx, dy, dz, dyaw, gripper]`` in ``base_link``.
    Translation and yaw are scaled by the configured per-step limits. Gripper
    ``-1`` closes and ``+1`` opens. A configurable proximity weld provides a
    deterministic contact latch for data generation; it is surfaced in every
    reset's metadata and is not represented as a pure-contact grasp.
    """

    metadata: ClassVar[dict[str, Any]] = {  # type: ignore[misc]
        "render_modes": ["rgb_array"],
        "render_fps": 25,
    }

    QUALITY_CLASSES: tuple[QualityClass, ...] = ("accepted", "scratch", "unknown")
    ACTION_DIM = 5
    ROBOT_STATE_DIM = 24

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

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
                "robot_state": spaces.Box(
                    -100.0, 100.0, shape=(self.ROBOT_STATE_DIM,), dtype=np.float32
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
        )
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
        self._grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._top_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
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

        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._home_arm_qpos = np.array([0.0, 0.20, -2.00, -1.00, 0.0], dtype=np.float64)
        self._workspace_low = np.array([-0.39, -0.18, 0.055], dtype=np.float64)
        self._workspace_high = np.array([0.39, 0.34, 0.40], dtype=np.float64)

        self._task = SortTask("accepted", "place the accepted part", "bootstrap", "train")
        self._step_count = 0
        self._episode_collision_steps = 0
        self._contact_count = 0
        self._wrong_pick = False
        self._wrong_bin = False
        self._stable_success_steps = 0
        self._grasped_object: int | None = None
        self._last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self._desired_eef_pos = np.zeros(3, dtype=np.float64)
        self._desired_yaw = 0.0
        self._action_queue: deque[np.ndarray] = deque()
        self._randomization: dict[str, Any] = {}

    @property
    def task(self) -> SortTask:
        return self._task

    @property
    def target_object_index(self) -> int:
        return self.QUALITY_CLASSES.index(self._task.target_class)

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

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
        self._task = sample_task(self.np_random, split=split, target_class=target_class)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        self.data.eq_active[:] = 0
        self._randomization = self.domain_randomizer.apply(self.np_random)
        self._sample_object_layout(ood=bool(options.get("ood_layout", False)))
        mujoco.mj_forward(self.model, self.data)

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
        delay = int(self._randomization["control_delay_steps"])
        self._action_queue = deque(
            [np.zeros(self.ACTION_DIM, dtype=np.float32) for _ in range(delay)]
        )

        observation = self._get_observation()
        info = self._info(success=False, collision=False, reward_terms={})
        info["reset_seed"] = seed
        return observation, info

    def _sample_object_layout(self, *, ood: bool) -> None:
        x_limit = 0.27 if ood else 0.21
        y_low, y_high = (-0.15, 0.085) if ood else (-0.11, 0.045)
        positions: list[np.ndarray] = []
        for _ in range(3):
            for _attempt in range(200):
                candidate = np.array(
                    [
                        self.np_random.uniform(-x_limit, x_limit),
                        self.np_random.uniform(y_low, y_high),
                        0.026,
                    ],
                    dtype=np.float64,
                )
                if all(np.linalg.norm(candidate[:2] - p[:2]) > 0.105 for p in positions):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError("could not sample a collision-free object layout")

        for object_index, position in enumerate(positions):
            address = int(self._object_qpos_adr[object_index])
            self.data.qpos[address : address + 3] = position
            yaw = float(self.np_random.uniform(-np.pi, np.pi))
            self.data.qpos[address + 3 : address + 7] = np.array(
                [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            )
            dof_address = int(self.model.jnt_dofadr[self._object_joint_ids[object_index]])
            self.data.qvel[dof_address : dof_address + 6] = 0.0

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        normalized = np.asarray(action, dtype=np.float32)
        if normalized.shape != (self.ACTION_DIM,):
            raise ValueError(f"expected action shape {(self.ACTION_DIM,)}, got {normalized.shape}")
        if not np.isfinite(normalized).all():
            raise ValueError("action contains NaN or infinity")
        normalized = np.clip(normalized, -1.0, 1.0)
        self._last_action = normalized.copy()
        self._action_queue.append(normalized)
        applied = self._action_queue.popleft()

        # Actions are Cartesian deltas from measured state, not a free-running
        # setpoint integrator. Re-anchoring every control step prevents wind-up
        # when a requested waypoint is near the reachable workspace boundary.
        self._desired_eef_pos = np.clip(
            self.gripper_position() + applied[:3] * self.max_translation_m,
            self._workspace_low,
            self._workspace_high,
        )
        measured_yaw = float(
            self.data.qpos[self._arm_qpos_adr[0]] + self.data.qpos[self._arm_qpos_adr[4]]
        )
        self._desired_yaw = float(np.clip(measured_yaw + applied[3] * self.max_yaw_rad, -2.8, 2.8))
        q_target = self._solve_ik(self._desired_eef_pos, self._desired_yaw)
        self.data.ctrl[:5] = q_target
        finger_target = float((applied[4] + 1.0) * 0.5 * 0.032)
        self.data.ctrl[5:7] = finger_target

        self._update_grasp_latch(gripper_command=float(applied[4]))
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._step_count += 1

        collision, new_contacts = self._detect_collision()
        self._contact_count += new_contacts
        if collision:
            self._episode_collision_steps += 1

        success = self._check_success()
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
            "success": 12.0 if success else 0.0,
            "time": -0.01,
        }
        reward = float(sum(reward_terms.values()))
        terminated = bool(success or self._wrong_bin)
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        observation = self._get_observation()
        info = self._info(success=success, collision=collision, reward_terms=reward_terms)
        return observation, reward, terminated, truncated, info

    def _solve_ik(self, target_position: np.ndarray, target_yaw: float) -> np.ndarray:
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

    def _detect_collision(self) -> tuple[bool, int]:
        collision = False
        contact_count = 0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self._robot_geom_ids or geom2 in self._robot_geom_ids:
                other = geom2 if geom1 in self._robot_geom_ids else geom1
                if other in self._obstacle_geom_ids:
                    collision = True
                    contact_count += 1
                elif other in self._object_geom_ids:
                    object_index = self._object_geom_ids[other]
                    if object_index != self.target_object_index:
                        collision = True
                        contact_count += 1
        return collision, contact_count

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

    def _robot_state(self) -> np.ndarray:
        qpos = self.data.qpos[np.concatenate((self._arm_qpos_adr, self._finger_qpos_adr))]
        qvel = self.data.qvel[np.concatenate((self._arm_dof_adr, self._finger_dof_adr))]
        wrist_world_yaw = float(qpos[0] + qpos[4])
        state = np.concatenate(
            (
                qpos,
                qvel,
                self.gripper_position(),
                np.array([np.sin(wrist_world_yaw), np.cos(wrist_world_yaw)]),
                self._last_action,
            )
        ).astype(np.float32)
        noise_std = float(self._randomization.get("robot_state_noise_std", 0.0))
        if noise_std > 0.0:
            state += self.np_random.normal(0.0, noise_std, size=state.shape).astype(np.float32)
        return state

    def _get_observation(self) -> dict[str, Any]:
        return {
            "rgb": self.render(),
            "robot_state": self._robot_state(),
            "instruction": self._task.instruction,
        }

    def _info(
        self,
        *,
        success: bool,
        collision: bool,
        reward_terms: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "success": success,
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
        }

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.image_size, width=self.image_size
            )
        self._renderer.update_scene(self.data, camera=self._top_camera_id)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(
    id="SmartPickSort-v0",
    entry_point="smartpick_vla.envs.smartpick_env:SmartPickEnv",
)
