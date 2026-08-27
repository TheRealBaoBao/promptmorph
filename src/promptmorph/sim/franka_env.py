"""Headless MuJoCo Franka tabletop adapter and bounded chunk executor."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import cast

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from promptmorph.geometry.se3 import matrix_to_quaternion, position_distance
from promptmorph.models import (
    ActionChunk,
    EntityKind,
    EntityState,
    ExecutionResult,
    GripperCommand,
    Pose,
    WorldFrame,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SceneLayout:
    marker_xyz: tuple[float, float, float]
    cup_xyz: tuple[float, float, float]


class FrankaTabletopEnv:
    """MuJoCo scene with a 7-DoF Panda chain and kinematic grasp abstraction.

    The arm is simulated through joint position actuators. Object attachment is kept
    deterministic for the perception/planning MVP; replacing it with contact grasping
    does not change the external observation or action contracts.
    """

    HOME_QPOS = np.array([0.0, -0.65, 0.0, -2.10, 0.0, 1.55, 0.75], dtype=np.float64)

    def __init__(self) -> None:
        asset = files("promptmorph.sim.assets").joinpath("franka_tabletop.xml")
        with as_file(asset) as asset_path:
            self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.data = mujoco.MjData(self.model)
        self._site_id = self.model.site("pinch").id
        self._marker_mocap_id = int(self.model.body("marker").mocapid[0])
        self._cup_mocap_id = int(self.model.body("cup").mocapid[0])
        self._joint_ids = np.array(
            [self.model.joint(f"panda_joint{index}").id for index in range(1, 8)],
            dtype=np.int32,
        )
        self._qpos_addresses = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self._joint_ids], dtype=np.int32
        )
        self._gripper_closed = False
        self._marker_attached = False
        self._attachment_offset = np.zeros(3, dtype=np.float64)
        self.reset(seed=0, randomize=False)

    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def marker_attached(self) -> bool:
        return self._marker_attached

    def reset(self, *, seed: int, randomize: bool = True) -> WorldFrame:
        rng = np.random.default_rng(seed)
        if randomize:
            marker = (-0.02 + rng.uniform(-0.06, 0.06), rng.uniform(-0.20, -0.08), 0.06)
            cup = (0.10 + rng.uniform(-0.02, 0.02), rng.uniform(0.04, 0.12), 0.05)
        else:
            marker = (0.02, -0.14, 0.06)
            cup = (0.10, 0.08, 0.05)
        return self.reset_to_layout(SceneLayout(marker_xyz=marker, cup_xyz=cup))

    def reset_to_layout(self, layout: SceneLayout) -> WorldFrame:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = self.HOME_QPOS
        self.data.ctrl[:7] = self.HOME_QPOS
        self.data.mocap_pos[self._marker_mocap_id] = np.asarray(layout.marker_xyz)
        self.data.mocap_pos[self._cup_mocap_id] = np.asarray(layout.cup_xyz)
        self._gripper_closed = False
        self._marker_attached = False
        self._attachment_offset[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self) -> WorldFrame:
        marker_position = tuple(
            float(value) for value in self.data.mocap_pos[self._marker_mocap_id]
        )
        cup_position = tuple(float(value) for value in self.data.mocap_pos[self._cup_mocap_id])
        return WorldFrame(
            timestamp_s=float(self.data.time),
            entities={
                "marker": EntityState(
                    entity_id="marker",
                    kind=EntityKind.OBJECT,
                    pose=Pose(position=marker_position),  # type: ignore[arg-type]
                    size_xyz=(0.024, 0.024, 0.12),
                ),
                "cup": EntityState(
                    entity_id="cup",
                    kind=EntityKind.TARGET,
                    pose=Pose(position=cup_position),  # type: ignore[arg-type]
                    size_xyz=(0.11, 0.11, 0.10),
                ),
            },
            gripper_pose=self.gripper_pose(),
            gripper_closed=self._gripper_closed,
        )

    def gripper_pose(self) -> Pose:
        position = tuple(float(value) for value in self.data.site_xpos[self._site_id])
        rotation = np.asarray(self.data.site_xmat[self._site_id], dtype=np.float64).reshape(3, 3)
        return Pose(
            position=position,  # type: ignore[arg-type]
            quaternion_xyzw=matrix_to_quaternion(rotation),
        )

    def move_cup(self, delta_xyz: tuple[float, float, float]) -> WorldFrame:
        self.data.mocap_pos[self._cup_mocap_id] += np.asarray(delta_xyz, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def command_gripper(self, command: GripperCommand) -> None:
        if command == GripperCommand.HOLD:
            return
        if command == GripperCommand.OPEN:
            self._gripper_closed = False
            self._marker_attached = False
            return

        self._gripper_closed = True
        marker_pose = self.observe().entity("marker").pose
        gripper_pose = self.gripper_pose()
        if position_distance(marker_pose, gripper_pose) <= 0.10:
            self._marker_attached = True
            self._attachment_offset = np.asarray(marker_pose.position) - np.asarray(
                gripper_pose.position
            )

    def attachment_offset(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._attachment_offset)  # type: ignore[return-value]

    def step(self) -> None:
        # Feed forward MuJoCo's generalized bias forces (gravity/Coriolis). The
        # position actuators then regulate tracking error instead of spending their
        # authority holding the arm up, matching a standard gravity-compensated arm.
        self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
        mujoco.mj_step(self.model, self.data)
        self._synchronize_attachment()

    def _synchronize_attachment(self) -> None:
        if self._marker_attached:
            self.data.mocap_pos[self._marker_mocap_id] = (
                self.data.site_xpos[self._site_id] + self._attachment_offset
            )
            mujoco.mj_forward(self.model, self.data)


@dataclass
class MujocoChunkExecutor:
    env: FrankaTabletopEnv
    damping: float = 0.03
    joint_step_limit_rad: float = 0.04
    ik_iterations: int = 120

    def execute(self, chunk: ActionChunk) -> ExecutionResult:
        start_time = float(self.env.data.time)
        steps = max(1, round(chunk.duration_s / self.env.timestep_s))
        start_qpos = self.env.data.qpos[self.env._qpos_addresses].copy()
        desired_qpos = self._solve_position_ik(chunk.target_pose)

        for step_index in range(steps):
            # Reach the joint target during the first 60% of the chunk and use the
            # remaining 40% as an explicit settling window. This keeps each command
            # at 250 ms while avoiding a high-velocity handoff at the chunk boundary.
            alpha = min(1.0, ((step_index + 1) / steps) / 0.60)
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            self.env.data.ctrl[:7] = start_qpos + smooth_alpha * (desired_qpos - start_qpos)
            self.env.step()

        self.env.command_gripper(chunk.gripper_command)
        self.env._synchronize_attachment()
        final_frame = self.env.observe()
        error_m = position_distance(final_frame.gripper_pose, chunk.target_pose)
        return ExecutionResult(
            chunk_id=chunk.chunk_id,
            start_time_s=start_time,
            end_time_s=float(self.env.data.time),
            simulation_steps=steps,
            reached_target=error_m <= chunk.position_tolerance_m,
            position_error_m=error_m,
            final_frame=final_frame,
        )

    def _solve_position_ik(self, target_pose: Pose) -> FloatArray:
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[:] = self.env.data.qpos
        mujoco.mj_forward(self.env.model, scratch)
        target = np.asarray(target_pose.position, dtype=np.float64)
        jacobian = np.zeros((3, self.env.model.nv), dtype=np.float64)
        regularizer = (self.damping**2) * np.eye(3, dtype=np.float64)
        limits = self.env.model.jnt_range[self.env._joint_ids]

        for _ in range(self.ik_iterations):
            mujoco.mj_jacSite(
                self.env.model, scratch, jacobian, None, self.env._site_id
            )
            error = target - scratch.site_xpos[self.env._site_id]
            if float(np.linalg.norm(error)) < 1e-5:
                break
            arm_jacobian = jacobian[:, :7]
            delta = arm_jacobian.T @ np.linalg.solve(
                arm_jacobian @ arm_jacobian.T + regularizer, error
            )
            delta = np.clip(delta, -self.joint_step_limit_rad, self.joint_step_limit_rad)
            scratch.qpos[self.env._qpos_addresses] += delta
            scratch.qpos[self.env._qpos_addresses] = np.clip(
                scratch.qpos[self.env._qpos_addresses], limits[:, 0], limits[:, 1]
            )
            mujoco.mj_forward(self.env.model, scratch)
        result = np.asarray(
            scratch.qpos[self.env._qpos_addresses], dtype=np.float64
        ).copy()
        return cast(FloatArray, result)
