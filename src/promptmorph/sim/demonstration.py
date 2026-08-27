"""Scripted physical prompt capture using the same action interface as deployment."""

from __future__ import annotations

import numpy as np

from promptmorph.data.recorder import EpisodeRecorder
from promptmorph.models import ActionChunk, Demonstration, GripperCommand, Pose
from promptmorph.planning.cartesian import CartesianWaypointPlanner
from promptmorph.runtime.interfaces import ChunkExecutor
from promptmorph.sim.franka_env import FrankaTabletopEnv


def record_marker_into_cup_demonstration(
    env: FrankaTabletopEnv,
    executor: ChunkExecutor,
    recorder: EpisodeRecorder,
    planner: CartesianWaypointPlanner,
) -> Demonstration:
    recorder.record_frame(env.observe())
    frame = env.observe()
    marker = np.asarray(frame.entity("marker").pose.position, dtype=np.float64)
    cup = np.asarray(frame.entity("cup").pose.position, dtype=np.float64)
    approach = Pose(position=tuple(marker + np.array([0.0, 0.0, 0.12])))
    grasp = Pose(position=tuple(marker))
    lift = Pose(position=tuple(marker + np.array([0.0, 0.0, 0.18])))

    sequences = (
        planner.plan(frame.gripper_pose, approach, prefix="demo-approach"),
        planner.plan(
            approach,
            grasp,
            prefix="demo-grasp",
            final_gripper_command=GripperCommand.CLOSE,
        ),
        planner.plan(grasp, lift, prefix="demo-lift"),
    )
    for sequence in sequences:
        _execute_and_record(executor, recorder, sequence)

    desired_marker = cup + np.array([0.0, 0.0, 0.05])
    attachment_offset = np.asarray(env.attachment_offset(), dtype=np.float64)
    gripper_goal = Pose(position=tuple(desired_marker - attachment_offset))
    _execute_and_record(
        executor,
        recorder,
        planner.plan(env.gripper_pose(), gripper_goal, prefix="demo-place"),
    )
    _execute_and_record(
        executor,
        recorder,
        tuple(
            ActionChunk(chunk_id=f"demo-place-settle-{index}", target_pose=gripper_goal)
            for index in range(1, 3)
        ),
    )
    release = ActionChunk(
        chunk_id="demo-release",
        target_pose=env.gripper_pose(),
        gripper_command=GripperCommand.OPEN,
    )
    _execute_and_record(executor, recorder, (release,))
    return recorder.demonstration()


def _execute_and_record(
    executor: ChunkExecutor,
    recorder: EpisodeRecorder,
    chunks: tuple[ActionChunk, ...],
) -> None:
    for chunk in chunks:
        result = executor.execute(chunk)
        recorder.record_action(chunk)
        recorder.record_frame(result.final_frame)
