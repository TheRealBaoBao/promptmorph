"""Closed-loop pick/place runtime with plan invalidation and bounded recovery."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from promptmorph.data.recorder import EpisodeRecorder
from promptmorph.geometry.se3 import position_distance
from promptmorph.models import (
    ActionChunk,
    GripperCommand,
    Pose,
    RelationType,
    RuntimeEvent,
    RuntimeStatus,
    Subgoal,
    TaskGraph,
)
from promptmorph.planning.cartesian import CartesianWaypointPlanner
from promptmorph.planning.goal_adapter import instantiate_world_goal
from promptmorph.runtime.interfaces import ChunkExecutor
from promptmorph.runtime.monitor import ReplanMonitor
from promptmorph.sim.franka_env import FrankaTabletopEnv

DisturbanceHook = Callable[[int, FrankaTabletopEnv], None]


@dataclass(frozen=True)
class RecoveryReport:
    status: RuntimeStatus
    reason: str
    replans: int
    chunks_executed: int
    final_position_error_m: float


@dataclass
class PickPlaceRecoveryRuntime:
    env: FrankaTabletopEnv
    executor: ChunkExecutor
    planner: CartesianWaypointPlanner
    monitor: ReplanMonitor
    maximum_replans: int = 4
    recorder: EpisodeRecorder | None = None

    def run(
        self,
        task: TaskGraph,
        *,
        disturbance_hook: DisturbanceHook | None = None,
    ) -> RecoveryReport:
        chunks_executed = 0
        if self.recorder is not None:
            self.recorder.record_frame(self.env.observe())

        pickup_chunks = self._pickup_chunks()
        chunks_executed += self._execute(pickup_chunks)
        if not self.env.marker_attached:
            return RecoveryReport(
                status=RuntimeStatus.FAILED,
                reason="grasp abstraction did not attach the marker",
                replans=0,
                chunks_executed=chunks_executed,
                final_position_error_m=float("inf"),
            )

        place = self._place_subgoal(task)
        replans = 0
        transport_chunks = 0
        planning_frame = self.env.observe()
        self.monitor.arm(planning_frame)
        pending = list(self._transport_chunks(place, replans))

        while pending:
            chunk = pending.pop(0)
            chunks_executed += self._execute((chunk,))
            transport_chunks += 1
            if disturbance_hook is not None:
                disturbance_hook(transport_chunks, self.env)

            current = self.env.observe()
            event = self.monitor.inspect(task, current, place.subgoal_id)
            self._record_event(event)
            if event.status == RuntimeStatus.FAILED:
                return RecoveryReport(
                    status=RuntimeStatus.FAILED,
                    reason=event.reason,
                    replans=replans,
                    chunks_executed=chunks_executed,
                    final_position_error_m=float("inf"),
                )
            if event.status == RuntimeStatus.REPLAN_REQUIRED:
                replans += 1
                if replans > self.maximum_replans:
                    return RecoveryReport(
                        status=RuntimeStatus.FAILED,
                        reason="replan budget exceeded",
                        replans=replans,
                        chunks_executed=chunks_executed,
                        final_position_error_m=float("inf"),
                    )
                self.monitor.arm(current)
                pending = list(self._transport_chunks(place, replans))

        release = ActionChunk(
            chunk_id="release-object",
            target_pose=self.env.gripper_pose(),
            gripper_command=GripperCommand.OPEN,
        )
        chunks_executed += self._execute((release,))
        final_frame = self.env.observe()
        object_goal = instantiate_world_goal(place, final_frame)
        error_m = position_distance(
            final_frame.entity(task.manipulated_object_id).pose, object_goal
        )
        succeeded = error_m <= place.position_tolerance_m
        status = RuntimeStatus.SUCCEEDED if succeeded else RuntimeStatus.FAILED
        reason = (
            "marker placed at the live object-relative goal"
            if succeeded
            else "marker remained outside the placement tolerance"
        )
        self._record_event(
            RuntimeEvent(
                status=status,
                reason=reason,
                active_subgoal_id="release-object",
            )
        )
        return RecoveryReport(
            status=status,
            reason=reason,
            replans=replans,
            chunks_executed=chunks_executed,
            final_position_error_m=error_m,
        )

    def _pickup_chunks(self) -> tuple[ActionChunk, ...]:
        frame = self.env.observe()
        marker = np.asarray(frame.entity("marker").pose.position, dtype=np.float64)
        approach = Pose(position=tuple(marker + np.array([0.0, 0.0, 0.12])))
        grasp = Pose(position=tuple(marker))
        lift = Pose(position=tuple(marker + np.array([0.0, 0.0, 0.18])))
        chunks = list(
            self.planner.plan(frame.gripper_pose, approach, prefix="approach-object")
        )
        chunks.extend(
            self.planner.plan(
                approach,
                grasp,
                prefix="grasp-object",
                final_gripper_command=GripperCommand.CLOSE,
            )
        )
        chunks.extend(self.planner.plan(grasp, lift, prefix="lift-object"))
        return tuple(chunks)

    def _transport_chunks(self, place: Subgoal, replan_index: int) -> tuple[ActionChunk, ...]:
        frame = self.env.observe()
        object_goal = instantiate_world_goal(place, frame)
        attachment_offset = np.asarray(self.env.attachment_offset(), dtype=np.float64)
        gripper_goal_position = np.asarray(object_goal.position) - attachment_offset
        gripper_goal = Pose(
            position=tuple(float(value) for value in gripper_goal_position),  # type: ignore[arg-type]
            quaternion_xyzw=frame.gripper_pose.quaternion_xyzw,
        )
        chunks = list(self.planner.plan(
            frame.gripper_pose,
            gripper_goal,
            prefix=f"place-r{replan_index}",
        ))
        # Two goal-hold chunks provide a bounded terminal convergence phase while
        # preserving the same 250 ms control/observation boundary.
        chunks.extend(
            ActionChunk(
                chunk_id=f"place-r{replan_index}-settle-{index}",
                target_pose=gripper_goal,
            )
            for index in range(1, 3)
        )
        return tuple(chunks)

    def _execute(self, chunks: Iterable[ActionChunk]) -> int:
        count = 0
        for chunk in chunks:
            result = self.executor.execute(chunk)
            count += 1
            if self.recorder is not None:
                self.recorder.record_action(chunk)
                self.recorder.record_frame(result.final_frame)
        return count

    def _record_event(self, event: RuntimeEvent) -> None:
        if self.recorder is not None:
            self.recorder.record_event(event)

    @staticmethod
    def _place_subgoal(task: TaskGraph) -> Subgoal:
        return next(
            subgoal
            for subgoal in task.subgoals
            if subgoal.relation == RelationType.PLACE_RELATIVE
        )
