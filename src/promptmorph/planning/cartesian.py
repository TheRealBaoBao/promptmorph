"""Small Cartesian waypoint planner for bounded receding-horizon execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from promptmorph.models import ActionChunk, GripperCommand, Pose


@dataclass(frozen=True)
class CartesianWaypointPlanner:
    maximum_step_m: float = 0.04
    action_chunk_s: float = 0.25

    def plan(
        self,
        start: Pose,
        goal: Pose,
        *,
        prefix: str,
        final_gripper_command: GripperCommand = GripperCommand.HOLD,
    ) -> tuple[ActionChunk, ...]:
        if self.maximum_step_m <= 0:
            raise ValueError("maximum_step_m must be positive")
        if not 0 < self.action_chunk_s <= 0.25:
            raise ValueError("action_chunk_s must be in (0, 0.25]")

        start_position = np.asarray(start.position, dtype=np.float64)
        goal_position = np.asarray(goal.position, dtype=np.float64)
        distance = float(np.linalg.norm(goal_position - start_position))
        count = max(1, int(np.ceil(distance / self.maximum_step_m)))
        chunks: list[ActionChunk] = []
        for index in range(1, count + 1):
            alpha = index / count
            position = start_position + alpha * (goal_position - start_position)
            chunks.append(
                ActionChunk(
                    chunk_id=f"{prefix}-{index:03d}",
                    target_pose=Pose(
                        position=tuple(float(value) for value in position),  # type: ignore[arg-type]
                        quaternion_xyzw=goal.quaternion_xyzw,
                    ),
                    gripper_command=(
                        final_gripper_command if index == count else GripperCommand.HOLD
                    ),
                    duration_s=self.action_chunk_s,
                )
            )
        return tuple(chunks)
