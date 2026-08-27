"""Runtime safety/progress gates for closed-loop execution."""

from dataclasses import dataclass

from promptmorph.geometry.se3 import position_distance
from promptmorph.models import RuntimeEvent, RuntimeStatus, TaskGraph, WorldFrame


@dataclass
class ReplanMonitor:
    target_displacement_threshold_m: float = 0.04
    minimum_perception_confidence: float = 0.70
    _planning_frame: WorldFrame | None = None

    def arm(self, planning_frame: WorldFrame) -> None:
        self._planning_frame = planning_frame

    def inspect(
        self, task: TaskGraph, current_frame: WorldFrame, active_subgoal_id: str
    ) -> RuntimeEvent:
        if self._planning_frame is None:
            raise RuntimeError("monitor must be armed with the planning frame before inspection")

        subject = current_frame.entity(task.manipulated_object_id)
        target = current_frame.entity(task.target_id)
        if min(subject.confidence, target.confidence) < self.minimum_perception_confidence:
            return RuntimeEvent(
                status=RuntimeStatus.FAILED,
                reason="perception confidence fell below the execution safety threshold",
                active_subgoal_id=active_subgoal_id,
            )

        original_target = self._planning_frame.entity(task.target_id)
        displacement = position_distance(original_target.pose, target.pose)
        if displacement > self.target_displacement_threshold_m:
            return RuntimeEvent(
                status=RuntimeStatus.REPLAN_REQUIRED,
                reason="target moved beyond the active plan validity region",
                active_subgoal_id=active_subgoal_id,
                target_displacement_m=displacement,
            )

        return RuntimeEvent(
            status=RuntimeStatus.RUNNING,
            reason="active plan remains valid",
            active_subgoal_id=active_subgoal_id,
            target_displacement_m=displacement,
        )

