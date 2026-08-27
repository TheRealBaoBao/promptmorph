"""Compile a short pick/place demonstration into an object-relative task graph."""

from __future__ import annotations

import numpy as np

from promptmorph.geometry.se3 import position_distance, relative_pose
from promptmorph.models import (
    Demonstration,
    EntityKind,
    RelationType,
    Subgoal,
    TaskGraph,
)


class CompilationError(RuntimeError):
    """Raised when the physical prompt is ambiguous or structurally invalid."""


class PickPlaceCompiler:
    """Deterministic compiler for the first narrowly scoped MVP capability.

    Object identity is inferred from motion magnitude and the target from final
    proximity. Perception confidence and ambiguity gates intentionally fail closed.
    """

    def __init__(self, minimum_motion_m: float = 0.05, ambiguity_margin_m: float = 0.015):
        self.minimum_motion_m = minimum_motion_m
        self.ambiguity_margin_m = ambiguity_margin_m

    def compile(self, demonstration: Demonstration) -> TaskGraph:
        first = demonstration.frames[0]
        last = demonstration.frames[-1]

        candidates = [
            entity_id
            for entity_id, entity in first.entities.items()
            if entity.kind in {EntityKind.OBJECT, EntityKind.TOOL}
        ]
        if not candidates:
            raise CompilationError("no manipulable object candidates were observed")

        motion = {
            entity_id: position_distance(first.entity(entity_id).pose, last.entity(entity_id).pose)
            for entity_id in candidates
        }
        manipulated_object_id = max(motion, key=motion.__getitem__)
        if motion[manipulated_object_id] < self.minimum_motion_m:
            raise CompilationError(
                f"largest object motion was {motion[manipulated_object_id]:.3f} m; "
                f"minimum is {self.minimum_motion_m:.3f} m"
            )

        target_candidates = [
            entity_id
            for entity_id, entity in last.entities.items()
            if entity_id != manipulated_object_id and entity.kind == EntityKind.TARGET
        ]
        if not target_candidates:
            raise CompilationError("no target candidates were observed")

        final_object_pose = last.entity(manipulated_object_id).pose
        ranked_targets = sorted(
            (
                (position_distance(final_object_pose, last.entity(target_id).pose), target_id)
                for target_id in target_candidates
            ),
            key=lambda item: item[0],
        )
        if len(ranked_targets) > 1:
            margin = ranked_targets[1][0] - ranked_targets[0][0]
            if margin < self.ambiguity_margin_m:
                raise CompilationError(
                    f"target is ambiguous: nearest-target distance margin is {margin:.3f} m"
                )

        target_id = ranked_targets[0][1]
        desired_relative_pose = relative_pose(final_object_pose, last.entity(target_id).pose)
        return TaskGraph(
            demonstration_id=demonstration.demonstration_id,
            manipulated_object_id=manipulated_object_id,
            target_id=target_id,
            subgoals=(
                Subgoal(
                    subgoal_id="grasp-object",
                    relation=RelationType.GRASP,
                    subject_id=manipulated_object_id,
                ),
                Subgoal(
                    subgoal_id="align-with-target",
                    relation=RelationType.ALIGN,
                    subject_id=manipulated_object_id,
                    reference_id=target_id,
                    desired_subject_in_reference=desired_relative_pose,
                    position_tolerance_m=0.04,
                ),
                Subgoal(
                    subgoal_id="place-relative",
                    relation=RelationType.PLACE_RELATIVE,
                    subject_id=manipulated_object_id,
                    reference_id=target_id,
                    desired_subject_in_reference=desired_relative_pose,
                    # The cup has a 43 mm inner radius and the marker a 12 mm
                    # half-width, leaving a conservative 35 mm center tolerance.
                    position_tolerance_m=0.035,
                ),
                Subgoal(
                    subgoal_id="release-object",
                    relation=RelationType.RELEASE,
                    subject_id=manipulated_object_id,
                ),
            ),
        )


def motion_correlation(object_positions: np.ndarray, gripper_positions: np.ndarray) -> float:
    """Diagnostic signal for future contact-event inference."""

    if object_positions.shape != gripper_positions.shape or object_positions.ndim != 2:
        raise ValueError("position arrays must have matching [time, xyz] shapes")
    object_delta = np.diff(object_positions, axis=0).reshape(-1)
    gripper_delta = np.diff(gripper_positions, axis=0).reshape(-1)
    if np.linalg.norm(object_delta) < 1e-8 or np.linalg.norm(gripper_delta) < 1e-8:
        return 0.0
    return float(np.dot(object_delta, gripper_delta) / (
        np.linalg.norm(object_delta) * np.linalg.norm(gripper_delta)
    ))
