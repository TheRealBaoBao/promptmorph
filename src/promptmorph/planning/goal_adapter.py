"""Adapt a compiled object-relative goal to the live scene geometry."""

from promptmorph.geometry.se3 import compose_pose
from promptmorph.models import Pose, RelationType, Subgoal, WorldFrame


def instantiate_world_goal(subgoal: Subgoal, live_frame: WorldFrame) -> Pose:
    if subgoal.relation not in {RelationType.ALIGN, RelationType.PLACE_RELATIVE}:
        raise ValueError(f"subgoal {subgoal.subgoal_id!r} does not define a Cartesian goal")
    if subgoal.reference_id is None or subgoal.desired_subject_in_reference is None:
        raise ValueError(f"subgoal {subgoal.subgoal_id!r} lacks a reference-relative pose")
    reference_pose = live_frame.entity(subgoal.reference_id).pose
    return compose_pose(reference_pose, subgoal.desired_subject_in_reference)

