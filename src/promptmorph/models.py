"""Typed domain models shared by perception, planning, and execution.

Keeping these models independent of MuJoCo prevents simulator details from leaking
into the task representation and makes a later ROS 2 adapter straightforward.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityKind(str, Enum):
    OBJECT = "object"
    TARGET = "target"
    TOOL = "tool"
    ROBOT = "robot"


class Pose(BaseModel):
    """Right-handed world pose using xyzw quaternion order."""

    model_config = ConfigDict(frozen=True)

    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @field_validator("quaternion_xyzw")
    @classmethod
    def quaternion_must_be_normalized(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        norm = float(np.linalg.norm(value))
        if norm < 1e-8:
            raise ValueError("quaternion norm must be non-zero")
        if not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError(f"quaternion must be normalized; received norm={norm:.6f}")
        return value


class EntityState(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    kind: EntityKind
    pose: Pose
    size_xyz: tuple[float, float, float] = (0.04, 0.04, 0.04)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class WorldFrame(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp_s: float = Field(ge=0.0)
    entities: dict[str, EntityState]
    gripper_pose: Pose
    gripper_closed: bool

    def entity(self, entity_id: str) -> EntityState:
        try:
            return self.entities[entity_id]
        except KeyError as exc:
            raise KeyError(f"entity {entity_id!r} is missing from frame") from exc


class Demonstration(BaseModel):
    model_config = ConfigDict(frozen=True)

    demonstration_id: str
    frames: tuple[WorldFrame, ...]

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, frames: tuple[WorldFrame, ...]) -> tuple[WorldFrame, ...]:
        if len(frames) < 3:
            raise ValueError("a demonstration needs at least three frames")
        timestamps = [frame.timestamp_s for frame in frames]
        if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
            raise ValueError("demonstration timestamps must be strictly increasing")
        entity_sets = [set(frame.entities) for frame in frames]
        if any(entity_set != entity_sets[0] for entity_set in entity_sets[1:]):
            raise ValueError("all demonstration frames must contain the same entity IDs")
        return frames


class RelationType(str, Enum):
    GRASP = "grasp"
    ALIGN = "align"
    PLACE_RELATIVE = "place_relative"
    RELEASE = "release"


class Subgoal(BaseModel):
    model_config = ConfigDict(frozen=True)

    subgoal_id: str
    relation: RelationType
    subject_id: str
    reference_id: str | None = None
    desired_subject_in_reference: Pose | None = None
    position_tolerance_m: float = Field(default=0.025, gt=0.0)


class TaskGraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    demonstration_id: str
    manipulated_object_id: str
    target_id: str
    subgoals: tuple[Subgoal, ...]


class RuntimeStatus(str, Enum):
    RUNNING = "running"
    REPLAN_REQUIRED = "replan_required"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: RuntimeStatus
    reason: str
    active_subgoal_id: str | None = None
    target_displacement_m: float = 0.0

