"""Simulator-independent relational features for execution-progress learning."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from promptmorph.models import TaskGraph, WorldFrame

PROGRESS_FEATURE_NAMES = (
    "object_to_target_x",
    "object_to_target_y",
    "object_to_target_z",
    "gripper_to_object_x",
    "gripper_to_object_y",
    "gripper_to_object_z",
    "object_target_distance",
    "gripper_object_distance",
    "gripper_closed",
    "minimum_pose_confidence",
)


def progress_features(frame: WorldFrame, task: TaskGraph) -> NDArray[np.float32]:
    subject = frame.entity(task.manipulated_object_id)
    target = frame.entity(task.target_id)
    subject_position = np.asarray(subject.pose.position, dtype=np.float32)
    target_position = np.asarray(target.pose.position, dtype=np.float32)
    gripper_position = np.asarray(frame.gripper_pose.position, dtype=np.float32)
    object_to_target = target_position - subject_position
    gripper_to_object = subject_position - gripper_position
    features = np.concatenate(
        (
            object_to_target,
            gripper_to_object,
            np.asarray(
                [
                    np.linalg.norm(object_to_target),
                    np.linalg.norm(gripper_to_object),
                    float(frame.gripper_closed),
                    min(subject.confidence, target.confidence),
                ],
                dtype=np.float32,
            ),
        )
    )
    if features.shape != (len(PROGRESS_FEATURE_NAMES),):
        raise RuntimeError(f"progress feature contract changed unexpectedly: {features.shape}")
    return np.asarray(features, dtype=np.float32)
