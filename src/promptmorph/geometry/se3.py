"""Small, audited SE(3) helpers with explicit quaternion conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from promptmorph.models import Pose

FloatArray = NDArray[np.float64]


def quaternion_to_matrix(quaternion_xyzw: tuple[float, float, float, float]) -> FloatArray:
    x, y, z, w = quaternion_xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: FloatArray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to normalized xyzw quaternion."""

    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)  # type: ignore[return-value]


def pose_to_matrix(pose: Pose) -> FloatArray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(pose.quaternion_xyzw)
    transform[:3, 3] = np.asarray(pose.position)
    return transform


def matrix_to_pose(transform: FloatArray) -> Pose:
    if transform.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, received {transform.shape}")
    return Pose(
        position=tuple(float(value) for value in transform[:3, 3]),  # type: ignore[arg-type]
        quaternion_xyzw=matrix_to_quaternion(transform[:3, :3]),
    )


def relative_pose(subject_world: Pose, reference_world: Pose) -> Pose:
    subject = pose_to_matrix(subject_world)
    reference = pose_to_matrix(reference_world)
    return matrix_to_pose(np.linalg.inv(reference) @ subject)


def compose_pose(reference_world: Pose, subject_in_reference: Pose) -> Pose:
    return matrix_to_pose(pose_to_matrix(reference_world) @ pose_to_matrix(subject_in_reference))


def position_distance(left: Pose, right: Pose) -> float:
    return float(np.linalg.norm(np.asarray(left.position) - np.asarray(right.position)))

