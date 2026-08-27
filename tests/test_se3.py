import numpy as np

from promptmorph.geometry.se3 import compose_pose, relative_pose
from promptmorph.models import Pose


def test_relative_pose_round_trip() -> None:
    reference = Pose(position=(0.5, -0.2, 0.1))
    subject = Pose(position=(0.6, -0.1, 0.2))
    relative = relative_pose(subject, reference)
    recovered = compose_pose(reference, relative)
    assert np.allclose(recovered.position, subject.position)
    assert np.allclose(recovered.quaternion_xyzw, subject.quaternion_xyzw)


def test_relative_pose_respects_reference_rotation() -> None:
    quarter_turn_z = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    reference = Pose(position=(0.0, 0.0, 0.0), quaternion_xyzw=quarter_turn_z)
    subject_in_reference = Pose(position=(0.1, 0.0, 0.0))
    world = compose_pose(reference, subject_in_reference)
    assert np.allclose(world.position, (0.0, 0.1, 0.0), atol=1e-7)

