import numpy as np

from promptmorph.compiler.pick_place import PickPlaceCompiler
from promptmorph.learning.features import PROGRESS_FEATURE_NAMES, progress_features
from promptmorph.models import Demonstration
from promptmorph.sim.mock_env import marker_cup_frame


def test_progress_features_are_object_relative() -> None:
    task = PickPlaceCompiler().compile(
        Demonstration(
            demonstration_id="demo",
            frames=(
                marker_cup_frame(0.0, (-0.2, 0, 0.06), (0.2, 0, 0.05)),
                marker_cup_frame(1.0, (0, 0, 0.15), (0.2, 0, 0.05)),
                marker_cup_frame(2.0, (0.2, 0, 0.08), (0.2, 0, 0.05)),
            ),
        )
    )
    left = marker_cup_frame(
        3.0, (-0.1, 0.0, 0.06), (0.2, 0.0, 0.05), gripper_xyz=(-0.2, 0, 0.1)
    )
    shifted = marker_cup_frame(
        3.0, (0.4, 0.5, 0.06), (0.7, 0.5, 0.05), gripper_xyz=(0.3, 0.5, 0.1)
    )
    assert len(progress_features(left, task)) == len(PROGRESS_FEATURE_NAMES)
    assert np.allclose(progress_features(left, task), progress_features(shifted, task))

