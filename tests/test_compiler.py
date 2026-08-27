import pytest

from promptmorph.compiler.pick_place import CompilationError, PickPlaceCompiler
from promptmorph.models import Demonstration
from promptmorph.sim.mock_env import marker_cup_frame


def test_compiler_extracts_object_relative_goal() -> None:
    demo = Demonstration(
        demonstration_id="demo",
        frames=(
            marker_cup_frame(0.0, (-0.2, 0.0, 0.06), (0.2, 0.0, 0.05)),
            marker_cup_frame(1.0, (0.0, 0.0, 0.15), (0.2, 0.0, 0.05)),
            marker_cup_frame(2.0, (0.2, 0.0, 0.08), (0.2, 0.0, 0.05)),
        ),
    )
    graph = PickPlaceCompiler().compile(demo)
    assert graph.manipulated_object_id == "marker"
    assert graph.target_id == "cup"
    assert [subgoal.subgoal_id for subgoal in graph.subgoals] == [
        "grasp-object",
        "align-with-target",
        "place-relative",
        "release-object",
    ]
    assert graph.subgoals[2].desired_subject_in_reference is not None
    assert graph.subgoals[2].desired_subject_in_reference.position == pytest.approx((0, 0, 0.03))


def test_compiler_fails_closed_when_nothing_moves() -> None:
    demo = Demonstration(
        demonstration_id="static",
        frames=(
            marker_cup_frame(0.0, (0.0, 0.0, 0.06), (0.2, 0.0, 0.05)),
            marker_cup_frame(1.0, (0.0, 0.0, 0.06), (0.2, 0.0, 0.05)),
            marker_cup_frame(2.0, (0.0, 0.0, 0.06), (0.2, 0.0, 0.05)),
        ),
    )
    with pytest.raises(CompilationError, match="largest object motion"):
        PickPlaceCompiler().compile(demo)

