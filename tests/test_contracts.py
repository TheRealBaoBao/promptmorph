import numpy as np
import pytest
from pydantic import ValidationError

from promptmorph.cli import main
from promptmorph.compiler.pick_place import (
    CompilationError,
    PickPlaceCompiler,
    motion_correlation,
)
from promptmorph.geometry.se3 import matrix_to_pose
from promptmorph.models import (
    Demonstration,
    EntityKind,
    EntityState,
    Pose,
    RelationType,
    RuntimeStatus,
    Subgoal,
    WorldFrame,
)
from promptmorph.planning.goal_adapter import instantiate_world_goal
from promptmorph.runtime.monitor import ReplanMonitor
from promptmorph.sim.mock_env import marker_cup_frame


def test_pose_rejects_invalid_quaternions() -> None:
    with pytest.raises(ValidationError, match="non-zero"):
        Pose(position=(0, 0, 0), quaternion_xyzw=(0, 0, 0, 0))
    with pytest.raises(ValidationError, match="normalized"):
        Pose(position=(0, 0, 0), quaternion_xyzw=(0, 0, 0, 2))


def test_demonstration_contract_rejects_bad_sequences() -> None:
    frame = marker_cup_frame(0.0, (0, 0, 0), (0.2, 0, 0))
    with pytest.raises(ValidationError, match="at least three"):
        Demonstration(demonstration_id="short", frames=(frame,))

    frame_one = marker_cup_frame(1.0, (0, 0, 0), (0.2, 0, 0))
    with pytest.raises(ValidationError, match="strictly increasing"):
        Demonstration(demonstration_id="order", frames=(frame, frame_one, frame_one))


def test_world_frame_reports_missing_entity() -> None:
    frame = marker_cup_frame(0.0, (0, 0, 0), (0.2, 0, 0))
    with pytest.raises(KeyError, match="missing"):
        frame.entity("bowl")


def test_matrix_to_pose_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="4x4"):
        matrix_to_pose(np.eye(3))


def test_compiler_rejects_scene_without_target() -> None:
    def frame(timestamp: float, x: float) -> WorldFrame:
        return WorldFrame(
            timestamp_s=timestamp,
            entities={
                "marker": EntityState(
                    entity_id="marker",
                    kind=EntityKind.OBJECT,
                    pose=Pose(position=(x, 0, 0.1)),
                )
            },
            gripper_pose=Pose(position=(x, 0, 0.2)),
            gripper_closed=x > 0,
        )

    demo = Demonstration(
        demonstration_id="no-target", frames=(frame(0, 0), frame(1, 0.1), frame(2, 0.2))
    )
    with pytest.raises(CompilationError, match="no target"):
        PickPlaceCompiler().compile(demo)


def test_motion_correlation_and_shape_gate() -> None:
    path = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], dtype=float)
    assert motion_correlation(path, path) == pytest.approx(1.0)
    assert motion_correlation(np.zeros_like(path), path) == 0.0
    with pytest.raises(ValueError, match="matching"):
        motion_correlation(path, path[:, :2])


def test_goal_adapter_rejects_non_cartesian_subgoal() -> None:
    frame = marker_cup_frame(0.0, (0, 0, 0), (0.2, 0, 0))
    grasp = Subgoal(
        subgoal_id="grasp", relation=RelationType.GRASP, subject_id="marker"
    )
    with pytest.raises(ValueError, match="does not define"):
        instantiate_world_goal(grasp, frame)

    malformed = Subgoal(
        subgoal_id="malformed", relation=RelationType.PLACE_RELATIVE, subject_id="marker"
    )
    with pytest.raises(ValueError, match="lacks"):
        instantiate_world_goal(malformed, frame)


def test_monitor_running_and_unarmed_branches() -> None:
    demo = Demonstration(
        demonstration_id="demo",
        frames=(
            marker_cup_frame(0.0, (-0.2, 0, 0.06), (0.2, 0, 0.05)),
            marker_cup_frame(1.0, (0.0, 0, 0.15), (0.2, 0, 0.05)),
            marker_cup_frame(2.0, (0.2, 0, 0.08), (0.2, 0, 0.05)),
        ),
    )
    task = PickPlaceCompiler().compile(demo)
    live = marker_cup_frame(3.0, (-0.1, 0, 0.06), (0.2, 0, 0.05))
    monitor = ReplanMonitor()
    with pytest.raises(RuntimeError, match="armed"):
        monitor.inspect(task, live, "grasp-object")
    monitor.arm(live)
    event = monitor.inspect(task, live, "grasp-object")
    assert event.status == RuntimeStatus.RUNNING


def test_cli_exposes_replanning_story(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    output = capsys.readouterr().out
    assert "Compiled:" in output
    assert "replan_required" in output
    assert "Replanned live goal" in output

