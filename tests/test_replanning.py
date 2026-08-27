import pytest

from promptmorph.compiler.pick_place import PickPlaceCompiler
from promptmorph.models import Demonstration, RelationType, RuntimeStatus
from promptmorph.planning.goal_adapter import instantiate_world_goal
from promptmorph.runtime.monitor import ReplanMonitor
from promptmorph.sim.mock_env import marker_cup_frame


def _task():  # type: ignore[no-untyped-def]
    return PickPlaceCompiler().compile(
        Demonstration(
            demonstration_id="demo",
            frames=(
                marker_cup_frame(0.0, (-0.2, 0.0, 0.06), (0.2, 0.0, 0.05)),
                marker_cup_frame(1.0, (0.0, 0.0, 0.15), (0.2, 0.0, 0.05)),
                marker_cup_frame(2.0, (0.2, 0.0, 0.08), (0.2, 0.0, 0.05)),
            ),
        )
    )


def test_goal_moves_with_live_target() -> None:
    task = _task()
    place = next(goal for goal in task.subgoals if goal.relation == RelationType.PLACE_RELATIVE)
    live = marker_cup_frame(3.0, (-0.1, 0.0, 0.06), (0.7, 0.3, 0.05))
    goal = instantiate_world_goal(place, live)
    assert goal.position == pytest.approx((0.7, 0.3, 0.08))


def test_monitor_requests_replan_when_target_moves() -> None:
    task = _task()
    original = marker_cup_frame(3.0, (-0.1, 0.0, 0.06), (0.2, 0.0, 0.05))
    moved = marker_cup_frame(3.2, (-0.1, 0.0, 0.06), (0.3, 0.0, 0.05))
    monitor = ReplanMonitor(target_displacement_threshold_m=0.04)
    monitor.arm(original)
    event = monitor.inspect(task, moved, "place-relative")
    assert event.status == RuntimeStatus.REPLAN_REQUIRED
    assert event.target_displacement_m == pytest.approx(0.1)


def test_monitor_stops_on_low_confidence() -> None:
    task = _task()
    original = marker_cup_frame(3.0, (-0.1, 0.0, 0.06), (0.2, 0.0, 0.05))
    uncertain = marker_cup_frame(
        3.2, (-0.1, 0.0, 0.06), (0.2, 0.0, 0.05), confidence=0.4
    )
    monitor = ReplanMonitor()
    monitor.arm(original)
    event = monitor.inspect(task, uncertain, "place-relative")
    assert event.status == RuntimeStatus.FAILED

