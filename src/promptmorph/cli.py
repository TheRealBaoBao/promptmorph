"""Deterministic end-to-end demonstration of compilation and replanning."""

from promptmorph.compiler.pick_place import PickPlaceCompiler
from promptmorph.models import Demonstration, RelationType
from promptmorph.planning.goal_adapter import instantiate_world_goal
from promptmorph.runtime.monitor import ReplanMonitor
from promptmorph.sim.mock_env import marker_cup_frame


def main() -> None:
    demonstration = Demonstration(
        demonstration_id="marker-into-cup-001",
        frames=(
            marker_cup_frame(0.0, (-0.18, 0.00, 0.06), (0.20, 0.05, 0.05)),
            marker_cup_frame(
                1.0,
                (0.00, 0.03, 0.16),
                (0.20, 0.05, 0.05),
                gripper_xyz=(0.00, 0.03, 0.18),
                gripper_closed=True,
            ),
            marker_cup_frame(2.0, (0.20, 0.05, 0.08), (0.20, 0.05, 0.05)),
        ),
    )
    task = PickPlaceCompiler().compile(demonstration)
    place_subgoal = next(
        subgoal for subgoal in task.subgoals if subgoal.relation == RelationType.PLACE_RELATIVE
    )

    live = marker_cup_frame(3.0, (-0.12, -0.08, 0.06), (0.12, -0.10, 0.05))
    initial_goal = instantiate_world_goal(place_subgoal, live)
    print(f"Compiled: {task.demonstration_id} -> {len(task.subgoals)} subgoals")
    print(f"Initial live goal: {initial_goal.position}")

    monitor = ReplanMonitor()
    monitor.arm(live)
    moved = marker_cup_frame(3.4, (-0.02, -0.08, 0.12), (0.24, -0.04, 0.05))
    event = monitor.inspect(task, moved, active_subgoal_id=place_subgoal.subgoal_id)
    print(f"Monitor: {event.status.value} ({event.reason})")
    if event.status.value == "replan_required":
        replanned_goal = instantiate_world_goal(place_subgoal, moved)
        print(f"Replanned live goal: {replanned_goal.position}")


if __name__ == "__main__":
    main()

