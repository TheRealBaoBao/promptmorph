"""One-command MuJoCo demonstration capture and cup-movement recovery."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from promptmorph.compiler.pick_place import PickPlaceCompiler
from promptmorph.data.recorder import EpisodeRecorder
from promptmorph.planning.cartesian import CartesianWaypointPlanner
from promptmorph.runtime.monitor import ReplanMonitor
from promptmorph.runtime.recovery import PickPlaceRecoveryRuntime
from promptmorph.sim.demonstration import record_marker_into_cup_demonstration
from promptmorph.sim.franka_env import FrankaTabletopEnv, MujocoChunkExecutor


def run_demo(*, seed: int, output_dir: Path) -> dict[str, object]:
    planner = CartesianWaypointPlanner(action_chunk_s=0.25)
    env = FrankaTabletopEnv()
    executor = MujocoChunkExecutor(env)

    env.reset(seed=0, randomize=False)
    demonstration_recorder = EpisodeRecorder(
        root=output_dir,
        episode_id="physical-prompt-001",
        seed=0,
        config={"action_chunk_s": 0.25, "scene": "franka_tabletop"},
    )
    demonstration = record_marker_into_cup_demonstration(
        env, executor, demonstration_recorder, planner
    )
    task = PickPlaceCompiler().compile(demonstration)
    demonstration_recorder.close(
        outcome="demonstration_recorded",
        extra={"task_graph": task.model_dump(mode="json")},
    )

    env.reset(seed=seed, randomize=True)
    rollout_recorder = EpisodeRecorder(
        root=output_dir,
        episode_id=f"recovery-seed-{seed:04d}",
        seed=seed,
        config={
            "action_chunk_s": 0.25,
            "target_displacement_threshold_m": 0.04,
            "maximum_replans": 4,
            "disturbance_delta_xyz": [0.06, -0.03, 0.0],
        },
    )
    disturbed = False

    def move_cup_once(completed_transport_chunks: int, live_env: FrankaTabletopEnv) -> None:
        nonlocal disturbed
        if completed_transport_chunks == 2 and not disturbed:
            live_env.move_cup((0.06, -0.03, 0.0))
            disturbed = True

    runtime = PickPlaceRecoveryRuntime(
        env=env,
        executor=executor,
        planner=planner,
        monitor=ReplanMonitor(target_displacement_threshold_m=0.04),
        maximum_replans=4,
        recorder=rollout_recorder,
    )
    report = runtime.run(task, disturbance_hook=move_cup_once)
    rollout_recorder.close(
        outcome=report.status.value,
        extra={"task_graph": task.model_dump(mode="json"), "report": asdict(report)},
    )
    return {
        "status": report.status.value,
        "reason": report.reason,
        "replans": report.replans,
        "chunks_executed": report.chunks_executed,
        "final_position_error_m": report.final_position_error_m,
        "output_dir": str(output_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(run_demo(seed=args.seed, output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
