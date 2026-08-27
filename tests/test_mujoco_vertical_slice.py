import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from promptmorph.compiler.pick_place import PickPlaceCompiler
from promptmorph.data.recorder import SCHEMA_VERSION, EpisodeRecorder
from promptmorph.models import (
    ActionChunk,
    GripperCommand,
    Pose,
    RuntimeEvent,
    RuntimeStatus,
)
from promptmorph.planning.cartesian import CartesianWaypointPlanner
from promptmorph.runtime.monitor import ReplanMonitor
from promptmorph.runtime.recovery import PickPlaceRecoveryRuntime
from promptmorph.sim.demonstration import record_marker_into_cup_demonstration
from promptmorph.sim.franka_env import FrankaTabletopEnv, MujocoChunkExecutor
from promptmorph.sim.mock_env import marker_cup_frame
from promptmorph.sim.mujoco_demo import run_demo


def test_action_chunks_are_bounded() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 0.25"):
        ActionChunk(chunk_id="too-long", target_pose=Pose(position=(0, 0, 0)), duration_s=0.3)

    planner = CartesianWaypointPlanner(maximum_step_m=0.04)
    chunks = planner.plan(
        Pose(position=(0, 0, 0)), Pose(position=(0.10, 0, 0)), prefix="bounded"
    )
    assert len(chunks) == 3
    assert all(chunk.duration_s == 0.25 for chunk in chunks)

    with pytest.raises(ValueError, match="positive"):
        CartesianWaypointPlanner(maximum_step_m=0).plan(
            Pose(position=(0, 0, 0)), Pose(position=(0.1, 0, 0)), prefix="invalid"
        )
    with pytest.raises(ValueError, match="0.25"):
        CartesianWaypointPlanner(action_chunk_s=0.3).plan(
            Pose(position=(0, 0, 0)), Pose(position=(0.1, 0, 0)), prefix="invalid"
        )


def test_franka_scene_and_executor_use_exact_control_horizon() -> None:
    env = FrankaTabletopEnv()
    assert env.model.nu == 7
    assert env.model.joint("panda_joint7").id >= 0
    start = env.gripper_pose()
    target = Pose(
        position=(start.position[0], start.position[1] - 0.02, start.position[2] - 0.02)
    )
    result = MujocoChunkExecutor(env).execute(
        ActionChunk(chunk_id="quarter-second", target_pose=target)
    )
    assert result.simulation_steps == 125
    assert result.end_time_s - result.start_time_s == pytest.approx(0.25)
    assert result.position_error_m < 0.01

    original_cup = env.observe().entity("cup").pose.position
    moved = env.move_cup((0.06, 0.0, 0.0))
    assert moved.entity("cup").pose.position[0] == pytest.approx(original_cup[0] + 0.06)


def test_episode_recorder_writes_versioned_atomic_artifacts(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path, "episode", 11, {"action_chunk_s": 0.25})
    frames = [
        marker_cup_frame(float(index), (0.1 * index, 0, 0.06), (0.2, 0, 0.05))
        for index in range(3)
    ]
    for frame in frames:
        recorder.record_frame(frame)
    action = ActionChunk(
        chunk_id="move",
        target_pose=Pose(position=(0.1, 0, 0.2)),
        gripper_command=GripperCommand.HOLD,
    )
    recorder.record_action(action)
    recorder.record_event(RuntimeEvent(status=RuntimeStatus.RUNNING, reason="valid"))
    assert recorder.demonstration().demonstration_id == "episode"
    episode_dir = recorder.close(outcome="succeeded")

    metadata = json.loads((episode_dir / "metadata.json").read_text())
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["frame_count"] == 3
    assert len((episode_dir / "frames.jsonl").read_text().splitlines()) == 3
    assert not list(episode_dir.glob("*.tmp"))
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.record_frame(frames[-1])


def test_scripted_demo_compiles_and_attaches_marker(tmp_path: Path) -> None:
    env = FrankaTabletopEnv()
    executor = MujocoChunkExecutor(env)
    planner = CartesianWaypointPlanner()
    recorder = EpisodeRecorder(tmp_path, "physical-prompt", 0, {})
    demonstration = record_marker_into_cup_demonstration(env, executor, recorder, planner)
    task = PickPlaceCompiler().compile(demonstration)
    assert task.manipulated_object_id == "marker"
    assert task.target_id == "cup"
    assert not env.marker_attached


def test_full_cup_movement_recovery_succeeds_and_persists_evidence(tmp_path: Path) -> None:
    summary = run_demo(seed=7, output_dir=tmp_path)
    assert summary["status"] == "succeeded"
    assert summary["replans"] == 1
    assert float(summary["final_position_error_m"]) < 0.01

    metadata = json.loads((tmp_path / "recovery-seed-0007" / "metadata.json").read_text())
    assert metadata["outcome"] == "succeeded"
    events = [
        json.loads(line)
        for line in (tmp_path / "recovery-seed-0007" / "events.jsonl").read_text().splitlines()
    ]
    assert any(event["status"] == "replan_required" for event in events)
    assert events[-1]["status"] == "succeeded"


def test_recovery_fails_closed_when_replan_budget_is_exceeded(tmp_path: Path) -> None:
    env = FrankaTabletopEnv()
    executor = MujocoChunkExecutor(env)
    planner = CartesianWaypointPlanner()
    demo_recorder = EpisodeRecorder(tmp_path, "demo-budget", 0, {})
    task = PickPlaceCompiler().compile(
        record_marker_into_cup_demonstration(env, executor, demo_recorder, planner)
    )
    env.reset(seed=3, randomize=True)

    def keep_moving_cup(_: int, live_env: FrankaTabletopEnv) -> None:
        live_env.move_cup((0.05, 0.0, 0.0))

    runtime = PickPlaceRecoveryRuntime(
        env=env,
        executor=executor,
        planner=planner,
        monitor=ReplanMonitor(target_displacement_threshold_m=0.04),
        maximum_replans=1,
    )
    report = runtime.run(task, disturbance_hook=keep_moving_cup)
    assert report.status == RuntimeStatus.FAILED
    assert report.reason == "replan budget exceeded"
    assert report.replans == 2
