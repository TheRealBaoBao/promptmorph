# Architecture

## Capability contract

The input is one temporally ordered demonstration containing object poses, gripper pose,
gripper state, timestamps, and confidence. The output is a task graph expressed in
object-relative coordinates. At deployment, the runtime converts that graph into live
world goals and executes bounded action chunks.

The v0.1 contract supports rigid pick/place tasks with one manipulated object and one
target. Unsupported or ambiguous prompts are rejected.

## Components

| Component | Responsibility | Must not do |
|---|---|---|
| Perception backend | Produce timestamped entities, poses, sizes, confidence | Select a task or command actuators |
| Prompt compiler | Infer manipulated object, target, and relations | Depend on live robot joints |
| Task graph | Store intent and tolerances | Reference MuJoCo body IDs |
| Goal adapter | Instantiate a relation in the live world frame | Choose motion strategy |
| Motion planner | Produce a bounded collision-aware action chunk | Declare task success |
| Runtime monitor | Validate confidence, plan freshness, progress, budgets | Silently ignore invalid state |
| Episode logger | Persist evidence and failure taxonomy | Change runtime behavior |

## Implemented protocols

The runtime boundary is expressed by these Python protocols:

```python
class PerceptionBackend(Protocol):
    def observe(self) -> WorldFrame: ...

class MotionPlanner(Protocol):
    def plan(self, frame: WorldFrame, goal: Pose) -> ActionChunk: ...

class ChunkExecutor(Protocol):
    def execute(self, chunk: ActionChunk) -> ExecutionResult: ...
```

MuJoCo implements perception and execution for the MVP. A future ROS 2 adapter should not require
changes to the compiler or task graph.

## MuJoCo control loop

Each `ActionChunk` is capped at 250 ms. The Franka adapter solves a position-only
damped-least-squares IK target, tracks a smooth joint trajectory through MuJoCo position
actuators, and feeds forward MuJoCo gravity/Coriolis bias forces. At a 2 ms timestep,
every full chunk advances exactly 125 physics steps before returning a fresh `WorldFrame`.

The monitor evaluates that frame before another chunk is committed. If the cup moved
beyond the active plan validity radius, the runtime discards all remaining waypoints,
re-instantiates the marker goal in the new cup frame, and generates a new bounded plan.

## Coordinate conventions

- Right-handed Cartesian frames
- SI units: meters, seconds, radians
- Quaternions use `(x, y, z, w)` everywhere
- Poses are named `subject_in_reference` or `entity_world`; avoid ambiguous `pose`
- Timestamps are monotonic simulation time for v0.1

Every external adapter must convert into this convention at its boundary.

## Replanning semantics

A plan is valid only relative to the world frame from which it was created. The runtime
invalidates it when:

- target translation exceeds the configured validity radius at a chunk boundary;
- perception confidence falls below threshold;
- observations become stale;
- the executed chunk does not make minimum progress;
- contact/grasp state contradicts the expected subgoal;
- collision, timeout, or replan budget is reached.

Replanning is an expected state transition, not an exception.
