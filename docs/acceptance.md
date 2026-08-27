# MVP acceptance matrix

All results use a frozen configuration and published seed list. No failed seeds may be
removed after observing results.

| ID | Scenario | Runs | Pass condition |
|---|---|---:|---|
| A1 | Demonstration scene replay | 10 | 10/10 completion |
| A2 | Random object and target translation | 30 | ≥27/30 completion |
| A3 | Cup moved once during approach | 30 | ≥24/30 recoveries |
| A4 | Cup moved twice within replan budget | 30 | ≥21/30 recoveries |
| A5 | Perception confidence forced below threshold | 10 | 10/10 safe stops |
| A6 | Target moved outside reachable workspace | 10 | 10/10 explicit failures |
| A7 | Exact rerun from stored seed/config | 10 | Identical task events |

## Episode success

An episode succeeds only when all of the following hold for five consecutive frames:

- manipulated-object position is within 2.5 cm of the instantiated relative goal;
- the object is released;
- the object remains stable;
- no collision or safety violation occurred;
- episode time and replan count remain within budget.

## Required artifacts per episode

- `metadata.json`: seed, configuration hash, commit, timestamps
- `task_graph.json`: compiled immutable prompt representation
- `events.jsonl`: subgoal changes, plan invalidations, replans, failures
- `trajectory.npz`: entity, gripper, and commanded states
- `video.mp4`: fixed camera plus diagnostic overlay

## Release gate

The v0.1 release is blocked by any of:

- clean Colab installation takes more than 10 minutes;
- a failure lacks a classified reason;
- geometry, compiler, or runtime unit tests fail;
- reported results cannot be regenerated from the stored seeds;
- the showcase video uses different code/configuration than the reported runs.

