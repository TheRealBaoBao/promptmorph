# Failure modes and responses

| Failure | Detection | Runtime response | Logged reason |
|---|---|---|---|
| Ambiguous manipulated object | Motion score margin too small | Reject prompt | `ambiguous_subject` |
| Ambiguous target | Final-distance margin too small | Reject prompt | `ambiguous_target` |
| Low-confidence pose | Confidence below threshold | Stop, do not plan | `low_confidence` |
| Stale observation | Observation age over budget | Stop and refresh | `stale_observation` |
| Target moved | Pose delta over validity radius | Cancel chunk and replan | `target_displacement` |
| Grasp lost | Object no longer follows gripper | Return to grasp subgoal | `grasp_lost` |
| No progress | Goal error fails to decrease | Reobserve then replan | `progress_stalled` |
| Collision | MuJoCo contact allowlist violated | Emergency stop episode | `collision` |
| Unreachable goal | IK/planner infeasible | Stop with diagnostic | `unreachable_goal` |
| Replan loop | Replan count over budget | Terminate episode | `replan_budget` |
| Timeout | Episode clock over budget | Terminate episode | `episode_timeout` |

The showcase must include at least one visible recovery and one safe failure. A system
that only displays successes does not demonstrate production readiness.

