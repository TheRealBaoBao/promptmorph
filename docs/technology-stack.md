# Technology stack

The stack is selected to support the capability rather than decorate the repository.

| Technology | PromptMorph responsibility |
|---|---|
| Python 3.11 | Shared perception, task compilation, planning, and experiment logic |
| PyTorch | Train native sensorimotor models from scratch on automatically labeled physical-interaction trajectories |
| MuJoCo | Franka joint dynamics, seeded scene disturbances, camera-ready state, and rollouts |
| NumPy | Geometry, episode arrays, and simulator-independent feature extraction |
| Pydantic | Runtime validation and versionable data contracts |
| Docker | Reproducible headless simulation and CI environment |
| GitHub Actions | Lint, strict typing, tests, coverage, and container builds |
| Ruff, mypy, pytest | Static and behavioral quality gates |

This intentionally mirrors the classes of work Generalist publicly describes: strong
PyTorch fundamentals for multimodal robot training; Python video/data pipelines; model
training followed by robot validation; GPU/NumPy/Python optimization; and containerized,
scalable infrastructure.

## PyTorch component

The first `SubgoalProgressNet` predicts one of four phases from object-relative state:

1. `approach`
2. `grasp`
3. `transport`
4. `place`

Training examples will come from MuJoCo rollouts and are task-layout invariant because
features are relative to the manipulated object and target. The one demonstration still
defines the desired relation; the learned component estimates execution progress.

The model is initialized from scratch. PromptMorph does not fine-tune CLIP, a VLM, or a
pretrained robot policy for v0.1. Once the data engine is stable, the next bounded experiment
is a small sensorimotor prompt transformer trained from scratch on prompt/current-state/action
sequences. It becomes an action-proposal source only if it improves the project-level goal.

The network is **not** allowed to override safety gates. Low confidence, stale state,
collisions, and exceeded budgets remain deterministic runtime stops. This separation lets
the project demonstrate ML training without making a small learned model the only safety
layer.

## MuJoCo component

`FrankaTabletopEnv` owns simulator names, joint addresses, mocap bodies, and ground-truth
observations. `MujocoChunkExecutor` exposes only typed `ActionChunk → ExecutionResult`
behavior to the rest of the system. Its 250 ms horizon is a deliberate receding-horizon
contract: no planner can queue a long open-loop motion without another observation and
plan-validity check.

The current Panda asset uses self-contained analytic geometry instead of downloaded mesh
assets. This keeps clean Colab and Docker installs deterministic. Kinematic marker
attachment isolates planning/recovery quality from contact-grasp tuning in this MVP.

## Container usage

```bash
docker build -t promptmorph:dev .
docker run --rm promptmorph:dev --seed 7
```

The image runs as an unprivileged user and sets `MUJOCO_GL=egl` for headless rendering.
GPU passthrough becomes relevant only after vision-model integration.
