# Goal-driven training principles

PromptMorph is not a project about attaching robot actions to whichever model family is
currently popular. The goal is concrete:

> Transfer the intent of one physical demonstration to a rearranged scene and recover
> closed-loop when the scene changes.

Perception, planning, learning, and control are tools for reaching that goal. Model labels
such as VLA or world model are not project requirements.

## 1. Native physical data

The trainable component consumes physical interaction signals produced by PromptMorph:

- object and target poses;
- gripper pose and state;
- object-relative geometry;
- action chunks;
- contact and collision events;
- progress and failure labels;
- prompt demonstration sequences.

The v0.1 model is trained from random initialization on this data. It is not a fine-tuned
vision-language model with an action head.

## 2. Build the data engine before scaling the model

Every MuJoCo rollout must produce a versioned episode containing observations, actions,
task graph, disturbances, contacts, success state, and failure reason. Dataset manifests
record generator version, configuration hash, seed range, feature schema, and train/test
split.

Train/test separation is by complete scene seed and perturbation configuration—not random
rows from the same trajectory—so temporal leakage cannot inflate results.

## 3. Start small and scale only from evidence

The initial model is a compact PyTorch progress network trained from scratch. A small
sensorimotor prompt transformer is the next candidate after the MuJoCo dataset exists.

The transformer would receive:

```text
[demonstration observation/action tokens] + [rolling live-state tokens]
    → next action-chunk proposal and progress distribution
```

The model stays small enough to train on a Colab GPU. We increase data diversity, context,
or capacity one axis at a time and record the resulting success, recovery, latency, and
calibration changes.

## 4. Learning complements hard constraints

Learned outputs may propose progress or actions. They may not bypass:

- workspace and joint limits;
- collision checks;
- perception-confidence thresholds;
- stale-state detection;
- timeout and replan budgets;
- task-success verification.

This lets the system use learning where generalization matters while retaining inspectable
runtime behavior.

## 5. The model must earn its place

The deterministic compiler/planner is the required baseline. The learned component ships
only if a frozen held-out study demonstrates at least one of:

- higher recovery success under target displacement;
- lower time-to-completion without lower reliability;
- fewer replans for the same success level;
- better progress calibration under observation noise.

Otherwise the negative result is reported and the simpler system remains the release.

## Claims discipline

PromptMorph may claim a model is trained from scratch on its simulated sensorimotor data.
It may not call that model a foundation model, claim physical AGI, or imply equivalence to
GEN-1/GEN-1.5. The relevant contribution is the complete one-shot, closed-loop system and
the quality of its evidence.

