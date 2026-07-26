# Phase 4 — Classical IK expert demos → LeRobot dataset → fine-tune MolmoAct2 on Modal

## What this phase is

Phase 3 ran the *pretrained* MolmoAct2-DROID checkpoint in a closed loop and it didn't
reliably do the task. Phase 4 flips the approach: instead of hoping a general checkpoint
works zero-shot, we **generate our own training data** for the specific task and
**fine-tune** the model on it.

The task: **pick up the green ball and put it in the green container.**

We produce the demonstrations *classically* — no teleoperation, no joystick, no policy
in the loop. A hand-written waypoint script drives the arm through the pick-and-place
motion, and a numerical inverse-kinematics (IK) solver turns each Cartesian waypoint
into arm joint angles. We record every step as a training example and write it out in
the **LeRobot dataset format**, which is exactly what MolmoAct2's fine-tuning code
consumes. Then we fine-tune MolmoAct2-DROID on that data on a rented **Modal** GPU (the
same way Phase 3 served inference on Modal).

End-to-end goal:

```
 collect IK demos  ->  LeRobot dataset  ->  fine-tune MolmoAct2-DROID on Modal  ->  serve it
 (phase4_collect_    (data/green_ball_    (phase4_modal_train.py)               (phase3_modal_finetuned.py)
  demos.py)           pick/)
```

The fine-tuned checkpoint speaks the same `/act` wire format as Phase 3, so
`phase3_closed_loop.py` can keep using the same POST payload. The saved artifact is not
the same *file format* as the released Hugging Face checkpoint, though: Phase 4 produced
an OLMo/PyTorch distributed training checkpoint (`config.yaml`, `model_and_optim/*.distcp`,
and `train/rank0.pt`). Use `phase3_modal_finetuned.py` to serve that raw checkpoint via
MolmoAct2's experiments policy server. Keep `phase3_modal.py` for the released base
`allenai/MolmoAct2-DROID` HF checkpoint.

---

## Key decisions (and why)

| Decision | Choice | Why |
| --- | --- | --- |
| IK method | **MuJoCo numerical IK** (Jacobian damped least squares), not real IKFast | No external toolchain (IKFast needs OpenRAVE codegen), and it handles the Panda's 7-DOF redundancy naturally. Standard for sim demo generators. |
| Robot / scene | **Reuse `scene_pick_place.xml`** (Panda + Robotiq 2F-85) + add a green ball | Least new surface area; we already trust this scene from Phase 3. |
| Output format | **LeRobot v2.1**, written directly (parquet + mp4 + meta json) | The format MolmoAct2's trainer reads. Written *without* installing the heavy `lerobot`+torch stack on this laptop, to keep the client light. |
| State/action convention | **8-D DROID: `[q1..q7, gripper_rad]`** | Identical to what `phase3_closed_loop.py` already sends/receives, so a model trained on this speaks the existing `/act` protocol. |
| Fine-tune start point | **`allenai/MolmoAct2-DROID`** | Our data is the DROID embodiment (Franka, absolute joint pose). Fine-tune the checkpoint we've already been serving. |

---

## The classical IK expert — how it works

File: **`phase4_collect_demos.py`**

### Inverse kinematics as a *planner*
The IK solver (`solve_ik`) does **not** teleport the robot. It runs on a throwaway copy
of the sim state (`scratch = mujoco.MjData`) to compute a target joint configuration for
a desired end-effector pose, then the real sim is driven toward that configuration by the
Panda's **position actuators**. This means the actual grasp happens through **real
contact physics** — the fingers close on the ball, friction holds it — not by moving the
ball's coordinates by hand.

- **Target site:** the `pinch` site on the Robotiq gripper (the fingertip convergence
  point). At the home pose this site's Z axis already points straight down, so top-down
  grasps just hold that orientation while moving in X/Y/Z.
- **Algorithm:** damped least squares. Each iteration computes the 6-D pose error
  (position + orientation), gets the site Jacobian (`mj_jacSite`), and takes a step
  `dq = Jᵀ (JJᵀ + λ²I)⁻¹ · err`, clipped and clamped to joint limits. `λ` (damping)
  keeps it stable near singularities.

### The pick-and-place waypoint script
`plan_episode` solves IK at each of these waypoints (chaining the seed so the elbow
doesn't flip between them), then interpolates joint space between them to get one
`(joint_target, gripper)` setpoint per control tick:

1. Pre-grasp — hover ~14 cm above the ball, gripper **open**
2. Descend onto the ball, still open
3. Close the gripper (arm holds still while fingers close)
4. Lift straight up
5. Transport over the container
6. Lower into the container
7. Release (open gripper)
8. Retreat up

### Recording (the training data)
The sim runs at 500 Hz; control is applied at **15 Hz** (DROID's rate) via a decimation
of 33 steps per tick. Every control tick, *before* stepping, we record:

- `external_cam` + `wrist_cam` RGB frames (the same two cameras Phase 3 uses),
- `state` (8,) = `[q1..q7, gripper_rad]` — the DROID server's own state schema,
- `action` (8,) = `[q1..q7 target, gripper_target_rad]` — absolute joint targets, the
  same thing MolmoAct2-DROID's `/act` endpoint returns.

### Diversity + success filtering
- **Ball position** is randomized per episode over a reachable patch in front of the arm.
- **Arm start pose** is jittered per episode (`--start-jitter`, default 0.05 rad) so demos
  don't all begin from the identical configuration.
- An episode is **kept only if it succeeds** — the ball ends up resting inside the green
  container's footprint (checked from the ball's final position).

### Verified
- IK grasp works: preview episode picks the ball and drops it in the container (confirmed
  by rendering a contact sheet and watching the mp4).
- **100% success across 40 randomized positions** (headless probe) and across full
  recorded runs with start-jitter on.
- Speed: ~25 s per episode (dominated by rendering 2 cameras × 121 steps + mp4 encode).

---

## The dataset format

File: **`lerobot_writer.py`** — a minimal LeRobot **v2.1** writer with no `lerobot`/torch
dependency (only `pyarrow` for parquet + `imageio[ffmpeg]` for mp4).

Layout produced under `--out` (default `data/green_ball_pick/`):

```
data/green_ball_pick/
  meta/
    info.json             # feature schema, fps, counts, path templates
    tasks.jsonl           # {task_index, task="pick up the green ball..."}
    episodes.jsonl        # {episode_index, tasks, length} per episode
    episodes_stats.jsonl  # per-episode min/max/mean/std/count for every feature
  data/chunk-000/
    episode_000000.parquet   # per-frame state/action/index columns (NO pixels)
    episode_000001.parquet
    ...
  videos/chunk-000/
    observation.images.external_cam/episode_000000.mp4
    observation.images.wrist_cam/episode_000000.mp4
    ...
```

- Images are stored as **video features** (mp4), not inline in the parquet. The parquet
  holds only the low-dimensional columns; a LeRobot loader decodes the matching mp4 by
  timestamp (`frame_index / fps`) at load time — hence one frame per parquet row.
- `episodes_stats.jsonl` holds the normalization statistics the trainer uses (state/action
  mean/std/min/max, and channel-first `(3,1,1)` image stats).

### Verified
Dataset reads back correctly: v2.1 layout, parquet columns
(`observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`,
`task_index`), 121-frame 256×256 mp4s, per-feature stats, and the task string.

---

## How MolmoAct2 gets trained on this

MolmoAct2's training code **is released** (in `molmoact2/experiments/` — note the
`molmoact2/CLAUDE.md` "coming soon" line is stale). Fine-tuning a new LeRobot dataset is
described as the repo's primary use case.

- Training is **behavior cloning**: given the two images + state + the language
  instruction, optimize the model to output the expert action the IK script took.
- MolmoAct2 predicts an **action chunk** (~10–15 future steps) via a continuous
  flow-matching action head. The trainer slices `action[t : t+horizon]` as the target at
  each frame `t` — trivial because we recorded one action per row in time order.
- Normalization comes from the stats in `episodes_stats.jsonl`.

We registered a mixture named **`green_ball_pick`** in
`molmoact2/experiments/launch_scripts/data_mixtures.py`, mirroring the built-in `droid`
tag (Franka, `control_mode="absolute joint pose"`, `normalize_gripper=False`,
`action_horizon=15`) but pointed at our dataset and our camera key names. Because it's the
DROID embodiment, we start fine-tuning from `allenai/MolmoAct2-DROID`.

**Three fine-tune modes** (increasing cost/scope):
- `ae_only` — freeze the vision-language model, train only the action expert. Cheapest,
  fits 1 GPU, and the right call when the new dataset mainly changes control (our case).
- `lora` — LoRA adapters on the VLM path + full action expert.
- `full` — full fine-tune of everything (Ai2's own recipe uses 8×80 GB GPUs).

---

## Running training on Modal

File: **`phase4_modal_train.py`** — the training counterpart to `phase3_modal.py`. It
wraps `experiments/launch_scripts/train_lerobot.py` in a Modal container and runs it under
`torchrun`. It's a **separate image** from the serving one because the trainer needs
`transformers>=5.3` while the inference server pins `4.57`.

Three Modal Volumes:
- `molmoact2-hf-cache` (`/cache/huggingface`) — shares the ~22 GB base checkpoint with the
  inference app.
- `molmoact2-lerobot-data` (`/data` = `LEROBOT_DATA_ROOT`) — where the dataset lives.
- `molmoact2-checkpoints` (`/checkpoints`) — training output.

### Prerequisites (do these first — not yet done)
1. Populate the `lerobot` submodule the trainer installs (it ships **empty**):
   ```bash
   cd molmoact2 && git submodule update --init lerobot && cd ..
   ```
2. Collect enough data and upload it to the data Volume at the repo_id path the mixture
   expects (`greenbox/green_ball_pick`):
   ```bash
   uv run python phase4_collect_demos.py --episodes 300 --out data/green_ball_pick
   modal volume create molmoact2-lerobot-data
   modal volume put molmoact2-lerobot-data data/green_ball_pick /greenbox/green_ball_pick
   ```
3. (Optional) W&B logging: `modal secret create wandb-secret WANDB_API_KEY=...` (omit to
   run wandb-disabled).

### Run
```bash
modal setup                                              # one-time auth
modal run phase4_modal_train.py --max-steps 20           # tiny smoke run first
modal run phase4_modal_train.py                          # action-expert-only, 1 GPU (cheapest)
modal run phase4_modal_train.py --mode lora              # LoRA
modal run phase4_modal_train.py --mode full --gpus 8     # full fine-tune (8×80 GB)
```
Pull the result and serve it:
```bash
modal volume get molmoact2-checkpoints checkpoints/finetune/ae_train ./out
modal deploy phase3_modal_finetuned.py
uv run python phase3_closed_loop.py --model-path droid \
    --server-url <printed-finetuned-modal-url>/act \
    --request-timeout 600 --chunks 5
```

### Status / honesty
The 500-step `ae_train` run has now been executed on Modal and completed. The final
checkpoint was pulled recursively from `molmoact2-checkpoints` into
`fine_tunes/pick_up_tasks/ae_train/run_20260719_ae500/checkpoints/step500`.

Verified local checkpoint contents:
- `config.yaml`
- `train/rank0.pt`
- `model_and_optim/.metadata`
- 16 `model_and_optim/__0_N.distcp` shards
- Total local size: approximately 25 GB

Training result:
- Final `train/action_flow_loss`: 0.0214 at step 500.
- Loss trended down from about 0.18 early in training to the 0.002-0.02 band late in
  training, noisy as expected with tiny batches.

Current serving status:
- `phase3_modal_finetuned.py` has been added to serve the raw DCP checkpoint from the
  Modal checkpoint volume using `molmoact2/experiments/scripts/serve_policy.py`.
- It has not yet been deployed or smoke-tested.
- No simulation has yet been run against the fine-tuned model.
- No training/download/simulation process is currently running.

---

## Files this phase added or changed

| File | What |
| --- | --- |
| `phase4_collect_demos.py` | **New.** The IK expert + episode executor + recorder. `--preview`, `--episodes`, `--start-jitter`, `--res`, `--seed`. |
| `lerobot_writer.py` | **New.** Minimal LeRobot v2.1 dataset writer (parquet + mp4 + meta). |
| `phase4_modal_train.py` | **New.** Modal wrapper that runs `train_lerobot.py` on a rented GPU. |
| `phase3_modal_finetuned.py` | **New.** Modal wrapper intended to serve the raw `ae_train/step500` distributed checkpoint via MolmoAct2's experiments policy server. Added but not yet deployed/smoke-tested. |
| `fine_tunes/pick_up_tasks/ae_train/run_20260719_ae500/run_info.md` | **New.** Metadata and result notes for the completed `ae_train` fine-tune. |
| `mujoco_menagerie/franka_emika_panda/scene_pick_place.xml` | **Changed.** Added the graspable `green_ball` (freejoint sphere, r=2 cm) as the pick target. |
| `molmoact2/experiments/launch_scripts/data_mixtures.py` | **Changed (vendored).** Registered the `green_ball_pick` mixture. |
| `pyproject.toml` | **Changed.** Added `pyarrow` + `imageio[ffmpeg]` for the dataset writer. |

## Quick command reference

```bash
# 1. See one episode as a video (no dataset written)
uv run python phase4_collect_demos.py --preview --seed 1        # -> assets/phase4_preview.mp4

# 2. Collect a dataset
uv run python phase4_collect_demos.py --episodes 300 --out data/green_ball_pick

# 3. (prereqs) populate lerobot submodule + upload dataset to Modal (see above)

# 4. Fine-tune on Modal
modal run phase4_modal_train.py --max-steps 20                  # smoke test
modal run phase4_modal_train.py                                 # real run (action-expert-only)

# 5. Retrieve + serve the fine-tuned checkpoint
modal volume get molmoact2-checkpoints checkpoints/finetune/ae_train ./out
modal deploy phase3_modal_finetuned.py
uv run python phase3_closed_loop.py --model-path droid \
    --server-url <printed-finetuned-modal-url>/act \
    --request-timeout 600 --chunks 5
```
