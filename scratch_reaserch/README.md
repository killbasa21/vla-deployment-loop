# greenbox

Fine-tune SmolVLA to put a green box in the green container, in robosuite, with
a Franka Panda. Built from scratch: scene, scripted expert, metrics, Modal
training and serving, closed-loop eval.

## The task

A green 4 cm cube sits in a random spot in an 11x11 cm square in the middle of
the table. Three open-topped trays sit at fixed slots -- left, right and top of
the `agentview` frame. **Their colours are permuted every episode**, and the
target is always the green one, so the correct tray cannot be memorised as a
position: it has to be read off the image. The instruction is fixed:

    put the green box in the green container

## Layout

| path | what |
|---|---|
| `greenbox/task_spec.py` | every constant the task is defined by: instruction, scene geometry, camera and feature keys. Single source of truth -- collection, training and serving all import it |
| `greenbox/objects.py` | the tray, a `CompositeObject` whose colour is a runtime `geom_rgba` write |
| `greenbox/env.py` | the robosuite environment |
| `greenbox/expert.py` | scripted OSC waypoint expert |
| `greenbox/policies.py` | action sources: expert, random, remote server |
| `greenbox/metrics.py` | per-episode staged metrics |
| `infra/modal_app.py` | Modal image, volumes, training function, policy server |
| `tools/` | preview, watch, collect, stats, score, checkpoint switch |

## Environments

Two, and they cannot be merged: **robosuite 1.4.1 pins numpy<2, lerobot pins
numpy>=2**, and lerobot 0.6 needs Python >=3.12.

1. **local** (`pyproject.toml`, `uv sync`) -- sim, expert, collection, scoring,
   viewer. Never loads a model, so torch stays out of it.
2. **Modal** (`infra/modal_app.py`) -- Python 3.12, `lerobot[smolvla]==0.6.1`.
   Everything that touches weights runs there.

## What gets fine-tuned

SmolVLA is 450.0 M parameters. With the base checkpoint's own recipe
(`freeze_vision_encoder=true`, `train_expert_only=true`) **99.9 M are trainable
(22.2%)**: the flow-matching action expert plus four projections
(`state_proj`, `action_in_proj`, `action_out_proj`, the action-time MLP). The
SigLIP vision tower and the 16-layer VLM decoder stay frozen and act as a
feature extractor. Measured, not assumed -- VLM hidden 960, expert hidden 720.

The stock checkpoint declares a 6-D-state / 6-D-action robot with three cameras;
ours is 9-D / 7-D with two. That needs no surgery, because SmolVLA pads state
and action to `max_state_dim`/`max_action_dim` (32) before projecting. Loading
the stock weights into our feature shapes reports **0 missing and 0 unexpected
non-normalization keys**.

## Interfaces

Action space is robosuite `OSC_POSE`, `control_delta=True`: 7-D, `[dx, dy, dz,
drx, dry, drz, gripper]`, first six in [-1,1] scaled to +-5 cm / +-0.5 rad per
step, gripper -1 open / +1 close, 20 Hz. This is LIBERO's convention on purpose.

Policy state is 9-D, LIBERO layout: eef pos (3), eef quat xyzw (4), finger qpos (2).

Cameras `agentview` and `robot0_eye_in_hand`, 256x256.

## Running it

```bash
uv sync --extra modal

# look at the scene
uv run python tools/preview_scene.py --episodes 4

# watch an episode in the MuJoCo viewer, with a live action HUD
uv run python tools/watch.py --episodes 3
uv run python tools/watch.py --policy server --server-url <url>

# expert sanity check, and the metric baselines
uv run python tools/score.py --policy expert --episodes 25
uv run python tools/score.py --policy random --episodes 20

# collect (rejection-sampled to successes only); 4 shards in parallel
uv run python tools/collect.py --episodes 75 --seed 100 --out data/demos/shard0

# stats, upload, train
uv run python tools/dump_stats.py
modal volume put greenbox-vol assets/stats.json /stats.json --force
modal volume put greenbox-vol data/demos /demos --force
modal run --detach infra/modal_app.py::train --run-name ft1 --steps 12000

# point the server at a checkpoint and confirm it cut over, then score
uv run python tools/serve_checkpoint.py --checkpoint /vol/checkpoints/ft1/step_012000
uv run python tools/score.py --policy server --server-url <url> --episodes 25
```

## Metrics

`tools/score.py` reports a staged chain rather than one success bit, so a failure
can be located: `reached -> grasped -> lifted -> transported -> released ->
placed -> complete`. `released` specifically means the box was let go **over the
target tray** -- dropping it anywhere else is not a release. Alongside those,
`placed_wrong` and `nearest_tray` separate a control failure from a colour
grounding failure, and continuous measures (`grasp_pos`, `grasp_rad`,
`place_pos`, `lift_height`) are reported as mean / median / p90 / min.

See `RESULTS.md` for the numbers.

## Things that cost time, recorded so they do not cost it twice

- **`quat2axisangle` does not take the short arc.** `q` and `-q` are the same
  rotation but only one unwraps short. A -0.77 rad wrist error came back as
  5.51 rad, saturated the rotation channels, and OSC's position/orientation
  coupling dragged the arm upward instead of down onto the box. Expert went
  0/10 -> 10/10 on one sign flip. Negate the error quaternion when `w < 0`.
- **Verify controller scaling against the controller**, not the docs.
  `output_max` is read off `env.robots[0].controller` so the expert's
  error-to-action conversion is right by construction.
- **`from __future__ import annotations` breaks `modal.parameter`** -- it turns
  the annotations into strings and Modal's type registry fails with
  `'str' object has no attribute '__name__'`.
- **lerobot 0.6 moved tokenization out of the policy.** `predict_action_chunk`
  reads `observation.language.tokens`, which only the preprocessor from
  `make_pre_post_processors` produces. Without it: `KeyError`.
- **The postprocessor is not optional.** `predict_action_chunk` returns
  normalized actions; skipping the postprocessor sends values like -16 into a
  [-1,1] action space.
- **Trays are static (`joints=None`)**, so position is a `sim.model.body_pos`
  write and colour is a `sim.model.geom_rgba` write. Both are runtime edits, so
  `hard_reset=False` and resets stay fast.
- **A policy that never grasps must not score points for "released".** Every
  stage flag is conditional on the one before it.
