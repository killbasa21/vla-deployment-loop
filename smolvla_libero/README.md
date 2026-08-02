# `smolvla_libero/` — SmolVLA-450M on the green-ball task

Running **`HuggingFaceVLA/smolvla_libero`** — HuggingFace's official SmolVLA-450M fine-tuned
on LIBERO — against this project's MuJoCo Panda scene.

Status, 2026-07-31: **first fine-tune done, on `a5`. It works and it is slow** — it
approaches, grasps, transports and places, but takes ~54 action chunks to do it.

That is the dataset, not the policy: `a5` episodes are 539 ticks / 27 s each, so the
policy is reproducing its expert faithfully. Full diagnosis in `libero/PROGRESS.md` §23;
the short version is that `OSC_SPEED_SCALE = 2.5` slowed the expert to stop action labels
clipping, which is the wrong knob for that (the ceiling is set by `DELTA_POS_SCALE`) and
did not even work — `a5` still clips `dx` on 3.07% of frames.

**`a6` is the re-collection**: `--delta-pos-scale 0.20`, distance-retimed segments,
shuffled bins. 161 ticks/episode, 0.00% saturation, and `dx q01 = -0.681` against released
LIBERO's -0.679. Retraining on it is the next step. Two things must match at serving time
or the run is not measuring `a6`:

```bash
uv run python libero/libero_closed_loop.py --delta-pos-scale 0.20 \
    --randomize-bins --randomize-ball --server-url <url>/act
```

## Why this checkpoint

`libero/README.md` picked MolmoAct2-**LIBERO** over MolmoAct2-DROID because LIBERO is a
robosuite/MuJoCo benchmark on a Franka Panda — same simulator family, same arm. That argument
applies here unchanged, and SmolVLA adds three things MolmoAct2 did not:

| | MolmoAct2-LIBERO | SmolVLA-LIBERO |
|---|---|---|
| params | 5.57 B | **0.45 B** (12x smaller) |
| serving GPU | L4, 24 GB | **T4, 16 GB** — the cheapest Modal offers |
| fine-tune | LoRA r32, ~$5/run, 200 s per checkpoint save | full fine-tune fits on one mid card |
| published LIBERO | — | 90 / 96 / 92 / 71, avg **87.3%** |

The size is the point. `docs/FINE_TUNE_LEARNINGS.md` §5.5 records the binding constraint on the
MolmoAct2 path: a $5 budget bought **150 steps = 0.06 epochs**, at which point the honest read
was that the checkpoint had "barely moved off the base". A 12x smaller model turns the same
budget into a real training run.

**Caveat, load-bearing.** The published 87.3% has a reproduction gap:
[lerobot#3264](https://github.com/huggingface/lerobot/issues/3264) reports the *official*
checkpoint scoring 63/93/81/56, avg **73.25%**. Treat 87.3% as unverified. If we need a
trusted baseline, `lerobot-eval --env.type=libero` measures it directly (see below).

## The conventions line up already

`smolvla_libero`'s `config.json` against what `libero_closed_loop.py` and `a4` already produce:

| | smolvla_libero | ours | |
|---|---|---|---|
| `observation.state` | `[8]`, MEAN_STD | `[eef_pos(3), axisangle(3), gripper_qpos(2)]` | ✓ exact |
| `action` | `[7]`, MEAN_STD | 6-D delta EE + gripper, `[-1, 1]` | ✓ exact |
| `observation.images.image` | `[3, 256, 256]` | `external_cam` @ 256 | ✓ exact |
| `observation.images.image2` | `[3, 256, 256]` | `eye_in_hand` @ 256 | ✓ **after rename** |
| `chunk_size` / `n_action_steps` | 50 / 1 | horizon 10 | server returns 10 |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Instruct` | — | |

Three of four match byte for byte, because `a4` was built against LIBERO's own convention in
the first place (`libero/fine_tune/README.md` §2). Only the wrist camera's *name* differs.

### That rename is not cosmetic

LeRobot's LIBERO docs: *"naming keys are encoded inside the normalization statistics layer."*
A policy built from this config looks up `observation.images.image2`. Hand it a dataset or a
request that only has `wrist_image` and it does **not** error — it finds no second camera and
the wrist view silently disappears. That is the same failure shape as `PROGRESS.md` §5's
hardcoded `NORM_TAG`, which produced "garbage actions of the correct shape".

So the rename is applied on both sides:
- **dataset** — `convert_dataset.py` (below)
- **wire** — `smolvla_modal.py`'s `WIRE_TO_FEATURE` maps the client's `wrist_image` onto
  `observation.images.image2`

## Files

```
smolvla_libero/
  README.md            this file
  convert_dataset.py   a4 -> data/a4_smolvla, renamed + re-statted
  smolvla_modal.py     Modal T4 server, /act protocol-compatible with libero_closed_loop.py
  data/a4_smolvla/     the converted dataset (gitignored; regenerate with the script)
```

### `convert_dataset.py`

Copies `libero/fine_tune/a4` and fixes two things.

**1. The key rename, in all four places it appears.** Renaming only the obvious one leaves a
dataset that fails deep inside the loader:

| file | what |
|---|---|
| `data/chunk-*/file-*.parquet` | the Arrow column **and** the `huggingface` schema metadata blob — rename the column alone and the struct is never decoded as an Image |
| `meta/info.json` | the `features` dict |
| `meta/stats.json` | the per-feature stats |
| `meta/episodes/chunk-*/file-*.parquet` | five flattened `stats/<key>/{min,max,mean,std,count}` columns |

**2. The stats swap.** `a4/meta/stats.json` did **not** hold a4's own statistics —
`pin_released_stats.py` had overwritten it with MolmoAct2's released LIBERO stats
(`count = 273465` vs a4's `19440`) so the MolmoAct2 fine-tune would inherit the pretrained
q01/q99 calibration. Correct there; wrong here, since SmolVLA normalises STATE/ACTION with
**MEAN_STD**, not q01/q99, against a different pretraining distribution. The script installs
`stats_measured.json` (the untouched backup of a4's real numbers) as `stats.json` and keeps
the MolmoAct2 one as `stats_molmoact2_released.json`.

*Deliberately not decided yet:* a fine-tune may instead want `HuggingFaceVLA/libero`'s stats,
so the action expert keeps the normalisation it was trained under. That only matters once
training starts — for the **stock** checkpoint the policy carries its own normalisation
buffers and this file's stats are never read.

```bash
uv run python smolvla_libero/convert_dataset.py            # convert + verify
uv run python smolvla_libero/convert_dataset.py --verify-only
```

Verified output: 90 episodes / 19,440 frames, v3.0, fps 20, features
`observation.images.image`, `observation.images.image2`, `observation.state`, `action`.

### `smolvla_modal.py`

Serves the stock checkpoint on the **cheapest GPU that can run it**.

450M is ~1.8 GB at fp32, so memory is nowhere near the constraint on a 16 GB T4 — the only
real question is Turing's lack of **bf16**, handled by loading in **float32** instead. Rungs
if the T4 disappoints: `L4` ($0.80/hr, Ada, bf16 fine — what `libero_modal.py` uses for the
5.57B model), then `A10G` ($1.10/hr). Override with `--gpu`.

**The sim client needs no changes.** The server speaks the same `/act` wire format
`libero_closed_loop.py` already implements, and returns **10** actions per call rather than
the checkpoint's `n_action_steps: 1` — matching LIBERO's action horizon, what the client
expects, and the `--policy.n_action_steps=10` LeRobot itself uses to reproduce published
LIBERO results. Re-querying every tick over HTTP would be pointless anyway: `PROGRESS.md` §2
measured transport at ~4x the inference cost.

## Running it

```bash
modal run    smolvla_libero/smolvla_modal.py     # ephemeral + self-test
modal deploy smolvla_libero/smolvla_modal.py     # persistent, prints a stable URL
```

Then the existing closed loop, unmodified:

```bash
uv run python libero/libero_closed_loop.py \
    --payload-keys libero \
    --server-url https://<printed-url>/act \
    --chunks 20 --randomize-ball --no-view \
    --run-id smolvla_stock_00
```

### Which controller this runs under

Nothing here needs wiring: this path drives `libero/libero_closed_loop.py` itself, so it
picked up the OSC port automatically. As of 2026-07-28 that client defaults to
**`--control-mode osc`** — a port of robosuite's own `OSC_POSE` writing joint torques on
`scene_libero_osc.xml` — instead of the old Route A IK-into-position-servos. See
`libero/README.md` "Control" and `PROGRESS.md` §22.

That matters more for SmolVLA than for MolmoAct2, for one reason: the argument for this
checkpoint is that its conventions *already* line up with LIBERO's, so the fewer places
we diverge from the plant it was trained on, the more meaningful a stock-checkpoint
number is. The controller was the largest remaining divergence.

```bash
# the pre-2026-07-28 controller, if a run needs to be compared against older logs
uv run python libero/libero_closed_loop.py --control-mode ik --payload-keys libero ...
```

**Consequence for `data/a4_smolvla`.** It descends from `libero/fine_tune/a4`, whose
labels were produced through Route A. The collector's design rule is that labels come
from the controller that will consume them, so a fine-tune trained on it and served
through OSC breaks that rule. Regenerating `a4` against OSC (free, local CPU) has to
happen before the fine-tune this directory is being set up for — and the conversion here
re-run on top of it. The key rename and stats logic in `convert_dataset.py` are
unaffected; only the underlying episodes change.

`--payload-keys libero` is required: it sends `{image, wrist_image, instruction, state}`.
The `droid` default sends `external_cam` / `wrist_cam`, which this server rejects with a 400
rather than silently running blind.

### Scoring

One rollout is one draw — SmolVLA's action expert is flow-matching, it **samples**
(`libero/README.md`; `PROGRESS.md` §18 has two runs from an identical start diverging
completely). Run several, then:

```bash
uv run python libero/score_runs.py assets/smolvla_libero/stock
```

`score_runs.py` works unchanged — this server emits the same log schema, because the client
writing those logs is the same client.

**Baseline to beat** (`README.md` §9, MolmoAct2-LIBERO stock on the corrected scene):
**0/3 placements, 1/3 grasp-and-lift.**

### Independent control

`lerobot-eval --env.type=libero` runs the real LIBERO suites in robosuite. That is the same
role `libero/libero_benchmark_eval.py` plays for MolmoAct2 — it separates "our scene is out
of distribution" from "the checkpoint is broken" — but as one command instead of a separate
robosuite venv:

```bash
lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero --env.task=libero_object \
  --policy.n_action_steps=10 --eval.n_episodes=10
```

It also settles the reproduction-gap caveat above with our own measurement.

## Not done yet

- **Nothing has been run against the sim.** The server is written and its conventions are
  checked against the checkpoint's own `config.json` at container start, but no rollout exists.
- **No fine-tune.** `data/a4_smolvla` is ready for one, but two things now gate it: the
  normalisation-stats question above, and the fact that its labels came from Route A while the
  client now serves through OSC (see "Which controller this runs under"). `a4` needs
  regenerating and converting again first.
- **`a4`'s rotation channels are ~2.5x narrower than released LIBERO's**
  (`libero/fine_tune/README.md` §6) — our task holds one top-down orientation throughout. Any
  fine-tune on it will see almost no rotational signal, and that is where overfitting shows
  first.
