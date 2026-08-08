# Setup and operation

Everything here runs **from the repo root** with `uv`. Scripts resolve scene XML, `assets/`
and `data/` relative to the root, so `cd`-ing into a subdirectory will break them.

```bash
git clone --recursive <this repo>      # --recursive: the scenes need mujoco_menagerie
uv sync
```

If you already cloned without `--recursive`:

```bash
git submodule update --init
```

## Running a policy

```bash
# 1. serve a policy on Modal (prints a URL)
modal setup                                     # one-time
modal deploy act/act_modal.py                   # ACT  (best current policy)
modal deploy libero/libero_modal.py             # MolmoAct2-LIBERO, stock
modal deploy libero/libero_modal_finetuned.py   # MolmoAct2-LIBERO, a fine-tune on the volume
modal deploy smolvla_libero/smolvla_modal.py    # SmolVLA
curl -s <url>/health                            # ALWAYS confirm which checkpoint answered

# 2. run the closed loop against it
uv run python libero/libero_closed_loop.py \
    --payload-keys libero --server-url <url>/act \
    --randomize-box --randomize-bins \
    --chunks 70 --no-view --run-id act_ck30000_00

# 3. score the run logs
uv run python libero/score_runs.py assets/act/act-green-ball_010000
```

`libero_closed_loop.py` is the client for **every** track, ACT and SmolVLA included — only the
server changes. Three flags decide whether the run measures anything:

- `--payload-keys libero` sends `{image, wrist_image, instruction, state}`. Required for ACT,
  SmolVLA and MolmoAct2-LIBERO; the `droid` default sends `external_cam`/`wrist_cam` and those
  servers 400 rather than run blind.
- `--delta-pos-scale` **must equal the value the dataset was collected at** — 0.10 for `a7`,
  0.20 for `a6`, 0.05 for `a5`. A mismatch does not measure that fine-tune at all. From
  2026-08-03 the value is 0.05 on both sides by default, so **do not pass the flag** for
  anything collected since.
- `--randomize-box --randomize-bins` — every fixed-layout number in the logs predates these
  and is not comparable across them.

## Collecting data and training

```bash
uv run python libero/fine_tune/collect_finetune_data.py --out libero/fine_tune/a8 --episodes 60
modal run act/act_modal_train.py::main --max-steps 1 --save-freq 1   # SMOKE FIRST, always
modal run act/act_modal_train.py::main --max-steps 60000 --save-freq 10000
```

Anything collected from 2026-08-03 onward is a **box** dataset — the collector reads the
target's name, geometry and instruction from the same place the closed loop does, so there is
no flag to set and no way for the two sides to disagree. `a1`–`a7` are ball datasets.

## The rebuilt task (`greenbox/`)

Separate project directory with its own lockfile, because **robosuite 1.4.1 pins numpy<2 and
lerobot pins numpy>=2**. They cannot share an environment.

```bash
cd greenbox && uv sync --extra modal

uv run python tools/preview_scene.py --episodes 4          # look at the scene
uv run python tools/watch.py --episodes 3                  # viewer + live action HUD
uv run python tools/score.py --policy expert --episodes 25 # expert sanity check
uv run python tools/collect.py --episodes 75 --seed 100 --out data/demos/shard0

uv run python tools/dump_stats.py
modal volume put greenbox-vol assets/stats.json /stats.json --force
modal volume put greenbox-vol data/demos /demos --force
modal run --detach infra/modal_app.py::train --run-name ft1 --steps 12000

export GREENBOX_SERVER_URL=<your modal deploy url>
uv run python tools/serve_checkpoint.py --checkpoint /vol/checkpoints/ft1/step_012000
uv run python tools/score.py --policy server --episodes 25
```

## Package management

Three dependency environments are in play, and mixing them is the mistake to avoid.

### 1. The local project env — `pyproject.toml` + `uv.lock`, python 3.12

The simulator, the closed-loop client, the dataset writers, the scorers. **This env never runs
a model**, so `torch`, `lerobot` and `transformers` are deliberately absent — they would add
multiple GB to a venv with no GPU to use them on.

```bash
uv sync                     # exact lockfile
uv sync --group dev         # + ruff
uv run ruff check .
```

A package goes here only if something under `libero/`, `act/`, `smolvla_libero/`, `droid/` or
`scripts/` imports it **locally**. Anything imported only inside a Modal function belongs to
env 2.

### 2. Modal images — `infra/modal_images.py`, built remotely

Every image the project deploys is defined **once** in that module:

| helper | python | used by |
|---|---|---|
| `lerobot_serve_image()` | 3.12 | `act/act_modal.py`, `smolvla_libero/smolvla_modal.py` |
| `lerobot_train_image()` | 3.12 | `act/act_modal_train.py`, `smolvla_libero/smolvla_modal_train.py` |
| `molmoact_serve_image()` | 3.11 | `libero/libero_modal.py` |
| `molmoact_experiments_image()` | 3.12 | `libero/libero_modal_train.py`, `libero/libero_modal_finetuned.py` |

`torch==2.5.1` + cu121 and `transformers 4.57.x` are pinned there and **nowhere else** — they
are what MolmoAct2's own `pyproject.toml` was validated against. Two python versions is
intentional: 3.11 for the MolmoAct2 serving path, 3.12 for everything LeRobot.

Sharing the definitions is not tidiness. Modal caches image layers by definition, so chains
meant to share the multi-GB torch pull silently stop sharing it the moment one drifts. Before
centralising, that pin was duplicated across five files.

Modal re-imports the app module **inside** the container, so any file importing from `infra/`
must ship it into the image. `with_infra(...)` does that and **must be the last layer** — no
build step may follow `add_local_python_source`.

`droid/` keeps its own inline image chains on purpose: the track is retired and its images are
frozen at what was last deployed.

### 3. The hf-libero env — a separate venv, outside this project

`libero/libero_benchmark_eval.py` needs `hf-libero`, which pins its own `robosuite` and
`mujoco`. **It must not be installed into the project env.**

```bash
MUJOCO_GL=egl <hf-libero-venv>/bin/python libero/libero_benchmark_eval.py \
    --server-url <url>/act --suite libero_object --task-id 0 --episodes 3
```

> **Name collision:** our directory `libero/` shadows the PyPI package `libero`. That file's
> `from libero.libero.envs import OffScreenRenderEnv` resolves only because it runs from the
> other venv with a different working directory. From the repo root it would import our
> directory instead. The import is deliberately deferred inside a function; keep it that way.

## Run artifacts

Every closed-loop run writes two things, grouped by the policy that produced them:

```
assets/
  <model>/<fine_tune>/
    logs/<run_id>.jsonl              one JSON object per action chunk
    images/<run_id>/                 camera_<timestamp_ms>_<cam>.png, both cams interleaved
```

`<model>/<fine_tune>` comes from `--model` / `--fine-tune`. Leave them off and
`libero_closed_loop.py` reads the server's `/health` `checkpoint` field and derives them. A
server that reports no checkpoint gives `unknown-model/unknown` — deliberately ugly, so an
unattributable run looks wrong in `ls`. Each log entry also carries `model` and `fine_tune` so
a file copied out of the tree still says what produced it.

Grouping by policy is what makes comparison possible: `score_runs.py` takes a directory and
recurses, so one policy is `score_runs.py assets/act/act-green-ball_010000` and everything ever
run is `score_runs.py assets/`.

Both files are written and flushed **during** the run — `tail -f` works, and frames land on
disk immediately after each `mj_step`. `--dry-run` writes the log but never renders. `assets/`
is gitignored; it is bulky regenerated debug output, and it was committed by accident once
already.

> Runs from before 2026-08-02 are in the old flat `assets/logs/` and `assets/images/`.
> `score_runs.py` still reads them by path.

## Conventions worth knowing before you touch anything

- **Smoke runs are 1 step.** `--max-steps 1 --save-freq 1` proves build → load → step → save.
  Any more burns GPU money for no extra signal.
- **`/health` before every evaluation.** A deployment that silently kept serving the old
  checkpoint has produced wrong conclusions here more than once.
- **Progress logs keep the wrong turns in.** Several conclusions in them were later reversed;
  knowing *which* were reversed is the useful part. Corrections are appended as new sections,
  never edited into old ones.
- **The last checkpoint is not the best one.** Score intermediate checkpoints.
- **A run log written by a live run is incomplete.** `score_runs.py` prints `INCOMPLETE`;
  believe it. A truncated log ends mid-carry and scores exactly like a release failure.
- **The instruction string is defined once,** in `infra/task_spec.py`. It used to be five
  literals. A prompt that differs between training and serving does not raise — it silently
  conditions the policy on something it never saw.
- **Never edit a file inside `mujoco_menagerie/`.** It is a pinned submodule. Our scenes live
  in `scenes/`; see [`scenes/README.md`](../scenes/README.md).
- **Verify plant parameters by compiling the model**, not by reading the XML.
