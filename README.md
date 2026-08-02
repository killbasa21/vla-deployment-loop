# greenbox — teaching a policy to pick up a green ball

A learning project about the **whole VLA deployment loop**, not about any one model: a
Franka Panda in MuJoCo on this machine, a policy server on a rented GPU, HTTP in between,
and a pick-and-place task that has to actually succeed for the loop to count as working.

**The task:** *pick up the green ball and put it in the green container.* Ball and bin
positions randomise every episode.

```
 ┌──────────── local (this repo) ─────────────┐        ┌──── Modal GPU ────┐
 │ MuJoCo: Franka Panda + green ball + bins   │  HTTP  │ policy server      │
 │  - renders external_cam + wrist_cam        │ ─────► │ POST /act          │
 │  - reads proprioception                    │  JSON  │ in:  images,       │
 │  - applies the returned action chunk       │ ◄───── │      instruction,  │
 │  - steps physics, logs every chunk         │        │      state         │
 └────────────────────────────────────────────┘        │ out: (N, 8) chunk  │
                                                       └────────────────────┘
```

Four policies have been run through that same loop. The interesting part of the project is
the *comparison*, and the fact that the winner is the smallest model.

## Status, 2026-08-02

| track | policy | best measured result | state |
|---|---|---|---|
| [`act/`](act/README.md) | **ACT**, from scratch on `a7` | **5/6 placed, 6/6 grasp-and-lift** (ck10000) | **best result so far**; training paused at 30 k / 60 k, resumable |
| [`libero/`](libero/README.md) | MolmoAct2-LIBERO, fine-tuned on `a5` / `a7` | 2/10 and 1/10 placed | fine-tunes are indistinguishable from the stock baseline |
| [`libero/`](libero/README.md) | MolmoAct2-LIBERO, stock | 0/3 placed, 1/3 lifted | the baseline everything is scored against |
| [`smolvla_libero/`](smolvla_libero/README.md) | SmolVLA-450M, LoRA on `a5` | completes the task, ~54 chunks | slow because the expert was slow; `a6` retrain never run |
| [`droid/`](droid/README.md) | MolmoAct2-DROID, stock + `ae_train` | never placed | **retired** — every run predates the decimation fix, so its ceiling was zero |

The binding constraint today is **not** geometry, reach or speed. It is the **gripper**:
across 20 MolmoAct2 rollouts, every lift and every placement came from a run where the
gripper fired at all, and it fired in exactly half of them
([`libero/PROGRESS.md` §25.1](libero/PROGRESS.md)). ACT is the one policy that closes
reliably; its remaining failure is the mirror image — it sometimes never *releases*
([`act/PROGRESS.md` §7.4](act/PROGRESS.md)).

## Quickstart

Everything is run **from the repo root** with `uv`. Scripts resolve scene XML, `assets/`
and `data/` relative to the root, so `cd`-ing into a subdirectory will break them.

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
    --delta-pos-scale 0.10 --randomize-ball --randomize-bins \
    --chunks 70 --no-view --run-id act_ck30000_00

# 3. score the run logs
uv run python libero/score_runs.py assets/act/act-green-ball_010000
```

`libero_closed_loop.py` is the client for **every** track, ACT and SmolVLA included — only
the server changes. Three flags decide whether the run measures anything:

- `--payload-keys libero` sends `{image, wrist_image, instruction, state}`. Required for
  ACT, SmolVLA and MolmoAct2-LIBERO; the `droid` default sends `external_cam`/`wrist_cam`
  and those servers 400 rather than run blind.
- `--delta-pos-scale` **must equal the value the dataset was collected at** — 0.10 for
  `a7`, 0.20 for `a6`, 0.05 for `a5`. A mismatch does not measure that fine-tune at all.
- `--randomize-ball --randomize-bins` — every fixed-layout number in the logs predates
  these and is not comparable across them.

Collecting a fresh dataset and training:

```bash
uv run python libero/fine_tune/collect_finetune_data.py --out libero/fine_tune/a8 --episodes 60
modal run act/act_modal_train.py::main --max-steps 1 --save-freq 1   # SMOKE FIRST, always
modal run act/act_modal_train.py::main --max-steps 60000 --save-freq 10000
```

## Repo layout

```
act/              ACT trained from scratch. Currently the best-performing track.
libero/           MolmoAct2-LIBERO port. Owns the scene, the OSC controller, the closed-loop
                  client and the scorer that every other track reuses.
  fine_tune/      Demo collection + LeRobot v3.0 writer. Datasets a1..a7 live here (gitignored).
  tools/          verify_osc.py — the plant checks (sag, penetration, tracking).
smolvla_libero/   SmolVLA-450M serving + LoRA training.
droid/            Retired MolmoAct2-DROID track: phases 0-4, the original closed loop,
                  the IK expert collector and the v2.1 LeRobot writer.
deck/             index.html — the presentation, with its images.
docs/             Historical plans and postmortems. See docs/README.md for what is still true.
scripts/          Small operational helpers.
fine_tunes/       Pulled checkpoints (gitignored except each run's run_info.md).
assets/           Per-run debug artifacts, grouped by policy:
                  <model>/<fine_tune>/logs/<run_id>.jsonl and .../images/<run_id>/. Gitignored.
infra/            modal_images.py — every Modal image, defined once.
data/             Locally generated datasets. Gitignored.
molmoact2/        Vendored reference repo, gitignored, not this project's code.
mujoco_menagerie/ Vendored reference repo + our scene XMLs. Gitignored — see the warning below.
```

> **The scene XMLs live inside a gitignored vendored repo.** `scene_libero_osc.xml`,
> `scene_pick_place.xml` and `panda_libero_hand.xml` are ours but are untracked and have no
> history, which has already cost one silent revert of the arm gains
> ([`libero/PROGRESS.md` §22](libero/PROGRESS.md)). **Verify plant parameters by compiling
> the model and reading `actuator_gainprm`, never by reading the XML.**

## Package management

There are **three** dependency environments in play, and mixing them is the mistake to
avoid.

### 1. The local project env — `pyproject.toml` + `uv.lock`, python 3.12

The simulator, the closed-loop client, the dataset writers, the scorers. **This env never
runs a model**, so `torch`, `lerobot` and `transformers` are deliberately absent from it —
they would add multiple GB to a venv with no GPU to use them on.

```bash
uv sync                     # exact lockfile
uv sync --group dev         # + ruff
uv run ruff check .
```

Rule: a package goes here only if something under `libero/`, `act/`, `smolvla_libero/`,
`droid/` or `scripts/` imports it **locally**. Anything imported only inside a Modal
function belongs to env 2.

### 2. Modal images — `infra/modal_images.py`, built remotely

Every image the project deploys is defined **once** in that module:

| helper | python | used by |
|---|---|---|
| `lerobot_serve_image()` | 3.12 | `act/act_modal.py`, `smolvla_libero/smolvla_modal.py` |
| `lerobot_train_image()` | 3.12 | `act/act_modal_train.py`, `smolvla_libero/smolvla_modal_train.py` |
| `molmoact_serve_image()` | 3.11 | `libero/libero_modal.py` |
| `molmoact_experiments_image()` | 3.12 | `libero/libero_modal_train.py`, `libero/libero_modal_finetuned.py` |

`torch==2.5.1` + cu121 and `transformers 4.57.x` are pinned there and **nowhere else** —
they are what MolmoAct2's own `pyproject.toml` was validated against. Two python versions
is intentional: 3.11 for the MolmoAct2 serving path matching the vendored repo, 3.12 for
everything LeRobot.

Sharing the definitions is not just tidiness. Modal caches image layers by definition, so
chains meant to share the multi-GB torch pull silently stop sharing it the moment one
drifts. Before centralising, that pin was duplicated across five files.

Modal re-imports the app module **inside** the container, so any file importing from
`infra/` must ship it into the image. `with_infra(...)` does that and **must be the last
layer** — no build step may follow `add_local_python_source`.

`droid/` keeps its own inline image chains on purpose: the track is retired and its images
are frozen at what was last deployed.

### 3. The hf-libero env — a separate venv, outside this project

`libero/libero_benchmark_eval.py` needs `hf-libero`, which pins its own `robosuite` and
`mujoco`. **It must not be installed into the project env.** Run it explicitly:

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
`libero_closed_loop.py` reads the server's `/health` `checkpoint` field and derives them:
`HuggingFaceVLA/smolvla_libero` → `smolvla_libero/stock`,
`/checkpoints/act/act-green-ball/checkpoints/010000/…` → `act/act-green-ball_010000`. A
server that reports no checkpoint gives `unknown-model/unknown` — deliberately ugly, so an
unattributable run looks wrong in `ls`. Each log entry also carries `model` and `fine_tune`
so a file copied out of the tree still says what produced it.

Grouping by policy is what makes comparison possible: `score_runs.py` accepts a directory
and searches it recursively, so one policy is `score_runs.py assets/act/act-green-ball_010000`
and everything ever run is `score_runs.py assets/`.

Both files are written and flushed **during** the run — `tail -f` works, and frames land on
disk immediately after each `mj_step` rather than being buffered to the end. `--dry-run`
writes the log but never renders. `assets/` is gitignored; it is bulky regenerated debug
output, and it was committed by accident once already.

> Runs from before 2026-08-02 are in the old flat `assets/logs/` and `assets/images/`.
> They were left where they are — `score_runs.py` still reads them by path.

## Reading order

New to the project, in order:

1. **This file** — what exists and what the scores are.
2. [`PROGRESS.md`](PROGRESS.md) — how it got here, one section per track, and what each
   track proved or disproved. The cross-track story.
3. The track you care about: [`act/README.md`](act/README.md),
   [`libero/README.md`](libero/README.md), [`smolvla_libero/README.md`](smolvla_libero/README.md),
   [`droid/README.md`](droid/README.md).
4. That track's `PROGRESS.md` — the chronological attempt log, wrong turns deliberately
   kept in.
5. [`docs/README.md`](docs/README.md) — the historical plans and postmortems, each labelled
   current / historical / superseded.

**Precedence when documents disagree**, newest wins in this order:
`libero/PROGRESS.md` and `act/PROGRESS.md` (measurements, newest at the bottom) →
subproject `README.md` (the spec) → `docs/` (history, partly superseded).
Numbers in `docs/` predate the OSC port and the decimation fix; treat them as evidence of
what was believed at the time, not as current fact. **Re-measure anything load-bearing.**

## Conventions worth knowing before you touch anything

- **Smoke runs are 1 step.** `--max-steps 1 --save-freq 1` proves build → load → step →
  save. Any more burns GPU money for no extra signal.
- **`/health` before every evaluation.** A deployment that silently kept serving the old
  checkpoint has produced wrong conclusions here more than once.
- **Progress logs keep the wrong turns in.** Several conclusions in them were later
  reversed; knowing *which* were reversed is the useful part. Corrections are appended as
  new sections, never edited into old ones.
- **The last checkpoint is not the best one** — demonstrated twice, on two architectures
  ([`act/PROGRESS.md` §7.5](act/PROGRESS.md)). Score intermediate checkpoints.
- **A run log written by a live run is incomplete.** `score_runs.py` prints `INCOMPLETE`;
  believe it. A truncated log ends mid-carry and scores exactly like a release failure.
