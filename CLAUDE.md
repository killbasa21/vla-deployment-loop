# CLAUDE.md

Guidance for Claude Code working in this repository. Current as of **2026-08-02**.

## What this repo is

A learning project about the full VLA deployment loop: a Franka Panda in MuJoCo locally, a
policy server on Modal, HTTP between them, and one task — *pick up the green ball and put
it in the green container* — that has to actually succeed.

Four policies have been driven through the same loop. **Read `README.md` for the current
scores and `PROGRESS.md` for how the project got here** before proposing any change to a
track; both are short and both are kept current.

| directory | track | state |
|---|---|---|
| `act/` | ACT trained from scratch on `a7` | **best results** — ck10000 scores 5/6 placed, 6/6 grasp-and-lift. Training paused at 30 k / 60 k, resumable. ck20000 unscored. |
| `libero/` | MolmoAct2-LIBERO. **Also owns the shared infrastructure** — the closed-loop client, the OSC controller, the scene and the scorer that every other track drives. | fine-tunes on `a5`/`a7` score 2/10 and 1/10, indistinguishable from the 0/3 stock baseline |
| `smolvla_libero/` | SmolVLA-450M LoRA | completes the task but slowly; the `a6` retrain was never run |
| `droid/` | MolmoAct2-DROID, phases 0-4 | **retired.** Every evaluation predates the decimation fix, so its ceiling was zero. Do not quote its numbers. |
| `infra/` | `modal_images.py` — every Modal image, defined once | — |
| `docs/` | historical plans and postmortems | see `docs/README.md`; each is labelled current / historical / superseded |

## Rules that have been learned the hard way

- **Smoke runs are 1 step.** `--max-steps 1 --save-freq 1` proves build → load → step →
  checkpoint-save. Anything more burns GPU money for no extra signal. Only a real training
  run gets a real step count.
- **`curl /health` before trusting any rollout.** A `modal deploy` that returns in six
  seconds has not necessarily cut over. Confirm the reported checkpoint is the one you
  meant to measure.
- **Verify plant parameters by compiling the model**, never by reading the XML. The scene
  files live in the gitignored `mujoco_menagerie/` and have no history; the arm gains were
  silently reverted once while their explanatory comment stayed in place
  (`libero/PROGRESS.md` §22).
- **Replay the expert through the inference path before blaming a policy.** That single
  check is what eventually exposed the decimation bug, after nine days of blaming models.
- **The last checkpoint is not the best one.** Demonstrated twice, on two architectures.
  Score intermediate checkpoints.
- **`--delta-pos-scale` at serving must equal the dataset's collection value** (0.10 for
  `a7`, 0.20 for `a6`, 0.05 for `a5`), and `--payload-keys libero` is required for every
  non-DROID server. A mismatch does not measure that fine-tune at all.
- **Progress logs keep the wrong turns in.** Append corrections as new sections; never edit
  a conclusion out of an old one. Knowing which claims were reversed is the point.
- **Run everything from the repo root.** Scene XML, `assets/` and `molmoact2/experiments`
  paths all resolve relative to the working directory, not to the script.

## Where documents disagree

Newest wins, in this order:

1. `libero/PROGRESS.md`, `act/PROGRESS.md` — the measurements, newest at the bottom
2. subproject `README.md` — the spec
3. `README.md`, `PROGRESS.md` — the cross-track summary
4. `docs/` — history, much of it superseded

Numbers written before 2026-07-27 predate the decimation fix; before 2026-07-28, the OSC
port. Both invalidated whole classes of measurement. Re-measure anything load-bearing
rather than quoting a figure.

## Run artifacts (`assets/`)

Every closed-loop run gets a `run_id` (default `<timestamp>_<pid>`, override `--run-id`)
and writes two things under `--assets-dir` (default `assets/`), grouped by the policy that
produced them:

```
assets/
  <model>/<fine_tune>/
    logs/<run_id>.jsonl                          # one JSON object per action chunk
    images/<run_id>/
      camera_<timestamp_ms>_image.png            # libero/act/smolvla camera names
      camera_<timestamp_ms>_wrist_image.png
```

- `--model` / `--fine-tune` set the two levels. Omitted, `libero_closed_loop.py` derives
  them from the server's `/health` `checkpoint` field (`derive_run_layout()`); an
  unreportable server yields `unknown-model/unknown`. `droid/phase3_closed_loop.py` takes
  the flags but does not probe — it only ever served one checkpoint family.
- Both files are written and flushed **as the run proceeds** so `tail -f` works and a run
  can be watched live. Frames hit disk right after each `mj_step`.
- Each log entry also carries `model` and `fine_tune`, so a log copied out of the tree is
  still self-describing.
- `score_runs.py` accepts directories and recurses, so one policy scores in one command.
- `--dry-run` writes the log but never renders frames or touches `data.ctrl`.
- **`assets/` is gitignored** — bulky, regenerated every run, and committed by accident
  once already (see the "deleted major assets" commit). To keep a run as a demonstration,
  copy it out and commit it under its own name (e.g. `deck/img/phase3_modal_run.gif`).
- Runs from before 2026-08-02 are still in the old flat `assets/logs/`, `assets/images/`.

## Packages

Three environments, described in full in `README.md` § "Package management":

1. **Local** (`pyproject.toml` + `uv.lock`, py3.12) — sim, client, writers, scorers. Never
   runs a model, so **torch/lerobot/transformers must stay out of it**.
2. **Modal images** — all defined in `infra/modal_images.py`. `torch==2.5.1` + cu121 is
   pinned there and nowhere else; do not relax it, and do not reintroduce a per-file image
   chain (that duplication is what the module exists to remove). `with_infra()` must be the
   **last** layer of any chain. `droid/` deliberately keeps its own frozen images.
3. **hf-libero** — a separate venv outside this project for `libero_benchmark_eval.py`.
   Never install it into the project env: it pins its own robosuite and mujoco. Note also
   that our `libero/` directory shadows the PyPI `libero` package, which is why that file's
   import is deferred inside a function.

## Vendored reference repos

`molmoact2/` and `mujoco_menagerie/` are gitignored clones, not this project's code.
`molmoact2/CLAUDE.md` documents that repo's own wire protocol and layout if you need the
server-side schema.
