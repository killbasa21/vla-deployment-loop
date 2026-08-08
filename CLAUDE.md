# CLAUDE.md

Guidance for Claude Code working in this repository. Current as of **2026-08-03**.

## What this repo is

A learning project about the full VLA deployment loop: a Franka Panda in MuJoCo locally, a
policy server on Modal, HTTP between them, and one task — *pick up the green box and put
it in the green container* — that has to actually succeed.

**The pick target changed on 2026-08-03 and the box is now the default everywhere.** 40 mm
sphere → 40 mm cube; instruction changed to match; scene identifiers renamed `green_ball*`
→ `green_box*` in all four scene XMLs; flags renamed `--randomize-ball` → `--randomize-box`
and `--ball-radius` → `--box-size` (still a half-extent). Rationale and the full change list
are in `libero/PROGRESS.md` §26. There is no ball mode and no flag to get one back — the
sphere is recoverable only from git history plus the scene XMLs, which are gitignored.

Everything that reads the scene picks this up with no argument: `libero_closed_loop.py`,
`libero/fine_tune/collect_finetune_data.py` (so **freshly collected demos are box demos by
default**), `score_runs.py`, all three policy servers, and the retired `droid/` scripts.

**Every score in this file and in `README.md`, and every dataset `a1`–`a7` plus their
`*_smolvla` conversions, was produced with the ball.** Do not compare a box run to them, and
do not rename their Modal volume paths — those still hold ball data.

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
- **Verify plant parameters by compiling the model**, never by reading the XML. The rule
  predates the fix and still holds: the scene files used to live inside the gitignored
  `mujoco_menagerie/` with no history, and the arm gains were silently reverted once while
  their explanatory comment stayed in place (`libero/PROGRESS.md` §22). **They now live in
  tracked `scenes/`** and `mujoco_menagerie/` is a pinned submodule — so a change is at
  least visible in a diff, but a diff still does not tell you what the model compiled to.
- **Replay the expert through the inference path before blaming a policy.** That single
  check is what eventually exposed the decimation bug, after nine days of blaming models.
- **The last checkpoint is not the best one.** Demonstrated twice, on two architectures.
  Score intermediate checkpoints.
- **`--delta-pos-scale` at serving must equal the dataset's collection value** (0.10 for
  `a7`, 0.20 for `a6`, 0.05 for `a5`), and `--payload-keys libero` is required for every
  non-DROID server. A mismatch does not measure that fine-tune at all.
- **`--delta-pos-scale` is 0.05 from 2026-08-03 on — LIBERO's own value, and the code
  default on both sides. Do not pass the flag.** The box-task collections and every
  fine-tune trained on them use it. The point is that 0.05 makes our action space
  *identical* to what the stock checkpoints were pretrained on rather than a rescale of it:
  same 7-D delta eef pose in [-1,1], same 8-D LIBERO-frame state, same Panda, same
  robosuite OSC_POSE port, same 20 Hz. The line above still governs `a5`–`a7`, which were
  collected at other values and must still be *served* at those values.
  Clipping is handled by `--motion-speed`, not by shrinking the scale: `--speed-scale` no
  longer stacks on top of the retiming (that double-slowdown was live at exactly 0.05).
  Not yet measured at 0.05 + `--motion-speed`: the label distribution. `a5` is the only
  0.05 collection and it predates `--motion-speed`, so its pinned `dx q01 = -1.000` is not
  evidence about this setting. Check `dx q01` against released LIBERO's −0.679 on the first
  box collection before trusting a score from it.
- **Progress logs keep the wrong turns in.** Append corrections as new sections; never edit
  a conclusion out of an old one. Knowing which claims were reversed is the point. This is
  why the ball→box change of 2026-08-03 rewrote the *code* everywhere but left every
  historical measurement, and its "ball" wording, exactly where it was.
- **The instruction string lives in `infra/task_spec.py`,** not in each track. It was copied
  into five files until 2026-08-03; a training/serving mismatch there does not error, it
  just conditions the policy on a prompt it never saw.
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

`molmoact2/` is a gitignored clone, not this project's code. `molmoact2/CLAUDE.md` documents
that repo's own wire protocol and layout if you need the server-side schema.

`mujoco_menagerie/` is a **pinned submodule** and must stay pristine upstream — never edit a
file inside it. Our scene XMLs live in tracked `scenes/`; see `scenes/README.md` for the
`meshdir` rule that makes that possible and for the two files that are deliberate forks of
upstream (`panda_robotiq.xml`, `scene_base.xml`).
