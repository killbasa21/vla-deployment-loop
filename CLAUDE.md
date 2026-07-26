# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo scope

This is a learning project working through a phased build of a closed-loop MolmoAct2-DROID
control demo for a simulated Franka Panda (MuJoCo). Phase context/history lives in
`PLAN.md`, `PHASE2_SETUP.md`, and `PHASE3_PLAN.md`. The two scripts actively worked on:

- `phase3_closed_loop.py` — the client: renders both cams, reads proprioception, POSTs to
  the remote `/act` endpoint, applies the returned action chunk, and writes debug
  artifacts (see "Run artifacts" below) every step so a run can be watched live.
- `phase3_modal.py` — wraps `molmoact2/examples/droid/host_server_droid.py` in a Modal
  `@app.cls` so the `/act` server runs on a rented Modal A100 instead of a manually
  rented GPU box. `modal serve`/`modal deploy` this; point `phase3_closed_loop.py
  --server-url` at the printed URL.
- `phase3_modal_finetuned.py` — intended serving wrapper for the Phase 4 raw training
  checkpoint. Unlike `phase3_modal.py`, it uses `molmoact2/experiments/scripts/serve_policy.py`
  so it can load the OLMo/PyTorch distributed checkpoint format directly from the
  `molmoact2-checkpoints` Modal volume. This file has been added, but has not yet been
  deployed or smoke-tested.

Phase 4 adds classical data collection + fine-tuning (see `PHASE4_PLAN.md`):
- `phase4_collect_demos.py` — IK-scripted expert demos of the green-ball pick-and-place
  task, recorded in the 8-D DROID convention.
- `lerobot_writer.py` — writes those demos as a LeRobot dataset.
- `phase4_modal_train.py` — fine-tunes MolmoAct2-DROID on a rented Modal GPU (counterpart
  to `phase3_modal.py`). Two local entrypoints: `::convert` (one-time v2.1→v3.0 dataset
  conversion on the volume) and `::main` (training).

Current Phase 4 status, 2026-07-19:
- `ae_train` completed a 500-step action-expert-only fine-tune on Modal, 1x H100.
- The final checkpoint was pulled locally to
  `fine_tunes/pick_up_tasks/ae_train/run_20260719_ae500/checkpoints/step500`.
- The pulled checkpoint is valid and large: about 25 GB, with `config.yaml`,
  `train/rank0.pt`, `model_and_optim/.metadata`, and 16 `model_and_optim/*.distcp`
  shards.
- No training, Modal download, or phase 3 simulation process is currently running.
- The fine-tuned model is not yet proven in simulation. Next required step is deploying
  `phase3_modal_finetuned.py`, checking `/health`, then running `phase3_closed_loop.py`
  against the printed `/act` URL.

**Smoke runs must be lean: 1 step.** When validating the Modal training pipeline end to
end (build → load → step → checkpoint-save), use `--max-steps 1 --save-interval 1` — a
single optimizer step + one checkpoint save is enough to prove the plumbing, and every
step past that burns GPU money for no extra signal. Only bump the step count for a real
training run, never for a smoke.

`molmoact2/` and `mujoco_menagerie/` are vendored reference repos (gitignored, not part
of this project's own code) — `molmoact2/CLAUDE.md` documents that repo's own wire
protocol and layout if you need the server-side schema.

## Run artifacts (`assets/`)

Every `phase3_closed_loop.py` run gets its own `run_id` (default
`<timestamp>_<pid>`, override with `--run-id`), and writes two things under
`--assets-dir` (default `assets/`) so a run in progress can be inspected without
waiting for it to finish:

```
assets/
  logs/
    <run_id>.jsonl              # one JSON object per action chunk
  images/
    <run_id>/
      camera_<timestamp_ms>_external_cam.png
      camera_<timestamp_ms>_wrist_cam.png
      ...
```

- **`assets/logs/<run_id>.jsonl`** — the run's log file: one JSON-lines entry per chunk
  (env state before/after, the model's returned actions, and network timing). Opened in
  `"w"` mode and `flush()`ed after every entry, so `tail -f` works while the run is live.
- **`assets/images/<run_id>/`** — every rendered camera frame from that run, external and
  wrist cams interleaved in one directory. Filenames are
  `camera_{timestamp_ms}_{camera_name}.png`, where `camera_name` is `external_cam` or
  `wrist_cam` and `timestamp_ms` is a millisecond epoch timestamp shared by both cameras
  for a given sim step — pair a frame with its counterpart, or with a `logs` entry's own
  `"timestamp"` field, by matching timestamps. Frames are written to disk immediately
  after each `mj_step`, not buffered to the end of the run, for the same "watch it live"
  reason as the log.
- Both are skipped in `--dry-run` mode except the log file itself (dry-run still logs
  the request/response, it just never touches `data.ctrl` or renders/saves frames).

`assets/` is gitignored — these are bulky, regenerated-every-run debug artifacts, not
project source. (They were committed by accident once; see the "deleted major assets"
commit. Don't repeat that — if you want to keep a specific run's output as a
demonstration, e.g. `phase3_modal_run.gif`, copy it out of `assets/` and commit it under
its own name instead.)

## Common commands

```bash
uv run python phase3_closed_loop.py --dry-run                      # check the server round trip only
uv run python phase3_closed_loop.py --chunks 5                      # run 5 chunks against SERVER_URL
uv run python phase3_closed_loop.py --chunks 5 --server-url <url>   # against a Modal deployment
uv run python phase3_closed_loop.py --chunks 0 --no-view            # run until killed, headless

modal setup                                                          # one-time Modal auth
modal serve phase3_modal.py                                          # ephemeral dev server, live-reloads
modal deploy phase3_modal.py                                         # persistent deployment, prints URL
modal deploy phase3_modal_finetuned.py                               # serve ae_train step500 from Modal volume

uv run python phase3_closed_loop.py --model-path droid \
  --server-url <printed-finetuned-modal-url>/act \
  --request-timeout 600 --chunks 5
```
