# `docs/` — historical plans and postmortems

Nothing in this directory is a current specification. These are the documents the project
was built from and the postmortems written along the way, kept because the *reasoning* is
still useful and because several of them record mistakes worth not repeating.

**Where a document here disagrees with a track's `PROGRESS.md` or `README.md`, the track
wins.** Numbers written before 2026-07-27 predate the decimation fix, and numbers written
before 2026-07-28 predate the OSC port; both invalidated large classes of measurement.
Re-measure anything load-bearing rather than quoting a figure from here.

File paths inside these documents were rewritten during the 2026-08-02 reorganisation
(`phase*.py` moved to `droid/`, the old top-level README became `SERVO_DROOP.md`). The
paths point at where the files live now, not at where they lived when the document was
written.

| file | what it is | status |
|---|---|---|
| [`PLAN.md`](PLAN.md) | The original project plan: architecture, the local/remote split, the phased build. | **Historical.** The architecture it describes is still exactly what runs. The task (red box) and the checkpoint (MolmoAct2-DROID) both changed. |
| [`PHASE2_SETUP.md`](PHASE2_SETUP.md) | Standing the MolmoAct2-DROID server up on a rented vast.ai RTX 5090, as actually done. | **Superseded** by Modal (`infra/modal_images.py`). Kept for the Blackwell/`sm_120` torch-compatibility procedure, which is still the right way to validate a rented box. |
| [`PHASE3_PLAN.md`](PHASE3_PLAN.md) | Plan for the closed-loop client. | **Historical.** Implemented as `droid/phase3_closed_loop.py`. Its "read their client code rather than guessing the units" instruction is the durable part. |
| [`PHASE4_PLAN.md`](PHASE4_PLAN.md) | IK expert demos → LeRobot dataset → Modal fine-tune. | **Historical.** The pipeline shape survives in `libero/fine_tune/`; the DROID-specific 8-D convention and the v2.1 writer do not. |
| [`PHASE5_PLAN.md`](PHASE5_PLAN.md) | Diagnosis of why the green-ball fine-tunes were failing, with a tiered fix list. | **Partly wrong, and usefully so.** `libero/PROGRESS.md` ends with an explicit "Corrections to `docs/PHASE5_PLAN.md`" section — read that alongside it. Its Tier B pad-friction number is robosuite's, not DROID's. |
| [`SERVO_DROOP.md`](SERVO_DROOP.md) | The 580-line postmortem on position-actuator droop poisoning the training labels. Was the top-level README until 2026-08-02. | **Superseded in its premise, one day after being written.** The bug only exists under position actuators; the OSC port removed it (sag 0.000 mm). Kept for its method — §5 documents how every number was produced — and for the general lesson that any label defined as `target − current` inherits the plant's tracking error. |
| [`FINE_TUNE_LEARNINGS.md`](FINE_TUNE_LEARNINGS.md) | Running record of the MolmoAct2-LIBERO fine-tune, started 2026-07-28. | **Historical.** §5.5's cost constraint is still quoted by the SmolVLA and ACT training scripts. Its conclusions about what would fix the task are superseded by `libero/PROGRESS.md` §25 — the bottleneck turned out to be the gripper, not the action space. |
| [`NEXT_STEPS_FOR_FINE_TUNE.md`](NEXT_STEPS_FOR_FINE_TUNE.md) | A paste-into-a-fresh-agent handoff prompt, written 2026-07-28. | **Dead.** It instructs the reader that `SERVO_DROOP.md` is "the current state of the world", which stopped being true the next day. Kept only as a record of what was believed at that moment. Do not follow it. |

## What replaced these

| for | read |
|---|---|
| What exists and what the scores are | [`../README.md`](../README.md) |
| How the project got here, across tracks | [`../PROGRESS.md`](../PROGRESS.md) |
| Scene, control law, conventions, wire format | [`../libero/README.md`](../libero/README.md) |
| Dataset format and collection | [`../libero/fine_tune/README.md`](../libero/fine_tune/README.md) |
| The measurements themselves | [`../libero/PROGRESS.md`](../libero/PROGRESS.md), [`../act/PROGRESS.md`](../act/PROGRESS.md) |
