# `droid/` — the MolmoAct2-DROID track (RETIRED)

> **Retired 2026-07-27. Do not use these numbers.** Every evaluation this track ever
> produced — stock checkpoint, `lora_train`, `ae_train` — ran through a client that called
> `mj_step` **once** per action, where the demos held each action for 33 steps. Each command
> got 2 ms of physics instead of 66 ms. Proven by oracle replay: the ground-truth expert
> setpoints, pushed through this client's own apply-and-step path, left the ball untouched
> from spawn. **The ceiling was zero regardless of policy.** See
> [`../libero/PROGRESS.md` §1](../libero/PROGRESS.md) and [`../PROGRESS.md` §2](../PROGRESS.md).

Kept because the code is the origin of everything that followed, because the phase 0-2
scripts are still the fastest way to sanity-check a fresh MuJoCo install, and because the
retired checkpoints are still on the Modal volume and someone may want to know what produced
them.

The live tracks are [`../act/`](../act/README.md) (best results),
[`../libero/`](../libero/README.md) (owns the scene, the client and the scorer) and
[`../smolvla_libero/`](../smolvla_libero/README.md).

## What is here

| file | what it does |
|---|---|
| `phase0_hello_panda.py` | Load the Panda, step it, print joint state. First thing to run on a fresh install. |
| `phase0_manual_control.py` | Drive `data.ctrl` by hand and watch the arm respond. |
| `phase1_render_check.py` | Render `external_cam` and `wrist_cam` and save the frames. The camera placement was verified by *looking* at these, not by trusting the maths — that habit is the reason phase 1 has no known bugs. |
| `phase1_playground.py` | Interactive `glfw` viewer on `scene_playground.xml`. |
| `phase3_closed_loop.py` | The original closed-loop client. `libero/libero_closed_loop.py` is a deliberate copy-paste-and-diverge of this, not an import. |
| `phase3_modal.py` | Serves MolmoAct2-DROID by wrapping `molmoact2/examples/droid/host_server_droid.py`. |
| `phase3_modal_finetuned.py` | Serves a raw OLMo/PyTorch distributed checkpoint via `serve_policy.py`. Added but never deployed or smoke-tested. |
| `phase4_collect_demos.py` | IK waypoint expert, recorded in the 8-D DROID convention (`[q1..q7, gripper]`, absolute joint positions). |
| `lerobot_writer.py` | Writes those demos as a LeRobot **v2.1** dataset. Superseded by `libero/fine_tune/lerobot_v30_writer.py`. |
| `phase4_modal_train.py` | Fine-tunes MolmoAct2-DROID on Modal. Entrypoints `::convert` (one-time v2.1→v3.0 conversion on the volume) and `::main`. |

## Running any of it

**From the repo root**, always — every path in these scripts (`mujoco_menagerie/...`,
`assets/`, `molmoact2/experiments`) resolves relative to the working directory, not to the
script.

```bash
uv run python droid/phase1_render_check.py
uv run python droid/phase3_closed_loop.py --dry-run          # round trip only, no physics
modal deploy droid/phase3_modal.py
```

## Why it was abandoned rather than fixed

The decimation bug is fixed in `libero/libero_closed_loop.py`, so this client *could* be
brought forward. It was not, for a reason that outlived the bug: MolmoAct2-**DROID** is
pretrained on real-robot footage of an **FR3**, and our scene is flat-shaded MuJoCo renders
of a **Panda** — a distribution gap on two axes at once. MolmoAct2-**LIBERO** is pretrained
on simulated Panda scenes, so `libero/` closes that gap by construction. See
[`../PROGRESS.md` §4](../PROGRESS.md).

## Its checkpoints

`ae_train` (25 GB) and `lora_train` (21 GB) are on the Modal volume
`molmoact2-checkpoints` under `/checkpoints/finetune/`, and are also copied locally under
`fine_tunes/pick_up_tasks/` (gitignored). That local copy is 46 GB of the repo's 46.6 GB of
checkpoints and is redundant with the volume — see each run's `run_info.md`, and
`modal volume get` to restore.

## Modal images

Unlike the live tracks, these two files keep their own inline `modal.Image` chains rather
than importing `infra/modal_images.py`. That is deliberate: the images are frozen at what
was last deployed, so the historical runs stay reproducible. If this track is ever revived,
move it onto the shared module first.
