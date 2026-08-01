# `act-green-ball` — ACT trained from scratch on `a7`

Local copies of checkpoints from the Modal run `act-green-ball-train`, 2026-08-01. Everything
under `fine_tunes/` except this file is gitignored; this is the record of what the binaries
are.

## The run

| | |
|---|---|
| script | `act/act_modal_train.py::main` |
| command | `--max-steps 60000 --save-freq 10000 --overwrite` |
| dataset | `libero/fine_tune/a7` — 60 episodes, 20,034 frames, `--delta-pos-scale 0.10`, shuffled bins |
| policy | `--policy.type=act`, from scratch (ResNet18 ImageNet init only) |
| shape | `chunk_size 50`, `n_action_steps 10` (baked for serving), 51,574,663 learnable params |
| optimiser | lr 1e-5, lr_backbone 1e-5, batch 16, constant LR (ACT's preset has no scheduler) |
| hardware | Modal L4, `cpu=16.0`, 12 dataloader workers, ~0.219 s/step |
| remote | volume `molmoact2-checkpoints`, `/act/act-green-ball/checkpoints/` |

Normalisation is built from `a7`'s **own** `meta/stats.json` — correct here, unlike the
MolmoAct2/SmolVLA paths, because ACT has no pretrained statistics to preserve. See
`act/PROGRESS.md` §2.

## What is stored locally

```
fine_tunes/act_green_ball_a7/
  run_info.md            this file
  010000/
    pretrained_model/    197 MB — weights + processor configs. ENOUGH TO REDEPLOY.
    training_state/      394 MB — optimizer, RNG, step counter. Needed only to RESUME.
```

`010000/training_state/training_step.json` reads `step: 10000, batch_size: 16`, and
`model.safetensors` holds 234 tensors / 51,620,487 values (the ~46 k above the learnable
count are non-trainable buffers — BatchNorm running stats and fixed positional embeddings).

Pulled because the Modal volume was the only copy. Later checkpoints (`020000`, `030000`, …)
remain on the volume; pull them the same way if they are worth keeping:

```bash
uv run modal volume get molmoact2-checkpoints \
    /act/act-green-ball/checkpoints/030000 fine_tunes/act_green_ball_a7/
```

Note the trailing `/` on the destination — `modal volume get` errors with
`[Errno 21] Is a directory` if the destination directory already exists as the exact target
name.

## Redeploying ck10000

The server reads an absolute path on the volume, so the normal path needs no local copy:

```bash
ACT_CHECKPOINT=/checkpoints/act/act-green-ball/checkpoints/010000/pretrained_model \
    modal deploy act/act_modal.py
curl -s <url>/health        # poll until it reports the checkpoint you meant
```

To serve **this local copy** instead — e.g. if the volume is ever lost — push it back first:

```bash
uv run modal volume put molmoact2-checkpoints \
    fine_tunes/act_green_ball_a7/010000 /act/act-green-ball-restored/checkpoints/010000
```

## Resuming from 10 k

`training_state/` is here, so a branch from step 10 000 is possible — but `lerobot-train`
resumes from `checkpoints/last` only, so `last` has to be repointed at `010000` on the volume
first. There is no flag to select a checkpoint by number. See `act/act_modal_train.py`'s
`--resume`.

## What ck10000 scores

`act/PROGRESS.md` §7: **5/6 placements, 6/6 grasp-and-lift** on the first pass, against a
baseline of 0/3 placed, 1/3 lifted. §7.4 records the one failure mode — the gripper never
opens when the carry runs *inward* (`green_bin_x < ball_x`), on 3 of 9 rollouts.
