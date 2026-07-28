# FINE_TUNE_LEARNINGS

Live log of the MolmoAct2-LIBERO fine-tune for the green-ball pick-and-place task.
Started 2026-07-28, from the handoff in `NEXT_STEPS_FOR_FINE_TUNE.md`.

Companion docs: `README.md` (servo droop, the dataset scoreboard, the measured stock-
checkpoint baseline), `libero/README.md` (conventions/scene/control law),
`libero/fine_tune/README.md` (dataset format), `libero/PROGRESS.md` (attempt log, partly
superseded).

**This file is the running record of what was done and why, including things that did not
work.** Where it disagrees with the older docs, this file and `README.md` win.

---

## 0. The two problems this fine-tune has to survive

1. **The task is not learned at all yet.** README §9: the stock checkpoint on the corrected
   scene scores **0/3 placements, 1/3 grasp-and-lift**. That is the baseline to beat.
2. **The dataset is tiny and single-task**, so the failure mode is not undertraining, it is
   **overfitting** — and README §6.1 says exactly where it will show first: the rotation
   channels, whose q01–q99 spans are 3–6× narrower than the released dataset's, collapsing
   toward zero.

Everything below is organised around not losing to (2) while fixing (1).

---

## 1. Anti-overfitting measures actually taken

Not aspirations — each of these is a decision with a reason, and a way to check it.

| measure | why | how it is checked |
|---|---|---|
| **3× more data** (`a4`, ~90 episodes vs `a3`'s 30) | demo collection is local CPU and free; the single largest lever against memorising 30 ball positions is more ball positions | episode count + per-channel quantiles in §3 |
| **LoRA, rank 32** (not the default 64) | fewer trainable parameters is a hard capacity bound on how much of 5k frames the adapter can memorise | trainable-param count printed at step 0 |
| **Short run, 1–2 epochs** | PROGRESS §4: 500 action-expert-only steps drove flow loss 0.176 → 0.01 with task success still at 0 — loss going to zero on this data means memorisation, not skill | steps × global_batch ÷ frames |
| **Frequent checkpoints, checkpoint SELECTION** | the honest defence: don't pick the last checkpoint, pick the best-scoring one in closed loop | multi-run eval per checkpoint, §6 |
| **`--img_aug full` kept on** (the trainer's default) | spatial + photometric augmentation; turning it off is the classic accidental overfit | flag in the launch command |
| **Pinned normalisation from the released dataset** | training on our data alone rebuilds the `libero` normaliser from a distribution 2.5× narrower than pretraining, so the adapter would spend capacity relearning an affine map that already exists | §4 |
| **Rotation-channel watch** | the predicted failure mode; a served policy whose `drx/dry/drz` std is ~0 has collapsed | compare action stats of served rollouts against `a4` labels |

Deliberately NOT done, and why:

- **A held-out validation split.** `train_lerobot.py` builds `evaluations = []` /
  `loss_evaluations = []` for LeRobot mixtures — there is no eval hook to feed a val split
  into, so a holdout would cost training signal and return nothing automatic. The
  closed-loop run on freshly sampled ball/bin layouts is a strictly better generalisation
  test and we run it anyway.
- **Mixing the released 273k-frame LIBERO dataset into the sampling mixture** (as replay).
  It is 35 GB. See §4 — we take its *statistics* for free instead, and spend the GPU budget
  on our own task. Revisit if the fine-tune visibly forgets.

---

## 2. What was wrong with `a3`, restated

From README §6.5, measured, not assumed:

```
                        dx       dy       dz
ACTION q01
  released           -0.679   -0.774   -0.873
  a3                 -0.072   -0.307   -0.258
q01-q99 SPAN, RATIO TO RELEASED
  a3                  0.376    0.372    0.439
```

`dx` is one-sided: over 5220 frames there is essentially no label that says *retreat in
−x*, which is the correction the closed-loop failures need. Noise sigma cannot fix it
(README §4.3: past σ≈0.09 episodes stop succeeding at all). So it has to come from the
expert trajectory.

---

## 3. `a4` — the dataset change

### 3.1 Retreat segments (the actual fix)

`libero/fine_tune/collect_finetune_data.py:waypoints()` gained two segments:

- **(a) back-off and re-approach**, inserted between the hover and the descent: the arm
  arrives above the ball, withdraws `RETREAT_BACKOFF = 0.12 m` toward the base (and 0.03 m
  up) over 0.45 s, then comes back in. This demonstrates the recovery manoeuvre *in the
  visual context where the policy needs it* — ball centred, gripper open.
- **(b) return to the start pose** after the release, replacing `a3`'s "rise 20 cm and
  stop". Bins are at x = 0.56 or 0.80 and the reset pose at x ≈ 0.45, so this is 0.11–0.35 m
  of −x travel, and it is what a real demonstration ends with anyway.

Sizing note, and it matters: **the retreat must not be slowed down to look gentle.**
README §3.2 shows the label ratio is `dt/tau` and is *independent of reference speed* — a
slower retreat produces a smaller label, not a cleaner one. 0.12 m in 0.45 s is ~0.27 m/s
mean, ~0.4 m/s at the smoothstep peak, ≈ 0.4 action units before servo lag, against a
standing droop bias of ≈ +0.10.

Trajectory is now 10.8 s / ~216 ticks (was 7.6 s / 174).

### 3.2 A consequence that had to be fixed with it

`RECOVER_KICK_WINDOWS` are *fractions of the episode*. The longer trajectory moves the
loaded phase to fraction 0.35–0.85 (gripper closes at 3.8 s, opens at 9.2 s), so `a3`'s
`(0.08, 0.40)` "free-arm" window would have fired onto a closed gripper. Re-derived to
`(0.06, 0.33)` and `(0.45, 0.78)`.

### 3.3 Results

A 6-episode pilot (seed 7) settles it — this is the fix, and it is a large one:

```
                    dx q01    dx q99    dx mean   span ratio dx   frames with dx < -0.2
released           -0.679     0.854      0.063        1.000              --
a3                 -0.072     0.504      0.098        0.376             0.02%
a4 pilot           -0.494     0.512      0.044        0.656             5.94%
```

`dy` improved as a side effect (q01 −0.307 → −0.483): the return-to-start segment crosses
in y whenever the green bin is at y = ±0.25. The gripper channel's mean moved 0.253 →
0.009 against released −0.050, because the added segments are open-gripper time. Nothing
clips: 0.00% at ±1 on all six pose channels (`a3` clipped `dz` on 0.06%).

Still not fixed, and deliberately: **`dz` remains one-sided** (q01 −0.259 vs released
−0.873) because our descent is slow and single-speed, and **rotation is still 3–7× narrow**
(span ratios drx 0.25, dry 0.18, drz 0.14). Both are honest properties of a single top-down
task, not bugs — see README §6.1. The rotation number is the one to watch after training.

Episode length 216 ticks (was 174), so a4 at 90 episodes is ~19.4k frames vs `a3`'s 5.2k.

---

## 4. Normalisation: pinned, without downloading 35 GB

Read out of the training code, not assumed:

- `train_lerobot.py` defaults `--norm_mode q01_q99`.
- `lerobot_utils/stats.py:_collect_tagged_stats` builds a tag's normaliser by merging the
  `LeRobotDatasetMetadata` stats of **every repo in that tag**, and
  `_merge_feature_stats` merges quantiles **count-weighted**. So a 273k-frame repo next to
  our ~19k-frame one would supply ~93% of the quantiles.
- Sampling is a **separate knob** from stats: `olmo/data/data_loader.py:_build_mixture`
  gives each dataset in a group a relative rate of `sqrt(len(dataset)) * sampling_rate`,
  and `DatasetWithArgs.sampling_rate` is per-repo. So a repo *can* anchor the normaliser
  while sampling rarely — the handoff's question is answered: **stats and sampling are
  genuinely independent.**
- **But** `_build_mixture` calls `get_dataset_by_name` on every repo in the mixture
  regardless of rate, so a repo at rate 0 is still fully materialised — 35 GB of inline-PNG
  parquet downloaded onto the Modal volume for statistics that are 400 kB of JSON.

**Decision:** take the released `meta/stats.json` and write it into `a4`'s metadata for the
`action` and `observation.state` keys only (the only two keys `_collect_tagged_stats`
reads), keeping the measured file beside it as `meta/stats_measured.json`. Same
normalisation as pretraining, no download, no sampling contention, and the divergence is
visible on disk rather than buried in a wrapper script.

Implemented as `libero/fine_tune/pin_released_stats.py` (with `--restore`). Run it on `a4`
before uploading. `libero_modal_train.py` prints the dataset's action q01/q99 and the stats
`count` at startup, so a run that forgot this step is visible in the first ten lines of the
log (`count` 273465 = pinned, ~19000 = not).

---

## 5. Training run

### 5.1 The pieces that had to be built

| file | what it is |
|---|---|
| `molmoact2/experiments/launch_scripts/data_mixtures.py` | new `libero_green_ball` mixture, tag **`libero`** |
| `libero/fine_tune/pin_released_stats.py` | pins released normalisation into a dataset |
| `libero/libero_modal_train.py` | Modal training wrapper (counterpart to `phase4_modal_train.py`) |
| `libero/libero_modal_finetuned.py` | Modal serving wrapper for the result |
| `libero/libero_closed_loop.py` | new `--payload-keys` (see §5.3) |

⚠️ **`molmoact2/` is a vendored, gitignored repo — the mixture entry is not under version
control.** It ships to Modal because the image does `add_local_dir("molmoact2/experiments")`,
but a fresh clone of that submodule silently loses it and the run dies with "Unknown
LeRobot mixture 'libero_green_ball'".

### 5.2 Configuration, and the reasoning behind each number

- **Mode `lora`** — LoRA on the VLM path *plus* a fully trained action expert. Not
  `ae_only`: PROGRESS §4 shows that run drove flow loss 0.176 → 0.01 with success at zero,
  because a frozen VLM cannot learn to *look* for a differently-placed ball, and README §9's
  failure (stalls 33–45 mm lateral, never closes the gripper) is a grounding failure.
- **`--lora_rank 32`**, half the trainer's default 64 — a hard cap on memorisation capacity.
- **`--norm_mode q01_q99`**, the trainer's default (phase 4 had to use `min_max` only
  because the repo-root v2.1 writer computes no percentiles; `lerobot_v30_writer.py` does).
- **`--img_aug full`** passed explicitly rather than inherited, so nobody flips it silently.
- **`--save_num_checkpoints_to_keep=-1`** — every checkpoint is kept, because selection
  among them is the main overfitting defence.
- **`OLMO_NUM_THREADS=4`** — phase 4's LoRA smoke run hung partway through its first
  checkpoint save (7 of 16 shards, then nothing); olmo's writer defaults to 16 concurrent
  shard writers and Modal Volumes are FUSE-backed.

### 5.3 A silent-failure trap found while wiring up serving

`serve_policy.py` turns the payload key it matched into the model feature name
`observation.images.<key>`, and looks those keys up in
`IMAGE_KEY_PRESETS["libero"] = ["image", "wrist_image"]`. Our client posts
`external_cam`/`wrist_cam`, which `host_server_droid.py` reads by name and forwards
positionally — fine there, **nothing at all** against `serve_policy`. Hence
`libero_closed_loop.py --payload-keys {droid,libero}`, defaulting to `droid` so the
existing released-checkpoint path is untouched. Same family as the `NORM_TAG` trap: the
wrong setting does not throw where you are looking.

### 5.4 GPU choice

**H100, single GPU.** With a *fixed* 1–2 h window, "cheapest per hour" and "most training"
are different objectives — a slower card fits fewer optimizer steps into the same window,
and Modal bills warm idle containers at the GPU rate, so a fast card that exits sooner can
cost less in total. Serving is a different question and stays on an L4 (24 GB), which
`libero/README.md` measured at no latency penalty against the A100 it started on.

Step count is budgeted backwards from the 1-step smoke run's measured seconds/step, not
guessed.

### 5.5 The $5 budget changes which risk is binding

The budget was set to **$5 total** after the smoke run, and that inverts the framing this
document opened with. Measured on the smoke run (H100, `$3.95/hr`):

```
trainable parameters                657,493,152   (LoRA r32 on the VLM + full action expert)
total / VLM / action expert         5.57B / 4.95B / 0.62B
peak GPU memory                     25 GB of 80    -> device_batch_size can go 1 -> 4
step 1 incl. dataloader warmup      63 s   at device_batch 1, global 16
checkpoint save                     200 s  (42 s base + 9 s LoRA + 150 s merged)
whole function                      333 s  ~= $0.47
```

$5 buys roughly one H100-hour, of which ~200 s per checkpoint save is not training. The
chosen run is **150 steps x global batch 8 = 1200 samples = 0.06 epochs** of a4.

**At 0.06 epochs the model cannot overfit — the risk has flipped to undertraining.** The
anti-overfitting measures in §1 all stay, because every one of them is free (more data was
CPU time; LoRA rank 32 and `img_aug=full` cost nothing; checkpoint selection is two saves).
But the honest reading is that §1's framing applied to the 1–2 hour budget in the handoff,
and under a $5 cap the thing to watch is a checkpoint that has barely moved off the base.

One deliberate consequence: **action-expert LR raised to 1e-4** (the trainer's own default)
from phase 4's 5e-5. With ~1200 samples a conservative LR mostly buys a checkpoint
indistinguishable from the base one. The VLM LoRA path stays at 5e-5 — that is where an
oversized step damages the general grounding we are relying on.

### 5.6 Runs

| run | config | outcome |
|---|---|---|
| `libero-smoke` | 1 step, save 1, db 1, gb 16, on `a3` | **PASS.** Wrote `step1`, `step1-lora-llm`, `step1-lora-vision`, `step1-merged`. Proved image build, mixture registration, dataset load, LoRA wrapping, optimizer step, and both checkpoint writers. |
| `libero-green-ball` | 150 steps, save 75, db 4, gb 8, on `a4` | *(running)* |

Smoke-run findings worth keeping:

- The merged directory is `stepN-merged` (the log line says "step1-lora-merged", which is a
  message, not the path). That is the one the serving wrapper must load — see
  `libero_modal_finetuned.py`, and `phase3_modal_finetuned.py` for why a raw `stepN` LoRA
  checkpoint cannot be served.
- `save_merged_lora_checkpoint` costs **150 s per save**, more than the base checkpoint
  write. At $3.95/hr each extra checkpoint is ~$0.22, which is 4% of this budget — that is
  the real price of "keep every checkpoint and select".
- The 10 GB base checkpoint did **not** re-download: `molmoact2-hf-cache` is shared with the
  serving deployments, so model load took 39 s.
- `sampling_rate/lerobot_tag/libero=100.00` in the log confirms the mixture resolved to our
  repo alone.

---

## 6. Evaluation

*(pending)*

Ground rules, from `libero/README.md` and README §9:

- The action expert is **flow-matching — it samples.** One rollout is one draw. Score
  success rate over several runs; two runs from an identical start have diverged
  completely before.
- Baseline to beat: **0/3 placements, 1/3 grasp-and-lift** on the corrected scene and
  stiffened plant.
- `libero/libero_benchmark_eval.py` (a real LIBERO task through robosuite's own OSC, 3/3 on
  the stock checkpoint) is the known-good control: if it regresses after the fine-tune,
  the adapter has damaged general competence.
- **The `norm_tag` trap.** `host_server_droid.py` hardcodes `NORM_TAG = "franka_droid"` at
  module level. Serving a LIBERO checkpoint with it yields garbage actions *of the correct
  shape* — silent failure. Any serving wrapper must retag after import, and `/health` must
  be confirmed to report `libero` before a rollout is believed.
