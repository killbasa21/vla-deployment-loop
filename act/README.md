# `act/` — ACT (Action Chunking Transformer) on the green-ball pick-and-place task

The control arm for the SmolVLA experiment. Same scene, same demos, same closed loop, same
scorer — a policy with **no pretraining and no language**, trained from scratch on our own
episodes.

Status, 2026-08-01: **plan and scripts written, nothing trained.** `a7` is collected and
repaired — 60 episodes, 20,034 frames, 334 ticks/ep (see `PROGRESS.md` §5 for the OOM that
cost it `stats.json` and `cohorts.json`).

## 1. Why ACT is worth the GPU hour

`libero/PROGRESS.md` §23.5 left the SmolVLA path with a diagnosis and two competing
explanations for why `a6` closes the gripper on air:

- **units** — one action unit is 0.20 m, so 0.9 normalised units of policy error is a 36 mm
  miss (the fix being `a7` at `--delta-pos-scale 0.10`);
- **grounding** — 30 episodes is too little for the policy to localise a randomised ball,
  and the LoRA targets the action expert, not the VLM.

ACT separates these cheaply. It is small (order 50-80 M — the ACT paper says 80 M; lerobot's
ResNet18 build is smaller, and the first training log prints the real number) and trained
end to end on our data with an
ImageNet ResNet18 vision backbone that **is** in the gradient path. If ACT grasps at the same
`--delta-pos-scale 0.10` where a SmolVLA fine-tune misses, the SmolVLA problem is adaptation
(frozen/narrow LoRA), not the scene, the labels, or the units. If ACT also misses laterally
by tens of mm, the problem is the data — 60 single-task episodes do not contain enough
ball-position variation to ground anything, and no policy choice fixes that.

Secondary reason: it is the honest accuracy floor. A 450 M VLA that cannot beat a ~50 M
from-scratch transformer on one table task is not earning its serving cost.

| | SmolVLA-LIBERO | ACT |
|---|---|---|
| params | 450 M | ~50-80 M |
| pretraining | LIBERO, 4 suites | **none** (ResNet18 ImageNet only) |
| language | yes, SmolVLM2 | **none** — task identity is implicit |
| head | flow-matching, **samples** | L1 + VAE KL, **deterministic** |
| training GPU | L4 | L4 |
| serving GPU | T4 | T4, comfortably |

The deterministic head matters for measurement. `PROGRESS.md` §18 has two SmolVLA rollouts
from an identical start diverging completely, which is why the scoring protocol is "run
several, then `score_runs.py`". ACT's decoder has no sampling at inference, so run-to-run
variance collapses to whatever the scene randomisation injects. Fewer rollouts per number.

## 2. What does NOT need building

Deliberately small surface. Three of the four pieces already exist.

| piece | reuse | why |
|---|---|---|
| demos | `libero/fine_tune/a7` **as-is** | no conversion step |
| sim client | `libero/libero_closed_loop.py --payload-keys libero` | unchanged wire format |
| scorer | `libero/score_runs.py` | same log schema, same client writes it |
| training | `act/act_modal_train.py` | new, but a near-copy of `smolvla_modal_train.py` |
| serving | `act/act_modal.py` | new, `/act`-compatible |

### 2.1 No `convert_dataset.py`

`smolvla_libero/convert_dataset.py` exists for one reason: `smolvla_libero`'s `config.json`
hardcodes `observation.images.image2` for the wrist camera, and LeRobot looks feature keys up
by name inside the normalisation layer, so a dataset carrying `wrist_image` trains with **no
second camera and no error** (that README's "the rename is not cosmetic").

ACT has no such contract. It is constructed from the dataset's own metadata and accepts *any*
key beginning with `observation.image`. So `act_modal_train.py` uploads
`libero/fine_tune/a7` verbatim and the wrist camera keeps its native name
`observation.images.wrist_image`. One fewer artefact, one fewer place for a silent rename bug.

That same name then has to appear on the wire side, and `act_modal.py`'s `WIRE_TO_FEATURE`
is therefore the identity map — but it is still *checked* against `cfg.input_features` at
container start, the same guard `smolvla_modal.py` has.

### 2.2 The normalisation decision inverts

`smolvla_modal_train.py`'s `--norm-stats checkpoint` default exists because that action
expert was pretrained under LIBERO's statistics and rebuilding the affine map from 30
episodes makes it relearn a transform it already had.

**ACT has no pretrained statistics.** There is no checkpoint to inherit from; `lerobot-train`
builds the STATE/ACTION/VISUAL normalisers from `a7/meta/stats.json` and that is correct.
Note `a7`'s `stats.json` must be its *own* measured stats — `libero/fine_tune/pin_released_stats.py`
overwrites `stats.json` with MolmoAct2's released LIBERO numbers for the MolmoAct2 path, and
`stats_measured.json` is the untouched backup (`smolvla_libero/README.md` §"the stats swap").
`act_modal_train.py` asserts `count` matches the dataset's own frame count before training,
so a pinned file fails loudly instead of quietly normalising into the wrong space.

## 2.5 The architecture, on our inputs

A CVAE whose decoder is a DETR-style encoder-decoder transformer. Sizes below are this
project's: two 256×256 cameras, `observation.state` 8-D, `action` 7-D, `chunk_size` 50,
`dim_model` 512, ResNet18.

```
                     ┌─── TRAINING ONLY ───────────────────────────────┐
  action chunk (50,7)│  vae_encoder_action_input_proj  -> (50, 512)    │
  state (8)          │  vae_encoder_robot_state_input_proj -> (1, 512) │
  [CLS]              │  vae_encoder_cls_embed          -> (1, 512)     │
                     │      concat (52, 512) + fixed sinusoidal pos    │
                     │      4x transformer encoder layers              │
                     │      CLS out -> latent_output_proj -> mu, logvar│
                     │      reparameterise -> z (32)                   │
                     └─────────────────────────────────────────────────┘
                                        |  at INFERENCE: z = zeros(32)
                                        v
  external_cam (3,256,256) -ResNet18-> (512,8,8) -1x1 conv-> 64 tokens
  wrist_cam    (3,256,256) -ResNet18-> (512,8,8) -1x1 conv-> 64 tokens   (same backbone)
  state (8)   -> encoder_robot_state_input_proj  -> 1 token
  z (32)      -> encoder_latent_input_proj       -> 1 token
                                        |
                            130 tokens x 512, + 2D sinusoidal pos on the
                            image tokens, learned 1D pos on the other two
                                        v
                            4x transformer ENCODER layers (8 heads, ff 3200)
                                        v
                            1x transformer DECODER layer
                            queries: 50 zeros + learned decoder_pos_embed
                                        v
                            action_head: Linear(512 -> 7)
                                        v
                                 actions (50, 7)
```

Four things about this shape are load-bearing here:

**`n_obs_steps = 1`.** ACT sees ONE frame. No history, no memory across calls — which is why
`act_modal.py` calls `policy.reset()` per request and why the wire protocol never had to
promise contiguous observations.

**The VAE encoder is training-only.** It reads the *ground-truth future actions*, so it
cannot exist at inference; `z` is set to zeros there. That is what makes the policy
deterministic at serve time, unlike SmolVLA's flow-matching sampler (`libero/PROGRESS.md`
§18 has two SmolVLA rollouts from an identical start diverging completely). Its job during
training is to absorb the multimodality of human/expert demos into `z` so the decoder is not
forced to average conflicting demonstrations — and `kl_weight = 10.0` is the knob that stops
it absorbing *everything* and turning the decoder into a latent-conditioned playback.

**Two cameras, one shared backbone, 64 tokens each.** ResNet18 at stride 32 turns 256×256
into 8×8. The wrist and external views are distinguished only by their 2D positional
embeddings and by what they contain — there is no per-camera embedding. Also note the ball is
~4.9 px at 256 (`libero/PROGRESS.md` "Corrections"), so after a stride-32 backbone it
occupies a fraction of ONE feature cell. That is the resolution the grounding question in §1
is actually being asked at.

**The state token is 1 of 130.** Numerically tiny, which is *not* reassurance — it is fully
connected to every decoder query through 4 encoder layers, and it is the only input that
tells the policy where in the episode it is. See §3.2.

Parameter count, by arithmetic (confirm against the first training log): ResNet18 11.2 M +
encoder 17.3 M + decoder 5.4 M + VAE encoder 17.3 M + projections ≈ **51 M for training,
~34 M served** once the VAE encoder is dropped.

## 3. The two real design choices

### 3.1 Chunk size, and why serving stays at 10

ACT defaults `chunk_size = n_action_steps = 100`: predict 100 actions, execute all 100.
`a7` episodes are 334 ticks, so a 100-chunk is 30% of the whole task predicted open-loop from
a single frame.

- **train** `chunk_size = 50`. Long enough to cover a phase (approach, descend, close),
  short enough that the decoder is not asked to memorise most of an episode from frame 0.
  Same number `smolvla_libero` trains at, which keeps the comparison clean.
- **serve** `n_action_steps = 10`. Matches LIBERO's action horizon, matches what
  `libero_closed_loop.py` expects, and matches every SmolVLA rollout already logged. The
  server slices the leading 10 of the 50-chunk exactly as `smolvla_modal.py` does.

**Temporal ensembling is off** (`temporal_ensemble_coeff = None`, the default). It requires
re-querying the policy every single tick (`n_action_steps = 1`) so the exponentially-weighted
average has overlapping chunks to average. Over HTTP that is 20 round trips per simulated
second against a measured transport cost of ~4x inference (`PROGRESS.md` §2). Revisit only
if the closed loop turns out to be jerky at chunk boundaries.

### 3.2 The state-shortcut risk, and the ablation that measures it

ACT conditions on `observation.state` (8-D) alongside the images. On a single task with a
stereotyped trajectory, the fastest way to drive the L1 loss down is to **ignore the cameras
and regress the action from proprioception plus phase** — which fits the training set
perfectly and produces exactly one trajectory at test time, i.e. the average over the ball
randomisation box. That is the same failure `PROGRESS.md` §4 describes for a frozen VLM,
arrived at from the opposite direction.

It is also directly measurable, and cheap: the miss distance under a *randomised* ball versus
a *nominal* ball is already the diagnostic `PROGRESS.md` §23.5 used. So run 2 as an
ablation is `--no-state`, dropping `observation.state` from the inputs and forcing the policy
through the cameras. ACT supports this — its only hard requirement is at least one
`observation.image*` key.

Do not run the ablation first. Run 1 is the standard recipe; the ablation is only worth GPU
time if run 1 shows the average-trajectory signature (tight on nominal, wide on randomised).

## 4. Phases

**P0 — `a7`. DONE.** 60 episodes, 20,034 frames, fps 20, 334 ticks/ep, `stats.json count ==
total_frames`. Its resolution is 23.5 mm per normalised action unit, between `a5`'s 15.5
(grasps) and `a6`'s 40.0 (misses) — the middle setting §23.5 asked for. `dx` clips on 1.35%
of frames. Details and the collection OOM in `PROGRESS.md` §5.

**P1 — upload + smoke.** `::upload` pushes `a7` to the `molmoact2-lerobot-data` volume under
its own path (`/data/greenbox/green_ball_a7_act` — never overwrite another run's repo, that
removes the ability to A/B). Then `--max-steps 1 --save-freq 1`, per CLAUDE.md's smoke rule:
one optimizer step and one checkpoint save proves build → load → step → save, and every
further step is money for no signal.

**P2 — run 1, the standard recipe.** 60 k steps, batch 16, lr 1e-5, `chunk_size 50`,
`save_freq 10000`. On `a7`'s 20,034 frames that is **48 epochs** — in the range ACT is
normally trained at, and far past what the SmolVLA LoRA runs saw. Cost estimate below.

**P3 — score every checkpoint, not just the last.** `PROGRESS.md` §23.5 measured ck3000 as
three times more laterally accurate than ck5000 on the same run; the final checkpoint is not
the best one. Deploy each, poll `/health` until it reports the checkpoint you meant — a
`modal deploy` returning in 6 s is *not* a cutover while the old container is inside its 300 s
`scaledown_window`, and §23.5 threw away a contaminated run to that exact trap.

**P4 — the comparison table.** Same axes §23.5 used, so the numbers slot in beside SmolVLA's:
chunks-to-completion, first-close chunk, lateral error at close, vertical error at close,
ball displacement. Both policies at `--delta-pos-scale 0.10`, `--randomize-bins`, and both
nominal and randomised ball.

**P5 — conditional.** Only if P4 says so: `--no-state` ablation (§3.2), or more data.

## 4.5 What a run actually looks like

No env in the loop, so there is nothing to watch but the loss. One `lerobot-train` process,
one line every `log_freq` (200) steps:

```
INFO step:2K smpl:32K ep:1.60 epch:1.60 loss:0.284 grdn:1.842 lr:1.0e-5 updt_s:0.184 data_s:0.001
```

- **`loss`** is L1 on the 50-action chunk **plus `kl_weight`(10.0) × the VAE KL**. It is not
  a success rate and not comparable to SmolVLA's flow-matching loss. Shape to expect: a fast
  drop over the first ~2 k steps as the arm's mean trajectory is learned, then a long slow
  grind where the ball-position-dependent part gets fitted. The grind is the part that
  matters and it is the part that looks like nothing is happening.
- **`data_s` vs `updt_s`** is the diagnostic that pays for itself. `updt_s` is GPU time,
  `data_s` is time the GPU spent *waiting for the dataloader*. Two PNG decodes per sample is
  why `cpu=16.0` and 12 workers are set; if `data_s` is a meaningful fraction of `updt_s`,
  raise workers before renting a bigger card.
- **No eval line.** `env_eval_freq` needs a LeRobot env and our simulator is a local MuJoCo
  scene, so training never scores the task. That is exactly why P3 scores every checkpoint in
  closed loop instead of trusting the last one.

Failure signatures, in the order they show up: an immediate crash at policy construction
(minutes, cheap — this is what the 1-step smoke buys); loss plateauing at a high value
(underfitting, usually LR or too few steps); loss going to near-zero while closed-loop
behaviour is one fixed trajectory regardless of ball position (the §3.2 state shortcut —
the one failure the loss cannot show you).

Each save writes weights plus optimizer state, ~1 GB per checkpoint; 6 saves is ~6 GB on the
`molmoact2-checkpoints` volume.

## 5. Cost

L4 at ~$0.80/hr. ACT at this size with two 256×256 cameras and batch 16 should land near 4-6
steps/s on an L4, so 60 k steps ≈ 3-4 h ≈ **$3**, plus a few cents of T4 serving per rollout
batch. The smoke run is under $0.10. If steps/s disappoints the bottleneck is likely CPU
PNG decode (two images per sample, PNG-in-parquet) rather than the GPU — hence `cpu=16.0`
and 12 dataloader workers, the same allocation `smolvla_modal_train.py` uses for the same
reason.

## 6. Commands

```bash
# P0 (done) — verify, and repair a dataset whose collector was killed in finalize()
python3 -c "import json; i=json.load(open('libero/fine_tune/a7/meta/info.json')); print(i['total_episodes'], i['total_frames'], i['fps'])"
uv run python libero/fine_tune/rebuild_stats.py libero/fine_tune/a7 --check

# P1
modal run act/act_modal_train.py::upload
modal run act/act_modal_train.py --max-steps 1 --save-freq 1        # SMOKE

# P2
modal run act/act_modal_train.py --max-steps 60000 --save-freq 10000

# P3
ACT_CHECKPOINT=/checkpoints/act/act-green-ball/checkpoints/060000/pretrained_model \
    modal deploy act/act_modal.py
curl -s <url>/health | python3 -m json.tool        # poll until it reports the right step

uv run python libero/libero_closed_loop.py \
    --payload-keys libero --server-url <url>/act \
    --delta-pos-scale 0.10 --randomize-bins --randomize-ball \
    --chunks 25 --no-view --run-id act_ck60000_00

# P4
uv run python libero/score_runs.py assets/act/act-green-ball_010000
```

`--payload-keys libero` is required (sends `{image, wrist_image, instruction, state}`); the
`droid` default sends `external_cam`/`wrist_cam` and the server 400s rather than run blind.
`--delta-pos-scale 0.10` and `--randomize-bins` are required to match how `a7` was collected —
a mismatch there does not measure `a7` at all.

## 7. Learnings

`act/PROGRESS.md`. Same role `libero/PROGRESS.md` plays for the MolmoAct2/SmolVLA path:
numbered sections, every claim carrying the measurement it came from, and corrections to this
README recorded rather than edited away.
