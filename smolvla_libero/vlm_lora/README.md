# `vlm_lora` — SmolVLA action expert + VLM adapter, on the green-**box** task

Written 2026-08-03. **Nothing is trained yet.** The recipe is implemented and smoke-tested;
the dataset is being collected. This file is the spec and the reasoning; scores go in
`libero/PROGRESS.md` when they exist.

Not to be confused with `--mode lora`, which is the mode that already scored 1–2/10.

---

## 1. What this is

One training mode in [`../smolvla_modal_train.py`](../smolvla_modal_train.py), selected with
`--mode vlm_lora`. It trains, all at LoRA rank 8:

| target | modules | why |
|---|---|---|
| VLM text tower `self_attn.{q,k,v,o}_proj` | 128 | the grounding fix — which image tokens the text stream attends to |
| vision→text connector | 1 | single bottleneck every visual token crosses into the LM |
| action expert `self_attn.{q,k,v,o}_proj` | 128 | learns *this* motion, not a new action space |
| state/action heads, **full rank** | 5 | 0.74 M total; LoRA on a 32→480 projection saves nothing |
| SigLIP vision tower (`--vision-lora`, **off**) | 48 | opt-in, see §5 |

**Measured on the stock checkpoint: 3,605,376 learnable of 608,539,552 total — 0.59%.**

## 2. The one thing it changes

`--mode lora` was documented as "LoRA adapters on the VLM + full action expert". It is not.
It uses lerobot's default SmolVLA target set, which never touches the VLM: expert `q/v` plus
the heads. Every `lora` run before 2026-08-03 measured that, and it scored **2/10 on `a5`,
1/10 on `a7`** (`libero/PROGRESS.md` §25).

So `lora` is the control, and `vlm_lora` moves **exactly one variable: the VLM gets
gradients.** Nothing else about the recipe changes. If the score moves, grounding was the
constraint; if it does not, grounding was not, and that is worth knowing before Phase 2 of
[`../PLAN_90PCT.md`](../PLAN_90PCT.md).

## 3. Why grounding is the suspect

Stock `smolvla_libero`, three randomised pick positions, closest lateral approach:
**10.7 mm, 108.2 mm, 46.4 mm.** Missing by 108 mm at an unseen position is a perception
failure, not an action-generation one. `libero/PROGRESS.md` §4: with the VLM frozen, the
expert can only learn the average trajectory over the randomisation box — which is exactly
that error pattern.

## 4. Why the expert is LoRA and not full rank

The first draft of this mode trained `lm_expert` at full rank (~81.5 M learnable), arguing
the action space had changed enough that a low-rank delta could not express the re-mapping.
**That argument is false.** From `libero/fine_tune/collect_finetune_data.py:9-11`:

| | ours | LIBERO |
|---|---|---|
| action | 7-D delta eef pose, `[-1,1]` | same |
| state | `[eef_pos(3), axisangle(3), gripper_qpos(2)]`, LIBERO frame | same |
| arm / controller | Franka Panda, robosuite `OSC_POSE` port | same |
| rate | 20 Hz | same |
| reset orientation | axis-angle `(3.140, 0, −0.089)` | `(3.141, 0.002, −0.090)` |

`action_in_proj` / `action_out_proj` / `state_proj` map the **identical** space they were
pretrained on. And with `--delta-pos-scale 0.05` (§6) the units match too, so it is not even
a rescale.

Three consequences:

1. **If perception is fixed, the action mapping is already correct.** The expert has to
   learn one motion, not a new space.
2. **Full-rank expert would move two variables at once** against a control in which only one
   was broken. Bad experiment design.
3. It removes a hazard `_peft_preflight` could not check. PEFT's `ModulesToSaveWrapper`
   deepcopies each wrapped module (peft 0.18.1, `utils/other.py:584`) and replaces the
   attribute, while SmolVLA reaches into `self.lm_expert.layers` **directly**
   (`smolvlm_with_expert.py:424`) rather than calling its forward. That works only because
   `ModulesToSaveWrapper.__getattr__` forwards to the active copy (`other.py:106-107`). peft
   is unpinned in the Modal image. The five modules still in `full_training_modules` are all
   invoked through their own forward, so none of them ride on that behaviour.

81.5 M trainable against 60 episodes was also a poor ratio. 3.6 M is defensible.

## 5. Why the vision tower is off

SigLIP has seen vastly more images than 60 episodes of one table; what it encodes about
*where a green blob is* is already better than this dataset can teach. What is missing is the
read-out — the connector and the text tower. It is also the expensive half: with vision
frozen and no adapters in it, **there is no backward pass through SigLIP at all**, which is
the single largest term in the step cost (§8). `--vision-lora` exists to test that reasoning
rather than assume it, at roughly 2× the step time for 0.6 M more parameters.

## 6. `--delta-pos-scale` is 0.05. Do not pass the flag.

0.05 is LIBERO's own value and the code default on both the collector and the client. Using
it is what makes §4's table exact rather than approximate. `a5`–`a7` were collected at other
values and must still be **served** at those values; that older rule still stands for them.

Clipping is handled by `--motion-speed`, not by shrinking the scale. **A bug was fixed here
on 2026-08-03**: `reference_track` applies both knobs — it retimes a segment to
`dist / motion_speed` and *then* multiplies by `speed_scale` — while `speed_scale`'s default
was derived from the same velocity ceiling the retiming already respects. At 0.05 that was
`2.5×` on top of an already-correct retime. Never noticed because the two were never
simultaneously non-trivial: `a5` is 0.05 but predates `--motion-speed`; `a6`/`a7` are high
enough that `max(1.0, …)` pinned it at 1.0.

**Open, and it gates believing any score:** the label distribution at 0.05 *with* the
retiming has never been measured. `a5` is the only 0.05 collection and its `dx q01` was
pinned at −1.000 — but it used `--speed-scale 2.5` and no retiming, so it is not evidence
about this setting. Saturated labels destroy gradient direction; an exact unit convention
does not save you from them. **Check `dx q01` against released LIBERO's −0.679 on the first
box collection before training on it.**

## 7. Guardrails

PEFT is silent about empty matches in both directions: a `target_modules` regex that
fullmatches nothing gives a LoRA over nothing, and `modules_to_save` entries that suffix-match
nothing are dropped unless `strict_module_check` is set, which lerobot does not set
(`utils/other.py:1041`). Either way the run reports a healthy loss and has not trained what
you asked for. This repo has paid for that class of bug repeatedly (§5 of
`libero/PROGRESS.md`; the `observation.images.image2` story in `../README.md`).

- **`_peft_preflight`** loads the stock checkpoint on CPU and counts VLM hits and expert hits
  **separately** — a non-empty total proves nothing, since lerobot's default target set also
  produces one. Either half empty is a hard exit. Disable with `--preflight false`.
- **`::dump_modules`** prints real `named_modules()` keys plus what each constant matches.
  Verified 2026-08-03: text 128, connector 1, expert 128, vision 48, each full-rank entry 1.
- **`--policy.train_expert_only=false` is load-bearing.**
  `SmolVLMWithExpertModel.train()` re-asserts `self.vlm.eval()` on every switch into train
  mode while that flag is set, and the stock checkpoint ships it `true`. Harmless while
  nothing in the VLM trains; the moment adapters are added it parks every one of them in
  eval mode, and the run reports nothing unusual.
- **Rank is also a learning-rate knob.** lerobot's `PeftConfig` CLI exposes only
  `target_modules, full_training_modules, method_type, init_type, r` — no `lora_alpha`, so
  alpha is PEFT's default 8. LoRA scales its update by `alpha/r`, so `r=8` gives 1.0× and
  `lora`'s `r=32` gave **0.25×**. Raising `--lora-r` *lowers* the effective LR. Compensate
  with `--lr`.

## 8. Cost

Per-sample FLOPs, estimated from the configs (not measured): SigLIP forward-only ~350 GFLOP
(2 cameras × 1024 tokens × 12 layers), text tower forward+backward ~380 GFLOP (~200-token
sequence after pixel-shuffle: 64 tokens/image), expert ~25 GFLOP (one flow-matching timestep,
50 action tokens). **~0.75 TFLOP/sample → ~12 TFLOP/step at batch 16.**

At 25–35% MFU on an L4 (121 TFLOPS bf16 peak) that is ~0.35 s/step of compute; call
**0.5–0.8 s/step** with the dataloader. **~30–40 min for 3000 steps, ~$0.50.** The expert
being LoRA rather than full rank saves memory (~1 GB → ~40 MB of optimizer state), not time.

Watch for the dataloader going input-bound: PNG-in-parquet means every sample decodes two
images on CPU, which is why the function asks for 16 cores.

## 9. Runbook

```bash
uv run modal run smolvla_libero/smolvla_modal_train.py::dump_modules --grep lm_expert
```

```bash
uv run python smolvla_libero/convert_dataset.py --src libero/fine_tune/<set> --out smolvla_libero/data/<set>_smolvla
```

```bash
uv run modal run smolvla_libero/smolvla_modal_train.py::upload --local-dir smolvla_libero/data/<set>_smolvla --dataset-root /data/greenbox/<set>
```

Smoke — 1 step, per repo law, proves build → load → step → checkpoint-save and nothing more:

```bash
uv run modal run smolvla_libero/smolvla_modal_train.py::main --mode vlm_lora --dataset-root /data/greenbox/<set> --exp-name smolvla-vlmlora-smoke --batch-size 2 --num-workers 4 --max-steps 1 --save-freq 1
```

Real run:

```bash
uv run modal run smolvla_libero/smolvla_modal_train.py::main --mode vlm_lora --dataset-root /data/greenbox/<set> --exp-name smolvla-vlmlora-<set> --max-steps 3000 --save-freq 500
```

Serve a checkpoint, then score it. `--save-freq 500` because the last checkpoint is not the
best one — demonstrated twice in this repo, on two architectures:

```bash
SMOLVLA_CHECKPOINT=/checkpoints/smolvla/smolvla-vlmlora-<set>/checkpoints/001500/pretrained_model uv run modal deploy smolvla_libero/smolvla_modal.py
```

```bash
uv run python libero/libero_closed_loop.py --payload-keys libero --randomize-box --randomize-bins --chunks 20 --no-view --server-url https://<...>.modal.run/act
```

`curl /health` first and confirm the reported checkpoint is the one you meant to measure — a
`modal deploy` that returns in six seconds has not necessarily cut over. No `--delta-pos-scale`
on that command: 0.05 is the default and the collection value (§6).

## 10. Smoke-test result, 2026-08-03

Fixture: first 2 episodes of `a5_smolvla`, cut by
[`../make_smoke_subset.py`](../make_smoke_subset.py) — 1078 frames, 27.9 MB, at
`/data/greenbox/smoke_a5_2ep`. Ball data on the ik-era plant; **irrelevant to any score**,
it exists only to exercise the path.

```
dataset : 2 episodes, 1078 frames, v3.0, fps 20.0
Using PEFT! Wrapping model.
Wrapped smolvla with PEFT (LoraConfig)
train_expert_only: False
num_learnable_params=3605376 (4M)
num_total_params=608539552 (609M)
Training: 100%|██████████| 1/1 [00:04<00:00,  4.30s/step]
Checkpoint policy after step 1
lerobot-train returned 0 after 34s
```

Confirmed: PEFT wraps, `train_expert_only` is `False` at the policy level, learnable count is
3.6 M (0.59%), one step runs, checkpoint saves, exit 0.

**Not confirmed by this, and not confirmable by a 1-step run:** that the adapters receive
useful gradient, that the VLM modules are in `train()` rather than `eval()` at forward time,
and that the loss curve behaves. Those need a real run. The learnable-parameter count does
prove the targeting is right, which is the failure mode that has historically cost the most.

## 11. Files

| path | what |
|---|---|
| [`../smolvla_modal_train.py`](../smolvla_modal_train.py) | `VLM_LORA_*` constants, `_peft_preflight`, `dump_modules`, the `vlm_lora` branch |
| [`../smolvla_modal.py`](../smolvla_modal.py) | serving; `SMOLVLA_CHECKPOINT` selects the weights |
| [`../convert_dataset.py`](../convert_dataset.py) | `wrist_image` → `image2` and the stats swap |
| [`../make_smoke_subset.py`](../make_smoke_subset.py) | N-episode prefix of a v3.0 dataset |
| [`../PLAN_90PCT.md`](../PLAN_90PCT.md) | the wider plan; this mode is its Phase 1 |
