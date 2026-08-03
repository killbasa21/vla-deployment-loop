# Plan — SmolVLA to >90% on the green-ball task

Written 2026-08-03. Nothing here has been run. Categories reference
`robot-learning/lehome-2026-study-index.md`.

---

## Reality check, before any of it

Three numbers from this repo set the bar honestly:

| policy | placed | source |
|---|---|---|
| ACT ck10000 (from scratch, ~50-80 M) | **5/6** | `act/PROGRESS.md` §7 |
| SmolVLA LoRA on `a5` | 2/10 | `libero/PROGRESS.md` §25 |
| SmolVLA LoRA on `a7` | 1/10 | `libero/PROGRESS.md` §25 |

**90% is above anything this repo has measured, on any policy.** ACT — the better vehicle
today — sits at 83% (n=6) and 71% release rate (n=7) at ck30000. So this plan is not
"tune SmolVLA a bit"; it is closing a 10% → 90% gap on the policy that is currently losing.

Two honest framings, pick deliberately:

- **If the goal is >90% on the task** — ACT is the closer starting point by a wide margin,
  and `ck20000` is *unscored* (`act/PROGRESS.md` §7.5). Cheapest possible progress.
- **If the goal is to make SmolVLA work** — that is a more valuable exercise (it is the VLA
  path, the generalist path, the thing that transfers), but treat 90% as a stretch target
  and Phase 3 as the realistic decision point.

This plan assumes the second. Phases are gated: **do not proceed past a gate that failed.**

---

## Phase 0 — Make the measurement trustworthy
*Category: T0 · evaluation protocol, biased-vs-unbiased accounting*

Nothing below is meaningful until this is done. `libero/PROGRESS.md` §25.3: five of twenty
rollouts died early with stdout sent to `/dev/null`, and the client exits rather than retries
on a transient server error. **Every rate in §25 carries that noise.**

| # | Step | What it does | How |
|---|---|---|---|
| 0.1 | Capture stderr + retry on transient failures | Removes the silent truncation that corrupts every rate | `libero_closed_loop.py`: retry the `/act` POST with backoff; stop sending stdout to `/dev/null` |
| 0.2 | Make `score_runs.py` refuse to score incomplete logs | A truncated log scores identically to a release failure | It already prints `INCOMPLETE` — make it exclude by default, `--include-incomplete` to override |
| 0.3 | Fix the evaluation seed set at n≥20 | At n=10, 9/10 vs 8/10 is noise. A 90% claim needs n≥20 minimum | Fixed seed list, committed to the repo, reused for every checkpoint forever |
| 0.4 | Separate "unbiased" from "diagnostic" runs | Replay/curriculum/hard-case runs must never enter a headline rate | Tag `run_id` by strategy; `score_runs.py` reports them separately |

**Gate:** re-score ACT ck10000 on the n≥20 protocol. If 5/6 does not roughly hold, the
measurement was the problem all along and everything downstream was noise.

---

## Phase 1 — Does SmolVLA adapt at all?
*Category: T1 · the adaptation question*

`act/PROGRESS.md` §7.1 is the sharpest result in the repo: ACT closes to 2.5–7.7 mm on the
**same** `a7`, same OSC plant, same `--delta-pos-scale 0.10` where the SmolVLA LoRA missed by
11–88 mm. It concludes: *"It was neither units nor grounding — it was adaptation."*

So the question is narrow: **can SmolVLA's perception be moved by our data at all?**

| # | Step | What it does | How |
|---|---|---|---|
| 1.1 | Unfreeze the vision encoder | ACT's win came with a vision backbone *in the gradient path*. SmolVLA's stock recipe freezes it (`freeze_vision_encoder: true`) and LoRA-on-VLM already failed | `smolvla_modal_train.py`, new `--mode full` — vision encoder + VLM + expert, no PEFT |
| 1.2 | Train on `a7`, not `a5` | `a7` matches ACT's dataset exactly, so the comparison isolates the policy | `data/a7_smolvla` is already converted |
| 1.3 | Smoke first | Repo law | `--max-steps 1 --save-freq 1` |
| 1.4 | Score every checkpoint | "The last checkpoint is not the best one" — demonstrated twice | `--save-freq` small; score each on the Phase 0 seed set |
| 1.5 | Sanity-check normalisation | Silent failure mode: right shape, wrong space | Keep `--norm-stats checkpoint`; verify against `a7` stats before trusting a number |

**Gate:** does *any* checkpoint reach ≥5/20 with closest-lateral under ~15 mm? If nothing
moves off 1–2/10 even with the whole network unfrozen, SmolVLA is the wrong vehicle for this
task and Phase 2+ is wasted money. Say so and switch to ACT.

---

## Phase 2 — Attack closure, not geometry
*Category: T0 reward/data design · T1 hard negative mining*

`libero/PROGRESS.md` §25.1: **every lift and every placement came from a run where the
gripper fired**, and it fired in exactly 5/10 for both fine-tunes. Closure is nearly
uncorrelated with whether the hand is on the ball (0.7 mm → never closed; 90.8 mm → closed).

§25.2 already names the cause, and it is a **data** cause:

> The expert only ever demonstrates closing on a stationary, perfectly-centred ball, and
> rejection sampling on `lifted and placed` deletes every episode where contact went wrong.
> The states the policy actually occupies at decision time are therefore absent from the data
> by construction.

That is exactly Larchenko's diagnosis of his own BC set — *"no failures and no recoveries...
a great base to refine but a poor source of recovery behaviour."*

| # | Step | What it does | How |
|---|---|---|---|
| 2.1 | **Stop rejection-sampling the collector** | Restores the near-miss states the policy actually occupies | `collect_finetune_data.py`: keep failed episodes, label outcome per episode |
| 2.2 | Collect a closure-focused set | Teaches *closure conditioned on having the object* — the missing capability | Start episodes near the grasp decision point with the ball offset ±0-40 mm; both outcomes kept |
| 2.3 | Weight the grasp window | The decision is ~3 chunks out of ~30; uniform sampling drowns it | Upweight frames near the gripper transition (paper's "success tail boost", §5.3) |
| 2.4 | Verify the label channel first | A policy faithfully reproducing a broken label is the cheaper explanation | ACT's check: `60/60 a7 episodes end commanded open`. Re-run it on any new set |

**Gate:** does gripper-fire rate rise above 5/10? That single number gates everything —
it has bounded every result in §25.

---

## Phase 3 — Inference-time optimization
*Category: T1 · free performance, zero training cost*

Completely untouched in this repo, and per the paper the **only** cluster with measured
evidence behind it. SmolVLA's expert is flow-matching and **samples** — `libero/PROGRESS.md`
§18 has two rollouts from an identical start diverging completely. That is textbook
multimodality, which is precisely what these tools address.

| # | Step | What it does | How |
|---|---|---|---|
| 3.1 | Sweep execution length | Larchenko's bandit converged to executing 3–5 of 30. Ours is hardcoded at 10 | `smolvla_modal.py` returns `n_action_steps`; sweep 3/5/10 on the Phase 0 seed set |
| 3.2 | Sweep noise temperature | τ<1 concentrates samples near the mode — directly targets the §18 divergence | Scale the flow seed noise by √τ; sweep 0.7/0.85/1.0 |
| 3.3 | Best-of-N (needs Phase 4) | Rejects disaster chunks at multimodal decision points | Sample N chunks off the shared prefix, rank by the Phase 4 head |
| 3.4 | Bandit the sweep instead of gridding it | Cheap, and the optimum moves as the policy trains | Thompson sampling over each knob independently, reward = success − mean |

**Gate — and this is the realistic decision point.** If Phases 0–3 land somewhere in
50–80%, that is a good result and Phase 4+ is a research project, not a tuning task. Decide
explicitly whether to spend it.

---

## Phase 4 — A success-prediction head
*Category: T1 · policy as its own value function*

The gateway to everything else. One sigmoid head, BCE on the binary outcome, reading **image
tokens only** (attention-group isolation — otherwise it overfits to proprioception).

| # | Step | What it does | How |
|---|---|---|---|
| 4.1 | Add the head | A value signal, with no separate critic to train or serve | Linear probe on a learned query token; small loss weight |
| 4.2 | Serve it alongside actions | Enables live failure detection mid-rollout | Return `p_success` per chunk in the `/act` response; log it |
| 4.3 | Use it to trigger snapshots | Hard negative mining — snapshot where predicted success drops off its running max | That is *where the policy ruins the episode*, i.e. exactly the Phase 2 data you want, found automatically |
| 4.4 | Use it to rank best-of-N | Unlocks 3.3 | Score candidates, execute the best |
| 4.5 | Expect the tail bias | Successes terminate instantly, near-misses linger → "almost done" frames come mostly from failures | Interpolate the last ~30 frames toward the known outcome offline |

**Note:** 4.3 closes the loop with Phase 2 — the value head becomes a data-collection
instrument. That pairing is the single highest-leverage idea in the paper for this repo.

---

## Phase 5 — RL post-training
*Category: T2 · only after the BC ceiling is real*

Do not start this until Phases 0–4 are done and a checkpoint has plateaued. Everything in
Larchenko's paper assumes *"a BC-pretrained policy with a non-zero success rate."*

| # | Step | What it does | How |
|---|---|---|---|
| 5.1 | Return-preserving dense reward | Credit assignment inside the episode with zero hacking surface | Checkpoints from the scorer's own conditions (lifted 50 mm, inside footprint); withdraw all reward on failure so `Σr = 1[success]` |
| 5.2 | GAE over the success head | Per-frame advantage from per-episode outcomes | γ=0.999, λ=0.99, dampen the baseline by α≈0.5 |
| 5.3 | AWR through the sampler | Trains more on high-advantage frames at no extra compute | `P(sample) ∝ e^clip(A,−2,2)`; **importance-weight the aux heads or they go biased** |
| 5.4 | Advantage conditioning + CFG | Unlocks guidance at inference | Advantage token, masked per the paper's schedule; then 3.3's guidance scale |

---

## Cheapest path, if you only do three things

1. **Phase 0** — without it no number below means anything, and it is a day's work.
2. **Step 1.1** (unfreeze the vision encoder) — the direct test of §7.1's conclusion, one
   training run, answers whether SmolVLA is viable at all.
3. **Step 2.1** (stop rejection-sampling) — one line in the collector, attacks the constraint
   §25 identified and no dataset has ever addressed.

## What would make me say "stop"

- Phase 1 gate fails → SmolVLA cannot adapt to this task; go to ACT ck20000.
- Gripper-fire rate stays at 5/10 after Phase 2 → the problem is the plant or the label
  channel, not the policy. Re-run the expert through the inference path before spending more
  (repo law, and it is what exposed the decimation bug).
- Any number improves without the Phase 0 protocol in place → do not believe it.
