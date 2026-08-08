# Results

Task: `put the green box in the green container`. Target tray colour is permuted
every episode, so the target cannot be memorised as a position.

All policy evals: 25 episodes, horizon 300, `--chunk-reuse 25` (re-observe every
25 of the 50 predicted actions). Expert and random baselines use the same scorer
and the same environment.

## Headline

| policy | complete | placed | wrong tray | grasp_pos_closest (mean) |
|---|---|---|---|---|
| random actions | 0% | 0% | 0% | 0.234 m |
| stock `lerobot/smolvla_base` | 0% | 0% | 0% | 0.211 m |
| ft1 @ 4 000 steps | 0% | 0% | 0% | 0.097 m |
| ft1 @ 8 000 steps | 12% | 12% | 0% | 0.061 m |
| **ft1 @ 12 000 steps** | **36%** | **36%** | **0%** | **0.035 m** |
| scripted expert (noise-free) | 100% | 100% | 0% | 0.001 m |

## Full stage breakdown

| stage | random | stock | ft1 4k | ft1 8k | ft1 12k | expert |
|---|---|---|---|---|---|---|
| reached | 0% | 0% | 4% | 20% | 48% | 100% |
| grasped | 0% | 10% | 28% | 40% | 84% | 100% |
| lifted | 0% | 0% | 0% | 12% | 36% | 100% |
| transported | 0% | 0% | 0% | 12% | 36% | 100% |
| released | 0% | 0% | 0% | 12% | 36% | 100% |
| placed | 0% | 0% | 0% | 12% | 36% | 100% |
| complete | 0% | 0% | 0% | 12% | 36% | 100% |

Continuous measures, ft1 @ 12 000 (mean / median / p90 / min):

```
grasp_pos_closest    0.0354  0.0270  0.0924  0.0017  m
grasp_pos_at_close   0.0453  0.0351  0.1006  0.0093  m
grasp_rad_closest    0.0480  0.0462  0.0870  0.0013  rad
grasp_rad_at_close   0.3941  0.3625  0.7026  0.0384  rad
place_pos_closest    0.1446  0.1682  0.2653  0.0013  m
place_pos_final      0.1553  0.1736  0.3088  0.0051  m
lift_height_max      0.0585  0.0000  0.1695  0.0000  m
```

## Reading the numbers

**Colour grounding is not the bottleneck.** Across every fine-tuned checkpoint,
`placed_wrong` is 0/25 — the policy never once put the box in a red or blue tray.
Successes are spread across target slots (left 5/11, right 1/4, top 3/10 at 12k),
so it is not succeeding by always driving to one favoured position. A frozen
SigLIP plus a frozen 16-layer decoder routes "green" to the action expert well
enough; unfreezing the vision encoder is not indicated.

**Grasping is the bottleneck**, and specifically wrist orientation.
`grasp_pos_at_close` is 0.045 m — the gripper is nearly on the box when it
closes — but `grasp_rad_at_close` is 0.394 rad (~23°) against a 0.048 rad best
achieved earlier in the same episodes. So the policy reaches the right *place*
with the right *orientation* at some point, then rotates away before closing.
That is what turns 84% grasped into 36% lifted. 16/25 episodes time out, almost
all of them stuck retrying the grasp.

**The trend is still rising at 12 000 steps.** complete goes 0% → 12% → 36% over
4k → 8k → 12k with training loss flattening only late (0.49 @ 1.6k, 0.37 @ 12k)
and the cosine schedule already decayed to 2.5e-6. This run was not trained to
convergence; it was trained to a fixed budget.

**Run-to-run spread.** Flow matching samples noise at inference, so the same 25
seeds do not give the same score twice. The 12 000-step checkpoint scored 32%
and 36% complete on two runs of identical episodes. Treat +-1 episode (4 points)
as noise at n=25.

## Setup

- 300 expert demonstrations, rejection-sampled to successes only (99% of attempts
  succeeded), 42 981 frames, collected with `--action-noise 0.05
  --waypoint-noise 0.004`.
- 12 000 steps, batch 16, lr 1e-4 cosine with 500 warmup, bf16, on one Modal L4
  at 2.09 steps/s (~96 min).
- 99.9 M of 450.0 M parameters trained (the action expert plus the four
  projections). Vision encoder and VLM frozen.

## A metric bug found and fixed mid-eval

The first 12 000-step score read `placed 32%` but `released 8%`, which is
impossible if release precedes placement. Cause was in the scorer, not the
policy: `released` was recorded at the *first* loss of gripper–box contact, and
contact breaks transiently during transport. Changed to the *last* let-go. After
the fix the chain is self-consistent (lifted = transported = released = placed =
complete = 9/25). The 8 000-step and earlier numbers are unaffected in
`released` terms because those runs had no successful transports to mis-attribute
— but note that the 8k row was measured under the old definition.

## Next levers, in the order worth trying

1. **Train longer.** The curve had not flattened. 30–40k steps on the same data.
2. **Attack the wrist-at-close error directly** — it is the single measured
   bottleneck. Either add wrist-perturbation demos so the policy sees recovery
   from bad orientations, or weight the rotation dimensions up in the loss.
3. **More data.** 300 demos is small; collection is cheap (~10 min per 75
   episodes per process, 4 in parallel).
4. **Shorter `--chunk-reuse`.** 25 steps of open loop is 1.25 s. Re-observing
   every 10 would react faster to a missed grasp, at 2.5x the requests.
