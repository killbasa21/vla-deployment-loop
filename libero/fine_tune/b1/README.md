# `b1` — box pick-and-place, for the SmolVLA-LIBERO fine-tune

First **box** dataset. `a1`–`a7` are ball datasets (`CLAUDE.md`); nothing here is comparable
to their scores.

## Generate

Run from the repo root.

```bash
MUJOCO_GL=egl uv run python libero/fine_tune/collect_finetune_data.py \
    --out libero/fine_tune/b1 \
    --control-mode osc \
    --delta-pos-scale 0.05 --speed-scale 1.0 \
    --no-backoff --bin-layout scene \
    --reach 20 --decoy 20 --drift 0 --noise 0 --recover 0 \
    --max-saturated-frac 0.002 --max-torque-frac 0.95 --max-sat-run 3 \
    --max-attempts-per-episode 12 --seed 0
```

~35 min, 40 episodes, ~16.5 k frames, ~410 ticks/episode.

Then convert for SmolVLA:

```bash
uv run python smolvla_libero/convert_dataset.py \
    --src libero/fine_tune/b1 --out smolvla_libero/data/b1_smolvla
```

## Why each flag

| flag | reason |
|---|---|
| `--delta-pos-scale 0.05` | LIBERO's own value. Fixed for this dataset. **The serving client must pass the same value** or the fine-tune is not being measured. |
| `--speed-scale 1.0` | **Required at 0.05.** The auto default is 2.5 and it *compounds* with the auto `--motion-speed`, giving ~1190-tick / 60 s episodes. `OSC_SPEED_SCALE` predates distance retiming and is redundant now. |
| `--no-backoff` | Drops the withdraw-and-re-approach (waypoints 1a/1b/1c). Approach is one clean descent. |
| `--bin-layout scene` | Pins bins to the scene XML positions — the only layout `libero_closed_loop.py` evaluates. |
| `--decoy 20` | See below. |
| `--max-saturated-frac 0.002` `--max-torque-frac 0.95` `--max-sat-run 3` | The three torque gates — see below. |
| `--reach 20` | Clean expert trajectories. |

Not passed, and correct as defaults in `--control-mode osc`: `--min-clearance` (off — OSC
yields on contact natively), `--motion-speed` (auto-derives 0.0738 m/s from the label
ceiling).

## Cohorts

- **`reach` (20)** — clean expert, randomised box position.
- **`decoy` (20)** — exactly 3 non-overlapping windows per episode, 8–20 ticks each,
  ≥15 ticks apart, all ending **before the box is released**. Inside a window the
  **executed** action points at a target 35 mm off the reference in a fixed random
  direction, so the arm deliberately moves the wrong way and then recovers.

**The one rule this cohort exists to respect:** the **label** always points at the *true*
reference, never at the decoy. That is what makes it recovery data. Put the wrong move in
`waypoints` instead and it becomes part of the reference — the label would then read "move
wrong", and the policy would faithfully learn to detour *and* correct, because both halves
were labelled correct. Same argument `libero/fine_tune/README.md` §4.1 makes for the noise
cohort. **The wrong move belongs in the execution and nowhere else.**

Why 35 mm: the corrective label peaks at `DECOY_OFFSET / DELTA_POS_SCALE` = 0.70 once the
arm reaches the decoy. At 50 mm that is exactly 1.0 and the label clips at the moment it
carries the most information.

Not used here: `--drift` (stochastic Gaussian windows, same placement machinery),
`--noise` (perturbed every tick), `--recover` (single-tick impulses).

## Torque gates

Every episode is checked at **every physics step, on every joint** — 25 steps × 7 joints
per control tick, ~70 k joint-steps per episode. Three separate gates, because one is not
enough:

| flag | rejects when | why the others miss it |
|---|---|---|
| `--max-saturated-frac 0.002` | >0.2 % of joint-steps hit the actuator limit | an episode that fights the limits throughout |
| `--max-torque-frac 0.95` | **any** joint at **any** step demands >95 % of its limit | a 3-step burst in 70 k is 0.004 % and sails through the fraction gate |
| `--max-sat-run 3` | >3 *consecutive* steps with any joint saturated | a joint losing authority for a stretch reads the same as isolated clipping to the other two |

`--max-torque-frac` measures **before** the clip (`OSCController.last_peak_frac`), so an
over-limit demand shows its true size. Post-clip torque cannot distinguish 0.99 of the
limit from 3× it — both read as exactly the limit.

Recorded per episode regardless of whether the gates are on: `saturated_frac`,
`saturated_steps`, `peak_torque_frac`, `peak_torque_tick`, `max_sat_run`.

**Measured at `--delta-pos-scale 0.05`, 3 episodes, per physics step:**

```
                  saturated joint-steps        peak |tau| / limit
inside  windows   0 / 6300,  0 / 6475,  0 / 6650      j2  52-61 %
outside windows   0 / 64400, 0 / 56875, 0 / 61950     j2  59-73 %
```

**Zero saturation anywhere**, and the decoy windows are *gentler* than the rest of the
episode on 5 of 7 joints. The j2 peak is gravity hold, present in clean episodes too and
unrelated to the decoy — a decoy is a goal displacement, not a force, and OSC's PD acts on
a 35 mm error smaller than the errors normal transport already produces.

The gates therefore never fire in practice. That is the intent: they are a tripwire for a
future change (a larger `DECOY_OFFSET`, a faster `--motion-speed`, a different scale), not
a filter doing daily work.

**A gate that never fires is not a verified gate**, so it was checked against a threshold
it cannot pass:

```bash
# --max-torque-frac 0.30 is below the gravity-hold peak; every episode must be rejected
uv run python libero/fine_tune/collect_finetune_data.py --out /tmp/gate_neg \
    --control-mode osc --delta-pos-scale 0.05 --speed-scale 1.0 --no-backoff \
    --reach 1 --max-torque-frac 0.30 --max-attempts-per-episode 2 --verbose
```

```
reject reach_00 try 1: lifted=True placed=True peak_tau=0.56@t119 OVER-TORQUE
reject reach_00 try 2: lifted=True placed=True peak_tau=0.56@t316 OVER-TORQUE
reach_00     DROPPED after 2 attempts
```

Note `lifted=True placed=True` on both — the gate rejects on torque independently of
whether the episode succeeded at the task, which is the property that matters. Re-run this
after any change to the gates.

## Verify after collecting

```bash
uv run python libero/fine_tune/rebuild_stats.py libero/fine_tune/b1 --check
uv run python scripts/dataset_to_video.py libero/fine_tune/b1 --episodes 0,20 --out /tmp/b1_vid
```

Measured on a 3-episode preview (1 `reach` + 2 `decoy`, 1243 frames, seed 0):

```
ch        q01     q50     q99     std   clip%  max|.|
dx     -0.219   0.008   0.451   0.122    0.00   0.548
dy     -0.413  -0.001   0.623   0.224    0.00   0.702
dz     -0.616   0.035   0.503   0.226    0.00   0.800
drx    -0.020   0.000   0.023   0.008    0.00   0.031
dry    -0.025   0.000   0.029   0.014    0.00   0.044
drz    -0.022   0.000   0.019   0.010    0.00   0.036
```

- **Clipping 0.00 % on every pose channel**, peak label 0.800. Anything above ~1 % means
  the expert is outrunning the label ceiling — check `--speed-scale` is really 1.0.
- Torque saturation 0.0000 % of joint-steps on all three episodes.
- Every episode `lifted=1 placed=1` on attempt 1, 0 rejected attempts.
- ~404–420 ticks/episode, ~48 s wall per episode.

The decoy cohort widens the label distribution exactly where intended — `dz` q99 goes
0.427 → 0.503 and q01 −0.489 → −0.616 against the clean-only preview, which is the
correction signal being added without any of it clipping.

**Known, accepted:** `dx q01 ≈ −0.22` against released LIBERO's `−0.679`. Removing the
backoff halved the −x label range (it was `−0.499` with it). The closing return-to-start
segment is now the only −x source. `libero/fine_tune/README.md` §8 argues thin −x labels
are what make the arm overshoot and never come back — `a3` had `−0.072` and failed that
way. `−0.22` is a deliberate trade for a clean trajectory, not an oversight.

## Preview videos

`libero/fine_tune/b_scratch/` holds rendered previews from the runs that set the numbers
above — `reach_00.mp4` (clean) and `decoy_00.mp4` / `decoy_01.mp4`. Regenerate any episode
with `scripts/dataset_to_video.py`. Not part of the dataset; safe to delete.

## Do not

- Do not pass `--delta-pos-scale` differently at collection and at serving.
- Do not omit `--speed-scale 1.0`.
- **Do not move the decoy detour into `waypoints`.** It looks like a simplification and it
  inverts what the cohort teaches — see the `decoy` section above.
- Do not compare any score here to `a1`–`a7`. Those are the ball.
