# Servo droop, and why it was poisoning the fine-tune data

Written 2026-07-28. This is the "read this to understand what happened" document for the
LIBERO fine-tune data problem. It covers what droop is, how it got into the training
labels, every calculation used to find and fix it, the things I got wrong on the way, and
what changed as a result.

Companion docs, in increasing detail: `libero/README.md` (the spec — conventions, scene,
control law), `libero/PROGRESS.md` (chronological attempt log), `libero/fine_tune/README.md`
(dataset format findings).

---

## 0. One-paragraph summary

The Panda is driven by position actuators that are heavily overdamped, so it never fully
arrives at the pose it is commanded to hold — it sits at a standing gravity **sag**, and it
only realises about **a third** of any commanded motion per control tick. The demo
collector labels every tick with `(target − current) / 0.05`, which is an *error* term, so
that sag got written into **every training label** as a near-constant offset: mean `dx`
**+0.174**, mean `dz` **+0.127**, against a released LIBERO distribution that is symmetric
about zero. The worst consequence is that `dx` became one-sided — its 1st percentile over
8700 frames is **−0.08** — so the data contained essentially no signal for retreating in
−x, which is exactly the recovery behaviour the closed-loop runs were failing at. The fix
is to stiffen the arm (`kp ×2`, `kd ×0.7`), which is a move *toward* LIBERO's tightly-
tracking OSC, not away from it. Dataset `a3/` is collected on the fixed plant; `a1/` and
`a2/` are superseded.

---

## 1. What droop is

Two distinct effects, often conflated. Both are properties of the plant, not of the policy.

**Static sag.** The arm holds a commanded joint angle by generating torque proportional to
the error, `tau = kp*(q_cmd - q) - kd*qdot`. Gravity pulls down continuously. At equilibrium
the servo must be producing enough torque to hold the arm up, and the only way it can do
that is to *sit at a nonzero error*. So the arm parks slightly below and behind wherever it
was told to go, forever. Bigger `kp` means less error is needed to make the same torque, so
stiffer means less sag.

**Tracking lag.** When the target is moving, the arm trails it. How badly depends on how
fast the servo responds relative to how fast the target moves.

The measured static sag at LIBERO's reset pose, `grip_site` height above the table:

```
pure forward kinematics, no dynamics    0.2728        <- where the joints say it should be
stock  kp4500 kd450                     0.2680        sag 4.84 mm
new    kp9000 kd315                     0.2704        sag 2.44 mm
x4     kp18000 kd225                    0.2716        sag 1.23 mm
LIBERO, measured from a live env        0.2733
```

### 1.1 A documentation error this exposed

`libero/README.md` and `libero/fine_tune/README.md` both record "eef above table at reset:
LIBERO 0.2733, ours 0.2728" — a claimed **0.5 mm** match. **0.2728 is the pure-FK number.**
It is where the joint angles put the eef with dynamics switched off. The pose the arm
actually settles at, and starts every single episode from, was **0.2680** — a **5.3 mm**
error against LIBERO, ten times what the docs claim. The gain change takes it to 2.9 mm.

Nothing was faked here; the FK number was measured correctly and then compared to a
dynamic one. It is a reminder to check whether a number came from `mj_forward` or from a
settled `mj_step` loop before comparing it to anything.

---

## 2. Why droop ends up inside the training labels

`libero/fine_tune/collect_finetune_data.py:expert_action` computes each label as:

```python
dpos = (target_pos - cur_pos) / L.DELTA_POS_SCALE       # 0.05 m per action unit
drot = L._orientation_error(target_mat, cur_mat) / L.DELTA_ROT_SCALE
```

That is a **P-controller error term, not a displacement.** It answers "how far is the arm
from where it should be", not "how far should the arm move".

So consider a tick where the reference trajectory is parked at a waypoint and the *intended*
motion is exactly zero. The arm is still sagging below and behind that waypoint. The
subtraction returns nonzero. A nonzero action is written to the dataset. The residual is
pure droop, and it appears in **every** label in the dataset.

Converting to physical units — action units are 0.05 m each:

```
mean dx = +0.174  ->  0.174 * 0.05 m = 8.7 mm of standing error in x
mean dz = +0.127  ->  0.127 * 0.05 m = 6.4 mm of standing error in z
```

### 2.1 Three measurements that prove it is a standing offset, not task signal

1. **Per-episode mean `dx` = +0.174, standard deviation across the 50 episodes only
   0.080.** Those episodes have randomised ball positions and shuffled bin layouts, so the
   actual motions differ a lot. Genuine task intent would not produce nearly the same mean
   in every episode. A constant plant bias would.

2. **Realised displacement is ~33% of commanded.** Per tick, across all 8700 frames of
   `a1`: commanded `norm(action[0:3]) * 0.05` averages **19.1 mm**, realised
   `norm(state[t+1][0:3] - state[t][0:3])` averages **6.4 mm**. Overshoot is essentially
   absent — only **2.9%** of ticks move further than commanded. A one-sided systematic
   shortfall, not noise around a correct value.

3. **Sign agreement between `action[t]` and the realised displacement over that tick is
   only 0.68 / 0.74 / 0.79** (x/y/z). Roughly a third of the time the arm moves *opposite*
   the commanded direction, because on small moves the constant bias outvotes the intent.

### 2.2 On the 33% figure

There is more than one defensible way to express that ratio, and they disagree at the
margin, so here they all are:

```
mean|ds| / mean|c|                       0.334    <- the straightforward reading
mean(|ds| / |c|) per-tick                0.365
mean projection of ds onto command dir   0.267
mean|ds| / (mean|ds| + mean|c - ds|)     0.305
```

The last one assumes realised motion is collinear with the command, which it is not (see
the sign agreement above). The projected figure is lower because some of the little motion
that does happen goes sideways. All four land in 0.27–0.37 and the conclusion is identical
either way; **0.33** is the number to quote.

### 2.3 Why this is not automatically fatal

The model is deployed through the *same* controller the labels were recorded on. A policy
that emits these actions reproduces this motion. So the pipeline is **self-consistent**,
and "always push +x a bit" is a perfectly learnable constant. Three things still make it
worth fixing:

- **It makes `dx` one-sided.** `a1`'s `dx` q01 is **−0.08** against the released dataset's
  **−0.679**. Over 8700 frames the policy is essentially never labelled to move in −x. The
  `recover` cohort exists to teach exactly that kind of correction, and the action
  statistics say it does not.
- **It saturates.** `dz` hard-clips at exactly **1.000** on **1.38%** of frames. The
  released data never exceeds **0.938** on any translation channel.
- **It moves the labels off the pretrained prior,** so a fine-tune spends capacity
  relearning an offset instead of learning the task.

---

## 3. The calculation that decided the fix

### 3.1 Model the servo

Treat one joint as a second-order system with effective inertia `M`, driven by
`M*qddot = kp*(q_d - q) - kd*qdot`. Then:

```
natural frequency   wn   = sqrt(kp / M)
damping ratio       zeta = kd / (2 * sqrt(kp * M))
```

For Panda joint 2 with `M ~ 2.5 kg m^2` and menagerie's stock `kp = 4500, kd = 450`:

```
wn   = sqrt(4500 / 2.5)           = 42 rad/s
zeta = 450 / (2 * sqrt(4500*2.5)) = 450 / 212 = 2.1     <- OVERDAMPED
```

`zeta = 2.1` is the whole problem. An overdamped system is *slow*: its dominant time
constant is `tau ~ 2*zeta/wn = 0.10 s`, and empirically the settling time is ~330 ms, which
matches what `PROGRESS.md` measured independently.

### 3.2 The ratio prediction

In steady-state ramp tracking, the arm ends up moving at the *reference* velocity `v`
while sitting a fixed distance behind. Writing `L` for that standing lag and `dt` for the
control tick:

```
label * 0.05 = target(t) - current(t)    = L      = v * tau
realised     = current(t+1) - current(t) = v * dt          (steady state)

ratio = realised / commanded = (v * dt) / (v * tau) = dt / tau
```

**The reference velocity `v` cancels.** This is the key result, and it predicts something
counterintuitive: **slowing the reference trajectory does not improve the ratio.** Both the
command and the realised motion shrink together.

Sanity check against the data: `dt = 0.05 s` (20 Hz), measured ratio 0.33, so
`tau ~ 0.15 s` — consistent with the 0.10 s estimate and the 330 ms settling time.

### 3.3 Confirming it experimentally

`probe_lag.py` runs one full episode per configuration and reports the action statistics.
Reference speed is varied by scaling every waypoint duration by `1/speed`; gains are varied
by writing `actuator_gainprm` / `actuator_biasprm` directly on the loaded `MjModel`.

```
config                       ticks  ratio  cmd_mm real_mm  mean_dx  mean_dz  q01_dx  clip_dz
baseline                       174  0.320    16.7     5.4    0.113    0.107   0.060    0.000
speed 0.5x                     348  0.245    11.1     2.7    0.099    0.121   0.069    0.000
speed 0.25x                    696  0.154     8.9     1.4    0.092    0.128   0.071    0.000
speed 2x                        87  0.364    27.7    10.1    0.141    0.059   0.054    0.103
kd 0.5 (toward critical)       174  0.441    12.3     5.4    0.104    0.120   0.061    0.000
kd 0.35                        174  0.493    11.1     5.5    0.101    0.123   0.059    0.000
kd 0.25                        174  0.531    10.3     5.5    0.099    0.125   0.056    0.000
kp 2x                          174  0.530    10.3     5.4    0.065    0.057   0.029    0.000
kp 2x kd 0.7                   174  0.618     8.8     5.5    0.062    0.060   0.027    0.000
kp 4x kd 0.5                   174  0.950     5.8     5.5    0.039    0.031   0.003    0.000
```

Read three things off this:

- **Slowing the reference makes the ratio worse**, 0.320 → 0.245 → 0.154, exactly as the
  algebra predicts and *opposite* to my first recommendation. The static sag does not scale
  with velocity while the commanded motion does, so the sag's share of the label grows.
  Note `mean_dx` only falls 0.113 → 0.092 across a 4x slowdown: that residual ~0.09 is the
  static sag floor, about 4.5 mm.
- **Speeding it up causes clipping** — `dz` saturates on 10.3% of ticks at 2x speed.
- **Stiffening works and is the only thing that does.** `kp 4x kd 0.5` reaches ratio 0.950
  with commanded per-tick falling 16.7 → 5.8 mm while realised holds at 5.5 mm. The servo
  finally arrives.

### 3.4 Choosing the operating point

`probe_gains.py`, 8 episodes per configuration (4 `reach` + 4 `noise`), also tracking task
success and contact force into the table:

```
config              placed  ratio  mean_dx  mean_dz  q01_dx  q99_dz  clip%   maxFn  pen_mm
baseline              5/8   0.355    0.142    0.128  -0.075   0.972   0.34    0.0 N    0.00
kp2 kd0.7             4/8   0.724    0.084    0.085  -0.181   0.715   0.17    0.0 N    0.00   <- chosen
kp3 kd0.6             4/8   0.899    0.077    0.087  -0.243   1.000   0.65    0.0 N    0.00
kp4 kd0.5             4/8   0.980    0.097    0.105  -0.258   1.000   1.05    0.0 N    0.00
kp6 kd0.45            4/8   1.040    0.121    0.135  -0.333   1.000   1.60    0.0 N    0.00
```

**`kp x2, kd x0.7` is the knee, not the extreme.** Reasoning:

- It more than doubles tracking (0.355 → 0.724) and roughly halves the bias
  (`mean_dx` 0.142 → 0.084).
- It is the first configuration where `dx` is meaningfully **two-sided**: q01 goes
  −0.075 → −0.181. That is the recovery signal that was missing.
- It has the **lowest clipping of anything tested** (0.17%), and `q99_dz` = 0.715, safely
  inside the released data's 0.938 ceiling.
- **Past `kp3` the system starts ringing.** `mean_dx` turns back *up* (0.077 → 0.097 →
  0.121) and `dz` saturates at 1.000. Chasing ratio → 1.0 buys a worse distribution.
- Task success is unchanged within the sample (4–5/8 is the raw rate; the collector uses
  rejection sampling with up to 8 attempts, which is why the shipped datasets are 100%).
- **Contact into the table stays at 0 N with zero penetration** in every configuration,
  because `--min-clearance` still holds the commanded target above the surface. Stiffening
  did not reintroduce the `PROGRESS.md` §19 sinking problem.

### 3.5 Why stiffening is a move toward LIBERO, not a hack

LIBERO drives the Panda through robosuite's `OSC_POSE`, a **force-based operational-space
controller that tracks its Cartesian setpoint tightly.** Our Route A stand-in — solve IK,
command joint positions — realising only a third of its setpoint is the *artefact*. Making
it actually arrive brings the plant closer to the one the checkpoint was trained against.
The gains being changed are menagerie's tuning for a free-standing arm; they were never a
LIBERO reference value.

---

## 4. What changed

### 4.1 `mujoco_menagerie/franka_emika_panda/panda_libero_hand.xml`

Arm actuators, `kp x2` and `kd x0.7`:

```
actuator1,2   gainprm 4500 -> 9000    kd 450 -> 315
actuator3,4   gainprm 3500 -> 7000    kd 350 -> 245
actuator5,6,7 gainprm 2000 -> 4000    kd 200 -> 140
```

`panda.xml` is **untouched**, so the DROID path and the phase-4 dataset are unaffected.

**This changes the plant at inference too, and that is required, not incidental.** The
collector's design rule is that labels are produced through the same controller that will
consume them, so `libero_closed_loop.py` must run on the same gains. It also means `a1/`
and `a2/` were collected on the old plant and are superseded by `a3/`.

### 4.2 `libero/fine_tune/collect_finetune_data.py` — stronger `recover` cohort

`a1`'s recover cohort was one kick of 0.85 in the first 45% of the episode, and it kept
10/10 episodes on the first attempt — the tracker absorbed it, so the cohort taught little.
A single kick during the approach has 100+ ticks to wash out and an empty gripper, so
nothing is at stake.

`a3` kicks **twice**:

```python
RECOVER_KICK         = 0.85   # free-arm, during the approach
RECOVER_KICK_LOADED  = 0.60   # mid-transport, with the ball actually in the gripper
RECOVER_KICK_WINDOWS = ((0.08, 0.40), (0.55, 0.80))
```

The second kick is the point. It lands where the observed closed-loop failures live —
`PROGRESS.md` §18, grasp achieved and retention not, run degenerating from chunk 5. A
loaded kick can genuinely drop the ball, rejection sampling then discards that episode, and
the attempt cost is the honest signal that the disturbance is doing work. It is smaller
than the free kick deliberately: at 0.85 with a ball in the gripper nearly every episode
fails and the cohort collects nothing.

### 4.3 `NOISE_SIGMA_POS` 0.15 -> 0.08, forced by the gain change

Sigma perturbs the **commanded** action, but what knocks the arm off the reference is the
perturbation that gets **realised** — and the new plant realises 72% of a command where the
old realised 33%. The same sigma therefore delivers more than twice the physical
disturbance per tick, compounding over 174 ticks. The first `a3` attempt at 0.15 **dropped
7 of 10 noise slots** after exhausting 8 attempts each.

Calibrated with 10 noise episodes per sigma on the new plant:

```
sigma   placed   q01 of dx    P(slot filled | 8 attempts)
0.05     5/10     -0.052           99.6%
0.07     5/10     -0.085           99.6%
0.09     3/10     -0.116           94%
0.11     0/10     -0.148           ~0
0.15     1/10     -0.215           <1%      <- a1's value
```

0.08 with `--max-attempts-per-episode 12`. Result: 30/30 slots filled, 14 rejected
attempts, all of them in `noise`.

### 4.4 `--selftest` now goes through `reset_episode`

So the guard reports the state an episode actually starts from, with objects placed and the
arm settled, rather than a hand-rolled near-equivalent. Both paths agree to <0.1 mm; this is
about the guard being the real thing.

---

## 5. How every number here was produced

All of it is CPU MuJoCo and costs nothing to re-run. No GPU, no Modal, no inference.

| what | how |
|---|---|
| dataset structure, dtypes, row counts | `pyarrow.parquet.read_table`, compare Arrow schema and `huggingface` metadata against the released dataset |
| action / state distributions | stack the `fixed_size_list` columns to `(N, 7)` / `(N, 8)`, take per-channel min/max/mean/std/quantiles |
| released reference distribution | `curl` `meta/stats.json` and `meta/info.json` from `huggingface.co/datasets/allenai/MolmoAct2-LIBERO-Dataset` — the real thing, not a remembered figure |
| commanded vs realised | per tick, `norm(action[t][0:3]) * DELTA_POS_SCALE` against `norm(state[t+1][0:3] - state[t][0:3])`, both straight out of the parquet |
| image orientation | decode the inline PNG bytes, tile 6 timesteps x 2 cameras into one strip, and **look at it** (`PROGRESS.md` §8: never conclude an orientation from numbers) |
| static sag | set `qpos[:7] = LIBERO_INIT_QPOS`, `mj_forward` for pure FK, then `mj_step` to convergence, and difference the two |
| settle convergence | same, sampled at 50/100/200/400/800/2000 steps — converged by 50 at both gain settings, so 200 in `reset_episode` is ample |
| gain sweeps | write `actuator_gainprm` / `actuator_biasprm` on the loaded `MjModel` and re-run whole episodes; **absolute** values, not multipliers |
| contact safety | `mj_contactForce` per contact per physics step, filtered to the table geom, tracking max normal force and worst `contact.dist` |

### 5.1 A mistake worth recording

The first sag sweep applied gain *multipliers* to a model that had **already** been edited
to the new gains, so every row was mislabeled by one step and it looked as though
stiffening made the reset match *worse*. Caught by printing the actually-loaded
`gainprm` and re-running with absolute values. This is the same class of error as
`PROGRESS.md` §7 (counting the green bin as the ball) and §19 (using a bounding-sphere
radius as a mesh's lowest point): **the measurement apparatus was wrong, not the system.**
When a result contradicts a mechanism you have already derived, suspect the probe first.

---

## 6. The other problem, which droop does not fix: normalisation

Independent of everything above, and confirmed by reading the training code rather than
guessing.

`molmoact2/experiments/launch_scripts/train_lerobot.py:298` defaults to
`--norm_mode q01_q99`, and `lerobot_utils/stats.py:417` `_collect_tagged_stats` builds the
normaliser for a tag from the `LeRobotDatasetMetadata` of **whatever repos are in the
mixture**. Fine-tune on `a3` alone and the `libero` tag's normaliser is rebuilt from our
episodes, discarding the pretrained calibration. Ratio of our q01–q99 span to the released
one, per channel, on `a1`:

```
dx    dy    dz    drx   dry   drz   grip
0.51  0.67  0.81  0.34  0.24  0.18  1.00
```

Inference stays self-consistent — `train_lerobot.py:715` preserves the normalisation
metadata in the saved config — but the pretrained flow-matching head's calibrated output
scale is thrown away, so the adapter spends capacity relearning an affine map.

**Recommended:** mix `allenai/MolmoAct2-LIBERO-Dataset` into the `libero` tag alongside
`a3`. At 273k frames against our ~5k it dominates the quantiles, restoring the pretraining
normalisation, and it doubles as replay against catastrophic forgetting. The alternative is
to inject the released `stats.json` directly.

### 6.1 Rotation is genuinely under-represented, and stiffening does not help

Our task holds one top-down orientation from reset to release; LIBERO's tasks reorient the
wrist. Rotation q01–q99 spans are **3–6x** narrower than released. This is real, not a bug,
and it means the fine-tune will not improve anything rotational — and if trained long
enough to overfit, the rotation channels are where it will collapse toward zero first.

### 6.2 `fps` is still unresolved

The released dataset declares `fps: 10`; LIBERO's env and our loop both run at 20 Hz.
`a2/` was built at 10 Hz specifically to test this and remains the way to answer it.

---

## 6.5 What `a3` actually came out like — including what did NOT work

30 episodes (10 reach / 10 noise / 10 recover), 5220 frames, 132 MB, 918 s, 14 rejected
attempts, no dropped slots. Every structural check that `a1` passed, `a3` passes.

```
                        dx       dy       dz
ACTION MEAN
  released            0.063    0.087   -0.090
  a1 (old plant)      0.174    0.001    0.127
  a3 (new plant)      0.098    0.000    0.072
ACTION q01
  released           -0.679   -0.774   -0.873
  a1 (old plant)     -0.080   -0.567   -0.473
  a3 (new plant)     -0.072   -0.307   -0.258
q01-q99 SPAN, RATIO TO RELEASED
  a1 (old plant)      0.510    0.669    0.813
  a3 (new plant)      0.376    0.372    0.439
```

**What worked.** The droop bias roughly halved on both channels (`dx` 0.174 → 0.098, `dz`
0.127 → 0.072), moving toward the released distribution. Spread of the per-episode bias
fell 0.080 → 0.053. Saturation past the released 0.938 ceiling collapsed from **1.38% to
0.06%**. Sign agreement is essentially unchanged (0.70/0.72/0.76).

**What did NOT work, and it was a reasoning error on my part.** Part of the justification
for the gain change was that `q01_dx` improved −0.075 → −0.181 in the `probe_gains` sweep,
i.e. the data would finally contain −x motion. **It came out at −0.072, unchanged from
`a1`.** The reason is §4.3: that probe ran at sigma 0.15, stiffening the plant made sigma
0.15 unfillable, and cutting it to 0.08 removed exactly the coverage the probe was
measuring. **The two-sidedness came from the noise magnitude, not from the stiffening.**
The two fixes fight each other, and this trade gave away the thing it was meant to buy.

**And one thing got worse.** All span ratios narrowed (`dx` 0.510 → 0.376). This is the
direct consequence of the servo no longer needing to be over-commanded 3x: commanded
magnitudes shrank, which is *physically correct* — the labels now mean what they say — but
it puts our distribution further inside the released one. **That makes the normalisation
problem in §6 more severe, not less.** Pinning the released stats is now more important
than it was for `a1`, not less.

**So the honest scoreboard:** the constant offset and the saturation are fixed, the
one-sided `dx` is not, and the normalisation gap widened. `a3` is a better dataset than
`a1` on two axes out of three.

### 6.6 How to actually get two-sided `dx`, since sigma cannot

Options, none of them yet tried:

- **Put retreat into the reference trajectory itself.** A deliberate back-off-and-
  re-approach segment produces negative `dx` in the *expert* labels, with no dependence on
  perturbation surviving rejection sampling. This is the cheap, direct fix and is what I
  would do next.
- **A high-sigma cohort with a large attempt budget,** accepting ~10% yield to buy tail
  coverage. Expensive in wall clock, but it is free CPU time.
- **Relax rejection for the noise cohort only** — a run that ends off-target still carries
  valid corrective labels on the way. This contradicts the collector's "a demonstration
  that fails the task is a demonstration of failing the task" rule, so it is a deliberate
  policy change, not a tweak.

---

## 7. Status of the datasets

| dataset | plant | contents | status |
|---|---|---|---|
| `a1/` | stock gains | 20 reach / 20 noise / 10 recover, 8700 frames | **superseded** — carries the droop bias |
| `a2/` | stock gains | 8 episodes at 10 Hz | **superseded** as data; still the `fps` probe design |
| `a3/` | `kp x2 kd x0.7` | 10 reach / 10 noise / 10 recover, 5220 frames | current, see §6.5 |

`a1`'s format was independently re-validated and is correct — v3.0, `video_path: null`,
inline PNG in `struct<bytes, path>` typed `Image`, 8-D state as
`[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]` (the `names` field in `info.json` is
legacy and wrong), mirrored `(+x, −x)` gripper, `int64` index columns, contiguous
`frame_index`, upright images. **The format was never the problem.** `a3` uses the same
writer.

---

## 8. Still open

- ~~Nothing has been run in closed loop since §21's scene corrections~~ **RESOLVED, and the
  answer is no.** See §9.
- **Normalisation must be pinned** (§6) before any training run. §6.5 makes this more
  pressing, not less — `a3`'s action span is narrower than `a1`'s relative to released.
- **`dx` is still one-sided** (§6.5, §6.6). The stiffening did not buy this and sigma
  cannot. Putting a retreat segment into the reference trajectory is the direct fix.
- **The `recover` cohort is still too easy.** With two kicks, one of them loaded
  mid-transport, it kept 9/10 slots on the FIRST attempt (10/10 within two). That is the
  same signal `a1` gave with one kick: the reference tracker absorbs the disturbance. If
  this cohort is meant to cover genuinely hard excursions, the kick has to get bigger or
  land later, and the honest read right now is that it is not doing much work.
- The `fps 10` vs 20 Hz question (§6.2).
- Transport is ~4x inference cost in the closed loop; encoding frames before POSTing is
  still the un-done latency fix.

---

## 9. The stock checkpoint on the corrected scene: measured, and it still fails

This was the gate everything else sat behind — the claim in §8 and in
`fine_tune/README.md` §9 that the closed-loop run should happen *before* fine-tuning,
because if the 100 mm table error was the real cause of the §17–19 failures then the
existing checkpoint might come good on its own. **It has now been run, and it does not.**

Three runs against the deployed `allenai/MolmoAct2-LIBERO` (L4, `norm_tag: libero`
confirmed at `/act`), on the corrected scene AND the stiffened plant:

```
run              chunks  best lateral   ball lift   gripper close   eef z range
fixedplant_01      15       44.9 mm       0.00 mm        6%         17-180 mm
fixedplant_02      12        7.2 mm     126.90 mm       30%         18-225 mm
fixedplant_03      12       32.6 mm       0.00 mm       24%         16-204 mm
```

**0/3 task success. 1/3 grasp-and-lift.** Statistically indistinguishable from §18's
result on the *old* scene (1 lift in 2 runs, no placement). Remember §18's warning: the
action expert is flow-matching, so it samples, and three runs is a small sample — but
three runs producing zero placements is enough to say the scene fix alone did not solve
the task.

**What genuinely improved.** eef z now arrests at **16-18 mm above the table in every
run** and never penetrates. The §17-19 signature — descend past the ball, jam into the
surface, then slide across it — is gone. So §21's 100 mm table correction was real and
load-bearing. It simply was not sufficient.

**What is still broken**, and it is the same two things as before:

- **The gripper barely closes.** 6% / 30% / 24% of actions across the three runs. In the
  failing runs it is essentially never commanded shut, exactly the signature §15 flagged.
- **Lateral approach stalls.** Best approach was 44.9 / 7.2 / 32.6 mm. Only the middle run
  got inside grasping range of a 40 mm ball, and that is the one that lifted it. The other
  two never got closer than 33 mm and then diverged monotonically — 444 mm and 229 mm
  final lateral error, with the arm reaching joint limits (`IK unreached 10/10`).

**Consequence for the plan.** The gate is closed: the fine-tune is required, not optional.
§20 already proved the checkpoint and the serving path are both fine by scoring 3/3 on a
real LIBERO task through robosuite's OSC, so the remaining gap is that our green-ball scene
is out of the checkpoint's distribution. That is what a fine-tune is for. The two
outstanding data problems in §6 and §6.5 — pinning the normalisation, and the one-sided
`dx` — are now the critical path.

### 9.1 A frame error of mine while reading these logs

The first pass at this table reported eef heights around **-720 mm**, i.e. most of a metre
below the table. That was nonsense: `eef_pos_after` in the run log is in OUR world frame
(table top `TABLE_TOP_Z = -0.012`), and I subtracted LIBERO's frame table height (0.900)
from it. Same class of mistake as §5.1 and as `PROGRESS.md` §13's `pinch`-site reading —
**mixing two coordinate frames that both look plausible.** The corrected column is the one
above. When a height comes out impossible, check the frame before believing the system is
broken.
