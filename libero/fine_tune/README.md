# `libero/fine_tune` — training data for a LoRA + action-expert fine-tune

> **Pick target changed 2026-08-03: 40 mm sphere → 40 mm cube**, instruction now *"pick up
> the green box and put it in the green container"*, scene identifiers renamed
> `green_ball*` → `green_box*`. Rationale and full change list in
> [`libero/PROGRESS.md` §26](../PROGRESS.md).
>
> **Everything measured on this page was measured on the ball**, including every dataset and
> checkpoint it names. The prose below has deliberately NOT been reworded — the measurements
> describe a sphere and would be false if restated about a cube. Command lines HAVE been
> updated, since the old flag names no longer exist.

Scripted expert demonstrations of the green-ball pick-and-place task, recorded in the
**LIBERO** convention and written as a **LeRobot v3.0** dataset that matches
`allenai/MolmoAct2-LIBERO-Dataset` field for field.

```
libero/fine_tune/
  README.md                    this file
  collect_finetune_data.py     the collector
  lerobot_v30_writer.py        LeRobot v3.0 writer (inline PNG, no mp4)
  a1/                          the requested dataset: 20 reach + 20 noise + 10 recover
  a2/                          a second dataset, 10 Hz — see "Why a2 exists"
```

---

## 1. Read this first: the scene was wrong, and it is now fixed

Before generating anything I checked the released dataset and a live LIBERO environment to
find out what the training format actually is. That turned up **four measured errors in our
own scene**, all of which would have been baked permanently into the dataset. They are
fixed, and the fixes change `libero_closed_loop.py`'s behaviour, so they are the first
thing to know.

Ground truth, measured from `OffScreenRenderEnv` across all four suites:

| quantity | LIBERO (measured) | ours, before | ours, now |
|---|---|---|---|
| eef at reset, relative to `link0` | `(0.4515, 0, 0.2613)` | `(0.4585, 0, 0.0891)` | `(0.4515, 0, 0.2608)` |
| eef axis-angle at reset | `(3.1408, 0.0018, -0.0899)` | `(2.1557, 2.1557, 0.1373)` | `(3.1403, 0, -0.0892)` |
| table top, relative to `link0` | `-0.012` | `-0.112` | `-0.012` |
| eef height above table at reset | `0.2733` | `0.2011` | `0.2728` |

### 1.1 The table was 100 mm too low

The scene put the table top 0.112 m below `link0`, from reading that LIBERO's table is at
`z=0.800` with the robot on a 0.912 m pedestal. A live env says `link0` is at 0.912 and the
table top at **0.900** — the surface is only **12 mm** below the base, not 112 mm.

This is the most consequential of the four. The policy commands a descent sized for a
surface ~0.27 m below the reset pose; ours was 0.10 m further down than that. The
descent ends in free space, the model sees it has not arrived, and it keeps commanding
down — which is precisely the "it picks up the ball but goes into the table" behaviour
from `PROGRESS.md` §17–19. **This is the best current explanation for that symptom**, and
it is a scene bug, not a controller bug.

Fixed in `scene_libero_hand.xml` and `scene_libero.xml`: table body `-0.512 → -0.462`,
half-height `0.4 → 0.45` (so the base still rests on the floor), and every object and bin
raised 100 mm with it. `TABLE_TOP_Z` in `libero_closed_loop.py` is now `-0.012`.

### 1.2 `grip_site` was 9.5 mm out and yawed 90°

`panda_libero_hand.xml` had `pos="0 0 0.1065" quat="0.707107 0 0 -0.707107"`. Sweeping the
site against the measured ground truth:

| site | position error | orientation error |
|---|---|---|
| `pos 0.097`, no quat | **0.5 mm** | **0.002 rad** |
| `pos 0.097`, `Rz(-90)` | 0.5 mm | 2.418 rad |
| `pos 0.097`, `Rz(+90)` | 0.5 mm | 5.842 rad |

So the frame needs **no** extra rotation — menagerie's −45° hand mount and robosuite's
`right_gripper` chain already agree — and the `Rz(-90)` that was there put a spurious 90°
yaw on **every `eef_axisangle` we have ever reported to the model**.

The `0.1065` was a deliberate trade (§16) to preserve a pad-centre relationship. It is the
wrong side of the trade: it costs 9.5 mm of error in the reported eef height, which is what
the model conditions on and servos to, to gain 9.5 mm of accuracy in pad geometry, which
only our own clamp heuristic reads. Now `0.097`, robosuite's literal value.

### 1.3 The reset pose was not LIBERO's

`LIBERO_INIT_QPOS` held robosuite's generic `Panda.init_qpos`. LIBERO resets to different
joints (identical in all four suites): `[0, -0.16103739, 0, -2.44459747, 0, 2.2267522,
0.78539816]`. The old pose started the eef 0.201 m above the table where LIBERO's starts it
0.273 m up — a 72 mm head start on being out of distribution before the first action.

### 1.4 `LIBERO_ORIGIN_OFFSET` had the wrong x

`-0.56 → -0.6`. `link0` sits at `x = -0.6` in every suite.

### 1.5 Consequences

`--min-clearance` default moved `0.013 → 0.016`, re-measured: with the site at 0.097 it now
rides **15.5 mm** above the lowest point of the hand (it was 6–12 mm at 0.1065, and the
whole difference is that move). A 16 mm floor leaves ~0.5 mm of hardware clearance at the
limit while still permitting a descent 4 mm past the ball's equator.

Verify any time with:

```bash
MUJOCO_GL=egl uv run python libero/fine_tune/collect_finetune_data.py --selftest
```

---

## 2. What the fine-tune engine actually requires

Read out of the vendored repo and the released dataset, not assumed.

### 2.1 The mixture entry

`molmoact2/experiments/launch_scripts/data_mixtures.py:319` (`build_molmoact2_libero`):

| field | value |
|---|---|
| `action_key` | `action` |
| `state_keys` | `["observation.state"]` |
| `camera_keys` | `["observation.images.image", "observation.images.wrist_image"]` |
| `normalize_gripper` | `False` |
| `action_horizon` / `n_action_steps` | `10` / `10` |
| `control_mode` | `delta end-effector pose` |
| `setup_type` | `single franka robotic arm in libero` |

### 2.2 The on-disk format

`lerobot_dataset.py:83` pins `CODEBASE_VERSION = "v3.0"` and raises
`BackwardCompatibilityError` on anything older.

```
meta/info.json                            codebase_version v3.0, video_path null
meta/tasks.parquet                        [task_index int64, task string], pandas index=task
meta/stats.json                           dataset-wide, with q01/q10/q50/q90/q99
meta/episodes/chunk-000/file-000.parquet  one row per episode + flattened per-episode stats
data/chunk-000/file-000.parquet           ALL episodes concatenated, rolled over at 100 MB
```

Two traps, both of which the repo-root `droid/lerobot_writer.py` falls into:

- **It writes v2.1.** That path only works via `droid/phase4_modal_train.py::convert`.
- **It writes mp4 video features.** The released LIBERO dataset has `"video_path": null`
  and stores every frame as **PNG bytes inline in the parquet**, in a
  `struct<bytes, path>` column typed `{"_type": "Image"}` by HuggingFace `datasets`.

`lerobot_v30_writer.py` here does it correctly. Its output validates against the released
dataset: identical column names, identical Arrow types, byte-identical `huggingface` schema
metadata, and matching key sets in `info.json`, `stats.json`, `tasks.parquet` and
`meta/episodes/`.

### 2.3 `observation.state` is 8-D, and `info.json` lies about what is in it

The released `info.json` names the state channels
`["x","y","z","rx","ry","rz","rw","gripper"]`, which reads as position + **quaternion** +
one gripper value. **That is wrong.** Decoding the actual parquet:

```
observation.state[0] = [-0.0534, 0.0070, 0.6783, 3.1408, 0.0018, -0.0899, 0.0388, -0.0388]
                        \__ eef_pos __/  \___ axis-angle ___/  \__ gripper_qpos __/
```

`‖state[3:6]‖ = 3.142` — an axis-angle of magnitude π, not a unit quaternion — and
`state[6:8]` is the mirrored `(+x, −x)` finger pair. So the layout is
`[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]`, matching
`env_processor.py`'s `LiberoProcessorStep` and matching what `libero_closed_loop.read_state`
already produces. The `names` field is legacy. Trust the data.

### 2.4 Images are stored upright

`LiberoProcessorStep` rotates env frames 180° "to match the HuggingFaceVLA/libero camera
orientation convention", which had left `--image-flip` unresolved. Decoding both settles it:

- raw robosuite `agentview` is **upside down**
- the stored dataset frame is **upright**

So the processor's 180° turns raw robosuite into upright, and **upright is what the model
consumes**. `mujoco.Renderer` already returns upright frames, so the correct setting for our
loop is **`--image-flip none`** — which is already the default. This corrects `PROGRESS.md`
§20, which recorded the conclusion as "180" by reading the processor without checking which
orientation its *input* was in.

### 2.5 There is no canonical absolute frame, and it matters less than we thought

Measured `link0` z: **0.912** in `libero_spatial` and `libero_goal`, **0.0** in
`libero_object`, **0.42** in `libero_10`. The suites do not share a world origin, which is
why the released dataset's eef z spans **0.008 … 1.366** (q01 0.519, q50 0.719, q99 1.043).

What *is* invariant across all four: eef at reset is `(0.4515, 0, 0.2613)` from `link0`, and
the table top is `link0 − 0.012`. **The model cannot be keying on absolute world z.** Our
`LIBERO_ORIGIN_OFFSET` therefore only has to land somewhere sane in that spread; it now
reproduces the spatial/goal frame exactly.

The same is true of the camera: `agentview` measured link0-relative is `(1.497, 0, 0.650)`
in `libero_object` and `(1.319, 0, 0.698)` in `libero_spatial`, with different pitches.
There is no single pose to match, so ours is left at the framing §11 measured its 16 px ball
at — raised 100 mm with the table so the view of the work surface is unchanged.

---

## 3. How the demonstrations are built

### 3.1 The one design rule

**Every recorded action is produced by, and executed through, `libero_closed_loop`'s own
`apply_action`** — the exact function that will consume the fine-tuned model's output at
inference. Nothing here re-implements the controller.

> **THIS RULE IS CURRENTLY VIOLATED, 2026-07-28.** `libero_closed_loop.py` now defaults
> to `--control-mode osc` (a port of robosuite's `OSC_POSE`, writing joint torques —
> `PROGRESS.md` §22), while `a1`–`a4` were all collected through `apply_action`, the
> Route A IK path. Training on `a4` and serving through OSC is exactly the mismatch this
> section exists to prevent.
>
> **`a4` must be regenerated against OSC before the next fine-tune.** It is free — local
> CPU, ~20 min — but it is not a re-run with the same flags:
> - the collector must call the OSC controller, not `apply_action`;
> - **`NOISE_SIGMA_POS` has to be re-calibrated from scratch.** README §4.3 cut it
>   0.15 → 0.08 because a stiffer plant realised more of each perturbation per tick; OSC
>   realises ~12.3% per tick where Route A managed ~33%, so the same sigma delivers a
>   different physical disturbance again. Carrying 0.08 over would be repeating the
>   original mistake with new numbers.
> - the `--min-clearance` clamp is off under OSC, so `reach` and `noise` episodes will no
>   longer log clamped ticks (§5's table) — compliance handles it instead.
>
> The *format* work in this file is unaffected: v3.0, inline PNG, the 8-D state layout,
> the mirrored gripper pair, the image orientation. Only the episodes change.

That is the whole point. §20's diagnostic scored 3/3 on a real LIBERO task through
robosuite's OSC, which proved the checkpoint and our serving are both fine and located the
failure in our environment. A fine-tune can absorb a controller mismatch, but **only** if
the actions it trains on are the actions that produce the recorded motion *in our
controller*. So the expert is a Cartesian reference trajectory, and the label at each tick
is the normalised delta from the arm's **actual** pose to that reference:

```python
dpos = (target_pos - current_pos) / DELTA_POS_SCALE      # 0.05 m per unit
drot = orientation_error(target_mat, current_mat) / DELTA_ROT_SCALE   # 0.5 rad per unit
action = [clip(dpos, -1, 1), clip(drot, -1, 1), gripper]
```

This is closed-loop, not a replayed open-loop plan. Servo lag, IK error and the table clamp
all end up *inside* the labels, which is where they have to be for the model to learn to
compensate for them.

### 3.2 The reference trajectory

Eleven segments: hover → descend → **dwell** → close → lift → transport → **dwell** →
lower → **dwell** → release → retreat. 7.6 s, 152 ticks at 20 Hz.

The dwells are not padding. The arm is a position servo with ~330 ms of settling, so a
waypoint reached *in the plan* is not a waypoint reached *by the hardware*. The first
calibration run had no dwells and got **12/15 grasps but only 5/15 placements** — the
gripper was opening while the arm was still short of the bin and moving, so the ball was
thrown rather than placed.

The grasp target is the **ball centre**, not an offset above it. That only became correct
once `grip_site` moved to 0.097: the site is now robosuite's own grasp point, so aiming it
at the object centre is exactly what LIBERO's demonstrations do.

### 3.3 A physics bug the dwells did not fix

Even with dwells and smoothstep easing, every demo lifted the ball and then dropped it
**0.10–0.12 m short of the bin**. Tracing one episode tick by tick:

```
t 65-81  fingers (0.0195, 0.0195)   <- correct: ball radius is 0.020
t 82     fingers (0.0188, ...)      <- still closing
t 86     fingers (0.0107, ...)      ball starts falling
t 88     fingers (0.0039, ...)      ball on the table, 154 mm from the site
t100+    fingers (0.0000, 0.0000)   <- fully closed on nothing
```

A 40 mm sphere between two flat pads has **no form closure** — it is held by friction
alone — and with menagerie's default contact softness a sustained squeeze slowly extrudes
it. Fixed on the ball geom:

```xml
priority="2" condim="4" friction="2 0.05 0.0001"
solref="0.004 1" solimp="0.99 0.999 0.001"
```

`priority=2` outranks the pads' `priority=1` so these parameters win the contact; `condim=4`
adds the torsional friction a sphere on flat pads needs to not roll out; the stiff
`solref`/`solimp` stops the squeeze at the surface. Placement went **5/15 → 5/5**.

The gripper `forcerange` was also changed from menagerie's ±100 N to robosuite's ±20 N,
but **that is not what fixed it** — swapping it produced bit-identical trajectories,
because a seated grasp never approaches either limit. It is kept because it is the correct
value, not because it did anything here.

### 3.4 Randomisation

- **Ball position** uniform in `x ∈ [0.46, 0.66]`, `y ∈ [-0.12, 0.12]`.
- **Bin layout** — the three bins are shuffled across three anchor slots with ±20 mm
  jitter each episode, so the model has to *look* for the green one rather than memorise a
  coordinate. The bins are welded, so this is written straight into `model.body_pos`.

### 3.5 Rejection sampling

An episode is kept only if the ball is lifted **and** ends inside the green bin. Failed
attempts are discarded, never written — including in the perturbed cohorts, where the point
is a *recoverable* excursion, not an unrecoverable one. If the attempt budget runs out the
slot is left empty and reported, rather than filled with a bad episode.

---

## 4. The cohorts in `a1`

| cohort | n | what it is |
|---|---|---|
| `reach_00` … `reach_19` | 20 | Nominal. Randomised ball and bin layout, LIBERO's reset pose, no perturbation. |
| `noise_00` … `noise_19` | 20 | DART-style noise injection. |
| `recover_00` … `recover_09` | 10 | My choice — large start jitter plus one hard disturbance. |

Cohort membership is recorded in the sidecar `meta/cohorts.json` (episode index → name,
cohort, and per-episode outcome), which is **not** part of the LeRobot spec and is kept out
of the parquet on purpose: adding columns the pretraining schema does not have is exactly
the silent divergence this writer exists to avoid.

### 4.1 `noise` — why the label is not the noisy action

Gaussian noise is added to the **executed** action; the **recorded label is the clean
expert action at the state the noise produced**:

```python
label    = expert_action(current_pose -> reference)     # <- recorded
executed = label + N(0, sigma)                          # <- actually applied
```

This is the standard noise-injected BC / DART construction, and it is the point of the
cohort. Plain behaviour cloning only ever shows the model states its own expert visits, so
at test time the first small error takes it somewhere it has no training signal for, and
errors compound — the drift in §17–19. Injecting noise into execution and labelling with
the *corrective* action pairs off-trajectory states with the action that fixes them.
Recording the noisy action instead would simply teach the model to be noisy.

`sigma = 0.15` on translation (7.5 mm/tick) and `0.04` on rotation (0.02 rad/tick). The
released LIBERO actions have a per-channel std around 0.33, so this is visible but a
minority of the signal. **The gripper channel is deliberately not perturbed** — flipping it
mid-grasp does not produce a recoverable state, it produces a dropped ball.

### 4.2 `recover` — the 10 I chose

Start-pose jitter of 0.09 rad per joint (vs 0 for the other cohorts), plus **one** near
full-scale disturbance (0.85 in action units, random direction) at a random tick in the
first 45% of the episode, then the expert drives back and completes the task.

Same motivation as `noise`, different part of the distribution: noise covers the
small-perturbation regime densely, this covers the tail. Our observed failures are tail
failures — the arm ends up somewhere the expert never goes and never comes back — so the
tail is worth sampling explicitly rather than hoping enough Gaussian draws stack up.

I considered a held-out validation split instead and rejected it: with 50 episodes of one
task, a 10-episode holdout costs 20% of the training signal to measure something the
closed-loop run in `libero_closed_loop.py` measures better and for real.

---

## 5. What was actually generated

`a1`, seed 0: **50 episodes, 8700 frames, 174 ticks each, 218 MB, ~9 min** on CPU. All
50 slots filled; nothing was dropped for running out of attempts.

| cohort | kept | attempts spent | mean table-clamped ticks | episodes touching the clamp |
|---|---|---|---|---|
| `reach` | 20 | 21 | 0.0 | 0 / 20 |
| `noise` | 20 | 29 | 8.0 | 20 / 20 |
| `recover` | 10 | 10 | 0.2 | 2 / 10 |

Three things worth reading off this table:

- **`reach` never touches the table clamp, in any episode.** With the corrected table height
  the scripted expert has no reason to descend past `table + 16 mm`, so the clamp is now
  genuinely a safety net rather than something that fires constantly. Under the old scene
  the clamp fired on up to 10/10 ticks of a chunk (§19).
- **`noise` touches it in every episode, ~8 ticks of 174.** Expected and wanted: that is the
  perturbation pushing the arm toward the surface and the clamp catching it, and those ticks
  are labelled with the corrective action. It is exactly the situation the cohort exists to
  teach.
- **`recover` succeeded on the first attempt 10/10 times.** Honest read: **the disturbance is
  probably too easy.** A single 0.85-magnitude kick during the approach, with 100+ ticks left
  to fix it, is well inside what the reference tracker absorbs. If this cohort is meant to
  cover genuinely hard excursions, raise `RECOVER_KICK`, kick more than once, or move the
  kick later — into the descent or the transport, where there is less time and a loaded
  gripper. I left it as generated rather than tuning it blind, but do not read 10/10 as
  evidence the cohort is doing much work.

`a2`, seed 1: 8 episodes, 696 frames, 87 ticks each (10 Hz), 17 MB, ~40 s.

### 5.1 Validation

Both datasets pass, checked programmatically against the downloaded released dataset:

```
data columns + Arrow types identical to released     PASS
huggingface schema metadata byte-identical           PASS
meta/episodes columns + types identical              PASS
meta/tasks columns identical                         PASS
info.json v3.0, video_path null                      PASS
stats.json key set identical                         PASS
state 8-D / action 7-D                               PASS
actions inside [-1, 1]                               PASS
axis-angle magnitude ~pi (top-down grasp)            PASS
gripper_qpos mirrored (+x, -x)                       PASS
row count == info.total_frames                       PASS
cohorts sidecar covers every episode                 PASS
every episode placed the ball                        PASS
```

The Arrow-type check is not redundant with the column-name check. A first pass matched
every column name while typing `stats/frame_index/min` (and `index`, `episode_index`,
`task_index`) as `double` where the released dataset uses `int64` — a divergence that
would have surfaced as a cast error inside the loader rather than as anything legible.

---

## 6. Distribution check against the released dataset

Per-channel action std, ours vs a representative released episode:

| channel | ours | released |
|---|---|---|
| dx, dy, dz | 0.19, 0.22, 0.38 | 0.17, 0.40, 0.32 |
| droll, dpitch, dyaw | 0.015, 0.013, 0.014 | 0.035, 0.037, 0.037 |
| gripper | 0.93 | 1.00 |

Translation and gripper sit right on top of the released distribution — a useful check that
`DELTA_POS_SCALE = 0.05` and the reference speeds are sane, since getting that scale wrong
would show up here as a factor-of-N offset.

**The rotation channels are ~2.5× smaller, and that is real, not a bug.** Our task holds
one top-down orientation from reset to release; LIBERO's tasks reorient the wrist. The
fine-tune will therefore see very little rotational signal, so do not expect it to improve
anything rotational — and if the adapter is trained long enough to overfit, the rotation
channels are where it will first collapse toward zero.

State `z` spans 0.915 … 1.201 against the released q01/q50/q99 of 0.519 / 0.719 / 1.043
(min 0.008, max 1.366). We sit at the upper end of the spread but inside it, which is the
expected consequence of §2.5 — the frame reproduces `libero_spatial`/`libero_goal`, the
higher of the suite origins.

---

## 7. Why `a2` exists

`a2` is the same collector at **10 Hz** instead of 20 Hz. It is a probe of a discrepancy I
found and could not resolve from the data alone:

**The released dataset declares `fps: 10`. LIBERO's robosuite env runs `control_freq = 20`,
and our loop runs at 20 Hz (decimation 25).**

If the demonstrations were subsampled by 2 before release, then one dataset action
corresponds to 0.1 s of motion, and replaying it at 20 Hz executes the same displacement in
half the time — i.e. everything the policy commands happens at double speed. That is a
plausible contributor to overshoot into the table, and it is cheap to test: fine-tune on
`a1` (20 Hz) and on `a2` (10 Hz) and compare.

I did **not** change the main loop's rate on this basis. It is a hypothesis with one piece
of evidence, and §20's diagnostic succeeded 3/3 while stepping the env at 20 Hz, which
argues against it mattering much. `a2` exists so the question can be answered instead of
argued.

`a2` is 8 episodes (4 reach / 2 noise / 2 recover) — enough to compare, not enough to spend
another hour of wall clock on.

---

## 8. Regenerating

```bash
# the requested dataset
MUJOCO_GL=egl uv run python libero/fine_tune/collect_finetune_data.py \
    --out libero/fine_tune/a1 --reach 20 --noise 20 --recover 10 --seed 0

# the 10 Hz probe
MUJOCO_GL=egl uv run python libero/fine_tune/collect_finetune_data.py \
    --out libero/fine_tune/a2 --control-hz 10 --reach 4 --noise 2 --recover 2 --seed 1

# frame checks against the measured LIBERO ground truth
MUJOCO_GL=egl uv run python libero/fine_tune/collect_finetune_data.py --selftest
```

Roughly 10 s per episode attempt, single-threaded, CPU only.

---

## 8.1 `a5` is slow, and so is anything trained on it (2026-07-31)

The first SmolVLA fine-tune picks the ball up and places it, but takes ~54 action chunks
to do it. That is not the policy: `a5` episodes are **539 ticks = 27 s each**, ~190 ticks
of that before the gripper even closes, and the policy reproduces its expert faithfully.

`a5` was collected with `OSC_SPEED_SCALE = 2.5`, which multiplies every segment duration.
It exists because the label is the tracking lag over `DELTA_POS_SCALE`, and OSC realises
only 12.3% of a commanded delta per 20 Hz tick, giving

```
v_max = DELTA_POS_SCALE * realised / dt = 0.05 * 0.123 * 20 = 0.123 m/s
```

which the ik-era timings exceed on the retreat (0.27 m/s), pinning `dx` at the ±1 bound.
The clipping was real. The clock was the wrong knob for it — the ceiling is set by the
action **scale**, so raising the scale raises the arm's top speed instead of lowering the
expert's. Both `collect_finetune_data.py` and `libero_closed_loop.py` now take
`--delta-pos-scale` (default LIBERO's 0.05, for stock-checkpoint compatibility);
`L.FINE_TUNE_DELTA_POS_SCALE = 0.20` is the paired value, and `a6` uses it. Measured,
4 episodes per setting, all placed:

| scale | timings | ticks/ep | dx saturated | dx q01 |
|---|---|---|---|---|
| 0.05 (`a5`) | speed-scale 2.5 | 539 | 3.07% | −1.000 |
| 0.125 | hand-set | 216 | 2.20% | −1.000 |
| 0.15 | hand-set | 216 | 0.93% | −0.987 |
| 0.15 | `--motion-speed` | 194 | 0.00% | −0.768 |
| **0.20** | **`--motion-speed`** | **161** | **0.00%** | **−0.681** |
| 0.25 | `--motion-speed` | 136 | 0.00% | −0.618 |
| 0.30 | `--motion-speed` | 126 | 0.00% | −0.553 |

Note the top row: the 2.5x slowdown **did not achieve its own goal**. It cost 2.5x the
episode length and still clipped `dx` more often than any full-speed collection.

0.20 is chosen for the `dx q01` column, not the tick column: −0.681 against released
LIBERO's −0.679 is a near-exact match to the distribution the checkpoint was pretrained
on. Past 0.25 the returns collapse (136 → 126 ticks for 20% more scale) while the label
distribution thins and OSC torque saturation climbs.

**`--motion-speed` is the third knob, and it is the one that removes the clipping
outright.** The hand-set durations were never set against a speed budget, so the same
trajectory both clipped and crawled — at scale 0.15 the retreat and the closing return
run at 0.275-0.285 m/s mean against a 0.246 budget, while the descent and the transport
run at 0.125-0.163, half of it. Retiming every motion segment to one target speed fixes
both ends; dwells keep their hand-set durations, because they are settling and
gripper-actuation time and their length is set by physics, not by distance. Default in osc
mode is 0.90 x the ceiling / 1.5 (smoothstep's peak-over-mean). `--motion-speed 0` restores
the hand-set durations.

Two things follow, and both matter:

- **The value is a wire convention shared by two processes.** The collector divides by it
  to make a label; the client multiplies by it to execute one. A mismatch silently
  rescales every motion the policy asks for, with no error. Pass the same value to both.
  It is recorded per-chunk in the run log under `timing.delta_pos_scale`.
- **The noise sigmas move with it.** They are in action units, so the physical disturbance
  per unit is proportional to the scale; `--noise-sigma-pos` is auto-scaled by
  `0.05 / --delta-pos-scale` (0.47 → 0.157 at 0.15) unless given explicitly. Leaving it
  at the `a5` value would triple the calibrated kick (§4.3) and stop episodes succeeding.

### 8.2 The bins are randomised in training and fixed at inference

Separate issue, found alongside the above, and it costs sample efficiency rather than
speed. `bin_layout()` permutes green/blue/yellow across three slots with ±2 cm of jitter
every episode. `libero_closed_loop.py` randomises nothing about the bins — it has
`--randomize-box` and no bin equivalent — so every evaluation runs the scene XML's own
layout, green at **(0.56, 0.25)**. In `a5` that layout drew **6 of 30 episodes**; the
remaining 24 demonstrate reaching toward bins that are not present at inference, and the
colour-grounding skill they buy is never tested.

Two coherent resolutions, and the mismatch is the only indefensible option:

- **Randomise both** (chosen for `a6`). `libero_closed_loop.py --randomize-bins` shuffles
  the bins at reset exactly as the collector does, drawing from `BIN_SLOTS` **imported
  from** `libero_closed_loop.py` rather than a second copy of the list — two copies is how
  the two sides end up sampling different distributions with nothing erroring. Keeps the
  colour-grounding skill and actually tests it.
- **Pin both.** `--bin-layout scene` pins the bins to the scene XML's positions and the
  client stays as it was. Buys ~5x the density on the one configuration under test, at the
  cost of a policy that has memorised where green is. Reasonable at 30-50 episodes if a
  working demo is the goal rather than colour grounding.

Note this is orthogonal to the ball, which is randomised on both sides already
(`BALL_SAMPLE_X/Y`, and `--randomize-box` at inference), so the grasp stays a real
closed-loop problem either way.

### 8.3 The speed/precision frontier — why `a6` is fast and misses (2026-08-01)

`a6` served at ~20 chunks per episode against `a5`'s ~54, with the phases in the right
order. It also **never moved the ball** in three runs: it closes on air, 36 mm laterally
off at the nominal ball position (11 mm but 40 mm high at checkpoint 3000).

That is a direct cost of raising `DELTA_POS_SCALE`. Under MEAN_STD normalisation the
metres one unit of *normalised* policy error carries is `std(dx) * DELTA_POS_SCALE`:

| dataset | scale | ticks/ep | dx std | mm per 1.0 normalised error |
|---|---|---|---|---|
| `a5` (slow, **grasps**) | 0.05 | 539 | 0.309 | **15.5** |
| `a6` (fast, **misses**) | 0.20 | 160 | 0.200 | **40.0** |
| `a7` | 0.10 | ~256 | ~0.27 | ~27 |

`a6`'s 36 mm miss is ~0.9 normalised units; the same 0.9 at `a5`'s resolution is 14 mm,
inside the ball's 20 mm radius. **The `a5`-grasps/`a6`-misses split is the units, not the
policy** — and no amount of extra training recovers a 2.6x change in what an action means.

So `DELTA_POS_SCALE` trades speed against terminal precision along a frontier the plant
sets (12.3% realised per tick), and 0.20 is past the knee. Pick it by what the task
needs at the *end* of a reach, not by episode length: a 40 mm grasp tolerance is the
budget here, and the scale has to leave the policy's own error inside it.

Regenerate and retrain:

```bash
# a6, as collected: 30 episodes, shuffled bins, scale 0.20, distance-retimed
MUJOCO_GL=egl uv run python libero/fine_tune/collect_finetune_data.py \
    --out libero/fine_tune/a6 --delta-pos-scale 0.20 --bin-layout shuffle \
    --reach 10 --noise 10 --recover 10 --seed 0 --max-attempts-per-episode 12
uv run python smolvla_libero/convert_dataset.py --src libero/fine_tune/a6 \
    --out smolvla_libero/data/a6_smolvla
# then serve, and run the client at the SAME scale, with bins randomised to match:
uv run python libero/libero_closed_loop.py --delta-pos-scale 0.20 --randomize-bins \
    --randomize-box --server-url <url>/act
```

Before spending that: `libero_closed_loop.py --action-scale 2.0` puts a gain on the
policy's pose channels at inference, clipped back to ±1. It is off-distribution and not a
fix, but if the existing `a5` checkpoint visibly speeds up under it, the diagnosis above
is confirmed against the deployed model rather than only against the dataset.

---

## 9. Known limitations

- **Untested against a training run.** The format is validated structurally against the
  released dataset, but nothing has loaded it through `LeRobotDataset` — the vendored
  `lerobot` needs torch and is not installed in this env.
- **Normalisation stats.** `meta/stats.json` here is computed from these 50 episodes, which
  is a much narrower distribution than the pretraining mixture. For a LoRA fine-tune you
  almost certainly want the **pretrained `libero` norm stats**, not these, or the action
  expert sees a different normalisation than it was trained under. The released stats are
  one `curl` away at
  `huggingface.co/datasets/allenai/MolmoAct2-LIBERO-Dataset/resolve/main/meta/stats.json`.
- **One task, one instruction string.** Every episode carries
  `"pick up the green ball and put it in the green container"`. Fine for a task-specific
  adapter; it will not preserve generality.
- **The scene fixes are unvalidated in closed loop.** §1 corrects four measured errors, and
  the corrected numbers match LIBERO to 0.5 mm and 0.002 rad — but no closed-loop run
  against the model has happened since. **That run is the obvious next step, and it should
  happen before any fine-tuning**: if the 100 mm table error was the real cause, the
  existing checkpoint may improve substantially on its own, which would change what the
  fine-tune is for.
