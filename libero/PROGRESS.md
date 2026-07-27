# LIBERO port — attempt log

Chronological record of what was tried and what actually happened, so the reasoning
behind each change is recoverable and dead ends aren't re-walked. Newest at the bottom.

Format per entry: **what we tried** → **what the result was**. Wrong turns are kept in,
including mine — several conclusions in this log were reversed by later measurements, and
knowing *which* were reversed is the useful part.

Companion docs: `libero/README.md` (current conventions, the spec), `PHASE5_PLAN.md`
(the diagnosis this came out of — see "Corrections to PHASE5_PLAN.md" at the bottom).

---

## 1. Decimation: inference ran the arm 33× too fast

**Tried.** `phase3_closed_loop.py` called `mj_step` exactly once per action. Demo
collection (`phase4_collect_demos.py:283-287`) holds each action for
`decimation = round(500/15) = 33` steps. So each command got 2 ms of physics at inference
instead of the 66 ms it was recorded with. Fixed by wrapping the step in a `decimation`
loop; added `--decimation` to override.

**Result.** Decisive, verified free by **oracle replay** — the same 121 expert setpoints
pushed through phase 3's own `apply_action` + step loop:

| path | final ball pos | in bin? |
|---|---|---|
| 1 step/action (old) | (0.490, 0.000, 0.020) — **untouched from spawn** | ✗ |
| 33 steps (fixed) | (0.682, −0.026, 0.024) vs bin (0.682, −0.019) | ✓ |

**Ground-truth expert actions scored zero through the old path.** So every prior
evaluation — base model, `lora_train`, `ae_train` — had a **ceiling of zero**. All three
looked identical because the environment discarded ~97% of every command regardless of
policy. Those runs measured the bug, not the models.

**Why it was so destructive** (measured, `panda.xml` kp=4500 kd=450 forcerange ±87):

```
held  33 steps ( 66 ms): joint2 −9% of the way   <- moving BACKWARD, gravity beats saturated motor
held  66 steps (132 ms): 57%
held 165 steps (330 ms): 88%
held 500 steps (1.00 s): 95%  (0.007 rad gravity-droop floor, never improves)
```

PD wants 738 N·m on a large step; the motor can deliver 87 — **8× saturated**. One step
didn't just give the arm less time, it stopped it during the phase where it's still
*losing* to gravity. Also self-reinforcing: arm doesn't move → `read_state` doesn't change
→ model re-emits the same absolute target → arm stays put.

**Corrected misconception.** 33 steps is *not* enough to converge, and isn't meant to be.
Settling takes ~330 ms ≈ 5 ticks. What makes it work is that `plan_episode` interpolates,
so the arm tracks a *ramp* with a steady ~12 mm trailing lag rather than chasing steps.
Matching collection is the goal, not convergence.

Commit `8eb2e460`.

---

## 2. Replanning: sequential by default

**Tried.** Originally the next `/act` request fired after action 0 to overlap inference
with execution. Considered firing mid-chunk instead (fresher state), plus dropping actions
whose tick had already elapsed.

**Result.** The arithmetic says firing early is worse on *both* axes. With H=15 actions,
firing after `f` leaves `k = 15 − 2f` ticks of staleness:

| k (stale) | f (fire after) | executed |
|---|---|---|
| 7 | 1 | 8 |
| 5 | 5 | 10 |
| 1 | 13 | 14 |

Early firing costs freshness *and* usable actions, buying only wall-clock — and in this
loop the sim is **frozen** while waiting, so wall-clock is physically free (the arm does
not droop; an earlier claim that it did was wrong). Conclusion: go fully sequential.
`--replan-at 0` = fire only after the whole chunk; staleness 0, all 15 actions used, half
the inference calls.

Also changed `--query-interval` default 10.0 → 0.0. At 10 s it added ~9 s of wall-clock
nothing between chunks. Never distorted physics, just made runs unwatchable.

**Bonus finding.** Sequential mode made the latency breakdown legible for the first time:

```
server dt (inference):  1394 ms
round trip:             6845 ms
unaccounted:            5451 ms  <- ~4x inference, pure transport of a 1.14 MB payload
```

The bottleneck is **shipping raw `json_numpy` arrays**, not the GPU. PNG/JPEG-encoding the
two frames should cut ~5 s/chunk. Not done — changes the wire format.

Commit `b1215085`.

---

## 3. Evaluating `lora_train` step500-merged through the fixed loop

**Tried.** First-ever valid evaluation of a fine-tune. 8 then 30 chunks against the
deployed LoRA checkpoint.

**Result.** Genuine progress and a genuine, *specific* failure:

- Arm now reaches the ball and **displaces it 14 cm**, (0.45, −0.10) → (0.58, −0.14).
  Under the old path the ball never moved a millimetre in any run.
- Ball `z` never leaves 0.020 — **never lifted**.
- Gripper spans up to **0.771 rad within a single chunk** (nearly the full 0→0.8 range)
  while the measured knuckle only reaches 0.104. It thrashes instead of committing.
- Arm-joint action spans shrink 0.73 → 0.13 rad over the run (settling); the gripper
  channel does not converge at all.

A real policy failure with a diagnosable signature, rather than an environment artifact.

---

## 4. Model choice: is action-expert-only enough?

**Tried.** Asked whether to re-run `ae_only` vs LoRA, and how many demos/steps.

**Result.** `ae_train/run_info.md` records `action_flow_loss` **0.176 → 0.01 in 500
steps**. The action expert already fit the data. More AE steps optimize a loss that is
already ~zero, so **AE-only cannot fix this at any step count** — with the VLM frozen it
can only learn the average trajectory over the ball's randomization box.

That matches the observed symptom (claw closing consistently to one side of the ball =
executing a learned average rather than localizing). So LoRA is *required*, not optional.

This also **contradicts `PHASE5_PLAN.md` §2.2** ("nothing converged"). The AE converged
fine. What didn't converge is task success — the classic BC gap the plan's own Rung 2
caveat warns about.

Also noted: demo collection is local CPU MuJoCo, so it **costs nothing**. Only training
steps cost money. Don't economize on demos.

---

## 5. Switching to `allenai/MolmoAct2-LIBERO`

**Tried.** MolmoAct2-DROID is pretrained on real-camera footage of an **FR3**. Found
`allenai/MolmoAct2-LIBERO` in the vendored repo — pretrained on LIBERO, a
**robosuite/MuJoCo benchmark on a Franka Panda**. Same simulator family, same arm.
Built `libero/` as a decoupled copy.

**Result.** Deployed and confirmed working. Conventions differ substantially from DROID:

| | DROID | LIBERO |
|---|---|---|
| `norm_tag` | `franka_droid` | **`libero`** |
| control mode | absolute joint pose | **delta end-effector pose** |
| action | 8-D, raw radians | **7-D, [−1,1]** |
| horizon | 15 | **10** |
| control rate | 15 Hz → dec 33 | **20 Hz → dec 25** |
| state | `[q1..q7, gripper]` | `[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]` |
| quat order | — | **(x,y,z,w)** — MuJoCo gives (w,x,y,z) |

Verified live: server returns **`(10, 7)`**, exactly LIBERO's horizon and action dim.

**Trap found.** `host_server_droid.py` hardcodes `NORM_TAG = "franka_droid"` at module
level and feeds it to `predict_action` to select un-normalization stats. Serving the LIBERO
checkpoint with the DROID tag yields **garbage actions of the correct shape** — silent
failure. `libero_modal.py` retags the module attribute after import. `REPO_ID` needed the
same treatment (it was reporting DROID while serving LIBERO — cosmetic but misleading).

**Consequence for §2:** dropping expired actions is only safe for *absolute* targets. With
deltas, skipping loses displacement. `--replan-at` is pinned to sequential here and warns
if set otherwise.

Route A chosen for control: convert the delta to a Cartesian target, solve one DLS IK step,
command joint positions. Keeps the existing position actuators. Not real OSC.

---

## 6. First LIBERO run: the arm drove through the floor

**Tried.** 15 chunks against the LIBERO checkpoint with the original
`scene_pick_place.xml` framing.

**Result.** Wrong, and the log said exactly why:

```
eef z:  0.469 → 0.386 → 0.290 → 0.220 → 0.115 → 0.020 → −0.008
commanded dz: −0.106, −0.201, −0.077, −0.289, −0.334   (relentlessly down)
ik_unreached = 0 on EVERY chunk
rotation deltas ~0.03 rad (negligible)
```

**The IK was never the problem** — it reached every pose asked of it, and rotation deltas
were tiny. The model was asking for the wrong thing. Diagnosis: we send `eef_pos` as
absolute world coordinates and the model conditions on them directly, but our objects sat
on the **floor at z=0.02** while LIBERO's sit on a **table at z=0.82**. Every z reported
was ~0.8 m outside the training distribution.

---

## 7. Camera investigation — two of my own errors

**Tried.** Measured the ball's pixel size by counting green pixels.

**Result — wrong twice.**

- First measured "28×21 px, so not sub-patch, the plan's 7 px figure is wrong." **The
  detector was counting the `green_bin`, which is also green.** Retracted.
- Analytic geometry is authoritative: at `scene_pick_place.xml`'s framing (1.445 m
  standoff, fovy 71.5) a 40 mm ball is **4.9 px @256, 7.3 px @378**. `PHASE5_PLAN.md`'s
  ~7 px was right all along. Sub-patch, unlocalizable.
- Worse: rendering at LIBERO's native 256² made it *worse* than 378².

Reliable method: project the ball centre and count green only within a local window.
Validated by agreeing exactly with analytic geometry on the new camera (16.0 vs 16×16 px).

**Camera match.** LIBERO's `agentview` (robosuite `table_arena.xml`, identical in every
`libero_*_base_style.xml`):

```xml
<camera mode="fixed" name="agentview" pos="0.5 0 1.35" quat="0.653 0.271 0.271 0.653"/>
```

No `fovy` → MuJoCo default **45°**. With the table top at 0.8 that is offset
(+0.5, 0, +0.55) from the table centre: 0.743 m standoff, ~45° below horizon, dead-on from
the front, framing 0.60 m. Ours was 1.445 m at fovy 71.5 → 2.08 m of framing: **1.9×
further with a 1.6× wider FOV = 3.4× coarser.**

---

## 8. `wrist_cam_rigid` — hand-derived `xyaxes` was 180° wrong

**Tried.** Added a rigid wrist camera by hand-deriving `xyaxes` from the forward vector.

**Result.** **Both axes negated** — a 180° roll. Rendered upside down (gripper at the
bottom, apparently staring at the skybox) while every forward-vector sanity check still
passed, which is why my verification missed it. Caught only when the view was actually
looked at.

Fix: read the values straight off the working camera's matrix instead of deriving them —
`Rl = R_base^T @ R_cam`, `xyaxes = (Rl[:,0], Rl[:,1])`. Now 0.002 mean pixel difference
from `wrist_cam` at home.

**Lesson applied since:** never hand-derive a camera orientation; compute it and *render*
it. Sign errors are invisible in the numbers and obvious in the image.

---

## 9. `mode="targetbody"` does not re-aim — plan correction

**Tried.** Assumed, per `PHASE5_PLAN.md` Tier B, that `wrist_cam` "re-aims every frame".

**Result.** False. The camera and its target `grasp_target` are children of the **same
`base` body**, so the view direction in the gripper frame is constant — measured across
three arm configurations the forward vector is **bit-identical** at
`(0, 0.86946, 0.49401)`. It never re-aims. Same for `external_cam`, whose target
`scene_center` is static in the world.

And a bolted-on real wrist camera would *also* hold a fixed gripper point at a fixed image
position. The drift carrying servo information is the **ball's** drift relative to the
gripper, which we already had.

What `targetbody` *does* get wrong is **roll**: MuJoCo keeps up world-referenced, so a
gripper twist counter-rotates the image instead of rotating with it (9.6e-2 deviation in
the rotation matrix while forward stays exact). Real, but far smaller than the plan claims.

---

## 10. robosuite ground truth: LIBERO's Panda is on a pedestal

**Tried.** Installed robosuite 1.5.2 in a scratch venv rather than guessing the robot
mount. (Env instantiation fails on a mujoco 3.10 API mismatch in `mj_fullM`, so the
constants were composed statically instead.)

**Result.** Overturned my assumption:

```
Panda.base_xpos_offset["table"](0.8) = (-0.56, 0, 0)
Panda.default_base                   = "RethinkMount"
RobotModel.bottom_offset             = (0, 0, -0.912)
set_base_xpos: link0 = pos - bottom_offset
  => link0 world pos                 = (-0.56, 0, 0.912)
TableArena table_offset              = (0, 0, 0.8)     [table TOP]
table_full_size                      = (0.8, 0.8, 0.05)
Panda.init_qpos = (0, 0.19635, 0, -2.61799, 0, 2.94159, 0.78540)
```

**LIBERO's Panda is not floor-mounted.** It sits on a 0.912 m pedestal, so its base is
**0.112 m above the table top** and the arm works *downward* onto a surface just below
itself. Our arm was bolted at z=0 reaching *down* to the floor — a different working
posture entirely, not merely a height offset. An earlier probe that read `link0` at z=0
had silently omitted the mount.

---

## 11. LIBERO-matched table scene

**Tried.** New `mujoco_menagerie/franka_emika_panda/scene_libero.xml`. `link0` can't be
moved (`panda.xml` pins it at the origin; editing that breaks the DROID path and
invalidates the collected dataset), so instead of raising the robot 0.912 m, everything
else was lowered by the same amount — **world origin at the robot base**. Includes
`panda.xml` directly rather than `scene.xml`, because `scene.xml`'s ground plane is at z=0
where the robot base is.

**Result.** Verified against robosuite's numbers:

| quantity | LIBERO | ours | |
|---|---|---|---|
| link0 above table top | +0.112 m | +0.112 m | ✓ |
| eef above table at `init_qpos` | 0.152 m | **0.148 m** | ✓ |
| ball on table | z = 0.82 | z = 0.82 (LIBERO frame) | ✓ |
| base→table gap | 0.16 m | 0.16 m | ✓ |
| agentview standoff / fovy | 0.743 m / 45° | 0.743 m / 45° | ✓ |
| 40 mm ball @256 | ~17 px | **16×17 px** | ✓ |

Absolute coordinates differ from LIBERO by a constant `(+0.56, 0, −0.912)`, so the loop
adds `(−0.56, 0, +0.912)` to `eef_pos` before sending — a change of origin, not a fudge.

**Open caveat.** The table is necessary but not sufficient as a physical stop. Route A
solves IK for whatever target it's given and writes joint positions, so it can command
*through* the surface; MuJoCo contact resists, so the arm presses hard rather than settling
compliantly the way robosuite's force-based OSC would.

---

## 12. First run on the table scene — the z-frame fix worked

**Tried.** Applied the three inference-side changes and ran against
`allenai/MolmoAct2-LIBERO`:
1. `LIBERO_ORIGIN_OFFSET = (-0.56, 0, +0.912)` added to `eef_pos` in `read_state`.
2. `LIBERO_INIT_QPOS` (robosuite's `Panda.init_qpos`) as the start posture, setting **both**
   `qpos` and `ctrl` — setting only `qpos` lets the position actuators drag the arm back to
   the old keyframe on the first step.
3. Objects moved onto the table; ball sampling box re-anchored to the table centre.

**Result — the divergence is gone.** Table top is at 0.800 in LIBERO's frame:

```
c0  eef_z 0.948 -> 0.871 | dz cmd mean -0.286 | ik_unreach 0 | mean|dpos| 33.2 mm
c1  eef_z 0.871 -> 0.784 | dz cmd mean -0.481 | ik_unreach 0 | mean|dpos| 28.0 mm
c2  eef_z 0.784 -> 0.788 | dz cmd mean -0.026 | ik_unreach 0 | mean|dpos| 19.0 mm
c3  eef_z 0.788 -> 0.790 | dz cmd mean -0.050 | ik_unreach 0 | mean|dpos| 24.4 mm
c4  eef_z 0.790 -> 0.795 | dz cmd mean +0.006 | ik_unreach 0 | mean|dpos| 30.1 mm
```

The model descends to the table and **stops commanding descent** — `dz` collapses from
−0.481 to ≈0 and `eef_z` stabilises at 0.79. Compare §6, where it drove to z = −0.008
(0.8 m too low) and never stopped. The out-of-distribution state really was the cause.

**Observed behaviour (from the viewer).** Approaches the ball, closes the gripper, ball
squirts away. Arm sits ~1 cm below the table surface. Approach is not well aimed.

**Diagnosis of what remains.** The pinch site settles at z ≈ 0.79 while the ball centre is
at 0.82 — so the fingers close ~3 cm *below* the ball's equator, from under the table
surface. Closing there squeezes the ball out sideways, which is exactly the reported
symptom. So the failed grasp is downstream of the penetration, not independent of it.

Penetration cause is **Route A's lack of compliance**, not decimation. robosuite's OSC is
force-based and yields on contact; Route A solves IK for whatever target it is handed and
commands joint positions, so the position actuators press into the table with up to
87 N·m. Flagged as an open caveat in §11 — now confirmed as the live blocker.

**On under-travel.** Commanded descent over chunks 0-1 was ~380 mm; achieved 164 mm, i.e.
**~43% of commanded per tick** (consistent with the isolated measurement of 17.7 mm
achieved for a 50 mm command). This is *probably authentic* rather than a bug: robosuite's
OSC at kp=150 also cannot traverse a full-scale delta in one 50 ms tick, so the model was
likely trained against similar under-travel, and the closed loop visibly compensates
(`dz` converges). Not to be "fixed" without evidence.

---

## 13. Decimation sweep (offline) + an end-effector site mismatch

**Tried.** Hypothesis raised: decimation is causing the over-penetration, so reduce it.
Tested for free by replaying the 50 **actual logged actions** from §12 at several
decimations — same commands, same start state, only physics-time-per-action varies. No GPU.

**Result.**

```
dec   ctrl_hz   final eef_z  min eef_z   penetration   travel realised
5     100.00    -0.0312      -0.0312     +80.8 mm      12% (67/574 mm)
10     50.00    -0.0888      -0.0888     +23.2 mm      22% (125/574 mm)
15     33.33    -0.1213      -0.1236     -11.6 mm      27% (157/574 mm)
25     20.00    -0.1171      -0.1278     -15.8 mm      27% (153/574 mm)   <- LIBERO's rate
40     12.50    -0.1159      -0.1278     -15.8 mm      26% (152/574 mm)
60      8.33    -0.1080      -0.1274     -15.4 mm      25% (144/574 mm)

table top = -0.1120     ball centre = -0.0920  (where the fingers should close)
```

Lower decimation does reduce penetration, and `dec=10` happens to land the pinch at
−0.0888, within **3 mm of the ball centre**. So the intuition has a real effect.

**But this replay is OPEN-LOOP and therefore not predictive.** The actions were recorded at
dec=25. Run live at dec=10, the model sees a higher eef, and will command *more* descent to
compensate — the closed loop may well converge to the same penetration depth, just more
slowly. Decimation is a global scale on realised motion; it does not add the compliance
that is actually missing. Treat the table above as physics characterisation, not a
prediction. Note also that realised travel saturates around 27% by dec=15 — beyond that,
extra physics time buys nothing, because the arm has reached its steady trailing lag.

**Separate real bug found: the eef site is defined differently.** From robosuite's
`panda_gripper.xml`:

```
<body name="eef" pos="0 0 0.097">      <- grip_site == robot0_eef, what LIBERO reports
<body name="leftfinger" pos="0 0 0.0524"> <body name="finger_joint1_tip" pos="0 0.0085 0.056">
                                       -> fingertip at 0.0524 + 0.056 = 0.1084
```

So **LIBERO's `robot0_eef` sits 11.4 mm BEHIND its fingertips.** Our `pinch` site
(`panda.xml:310`, `pos="0 0 0.145"`) *is* the fingertip convergence point. Reporting `pinch`
as eef therefore claims a point ~11 mm further out than LIBERO ever would, so when the
model servos its eef to a height, our fingertips land ~11 mm lower than LIBERO's would.

That is ~11 mm of the ~30 mm discrepancy (pinch settling at 0.79 vs ball centre 0.82), in
the right direction, and it pushes the fingers below the ball's equator — where a frictional
sphere grasp squeezes the ball out sideways rather than holding it. Consistent with the
reported symptom.

**So there are (at least) two independent contributors to the failed grasp:**
1. **No compliance** in Route A — position actuators press into the table with up to 87 N·m
   where robosuite's force-based OSC would yield.
2. **eef site offset** of 11.4 mm — a pure convention mismatch, cheap to correct by
   reporting the state at a point 11.4 mm back along the gripper approach axis.

Neither is decimation. Decimation at 25 is correct for LIBERO's 20 Hz and was verified
against `control_freq`.

---

## 14. Decimation 3, and the image-flip experiment — hypothesis disproven

**Tried (a).** Ran at `--decimation 3` (167 Hz, 8× LIBERO's rate) on the hypothesis that
decimation was causing the over-penetration.

**Result (a).** Decimation is a rate knob, not a fix — the closed loop simply compensates.
The arm descended at ~14 mm/chunk instead of ~80 and was still heading for the table after
80 actions (z = 0.838 in LIBERO frame vs table top 0.800), with `dz` commands holding
steady at −0.28…−0.44 rather than decaying. It would have reached the same penetration
depth, just slower. Confirms the offline read in §13: the missing ingredient is compliance,
not less physics time.

That run also showed a **78 mm shortfall in x** while z was nearly solved, and the model's
x command *decaying* (+0.559 → +0.069) as though converging. Since `agentview` looks back
along −x, x maps to the image's vertical axis — so this looked like an image-orientation
error, and `--image-flip` had never been verified.

**Tried (b).** Three runs, 8 chunks each, `--decimation 25`, identical start state, one per
flip mode. 24 inference calls.

**Result (b).** Distance from eef to ball centre, per chunk end (mm):

```
flip       c0   c1   c2   c3   c4   c5   c6   c7    closest  final
none       55   50   50   89  108  114  125  103      50      103
180       101   76   44   87  180  228  243  279      44      279
vertical   93   78   64   81  137  142  138   84      64       84
```

Lateral (xy only) distance to the ball at closest approach: **none 7 mm, vertical 8 mm,
180 14 mm.**

**The hypothesis was wrong.** All three flips localise the ball *well* — 7-14 mm lateral at
first approach. The model can see the ball and aims at it. `180` is clearly worst
(diverges to 279 mm) so it is ruled out; `none` and `vertical` are comparable and `none`
had the best single approach, so **keep the default `none`**. The x shortfall in §14(a) was
an artefact of dec=3 never letting the approach finish, not an orientation error.

**What the data actually shows.** Correlating z against lateral error exposes the real
sequence (`flip=none`, table top −0.112, ball centre −0.092):

```
c0  z=-0.038  +74.5mm above table              lateral to ball   7.2mm   <- excellent aim
c1  z=-0.128  -15.9mm PRESSED INTO TABLE        lateral to ball  35.4mm
c2  z=-0.125  -13.1mm PRESSED IN                lateral to ball  36.9mm
c3  z=-0.121   -9.1mm PRESSED IN                lateral to ball  84.3mm
c4  z=-0.119   -7.0mm PRESSED IN                lateral to ball 104.5mm
c5  z=-0.117   -4.8mm PRESSED IN                lateral to ball 111.2mm
c6  z=-0.111   +1.4mm                           lateral to ball 123.6mm
c7  z=-0.076  +36.4mm above table               lateral to ball 102.1mm
```

The failure is a **sequence**, not a bad aim:

1. Chunk 0 arrives 7 mm from the ball laterally, hovering 74 mm up. Aim is essentially
   perfect.
2. Chunk 1 descends 90 mm in one go — straight **past** the ball centre (−0.092) and 16 mm
   **into** the table.
3. Once jammed, every later command slides the gripper *across* the table instead of
   lifting and repositioning. Lateral error grows monotonically 35 → 124 mm, and the ball
   gets shoved along the way.

So "approaches the ball, closes, ball moves away" is the overshoot-then-drag signature. The
approach is fine; the terminal descent is not.

**Overshoot budget** (~36 mm past ball centre), best current accounting:
- **11.4 mm** from the eef-site convention mismatch (§13) — pure bookkeeping, cheap to fix.
- remainder from **no compliance**: the position actuators keep pressing at up to 87 N·m
  where robosuite's force-based OSC would yield on contact, and there is no intra-chunk
  feedback to arrest the descent (all 10 actions execute before the model sees state again,
  which matches LIBERO's `n_action_steps=10`, so it is not itself the bug).

**Next, in priority order** — *superseded by §15/§16; kept for the record. Item 1's
magnitude and sign were both wrong (the 11.4 mm compares `grip_site` to the finger tip
BODY, not to the pads), item 2 was aimed at a penetration that §15 showed does not exist,
and item 3 turned out to be robosuite's number rather than DROID's:*
1. Correct the reported eef to LIBERO's `grip_site` convention (11.4 mm back along the
   gripper approach axis). Cheap, principled, removes a known bias.
2. Add compliance. Either clamp the IK target so it cannot go below the table surface
   (blunt but immediate), or lower the actuator gains / add force limiting (closer to OSC).
3. Only then revisit pad friction (`PHASE5_PLAN.md` Tier B: 0.7/0.6 vs DROID's 2.0).

---

## 15. Contact instrumentation — §14's step 2 was aimed at a non-problem

**Tried.** Before adding compliance, checked the premise. A steady 16 mm of penetration is
not consistent with the pad contact parameters (`priority=1`, `solref="0.004 1"`,
`solimp="0.95 0.99 0.001"` — stiffens within 1 mm), so either contact was being overwhelmed
or it wasn't happening. Replayed the 80 logged actions from `flip_none.jsonl` at dec=25
with `d.ncon`, per-contact `mj_contactForce`, and the true lowest hand-geom corner logged
every step. Free, no GPU. The z-trajectory reproduces the live log to <0.5 mm at every
chunk, so the replay is faithful for anything z- or contact-related.

**Result — there is no penetration.**

```
chunk  pinch vs table   TRUE hand vs table   steps in contact   max Fn
0          +74.5 mm          +69.2 mm               0            0.0 N
1          -15.9 mm           -0.5 mm              78          662.4 N
2          -15.9 mm           -0.2 mm             224          312.8 N
7           +1.4 mm           -0.1 mm              46          156.0 N
```

Real hardware penetration is **0.1–0.5 mm**, exactly what the pad `solimp`/`solref`
predict. `left_pad1`/`right_pad1` contact normally, and the arm is **not saturated** while
pressing (j2 at −56.9 of 87 N·m at the deepest moment).

The "pressed into the table" reading in §12–§14 was **the `pinch` site, which is not
hardware** — a virtual point on the finger centreline ~5 mm past the pad faces, and with
the fingers tilted it projects ~15 mm below the pads' lowest corner. It reads as
penetration while the pads sit on the surface. **So Route A's lack of compliance was never
the blocker**, and §14's step 2 would have been tuning a non-problem. Struck, not
reordered.

**What was actually happening.** Second probe, hand↔ball contacts on the same replay:

- The ball is struck **once**, chunk 1, by `right_pad1` alone at 3.5 N, with the pinch
  23.4 mm *below* the ball centre and 32 mm lateral. One pad, edge-on — a swipe, not a
  grasp. It shoves the ball 26 mm.
- Chunk 0 aims well (7.2 mm lateral, 74 mm up). Chunk 1's 90 mm descent carries ~27 mm of
  lateral drift with it, so the open jaw arrives *beside* the ball, catches it with one
  pad, and bottoms out on the table.
- **The gripper is commanded fully open for all 80 actions** — mean −0.995, never once
  positive.

That last one across every LIBERO run:

```
run                gripper channel        frac commanding close
libero_view        -1.000 .. -0.984              0%
libero_dec3        -1.000 .. -0.984              0%
flip_none          -1.000 .. -0.984              0%   <- the mode sec.14 recommends
libero_table       -1.000 .. +0.977              6%
flip_vertical      -1.000 .. +0.996             31%
```

**This undercuts §14's flip conclusion.** That comparison scored flip modes on lateral
distance to the ball, and the winner (`none`) is the one mode that **never attempts a
grasp**. A run that never closes cannot succeed however well it aims. `flip_vertical` is
the only mode that tries. Needs re-running on a success-shaped metric.

Caveat: the ball's position diverges from the live run after chunk 1 (it rolls off the
table in replay) — open-loop, as §13 warned. Chunks 0–1 ball numbers are faithful; later
ones are not. The z, contact and gripper-channel findings hold throughout.

---

## 16. Mounting the stock Franka hand (and dropping to an L4)

**Tried.** §15 left "the gripper never closes" as the live question, and the gripper was
the one part of the robot still not matching LIBERO. Measured the two against each other:

| | robosuite `panda_gripper.xml` | ours (2F-85) |
|---|---|---|
| kinematics | 2 prismatic slides, 0–0.04 m | 4-bar linkage, knuckle 0–0.8 rad |
| pads | flat boxes, parallel | curved, two per finger, tilting |
| pad friction | **2 0.05 0.0001** | **0.7 / 0.6** |
| pad `solref` | `0.01 0.5` (soft, underdamped) | `0.004 1` (stiff) |
| reported point vs pad centre | `grip_site` **+3.6 mm** | `pinch` **+14.4 mm** |

**Result.** Swapped to the stock hand — `panda_libero_hand.xml` (upstream menagerie's
`panda.xml`, recovered from the submodule's git history, plus robosuite's `grip_site`,
`eye_in_hand` camera and pad parameters) and `scene_libero_hand.xml`. `panda.xml` and
`scene_libero.xml` untouched, so the DROID path and the earlier runs are unaffected.

Four approximations collapse at once, and two of them were **bugs**:

1. **`gripper_qpos` was half wrong.** robosuite mirrors its fingers with opposite joint
   *ranges*, so `robot0_gripper_qpos` is `[+x, −x]`; we reported `[+x, +x]`. One of the two
   gripper state channels was wrong across its whole range, in the channel that decides
   grasping — the exact thing §15 flagged. Now `[0.04, −0.04]` open, verified.
2. **Actuator sense is inverted between the two hands.** `actuator8` is `255 = OPEN` for
   the stock hand; the 2F-85's was `0 = open`. Carrying the old polarity over would have
   closed the gripper on every open command. Caught before the first run.
3. **eef convention** is now exact rather than an 11 mm estimate.
4. **Pad friction/softness** are robosuite's numbers, not guesses. Item 3 of §14's list,
   done as a side effect.

Also replaced our own wrist camera (22 cm side-mount, `fovy 56.7`, never matched anything)
with robosuite's `eye_in_hand`, copied verbatim — closing the "never chased down" item.
Rendered and looked at it, per §8: fingers framed bottom-of-view, ball centred, right way
up.

**Verified against robosuite's own FK** (its `robot.xml` loaded straight into MuJoCo,
rather than composing constants by hand): `right_hand` relative to `link0` agrees to
**0.5 mm**, and the finger opening axis lands on the eef frame's x in both.

**§11's table has a bad row.** It records "eef above table at `init_qpos`: LIBERO 0.152,
ours 0.148 ✓". robosuite's own FK gives **0.211**. The 0.152 reference was wrong, and the
2F-85's longer reach happened to agree with it — two wrong numbers confirming each other.
Ours is now 0.197 after gravity sag (0.201 ideal; the remaining 9.9 mm is the deliberate,
documented pad-mesh offset in `grip_site`).

**GPU.** Dropped `libero_modal.py` from A100-40GB to **L4**. The checkpoint is 5B params ≈
10 GB at bf16, so 40 GB was ~4× oversized. T4 is cheaper but is Turing — no bf16 at all,
so it would fail rather than run slow. The saving is larger than the $2.10→$0.80/hr gap
suggests: §2 measured inference at 1394 ms inside a 6845 ms round trip, so transport
dominates and a slower card barely moves wall clock, while `scaledown_window=300` means
most billed time is a *warm idle* container charged at the GPU rate regardless. Idle time,
not FLOPs, is the bill.

**Not yet run against the model.** Everything above is geometry, contact and unit checking.

---

## 17. First run on the stock hand — the gripper closes, the grasp is ~7 mm too low

**Tried.** Deployed `libero_modal.py` on the L4 and ran 8 chunks against
`scene_libero_hand.xml` (`--run-id stockhand_01`). First inference on the stock hand.

**Result.**

```
chunk  eef_z    vs table  lateral(mm)  ball moved(mm)  ball z    gripper frac>0
0      0.0212    133.2       27.1           0.0        -0.0924        0%
1     -0.0883     23.7       11.5           0.0        -0.0924        0%
2     -0.1063      5.7       10.5           0.0        -0.0924       40%
3     -0.1041      7.9        9.6           0.0        -0.0924       60%
4     -0.1030      9.0       19.9          26.8        -0.0924       50%
5      0.0061    118.1       16.6           5.8        -0.0924      100%
6     -0.0706     41.4       23.0          39.7        -0.0924       10%
7     -0.1053      6.7        6.7          48.9        -0.0925        0%
```

**The gripper channel is alive.** 32% of the 80 actions command a close, against **0 of 80**
in `flip_none` and 0% in three of the five earlier runs (§15). Reported `gripper_qpos` is
`[0.04, -0.04]` open, robosuite's convention exactly.

Two other qualitative changes:

- **No more jam-and-drag.** Lateral error goes 27 → 11.5 → 10.5 → 9.6 mm and *stays*
  single-to-low-double digits. In §14 it grew monotonically 35 → 124 mm once the gripper
  bottomed out. The arm now hovers and re-approaches instead of sliding across the table.
- **A real grasp-and-lift attempt.** Chunk 5 closes on 100% of actions and lifts the eef
  118 mm off the table. The ball does not come with it — `ball z` never leaves −0.0924 in
  any chunk.

**Why the lift comes up empty.** The eef settles at −0.103…−0.106, i.e. 6–9 mm above the
table top. `grip_site` sits 3.6 mm ahead of the pad centre (approach points down), so the
pads close at roughly **−0.099** while the ball centre is at **−0.092**: about **7 mm below
the equator**. A frictional sphere grasp closed below the equator squeezes the ball out
and down rather than holding it — which is what the ball-displacement column shows
(0 → 26.8 → 48.9 mm, all lateral, z untouched).

That is the same failure mode as §12, but the error has shrunk from ~36 mm below centre to
~7 mm. Remaining gap is small enough that it is no longer obviously a convention bug.

**New signal: `IK unreached on 6/10`** on chunks 4 and 7. Never appeared in any 2F-85 run
(`ik_unreached = 0` on every chunk of §6 and §12). The stock hand's shorter reach puts the
requested poses closer to the arm's limits. Not yet chased.

**Attribution caveat.** Four things changed at once — hand geometry, `gripper_qpos` sign,
actuator polarity, wrist camera, pad friction. This run says the *combination* works
better; it does not say which one mattered. The polarity fix alone would explain a gripper
that never appeared to close, so it is the leading candidate, but that is inference, not
measurement.

**Unexpected: the L4 is not slower.** Server dt averaged **733 ms** and round trip
3.1–4.2 s, against §2's 1394 ms / 6845 ms on the A100-40GB. Rather than claim an L4 beats
an A100 at compute, the honest reading is that §2's numbers were measured under conditions
that no longer apply. Either way there is no latency penalty from the downgrade, and the
cost is 2.6x lower.

---

## 18. Second stock-hand run — the ball leaves the table

**Tried.** Repeat of §17, same command, `--run-id stockhand_02`. Intended as a
reproducibility check.

**Result — the first successful grasp and lift in this project.**

```
             stockhand_01                      stockhand_02
chunk  lateral  ball moved  ball z  grip   lateral  ball moved  ball z   grip
0        27.1       0.0    -0.0924    0%     30.9       0.0    -0.0924    0%
1        11.5       0.0    -0.0924    0%      4.8       0.0    -0.0924    0%
2        10.5       0.0    -0.0924   40%      7.0       0.0    -0.0924   30%
3         9.6       0.0    -0.0924   60%      2.6       1.9    -0.0924   70%
4        19.9      26.8    -0.0924   50%     11.0      83.5   +0.0114   100%   <- LIFTED
5        16.6       5.8    -0.0924  100%     89.0      54.0    -0.0924  100%
6        23.0      39.7    -0.0924   10%    186.6     200.8    -0.0924    0%
7         6.7      48.9    -0.0925    0%    467.4     348.9    -0.0924    0%
```

Chunk 3 closes to **2.6 mm lateral**, chunk 4 closes the gripper on 100% of its actions,
and the ball goes from z = −0.0924 to **+0.0114** — **104 mm off the table**, 123 mm above
the surface. Every previous run in this log had `ball z` pinned to its spawn value; §3's
best result displaced the ball 14 cm sideways but "never lifted". This one lifts it.

It does not keep it. By chunk 5 the ball is back on the table and the run degenerates —
lateral error 89 → 187 → 467 mm, ball flung 349 mm. So: **grasp achieved, retention not.**

**The two runs diverge, and that is itself a finding.** Identical command, identical
deterministic start state, yet they part company from chunk 3 onward. MolmoAct2's action
expert is a **flow-matching** head, i.e. it samples — the policy is stochastic, not a
deterministic function of the observation. Any single 8-chunk run is one draw. An earlier
expectation in this session that a repeat would track §17 closely was wrong, and every
per-chunk comparison in §12-§17 should be read as one sample rather than as the model's
behaviour. Success rate over N runs is the only meaningful metric from here.

That also means the §14 flip comparison (§15 already reopened it on other grounds) is
weaker still: it compared three single runs.

**Where the marginal 7 mm sits.** §17's reading — pads closing ~7 mm below the ball's
equator — is consistent with a grasp that succeeds sometimes and slips often. Two runs is
not enough to rate it, but it makes the `grip_site` height (the one deliberate 9.5 mm
departure, `0.1065` vs robosuite's literal `0.097`) the obvious first parameter to sweep,
now that there is a success signal to sweep *against*.

---

## 19. The stock hand really does sink into the table — cause and fix

**Tried.** Observed in the viewer: the gripper visibly enters the table. §15 measured
0.1–0.5 mm for the 2F-85 and concluded compliance was not the blocker, so this needed
re-measuring on the new hand rather than assuming the old result carried over. Re-ran the
§15 instrumentation on `stockhand_02`'s actions.

**A measurement error of my own, first.** The initial probe reported 37 mm of penetration.
That was wrong: it estimated mesh geoms' lowest point as `geom_xpos − geom_rbound`, a
bounding-*sphere* radius, which grossly overstates how low a mesh reaches. Exactly the
same class of mistake as reading the `pinch` site as hardware in §13. Redone against
actual mesh **vertices** and MuJoCo's own `contact.dist`:

```
chunk  eef vs tbl  TRUE hand vs tbl  worst contact dist  steps in contact  max Fn
2          +6.0          -2.9             -2.88 mm            212          66.1 N
3         +11.0          -2.4             -2.40 mm            210          69.7 N
7          +9.6          -1.0             -0.96 mm             15          42.9 N
```

**2.9 mm, real, and ~6x the 2F-85's.** So §15's conclusion was correct *for that gripper*
and did not survive the swap.

**Cause.** The geom reaching the table is `geom76` — the **finger mesh**, not the
fingertip pads. It inherits menagerie's `collision` class, i.e. MuJoCo's default
`solref="0.02 1"`: a 20 ms, soft contact. The Robotiq pads it replaced were
`solref="0.004 1"`, five times stiffer. robosuite ships the same soft default and gets
away with it because its OSC is force-based and yields; Route A commands joint positions
and presses at ~70 N, so the softness shows up as visible sinking. The pad parameters
added in §16 were irrelevant here — those geoms never touch the table first.

**Fix, both halves, because "not at all" needs both:**

1. `panda_libero_hand.xml` — `solref="0.004 1"` on the finger mesh, matching the stiffness
   the 2F-85 had.
2. `libero_closed_loop.py` — a table floor on the *commanded* target, clamped before IK so
   the solver never sees an impossible request. `--min-clearance`, default **13 mm**,
   measured: over `stockhand_02`'s 80 control ticks `grip_site` rode 6.1–12.1 mm above the
   lowest point of the hand (median 7.6). A **negative** `--min-clearance` switches it off.

   **Bug in the first version of this flag:** `0` was documented as "off" but is not --
   it floors the target at the table top, which still clamps and still puts the hand
   through the surface (grip_site held at the surface, hardware 6-12 mm below it). The
   first A/B sweep's control arm was therefore also clamped, and its `clamp0_1` run logged
   `table-clamped 4/10` while supposedly unclamped. Fixed: off is now a negative value.

**Verified** by replaying the same actions through the new code:

```
worst contact dist  +0.00 mm      steps in table contact  0
TRUE hand vs table   min +0.6 mm  (was -2.9 mm)
```

Zero contacts, zero penetration, hardware stays above the surface throughout.

**The clamp does not cost grasp height — it gains some.** The pad centre sits 3.6 mm above
`grip_site` with the approach pointing down, so a 13 mm site floor puts the pads at
table+16.6 mm against a ball centre at table+20 mm: **3.4 mm below the equator**, versus
the ~7 mm below that §17 measured unclamped. It pushes the grasp toward the equator.

It is still a departure from LIBERO's control law — the model asks for a pose and we
decline part of it. Worth remembering when reading later runs.

**Not yet run live against the model.** The verification above is an open-loop replay,
which §13 established is not predictive of closed-loop behaviour.

---

## 20. Benchmark diagnostic — the checkpoint and our serving are both fine

Everything up to here debugs our scene. §20 asks the prior question: does the served
checkpoint work *at all*, under a driver that shares nothing with `libero_closed_loop.py`
except the wire format?

`libero/libero_benchmark_eval.py` (new) drives LIBERO's own `OffScreenRenderEnv` — real
robosuite `OSC_POSE`, a task from the suite the checkpoint was trained on, LIBERO's own
observation assembly. It duplicates the state/image code deliberately rather than importing
it, so a shared bug cannot hide in both.

Result, `libero_object` task 0 ("pick up the alphabet soup and place it in the basket"),
3 episodes, same L4 deployment we run against:

| episode | outcome | steps |
|---|---|---|
| 0 | SUCCESS | 163 |
| 1 | SUCCESS | 143 |
| 2 | SUCCESS | 140 |

**3/3. 100%.** Server dt ~740-760 ms, matching §16.

What this rules out, as a group: the serving path, the checkpoint itself, the norm stats
(`norm_tag=libero`), the 180° image rotation, the 8-D state layout, the quaternion order,
and the chunk-execution schedule. All of those are shared with the main loop and all of
them are now known-good.

~~`--image-flip` is settled at 180°~~ — **wrong, corrected in §21.** This driver flips
robosuite frames 180° and succeeds, but that says what the MODEL wants, not what OUR LOOP
should send: robosuite renders upside down and `mujoco.Renderer` does not. Decoding the
released dataset settles it — its stored frames are upright, so upright is what the model
consumes, so our correct setting is `none`, which is already the default.

What it does *not* settle: this run changed two variables at once — controller (OSC, not
Route A) and task (in-distribution, not our ball). So it confirms the problem is on our
side of the wire, but does not say which of the two.

To separate them, one more run is needed: **Route A driving a LIBERO task**. If our stiff
IK servo fails a task OSC just passed 3/3, the controller is proven guilty on its own and
the `--control-mode` OSC port is the clear buy. If it also succeeds, the controller is
survivable and our green-ball task is simply out of distribution — which points at
fine-tuning, not at OSC.

Setup note for repeating this: `hf-libero` pins its own `robosuite` 1.4.0 and `mujoco`
3.8.1 and must live in a separate venv, not the project env. It also prompts interactively
on first import; write `~/.libero/config.yaml` by hand to skip that. And do **not** call
`json_numpy.patch()` inside that venv — something in the LIBERO import chain installs a
global `json` object_hook returning `SimpleNamespace`, which json_numpy then composes with
and dies on. Encode and decode explicitly.

Two incidental confirmations from the smoke test: LIBERO reports
`controller: OperationalSpaceController`, and `gripper_qpos [0.0208, -0.0208]` — an
independent check of §16's `[+x, −x]` sign fix.

---

## 21. Building the fine-tune dataset found four measured errors in our own scene

**Tried.** Before generating training data, checked what the fine-tune engine actually
requires: the mixture entry in `data_mixtures.py`, the LeRobot version the loader pins, and
the released `allenai/MolmoAct2-LIBERO-Dataset` itself (downloaded and decoded). Then
measured our scene against a live `OffScreenRenderEnv` instead of against robosuite's XML.

**Result.** The scene was wrong in four independent, measurable ways. Every one of them
would have been baked permanently into the dataset.

| quantity | LIBERO (measured) | ours, before | ours, now |
|---|---|---|---|
| eef at reset, rel. `link0` | `(0.4515, 0, 0.2613)` | `(0.4585, 0, 0.0891)` | `(0.4515, 0, 0.2608)` |
| eef axis-angle at reset | `(3.1408, 0.0018, -0.0899)` | `(2.1557, 2.1557, 0.1373)` | `(3.1403, 0, -0.0892)` |
| table top rel. `link0` | `-0.012` | `-0.112` | `-0.012` |
| eef above table at reset | `0.2733` | `0.2011` | `0.2728` |

**21.1 The table was 100 mm too low.** §11 put the top 0.112 m below `link0`, reading
LIBERO's table as `z=0.800` under a robot at `0.912`. Live env: `link0` 0.912, table top
**0.900** — a 12 mm gap, not 112 mm. The policy commands a descent sized for a surface
~0.27 m down; ours was 0.10 m further. The descent ends in free space, the model sees it
has not arrived, and it keeps commanding down. **That is the best explanation yet for
§17-19's "picks up the ball but goes into the table"** — and it is a scene bug, not the
controller mismatch §20 pointed at.

**21.2 `grip_site` was 9.5 mm out and yawed 90°.** Sweeping the site against ground truth:

| site | position error | orientation error |
|---|---|---|
| `pos 0.097`, no quat | **0.5 mm** | **0.002 rad** |
| `pos 0.097`, `Rz(-90)` | 0.5 mm | 2.418 rad |
| `pos 0.097`, `Rz(+90)` | 0.5 mm | 5.842 rad |

The frame needs no extra rotation — §16's `Rz(-90)` put a spurious 90° yaw on **every
`eef_axisangle` ever reported to the model**. And §16's deliberate `0.1065` (to preserve a
pad-centre relationship) is the wrong side of its own trade: it buys 9.5 mm of pad accuracy,
which only our clamp reads, by paying 9.5 mm of error in the reported eef height, which is
what the model conditions on. Now robosuite's literal `0.097`.

**21.3 The reset pose was not LIBERO's.** `LIBERO_INIT_QPOS` held robosuite's generic
`Panda.init_qpos`. LIBERO resets to `[0, -0.16103739, 0, -2.44459747, 0, 2.2267522,
0.78539816]` in all four suites — 72 mm higher above the table.

**21.4 `LIBERO_ORIGIN_OFFSET` x was `-0.56`,** should be `-0.6`. (The `+0.912` z was right.)

Consequence: `--min-clearance` default `0.013 → 0.016`, re-measured — with the site at
0.097 it rides **15.5 mm** above the lowest hand point, not 6-12 mm.

**There is no canonical absolute frame.** Measured `link0` z is **0.912** (spatial, goal),
**0.0** (object), **0.42** (libero_10). The suites do not share a world origin, which is why
the released dataset's eef z spans **0.008 … 1.366**. What IS invariant in all four is the
pose relative to the robot. **So the model cannot be keying on absolute world z** — a
premise several earlier entries leaned on. Same for the camera: `agentview` is per-scene,
`(1.497, 0, 0.650)` in object vs `(1.319, 0, 0.698)` in spatial. There is nothing to match
exactly.

**21.5 A physics bug found while collecting.** Demos lifted the ball every time and then
dropped it 0.10-0.12 m short of the bin. Traced tick by tick: the fingers seat correctly at
0.0195 m (ball radius 0.02) for ~17 ticks, then drive on through to 0.0000 and the ball
squirts out. A 40 mm sphere between flat pads has no form closure, and menagerie's default
contact softness lets a sustained squeeze extrude it. Fixed on the ball geom with
`priority=2 condim=4 friction="2 0.05 0.0001" solref="0.004 1" solimp="0.99 0.999 0.001"` —
placement went 5/15 → 5/5. (The gripper `forcerange` was also set to robosuite's ±20 N from
menagerie's ±100, but that is **not** what fixed it: swapping it gave bit-identical
trajectories, because a seated grasp never approaches either limit.)

**Not yet validated in closed loop.** The corrected numbers match LIBERO to 0.5 mm and
0.002 rad, but no run against the model has happened since. That run should come before any
fine-tuning: if 21.1 was the real cause, the existing checkpoint may improve on its own.

See `libero/fine_tune/README.md` for the format findings (v3.0, inline PNG, the state
layout `info.json` misnames, and why `--image-flip none` is correct).

---

## Corrections to `PHASE5_PLAN.md`

Things the plan asserts that later measurement contradicted:

| plan says | actually |
|---|---|
| §2.1 mechanism: position actuators "never converge on any commanded target" | True but incomplete — they don't converge in *collection* either. Settling is ~330 ms ≈ 5 ticks. What works is ramp-tracking with a steady ~12 mm lag. |
| §2.2 "Nothing converged" | The action expert converged fine (flow loss 0.176 → 0.01 in 500 steps). Task success didn't. Different problem, different fix. |
| Tier B: wrist cam "re-aims every frame", "deleted the best cue" | It never re-aims — camera and target share a body. Only **roll** is wrong, a much smaller effect. |
| Tier B: ball ≈ 7 px | Correct (7.3 px @378, 4.9 px @256). My mid-chat "correction" of this was itself wrong. |
| Tier A: FR3-vs-Panda link lengths are the one Tier-A gap | Moot under LIBERO — it *is* a Panda, and delta-EE actions are embodiment-independent anyway. |
| Tier-1 item 8: perturb, re-solve IK from perturbed state | Produces a single-tick jump the arm can't execute (~330 ms settling). Ramp the correction over ~5 ticks or bound it to ~1 cm. |

---

## Still open

- **Nothing has been run in closed loop since §21's four scene corrections.** This is now
  the single most valuable next action, ahead of both the OSC port and the fine-tune: the
  table was 100 mm too low and every reported `eef_axisangle` was yawed 90°, so §17-19's
  failures were measured against a scene that disagreed with the training distribution in
  ways the model would feel directly.
- **Why the gripper never closes** is still open. §16 fixed two things that could plausibly
  cause it (the `[+x, +x]` state sign, and an actuator polarity that was about to be
  inverted), but neither is confirmed as the cause until a run says so.
- ~~Gripper state is an approximation~~ **RESOLVED in §16**: stock Franka hand mounted,
  `gripper_qpos` read from the same two joints LIBERO reports, in the same metres.
- ~~`--image-flip` resolved in §14~~, ~~REOPENED by §15~~ **RESOLVED in §20**: `180` is
  correct. The benchmark driver rotates both cameras 180° and scores 3/3 on a real task.
  §14's contrary result came from scoring on lateral distance in a scene where nothing
  ever grasps.
- **Which side of our scene is at fault is still open.** §20 proves the fault is ours, not
  the checkpoint's, but changed controller and task together. Next run to settle it: Route
  A driving a LIBERO task.
- ~~Wrist camera pose is our own design~~ **RESOLVED in §16**: replaced with robosuite's
  `eye_in_hand`, copied verbatim.
- ~~Pad friction 0.7/0.6 vs 2.0~~ **RESOLVED in §16**: the stock hand carries robosuite's
  own `friction="2 0.05 0.0001"` / `solref="0.01 0.5"`. (The 2.0 is robosuite's number,
  not DROID's, as `PHASE5_PLAN.md` Tier B has it.)
- **Fine-tune data is generated but unproven.** `libero/fine_tune/a1` (20 reach / 20
  noise / 10 recover) and `a2` (10 Hz probe) are written in the released dataset's exact
  v3.0 schema, but nothing has loaded them through `LeRobotDataset` and no training run has
  used them. Also unresolved: the released dataset declares `fps: 10` while LIBERO's env
  and our loop both run at 20 Hz — `a2` exists to test that.
- ~~Route A is not OSC — no compliance~~ **DEMOTED by §15**: measured hardware penetration
  is 0.1–0.5 mm with actuators unsaturated, so this is not the blocker it was thought to
  be. `--control-mode` hook still not built.
- Transport is ~4× inference cost (§2). Encode frames before POSTing. With the drop to an
  L4 this is now, even more clearly, the only latency worth optimising.
