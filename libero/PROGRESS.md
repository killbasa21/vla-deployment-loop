# LIBERO port — attempt log

Chronological record of what was tried and what actually happened, so the reasoning
behind each change is recoverable and dead ends aren't re-walked. Newest at the bottom.

Format per entry: **what we tried** → **what the result was**. Wrong turns are kept in,
including mine — several conclusions in this log were reversed by later measurements, and
knowing *which* were reversed is the useful part.

Companion docs: `libero/README.md` (current conventions, the spec), `docs/PHASE5_PLAN.md`
(the diagnosis this came out of — see "Corrections to docs/PHASE5_PLAN.md" at the bottom).

---

## 1. Decimation: inference ran the arm 33× too fast

**Tried.** `droid/phase3_closed_loop.py` called `mj_step` exactly once per action. Demo
collection (`droid/phase4_collect_demos.py:283-287`) holds each action for
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

This also **contradicts `docs/PHASE5_PLAN.md` §2.2** ("nothing converged"). The AE converged
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
  standoff, fovy 71.5) a 40 mm ball is **4.9 px @256, 7.3 px @378**. `docs/PHASE5_PLAN.md`'s
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

**Tried.** Assumed, per `docs/PHASE5_PLAN.md` Tier B, that `wrist_cam` "re-aims every frame".

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
3. Only then revisit pad friction (`docs/PHASE5_PLAN.md` Tier B: 0.7/0.6 vs DROID's 2.0).

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

## 22. Route A removed: native OSC, and the arm gains found reverted

**Tried.** Every entry from §11 onward carries the same caveat — Route A (solve IK, command
joint positions) is a stand-in for the operational-space controller LIBERO actually uses, and
§20 left "which side of our scene is at fault" open precisely because it changed controller
and task together. Rather than run the isolating diagnostic first, the controller was
replaced outright.

**What `ctrl` actually is, which is where the confusion lived.** `data.ctrl` is MuJoCo's
actuator input vector; what a number in it *means* is set by the actuator type:

| | robosuite/LIBERO | ours (before) |
|---|---|---|
| declaration | `<motor ctrlrange="-80 80">` | `<general biastype="affine" gainprm="4500">` |
| `ctrl[0:7]` is | **torque, N.m** | **joint angle, rad** |
| set by | OSC: `tau = J^T Lambda F + qfrc_bias + nullspace` | DLS IK solution |

So LIBERO never commanded joint angles. The policy's 7-D delta is not "an OSC thing" either
— it is embodiment-independent by design, and the controller under it is an implementation
detail the model never sees. That is what made swapping it legitimate.

**Built.**

| file | what |
|---|---|
| `mujoco_menagerie/.../panda_libero_osc.xml` | `panda_libero_hand.xml` + torque `<motor>` arm actuators (ADDITION 7), zeroed arm entries in the home keyframe |
| `mujoco_menagerie/.../scene_libero_osc.xml` | `scene_libero_hand.xml` with that include swapped; everything else byte-identical |
| `libero/osc_controller.py` | port of robosuite **1.4.0**'s `OSC_POSE` (the version LIBERO pins — 1.5.2 has an `input_ref_frame` that 1.4.0 lacks and we must not introduce) |
| `libero/tools/verify_osc.py` | four standalone checks, all CPU |
| `libero/libero_closed_loop.py` | `--control-mode {osc,ik}`, default **osc** |

**The timing detail that would have been a silent bug.** robosuite does *not* compute one
torque per policy action and hold it. `environments/base.py:454` loops
`control_timestep/model_timestep` times calling the controller **every physics step**, with
`policy_step=True` only on the first — so `set_goal` fires at 20 Hz and the PD runs at
500 Hz. Holding one torque across a 25-step tick is a different, unstable controller: the
damping term goes blind to the velocity it exists to damp. The stepping loop had to be
restructured for this; IK mode's write-once-then-step is correct only because it commands an
absolute joint target.

**Measured (`verify_osc.py`, all four pass):**

```
[1] velocity convention   J@qvel vs mj_objectVelocity   max|diff| 1e-16    PASS
[2] station keeping       sag 0.000 mm  (position servo: 4.84 mm stock, 2.44 stiffened)
                          settled eef above table 0.2728 vs LIBERO 0.2733
[3] step response         12.3% realised per tick; analytic 1-(1+wn*t)e^-wn*t = 12.6%
[4] contact compliance    120 ticks driving down: -1.11 mm at 35.4 N, rests +14.7 mm
                          above the table   (position servo: -2.9 mm at ~70 N, sec.19)
```

Three things this closes:

- **Droop is gone, exactly.** Sag is 1e-16, not "small". docs/SERVO_DROOP.md is *entirely* about a bug
  that only exists because an overdamped position servo never arrives — the collector's
  `(target - current)` label carried that sag into every frame. A torque controller has no
  joint setpoint to lag behind. It also incidentally makes README §1.1's documentation error
  moot: the claimed "0.2728 vs 0.2733, 0.5 mm" was a *pure-FK* number compared against a
  dynamic one, and the settled pose really was 0.2680. Under OSC the settled pose **is** the
  FK pose, so the claim is now true.
- **Compliance is native**, so `--min-clearance` defaults to OFF in osc mode. It was always a
  departure from LIBERO's control law, and `libero/README.md` already suspected it cost more
  than it bought.
- **`IK unreached` cannot happen.** `opspace_matrices` uses `pinv`, so near a singularity the
  wrench degrades smoothly. Replaced in the logs by a torque-saturation counter.

**One number that is NOT an improvement, and must not be "fixed".** OSC realises **12.3%** of
a commanded delta per tick against the position servo's ~33%. That is correct: it matches the
analytic critically-damped step response at `wn = sqrt(150)` to 0.3 points, and the policy was
trained through exactly this response. A value near 100% would mean the port is **wrong**. My
own pre-measurement guess of "~45%" was wrong, and the analytic check is what caught it —
"does it move a plausible amount" is not falsifiable.

**Smoke test, fake server, identical scripted actions through both modes:**

```
        chunk0 eef_z   chunk1   chunk2   chunk3   clamped
osc        0.187       0.108    0.029    0.003    none
ik         0.121       0.005    0.003    0.003    3/10, 10/10, 10/10
```

Route A slams down and then spends every remaining action against the clamp — the pathology
`libero/README.md` flagged from `stockhand_03`. OSC descends smoothly and never clamps. This
is a scripted-action smoke test, **not** evidence about the policy.

**A measurement error found on the way, and it is not mine this time.**
`panda_libero_hand.xml`'s ADDITION 6 documents the `kp x2, kd x0.7` stiffening at length, and
README §4.1 records it as landed — but the **compiled model reports menagerie's stock 4500/450**.
The actuator lines were reverted at 17:11 on 2026-07-28 while the comment was left in place;
nothing applies gains in Python (grepped). The file is **untracked** inside the gitignored
`mujoco_menagerie` submodule, so the edit never had version control and there is no history
to date it against. Timeline: `a3` 04:24, `a4` 14:27, `ft150` eval 15:18, revert 17:11 — the
datasets and the eval predate it. Moot for OSC (no arm position gains exist there), but it
means an IK-vs-OSC comparison run *today* is against a stock-gain Route A. Verified by
compiling the model and reading `actuator_gainprm`, not by reading the XML — which is now the
standing rule for this directory.

### 22.1 First inference on the OSC path (`osc_molmoact_01`)

12 chunks against the deployed stock `allenai/MolmoAct2-LIBERO` (L4, `GET /act` confirmed
`norm_tag: libero`, `repo_id: allenai/MolmoAct2-LIBERO`). Default settings, so
`--min-clearance` OFF. Server dt ~715 ms, round trip ~3.0-4.0 s warm (first call 37.9 s, cold).

```
 ch  eef_z_mm  ball_moved  ball_z_mm  grip%   sat  clamp
  0     249.6         0.0       20.0      0     0      0
  3      66.7         0.0       20.0      0     0      0
  4      14.8         0.0       20.0      0     0      0
  5      17.2         7.6       19.9     80     0      0
  6      50.2        47.6       20.0    100     0      0
  8      14.8        96.9       19.8      0     0      0
 11      15.4       103.0       20.0      0     0      0
```

**The mechanical pathologies are gone.** Zero table-clamped ticks and zero torque saturation
across all 120 control ticks, *with the clamp disabled* — the arm descends to ~15 mm above the
table and arrests there on its own. Compare §17-19, where the eef drove through the surface
and then dragged across it, and README §9, where two of three runs ended at `IK unreached
10/10` with 444 mm and 229 mm of lateral divergence. None of that happens here. Best lateral
approach **8.1 mm**, against 44.9 / 7.2 / 32.6 mm for the three Route A runs.

**The task still fails.** `ball_z` never leaves 20.0 mm: no lift, no placement. The ball is
shoved 103 mm laterally instead. Gripper closes on 25.8% of actions overall and commits
properly in chunks 5-6 (80%, 100%), so this is not the "never closes" failure of §15 — it is
the *closes beside the ball* failure. `grip_site` arrests at 14.8 mm with the pad centre
3.6 mm behind it, i.e. pads at ~18.4 mm against a ball centre at 20 mm — about **1.6 mm below
the equator**, down from §17's ~7 mm, and still the wrong side of it.

Rotation channel std is healthy (drx/dry/drz = 0.016 / 0.040 / 0.017 against released LIBERO's
0.039 / 0.063 / 0.078), so the stock checkpoint is not collapsed — as expected, it is stock.

**This is ONE run and the policy samples** (§18). It is not evidence of a rate, and it is not
evidence that OSC beats Route A on task success. What it does establish is that the OSC path
runs end to end against the real checkpoint, and that the contact/IK/clamp failure modes which
confounded §12-§19 are absent rather than merely smaller. Scoring the two controllers against
each other needs N runs each through `score_runs.py`.

**Consequence for the datasets, and it is not small.** `a3`/`a4` were collected through
`apply_action`, i.e. Route A, and the collector's design rule is that labels are produced by
the controller that will consume them. Training on them and serving through OSC breaks that
rule. They need regenerating against OSC — free CPU time, but `NOISE_SIGMA_POS` in particular
must be **re-calibrated**, not carried over: it was cut 0.15 -> 0.08 *because* a stiffer plant
realised more of each perturbation, and OSC's per-tick response is different again.

---

## 23. The SmolVLA fine-tune was slow because the expert was — two knobs, one ceiling

**Tried.** The first SmolVLA fine-tune (on `a5`) does the task: it approaches, grasps,
transports, places. It just takes ~54 action chunks to get there. Looked for the cause in
the policy, the chunk horizon, and the serving path. It was in none of them — it was in
the dataset, and it was visible without running anything:

```
a5: 16170 frames / 30 episodes = 539 ticks = 27 s per episode   (a4 was 216 ticks / 10.8 s)
```

The grasp closes around tick 190, i.e. chunk 19. **The policy is reproducing its expert
faithfully. The expert is slow.** Behaviour cloning has no notion of "do this, but
quicker"; episode duration is a property of the data, not a hyperparameter.

**Where the 2.5× came from, and why it was defensible.** `OSC_SPEED_SCALE = 2.5`
multiplies every waypoint duration. It was introduced with §22's OSC port because the
label is the tracking lag over `DELTA_POS_SCALE`, and OSC realises only 12.3% of a
commanded delta per tick:

```
label = lag / DELTA_POS_SCALE,   lag ~= v * dt / realised
v_max = DELTA_POS_SCALE * realised / dt = 0.05 * 0.123 * 20 = 0.123 m/s
```

The ik-era timings exceed that on the retreat (0.27 m/s), so `dx` pinned at −1.000 in
every cohort. Saturated labels destroy direction information — every clipped tick reports
"full scale" regardless of the true error — so the concern was real and correctly
identified.

**Result: the fix did not work, and cost 2.5× the episode length to not work.** Measured
on `a5` itself: `dx` is still saturated on **3.07%** of frames, `q01` still exactly
−1.000. Slowing the reference lowers the lag *and* the ticks-per-metre proportionally, so
past a point it buys nothing — the ceiling is set by the action **scale**, not the clock.
Two knobs, one constraint, and the clock was the wrong one:

| | | ticks/ep | dx saturated | dx q01 |
|---|---|---|---|---|
| `a5` | scale 0.05, speed-scale 2.5 | 539 | 3.07% | −1.000 |
| | scale 0.125, hand-set timings | 216 | 2.20% | −1.000 |
| | scale 0.15, hand-set timings | 216 | 0.93% | −0.987 |

Raising `DELTA_POS_SCALE` raises the arm's top speed instead of lowering the expert's, and
SmolVLA normalises actions MEAN_STD from the dataset's own stats — so the absolute unit is
invisible to training. Only saturation is not.

**Then: the timings were uneven, which neither knob addresses.** Printing per-segment mean
speed against the ceiling showed the durations had never been set against a speed budget
at all:

| segment | dist | mean v | vs 0.246 m/s budget @ scale 0.15 |
|---|---|---|---|
| retreat / re-approach | 0.124 m | 0.275 | **over** — this is the residual clipping |
| closing return to start | 0.285 m | 0.285 | **over** |
| descent onto the ball | 0.100 m | 0.125 | half the budget — wasted ticks |
| transport | 0.252 m | 0.126 | half the budget — wasted ticks |

So the same trajectory simultaneously clipped labels *and* crawled. `--motion-speed`
retimes every motion segment to one target mean speed, leaving dwells alone (they are
settling and gripper-actuation time, set by physics, not distance). The default is 0.90 ×
the ceiling ÷ 1.5, the 1.5 being smoothstep's peak-over-mean.

**Result, 4 episodes per setting, all placed:**

| scale | motion speed | ticks/ep | dx saturated | dx q01 | OSC torque saturation |
|---|---|---|---|---|---|
| 0.15 | 0.221 | 194 | **0.00%** | −0.768 | 0.05% of steps |
| **0.20** | **0.295** | **161** | **0.00%** | **−0.681** | 0.11% |
| 0.25 | 0.369 | 136 | 0.00% | −0.618 | 0.14% |
| 0.30 | 0.443 | 126 | 0.00% | −0.553 | 0.21% |

Retiming removes the clipping **entirely** at every scale — the unevenness was the whole
of the residual, not the scale. Chose **0.20**: `dx q01 = −0.681` against released LIBERO's
−0.679, i.e. a near-exact match to the distribution the checkpoint was pretrained on, at
**161 ticks — 3.3× shorter than `a5`**. Past 0.25 the returns collapse (136 → 126 ticks for
20% more scale) while the label distribution thins and torque saturation climbs.

**The learning worth keeping.** Three constants were tuned against each other without a
shared frame: `DELTA_POS_SCALE` (what one action unit means), the waypoint durations (how
fast the expert moves), and `OSC_SPEED_SCALE` (a correction bolted onto the second when the
plant changed). They all resolve to one physical quantity — commanded speed against
`v_max = scale * realised / dt` — and none of them was written in those terms, so the
correction landed on the knob that was easiest to reach rather than the one that binds.
**When a constant gets a correction factor, check whether the correction belongs to a
different constant.** The tell here was available from the start and unread: `a5` was
collected specifically to stop `dx` clipping, and its own `stats.json` shows `dx` min =
−1.000.

### 23.1 The bins were randomised on one side only

Found while checking the above. `bin_layout()` permutes green/blue/yellow across three
slots with ±2 cm jitter every episode, so the policy has to find green by colour.
`libero_closed_loop.py` had `--randomize-ball` and **no bin equivalent** — every run since
the scene was built evaluated the XML's single layout, green at (0.56, 0.25). In `a5` that
layout drew **6 of 30 episodes**:

| green bin at | a5 episodes | ever evaluated? |
|---|---|---|
| (0.80, 0.00) | 14 | no |
| (0.56, −0.25) | 7 | no |
| (0.56, +0.25) | 6 | **yes — all of it** |
| jittered variants | 3 | no |

So 80% of training taught reaching toward bins that are not on the table at inference, and
the colour-grounding the shuffle exists to buy was never once tested. Not a bug in either
file — a disagreement between two files that each looked correct alone, and the kind that
cannot produce an error message. `--randomize-bins` now exists on the client, drawing from
`BIN_SLOTS` **imported from** `libero_closed_loop.py` rather than a second copy;
`--bin-layout scene` is the opposite resolution (pin both) if colour grounding is not the
goal. Randomise both or pin both; randomising one is the only indefensible option.

**Generalises past this repo:** domain randomisation is a property of the *pair*, not of
the collector. Any axis randomised in training and fixed at evaluation silently converts
into wasted sample budget, and the eval still passes — it just measures one draw.

### 23.2 `a6` as collected

30 episodes (10 reach / 10 noise / 10 recover), scale 0.20, distance-retimed, shuffled
bins, seed 0. **30/30 placed, 1 rejected attempt, 447 s of CPU.**

| | `a4` | `a5` | **`a6`** |
|---|---|---|---|
| ticks / episode | 216 | 539 | **161** |
| frames | 19440 | 16170 | 4825 |
| dx saturated | 0.35% | 3.07% | **0.00%** |
| dx q01 (released LIBERO: −0.679) | −0.679 | −1.000 | **−0.807** |
| plant | ik | osc | osc |

**One thing the 4-episode sweep missed, because I was only checking the translation
channels: `ry` saturates on 0.54% of frames** (std 0.146 against `a4`'s 0.031). Cause is
mine — I raised `DELTA_POS_SCALE` and left `DELTA_ROT_SCALE` at 0.5, so faster motion makes
the wrist's orientation lag further while the divisor stayed put. All 26 saturated frames
land at **episode fraction 0.96-0.99**, i.e. inside the closing return-to-start, after the
ball is already in the bin. Shipped as-is: it is post-task, and `--delta-rot-scale` would
be a second wire convention to keep in sync across two processes for no task-relevant gain.
Worth revisiting only if the retrain shows orientation problems on the return.

**Method note.** The sweep that chose 0.20 printed `dx` and `dz` saturation and called the
setting clean. A per-channel check over all six would have caught `ry` before the
collection ran, not after. When a change scales one axis of an action space, the summary
has to cover the axes that were *not* scaled — those are where the coupled error shows up.

### 23.3 `--policy.use_peft=true` never created a LoRA

The first `a6` training smoke died 29 s in, before any GPU work:

```
ValueError: Can't find 'adapter_config.json' at 'HuggingFaceVLA/smolvla_libero'
```

`--policy.use_peft=true` had been in `smolvla_modal_train.py`'s `lora` mode since it was
written, with a comment asserting it was "SmolVLAConfig's own switch ... confirmed against
the installed 0.6.0 dataclass fields". The field exists; it does not mean what the comment
assumed. In lerobot 0.6.0 `factory.make_policy` reads it as *"this checkpoint IS a PEFT
adapter — load it"*, so it looks for an `adapter_config.json` beside the base checkpoint
and dies when the stock policy has none. Creating a **new** adapter is a top-level train
field, `--peft.*`, which `lerobot_train.py` turns into `policy.wrap_with_peft(...)` — and
that call sets `config.use_peft = True` itself on the way out, which is what makes the
saved checkpoint loadable with `use_peft` later. Passing it on the way in is the error.

Confirming the field exists is not confirming what it does. The dataclass check that was
run would pass identically for both meanings.

Second thing the fix surfaced: SmolVLA's default LoRA targets are

```
(model.vlm_with_expert.lm_expert.*.(q|v)_proj | model.(state_proj|action_*_proj|action_time_mlp_*))
```

— the **action expert's** attention projections and the state/action heads, and *not* the
VLM. So `lora` mode never adapted the VLM the way its comment claimed. Left at the default
deliberately: episode speed is encoded in action magnitudes, which is the expert's job, not
a grounding problem the VLM has to relearn. `freeze_vision_encoder` / `train_expert_only`
were dropped from the mode at the same time — `wrap_with_peft` calls `requires_grad_(False)`
on every base parameter regardless, so they were dead flags that read as live ones.

Second smoke passed: PEFT wrapped, **2.35 M trainable of 607 M**, 4825 frames / 30
episodes, one step, checkpoint saved, 65 s wall.

### 23.4 The `a6` retrain

5000 steps, batch 16 (16.6 epochs over a6), LoRA r=32 on SmolVLA's default targets, L4,
50m35s at 1.65 step/s, exit 0. Checkpoints every 1000 on
`molmoact2-checkpoints:/smolvla/smolvla-a6-lora/checkpoints/`.

| step | epoch | loss | grdn | lr |
|---|---|---|---|---|
| 200 | 0.66 | 1.313 | 0.254 | 1.0e-05 |
| 1000 | 3.98 | 0.617 | 0.333 | 8.9e-05 |
| 2000 | 7.30 | 0.492 | 0.427 | 6.3e-05 |
| 3000 | 10.61 | 0.440 | 0.458 | 3.3e-05 |
| 4000 | 13.93 | 0.429 | 0.473 | 1.0e-05 |

Flat by 3000 (0.440 → 0.429 over the next 1000 steps, with the LR already decayed to
floor), so the last fifth of the run bought nothing measurable.

**This is training loss on 30 episodes with no held-out split.** It says the adapter fit
`a6`; it says nothing about closed-loop behaviour, and `a5`'s fine-tune also trained
cleanly before serving slow. The 3000 and 4000 checkpoints are the A/B candidates if 5000
disappoints — with 3.3x fewer frames than `a5` at the same step count, overfitting is now
the plausible failure where `a5` was underfitting.

### 23.5 The retrain IS fast, and it misses the ball

Served checkpoint 005000 (and 003000) through `smolvla_modal.py` on a T4 and ran the closed
loop at `--delta-pos-scale 0.20`. Serving needed one change: a fine-tune is a LoRA
**adapter**, so `load()` now branches on `adapter_config.json` existing -- base policy from
`base_model_name_or_path`, adapter applied, `merge_and_unload()`. Note the symmetry with
sec.23.3: `use_peft=true` was wrong at TRAIN time (no adapter existed yet) and is right at
SERVE time (one does), which is why the branch tests for the file rather than a flag.

**The speed problem is solved.** Both runs execute the entire pick-place-return sequence in
~20 chunks against `a5`'s ~54, with the phases in the right order and the right proportions:

| | close gripper | lift | at bin | released | back at start |
|---|---|---|---|---|---|
| `a5`-trained (reported) | ~chunk 19 | — | — | — | — |
| `a6` ck5000 | chunk 3-4 | chunk 5 | chunk 8 | chunk 15-16 | chunk 18-19 |

**And the ball never moves.** Three runs, `tracked_object_xpos` identical at the first and
last chunk in all three:

| run | ball | first close | eef at close | lateral err | vertical |
|---|---|---|---|---|---|
| ck5000, randomised ball | (0.569, 0.104) | chunk 10 | (0.605, 0.024, 0.004) | **88 mm** | on the table |
| ck5000, nominal ball | (0.560, 0.000) | chunk 3 | (0.526, 0.010, 0.007) | **36 mm** | on the table |
| ck3000, nominal ball | (0.560, 0.000) | chunk 4 | (0.570, 0.004, 0.048) | **11 mm** | **40 mm high** |

So the policy has learned the *shape* of the task and the *timing* of it, and closes on
air. ck3000 is laterally three times more accurate than ck5000 (11 mm vs 36 mm), which is
what over-training on 30 episodes looks like — but it stops its descent 40 mm above the
ball, so it fails too.

**Leading explanation, and it is a cost of sec.23's own fix.** One action unit is now
0.20 m instead of LIBERO's 0.05, so every action the policy emits is **4x coarser in
metres**. The label distribution shrank to match — measured across the scale sweep, dx std
went 0.232 (scale 0.15) -> 0.154 (scale 0.30) — and terminal positioning is exactly where
the residual error is smallest and therefore where the labels are closest to zero. A 10 mm
correction was a 0.20 action at LIBERO's scale and is a 0.05 action at ours.

The decisive comparison is the one already in hand: **`a5` was slow and grasped the ball;
`a6` is fast and misses it.** Same scene, same collector, same cohorts, same plant. What
changed between them is the scale and the timing.

Quantified, this is not hand-waving. Under MEAN_STD normalisation the metres an action
carries per unit of *normalised* policy error is `std(dx) * DELTA_POS_SCALE`:

| dataset | scale | ticks/ep | dx std | mm per 1.0 normalised error |
|---|---|---|---|---|
| a4 | 0.05 | 216 | 0.205 | 10.2 |
| **a5** (slow, **grasps**) | 0.05 | 539 | 0.309 | **15.5** |
| sweep | 0.15 | 194 | 0.232 | 34.8 |
| **a6** (fast, **misses**) | 0.20 | 160 | 0.200 | **40.0** |
| sweep | 0.30 | 126 | 0.154 | 46.1 |

`a6`'s measured 36 mm miss is ~0.9 units of normalised error. The same 0.9 at `a5`'s
resolution is 14 mm — inside the ball's 20 mm radius, i.e. a grasp. The
`a5`-grasps/`a6`-misses split falls out of this table with no additional assumption, and
the ratio (2.6x) is not something a policy trained longer can recover: it is the units.

So there is a **speed/precision frontier** here, set by the plant's 12.3% realised
fraction, and sec.23 moved along it rather than beating it. 0.20 is past the knee.

If that reading is right, the fix is a middle setting rather than a return to 0.05. At `--delta-pos-scale 0.10`, `--motion-speed` retiming gives
~0.147 m/s and ~256 ticks/episode: still 2x faster than `a5`, with 2x the terminal
resolution of `a6`. Not yet run.

**The competing explanation has not been excluded**: 30 episodes may simply be too few for
reliable grounding, in which case the scale is innocent and the fix is more data or a LoRA
that actually adapts the VLM (sec.23.3 -- the current one targets only the action expert).
A run of `a6` at `--delta-pos-scale 0.05` would separate these: if a scale the model was
not trained at still grasps better, the coarseness reading is wrong.

**Procedural note.** The first ck3000 run was thrown away: `modal deploy` returned in 6 s
and `/health` still reported ck5000, because the previous container was inside its 300 s
`scaledown_window` and answered from the old deployment. A deploy is not a cutover. Poll
`/health` until it reports the checkpoint you meant to test — and note that the discarded
run showed the ball moving 178 mm, i.e. it looked like the most successful run of the set.
Contaminated results do not announce themselves by looking wrong.

 The failure mode this cannot fix is a
policy that is slow because it is *uncertain* — hedging toward a small action because the
chunk distribution is multimodal — and that would look identical from the outside. If `a6`
retrains to ~161-tick episodes, the diagnosis holds; if it stays at 500+, the slowness was
never the expert's.

---

## 24. `a7` collection killed the terminal twice — the writer holds every raw frame

**Symptom.** Two `a7` runs (60 episodes) simulated all 60 episodes, printed every
`ticks=... placed=1` line, and then died without writing `stats.json` or `cohorts.json`.
Disk was never the problem: 69 GB free throughout. Nothing appeared in the kernel log,
which is why it read as "the terminal crashed" rather than as an OOM.

**Cause.** `lerobot_v30_writer.py` buffers **raw uint8 frames** for every episode until
`finalize()`, and its own docstring said so: *"a few hundred MB for the 50-ish short
episodes this is used for; it would need a streaming rewrite for thousands."* a7 is not
thousands of episodes, but it is 4x a6's frames, and the frames are what matter:

| | frames | x2 cameras, raw @ 196.6 kB | fits in 15 GB? |
|---|---|---|---|
| a6 | 4825 | 1.9 GB | yes |
| a5 | 16170 | 6.4 GB | just |
| **a7** | **20034** | **7.9 GB** | **no** |

Then `finalize()` asks for more, on top of that 7.9 GB: PNG-encoding every frame into
Arrow buffers, and `_image_stats` materialising `np.asarray(frames, np.float32) / 255.0`
— a float32 copy 4x the size of the uint8 it comes from, 0.26 GB per camera per 334-frame
episode, and a 3.1 GB spike for the dataset-wide pass. It died at the single most
expensive moment available: after every episode had been simulated, before anything was
recoverable.

**Fix — encode and measure at ingest.** `add_episode` now PNG-encodes each frame while the
caller still holds it and stores only the bytes, and folds the same frame into a streaming
`ImageStatsAccumulator` (sum / sumsq / min / max in float64, O(1) memory). Nothing raw
survives the call.

```
raw uint8 256x256x3   196.6 kB/frame      a7: 7.9 GB
PNG of this scene      ~12   kB/frame      a7: ~0.5 GB   <- 16x
```

Encoding at ingest is not extra work — every frame is PNG-encoded exactly once either
way. It just happens spread out instead of in one spike. Measured on a 4-episode /
1340-frame collection: **peak RSS 780 MB**.

Two things fell out of the rewrite:

- **The old image stats were less accurate, not more.** Summing 3.2M pixels in float32
  loses precision: against an exact float64 reference the old path errs 1.5e-05 on the
  mean and 2.9e-04 on the std, the accumulator 2.7e-14 and 5.3e-13. The "mismatch" when
  comparing old against new is the old code.
- **`_image_stats`'s docstring described an optimisation that was not in the code** —
  "frames are subsampled because the exact mean of every pixel of every frame is not worth
  the memory", in a function that converted all of them. The dataset-wide pass did
  subsample (every 5th frame); the per-episode one never did. The accumulator makes the
  question moot and the global stats now cover every frame.

### 24.1 The dead `a7` was recovered rather than re-collected

A parallel agent wrote `libero/fine_tune/rebuild_stats.py`, which recomputes `stats.json`
from the parquets, and repaired the tree. Its reasoning matched this section's
independently, but "the parquets are complete and authoritative" is exactly the claim a
crash *during the parquet-writing stage* puts in doubt, so it was checked rather than
taken:

- 20034 rows = `info.total_frames`; episodes 0..59 all present
- per-episode lengths match `meta/episodes`; `index` contiguous 0..N-1; `frame_index`
  restarts at 0 per episode
- frames decode at (256,256,3) **including the last three rows**, where a truncated write
  would show
- rebuilt `stats.json` matches the data to float32 precision (action mean 1.0e-08)

Complete. `meta/cohorts.json` is not recoverable — it lives only in the collector's memory
— so `a7` cannot be split by cohort for ablations, which no training or serving path needs.

**`a7` as it stands:** 60 episodes, 20034 frames, scale 0.10, 334 ticks/episode.
Resolution 23.5 mm per unit of normalised error, between `a5`'s 15.5 and `a6`'s 40.0.
`dx` clips on 1.35% of frames against `a5`'s 3.07% — the noise cohort's sigma scales
inversely with the action scale, so a lower scale perturbs harder in action units.

---

## 25. Both fine-tunes measured properly — the gripper is the bottleneck, not the geometry

**Protocol.** 10 rollouts each, identical seeds 1-10, 70 chunks, `--randomize-ball
--randomize-bins`, each served at the scale its dataset was collected at, scored by
`score_runs.py` (ball lifted 50 mm at some point, and ending inside the green bin's
footprint).

| | placed | grasp-and-lift | gripper ever closed |
|---|---|---|---|
| `a5` (scale 0.05, 539-tick expert) | **2/10** | **3/10** | 5/10 |
| `a7` (scale 0.10, 334-tick expert) | **1/10** | **2/10** | 5/10 |
| stock checkpoint (README sec.9, n=3) | 0/3 | 1/3 | — |

2/10 against 1/10 at n=10 is noise, and `a7` had three runs truncated early (21, 13, 65
chunks) which could not succeed, so its denominator is if anything unfair. **The two
fine-tunes are indistinguishable on task success, and neither is clearly above a
three-run stock baseline.**

**The speed fix is real and did not help.** On the two seeds where both models succeeded,
`a7` closed the gripper at chunks 10 and 7 where `a5` took 31 and 20 — the ~3x the dataset
predicted. It converts into no additional placements.

### 25.1 What actually gates success

`score_runs.py`'s `close` column decides every run. Across **all 20 rollouts, every single
lift and every single placement came from a run where the gripper fired at all** — and it
fired in exactly 5 of 10 for both models. In the other half it never closed, and the arm
either hovered over the ball or pushed it hundreds of millimetres across the table.

Closure is also close to uncorrelated with whether the hand is on the ball:

| | closest lateral | gripper |
|---|---|---|
| `a5` run 8 | **2.1 mm** | never closed |
| `a5` run 1 | **0.7 mm** | closed at chunk 58, far too late |
| `a5` run 9 | 37.0 mm | closed at chunk 16 |
| `a7` run 4 | 90.8 mm | closed at 52, reopened at 53 |
| `a7` run 8 | 39.1 mm | full close-transport-release on an empty hand |

The two models fail in mirror-image ways — `a7` commits early (chunks 7, 8, 8, 10)
whether or not the ball is there, `a5` closes late or not at all — which is the same
missing capability, *closure conditioned on actually having the object*, expressed as
opposite biases.

### 25.2 Where that leaves sec.23-24

The resolution model (15.5 / 23.5 / 40.0 mm per unit of normalised error for a5 / a7 / a6)
is arithmetically correct and explains why `a6` could not touch the ball at all. It does
**not** explain this data: `a7` has finer resolution than `a6` and reaches *worse than
`a5`* on matched seeds (closest approaches 102 / 12.6 / 49.3 / 90.8 mm against `a5`'s
0.7 / 20.6 / 34.7 / 15.1 mm). A model that predicts the opposite of what is observed is
not the operative mechanism here, whatever its arithmetic.

Honest summary of the whole sec.23-25 arc: **the slowness was real, was diagnosed
correctly, was fixed — and was not the thing standing between this policy and the task.**
Three datasets and two fine-tunes were spent on the action space's geometry while the
binding constraint was a binary channel that neither dataset teaches reliably.

### 25.3 Two things before any further training

- **Runs truncate and I cannot say why.** Five rollouts across the two batches died early
  (a5: 66, 61; a7: 21, 13, 65) with stdout sent to `/dev/null`. `libero_closed_loop.py`
  exits rather than retries when a request fails, so one transient server error ends a
  rollout. Every rate in this section carries that noise. Capture stderr, and add a retry.
- **Attack closure, not geometry.** The expert only ever demonstrates closing on a
  stationary, perfectly-centred ball, and rejection sampling on `lifted and placed`
  deletes every episode where contact went wrong (sec.23.5). The states the policy
  actually occupies at decision time are therefore absent from the data by construction.

---

## 26. The pick target is a cube now, not a sphere

**Changed, 2026-08-03.** `green_ball` (sphere, r=0.02) → `green_box` (box, half-extent
0.02 0.02 0.02) in all four scene XMLs. Same 40 mm across a face, same 0.05 kg, same
material, same contact params. The instruction changed with it:

```
- pick up the green ball and put it in the green container
+ pick up the green box  and put it in the green container
```

**Nothing has been run yet.** This section records a change and its reasoning, not a
result. Every number in sec.1-25 was measured on the sphere.

### 26.1 Why

sec.25.1 is the whole argument. Across all 20 rollouts of the two MolmoAct2 fine-tunes,
every lift and every placement came from a run where the gripper fired at all, and it
fired in exactly 5 of 10 for both models. Closure was near-uncorrelated with whether the
hand was on the object: `a5` run 8 reached **2.1 mm** and never closed; `a7` run 8 did a
full close-transport-release on an empty hand.

A sphere is the worst possible object for that failure. It has one contact point, no face
to align to, and it rolls out from under a closing gripper — which is exactly why
`scene_libero_*.xml` carries a block of hand-tuned contact params (`priority 2`,
`condim 4`, stiff `solref`/`solimp`) that exists *only* because "a 40 mm sphere between
two flat pads has no form closure". Those params were added on 2026-07-28 after tracing
the fingers seating at 0.0195 m for ~17 ticks and then driving through to 0.0000 with the
ball shooting out sideways.

A cube has form closure. It also moves the task **toward** the pretraining distribution
rather than away from it: every object in LIBERO's 130 tasks is a box, a can or a cylinder,
and sec.20's 3/3 on `libero_object` was picking up a rectangular soup carton.

Third, smaller reason: a 40 mm sphere is 16.0 px at the matched agentview. A cube of the
same width is the same across a face and slightly more across a diagonal, so the pixel
budget does not get worse.

### 26.2 What this costs

**Every dataset in the repo is now a dataset of a different task.** `a1`-`a7`, the
converted `a4_smolvla` / `a7_smolvla`, `green_ball_pick`, and every checkpoint trained on
any of them. That includes ACT ck10000's 5/6 placed — the best result the project has —
and the 2/10 and 1/10 of sec.25.

None of it was deleted and none of it was reworded. The Modal volume paths keep their
`green_ball` names on purpose: they hold ball data, and renaming them would make a real
artifact lie about its contents. New box datasets get new names.

Re-collection is free CPU time, so the cost is wall-clock, not money. The cost that is
*not* recoverable is that sec.25's comparison table no longer has a live opponent: the
next box number has nothing to be compared against until at least one policy is retrained.

### 26.3 What changed in the code

- **Scenes** — `scene_libero.xml`, `scene_libero_hand.xml`, `scene_libero_osc.xml`,
  `scene_pick_place.xml`. Verified by compiling all four: `geom_type == mjGEOM_BOX (6)`,
  `geom_size == [0.02, 0.02, 0.02]`. Not by reading them, per the rule in sec.22.
- **The contact params were deliberately NOT retuned.** A cube does not need `condim 4`'s
  torsional friction. Keeping them holds the box scene one variable away from the ball
  scene; each XML now says so where the old rationale used to be.
- **Identifiers** — `BALL_RADIUS` → `BOX_HALF`, `BALL_SAMPLE_*` → `BOX_SAMPLE_*`,
  `set_ball_radius` → `set_box_half_extent`, `--randomize-ball` → `--randomize-box`,
  `--ball-radius` → `--box-size` (still a half-extent, so the numeric meaning is
  unchanged), `build_sim(randomize_ball=, ball_radius=)` → `(randomize_box=, box_half=)`.
- **`--box-size` now checks the DIAGONAL**, `2·half·√2`, not the face width. A cube at an
  arbitrary yaw presents up to 56.6 mm where the face presents 40 mm, and after a failed
  grasp the yaw is arbitrary. A sphere had no such distinction. Verified: `--box-size 0.03`
  is now refused (60 mm box, 85 mm diagonal, 80 mm hand) where the old face-width test
  would have accepted it.
- **Log key** — `ball_radius` → `box_half`. `score_runs.py` reads either, so old ball logs
  still score correctly; the key that is present is also the only thing in the log format
  that records which object a run used, which is why they were not collapsed into one name.
- **The instruction moved to `infra/task_spec.py`** and is imported by
  `libero_closed_loop.py`, `act_modal.py`, `smolvla_modal.py`, `phase3_closed_loop.py` and
  `phase4_collect_demos.py`. It was five independent string literals. A prompt that differs
  between collection and serving does not raise — it silently conditions the policy on
  something it never trained on, the same failure shape as sec.5's hardcoded `NORM_TAG`.

Incidental fix found on the way: `collect_finetune_data.py --help` crashed with
`ValueError: unsupported format character` on four unescaped `%` in argparse help strings.
Pre-existing, unrelated, fixed.

### 26.4 What to expect, stated in advance so it can be wrong

The prediction this change is worth making, before any measurement:

1. **The expert's keep rate should go up**, because rejection sampling is currently
   deleting episodes where a sphere squirted out of the grasp (sec.23.5).
2. **Closure should become better correlated with lateral distance**, because a cube that
   is contacted off-centre gets pushed rather than rolled away.
3. **The stock-checkpoint baseline should improve at least slightly**, because the object
   is now the shape LIBERO pretrained on.

If (1) holds and (3) does not, the object shape was never the constraint and sec.25.1's
diagnosis needs revisiting. If none of them hold, this change bought nothing but a cleaner
scene, and that should be written down here rather than quietly absorbed.

**Do not read any of this as a result.** Re-collect, retrain, re-score.

---

## Corrections to `docs/PHASE5_PLAN.md`

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
- **Which side of our scene is at fault is still open**, and §22 changed how to settle it.
  §20 proved the fault is ours rather than the checkpoint's, but changed controller *and*
  task together. The proposed isolating run was "Route A driving a LIBERO task" — that
  question is now largely retired, since the controller under test *is* OSC. The remaining
  form is: **OSC on our green-ball scene**. If it still fails where `libero_benchmark_eval.py`
  scores 3/3, the controller was never the blocker and the scene/task is simply out of
  distribution, which points squarely at the fine-tune.
- ~~Wrist camera pose is our own design~~ **RESOLVED in §16**: replaced with robosuite's
  `eye_in_hand`, copied verbatim.
- ~~Pad friction 0.7/0.6 vs 2.0~~ **RESOLVED in §16**: the stock hand carries robosuite's
  own `friction="2 0.05 0.0001"` / `solref="0.01 0.5"`. (The 2.0 is robosuite's number,
  not DROID's, as `docs/PHASE5_PLAN.md` Tier B has it.)
- **Fine-tune data is generated but unproven.** `libero/fine_tune/a1` (20 reach / 20
  noise / 10 recover) and `a2` (10 Hz probe) are written in the released dataset's exact
  v3.0 schema, but nothing has loaded them through `LeRobotDataset` and no training run has
  used them. Also unresolved: the released dataset declares `fps: 10` while LIBERO's env
  and our loop both run at 20 Hz — `a2` exists to test that.
- ~~Route A is not OSC — no compliance~~ ~~**DEMOTED by §15**~~ **RESOLVED in §22**:
  `--control-mode osc` runs a port of robosuite 1.4.0's `OSC_POSE` on torque actuators.
  Sag 0.000 mm, penetration −1.11 mm at 35.4 N, no `IK unreached`. Not yet run against
  the model.
- ~~The OSC path has never seen inference~~ **DONE, §22.1**: 12 chunks against the stock
  checkpoint. No clamping, no saturation, no IK failures, best lateral 8.1 mm — and still
  0/1 on lift and placement. **The open item is now the RATE**: N runs of osc vs ik through
  `score_runs.py`, against README §9's 0/3 placed, 1/3 lifted. One run is one draw.
- **The grasp closes ~1.6 mm below the ball's equator** (§22.1), down from ~7 mm. That is
  the remaining mechanical gap between "shoves the ball" and "lifts it", and it is now
  small enough to be worth a deliberate sweep of where the eef arrests.
- ~~**`a3`/`a4` are controller-mismatched**~~ **DONE**: `a5` was the OSC re-collection, and
  `a6` (§23) supersedes it — same plant, `DELTA_POS_SCALE 0.20`, distance-retimed, 161
  ticks/episode against `a5`'s 539.
- **Does the retrain actually get faster?** (§23) `a6` exists; nothing has trained on it.
  The diagnosis predicts ~161-tick episodes, i.e. ~16 chunks against the current ~54. If it
  retrains slow anyway, the slowness was never the expert's and the next suspect is chunk
  multimodality — a policy hedging toward small actions because the action distribution it
  fits is multimodal.
- **`--randomize-bins` has never been run against a model** (§23.1). It corrects a
  train/eval mismatch that was silently costing 80% of the sample budget, but every number
  in this log predates it, so no baseline is comparable across it.
- **`--action-scale` is unmeasured.** The cheap check on the deployed `a5` checkpoint —
  gain 2.0 on the pose channels — would confirm §23's diagnosis against the model rather
  than only against the dataset. It costs one run and has not been done.
- **The arm gains in `panda_libero_hand.xml` are stock, not stiffened** (§22), contrary to
  its own comment and top-level README §4.1. Untracked file in a gitignored submodule, so
  no history. Decide whether the stiffening should be restored for the `ik` path, or
  whether that path is now purely a reproducibility fallback and should stay as-is.
- Transport is ~4× inference cost (§2). Encode frames before POSTing. With the drop to an
  L4 this is now, even more clearly, the only latency worth optimising.
