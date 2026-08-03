# `libero/` — MolmoAct2-LIBERO closed loop

> **Pick target changed 2026-08-03: 40 mm sphere → 40 mm cube**, instruction now *"pick up
> the green box and put it in the green container"*, scene identifiers renamed
> `green_ball*` → `green_box*`. Rationale and full change list in
> [`libero/PROGRESS.md` §26](PROGRESS.md).
>
> **Everything measured on this page was measured on the ball**, including every dataset and
> checkpoint it names. The prose below has deliberately NOT been reworded — the measurements
> describe a sphere and would be false if restated about a cube. Command lines HAVE been
> updated, since the old flag names no longer exist.

Self-contained port of the phase-3 closed loop to the **`allenai/MolmoAct2-LIBERO`**
checkpoint. Deliberately decoupled from `droid/phase3_closed_loop.py` (copy-paste, not import)
so the DROID path keeps working untouched while this one is in flux.

> **2026-07-28 (later): Route A replaced by a native OSC port — see `PROGRESS.md` §22
> and "Control" below.** `--control-mode osc` is the default, on the new
> `scene_libero_osc.xml` (torque actuators). Servo droop, table penetration and
> `IK unreached` all go away; `--min-clearance` defaults to off. **`a3`/`a4` were
> collected through Route A and must be regenerated before the next fine-tune** — the
> collector's rule is that labels come from the controller that consumes them.
> Also found while doing it: `panda_libero_hand.xml`'s arm gains read menagerie's stock
> 4500/450 in the *compiled model*, despite its own `ADDITION 6` comment and README §4.1
> documenting a `kp×2 kd×0.7` stiffening. Untracked file, no history. Verify gains by
> compiling, not by reading the XML.

> **2026-07-28: four measured scene corrections landed — see `PROGRESS.md` §21.**
> The table was 100 mm too low relative to the robot base, `grip_site` was 9.5 mm out and
> yawed 90°, `LIBERO_INIT_QPOS` was not LIBERO's reset pose, and `LIBERO_ORIGIN_OFFSET`
> had the wrong x. Numbers in this file that predate that are stale. `--image-flip` is
> resolved: leave it at `none`. Format and dataset findings are in `fine_tune/README.md`.

## Why this checkpoint

MolmoAct2-**DROID** is pretrained on real-robot teleop footage of an **FR3**. Our scene is
MuJoCo renders of a **Panda**, which is a distribution gap on two axes at once (photoreal
vs flat-shaded, and different link lengths).

MolmoAct2-**LIBERO** is pretrained on [LIBERO](https://libero-project.github.io/), a
**robosuite/MuJoCo benchmark on a Franka Panda** — same simulator family, same arm, same
two-camera layout. The sim-to-real visual gap and the FR3/Panda mismatch both disappear.

## Conventions (verified against the vendored repo)

| | DROID (`droid/phase3_closed_loop.py`) | LIBERO (here) | source |
|---|---|---|---|
| `norm_tag` | `franka_droid` | **`libero`** | `data_mixtures.py:322` |
| `setup_type` | `single franka robotic arm in droid` | `single franka robotic arm in libero` | `data_mixtures.py:331` |
| `control_mode` | `absolute joint pose` | **`delta end-effector pose`** | `data_mixtures.py:332` |
| action dim | 8 = `[q1..q7, gripper_rad]` | **7** = `[dx,dy,dz,drx,dry,drz,grip]` | `envs/libero.py:85` |
| action range | raw radians | **[-1, 1]** normalized | `envs/libero.py:86-87` |
| `action_horizon` | 15 | **10** | `data_mixtures.py:333` |
| control rate | 15 Hz | **20 Hz** (robosuite default) | robosuite OSC docs |
| state dim | 8 = `[q1..q7, gripper_rad]` | 8 = `[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]` | `env_processor.py:68-75` |
| quat order | — | **(x, y, z, w)** — MuJoCo gives (w,x,y,z), must reorder | `env_processor.py:120` |
| cameras | `external_cam`, `wrist_cam` | `agentview_image`→`image`, `robot0_eye_in_hand_image`→`wrist_image` | `envs/libero.py:140` |
| render size | 378×378 | **256×256** | `envs/libero.py:109-110` |
| agentview mount | `targetbody`, 1.445 m, fovy 71.5 | **`fixed`, 0.707 m, fovy 45** | robosuite `table_arena.xml` |
| 40 mm ball @256 | 4.9 px (sub-patch) | **16.0 px** | measured + analytic, agree exactly |
| no-op action | — | `[0,0,0,0,0,0,-1]` → gripper **-1 = open** | `envs/libero.py:82` |

### Delta scaling

robosuite's `OSC_POSE` maps the normalized action onto a physical delta with
`output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`:

```
delta position  = action[0:3] * 0.05   metres
delta rotation  = action[3:6] * 0.5    radians (axis-angle, world frame)
```

So one action is at most 5 cm of translation and 0.5 rad of rotation. At 20 Hz that caps
the arm at 1 m/s, which is why a full-scale action is a big move, not a nudge.

## Control: native OSC (default), or Route A

> **2026-07-28: `--control-mode osc` is now the default and Route A is the fallback.**
> The sections below describe both. See `PROGRESS.md` §22 for the port and its
> verification.

LIBERO drives its Panda with robosuite's `OSC_POSE`, which computes joint **torques**.
Route A instead solved IK and commanded joint **angles** — the same `data.ctrl` array,
a different physical quantity. `libero/osc_controller.py` is a port of robosuite
**1.4.0**'s `OSC_POSE` (the version LIBERO pins; 1.5.2 restructured it and added an
`input_ref_frame` that 1.4.0 lacks), so the controller now matches the one the
demonstrations were recorded through.

| | `--control-mode osc` (default) | `--control-mode ik` |
|---|---|---|
| scene | `scene_libero_osc.xml` | `scene_libero_hand.xml` |
| actuators | `<motor>`, `ctrl` = **torque N·m** | position servo, `ctrl` = **rad** |
| law | `tau = Jᵀ Λ F + qfrc_bias + nullspace` | one DLS IK solve per tick |
| gains | `kp=150`, critically damped, `uncouple_pos_ori` | n/a |
| on contact | yields | presses |
| unreachable | wrench degrades (`pinv`) | `IK unreached` |
| `--min-clearance` | **off** by default | 16 mm |

A mismatched scene/mode pair is a **hard error**, not a warning: writing torques into
position actuators reinterprets N·m as radians and produces plausible-looking garbage.
The check reads the *compiled* model's `actuator_biastype`, not the filename.

### The timing detail

robosuite recomputes the torque **every physics step** — `environments/base.py:454` loops
`control_timestep/model_timestep` times calling the controller, with `policy_step=True`
only on the first. So the goal is set at 20 Hz and the PD runs at 500 Hz. Holding one
torque across a whole 25-step tick is a different and unstable controller, because the
damping term stops seeing the velocity it exists to damp.

### What it bought, measured

`MUJOCO_GL=egl uv run python libero/tools/verify_osc.py` — four checks, all CPU, free to
re-run after any change:

| | position servo | OSC |
|---|---|---|
| standing sag at reset | 4.84 mm (stock gains) | **0.000 mm** |
| settled eef above table | 0.2680 | **0.2728** (LIBERO 0.2733) |
| penetration driving into the table | −2.9 mm at ~70 N | **−1.11 mm at 35.4 N**, rests +14.7 mm above |
| realised motion per 20 Hz tick | ~33% | **12.3%** |

The first row is the significant one: the top-level `README.md` is *entirely* about a bug
that exists only because an overdamped position servo never arrives, so the collector's
`(target − current)` label carried a standing sag into every training frame. That failure
class does not exist here.

**The last row is not a regression and must not be "fixed".** 12.3% matches the analytic
step response of a critically damped system at `wn = √150` (12.6%) to within 0.3 points,
and the policy was trained through exactly this response, so its commanded magnitudes
already account for it. A value near 100% would mean the port is wrong.

### Route A: differential IK

Kept as `--control-mode ik` so every run and dataset produced before 2026-07-28 stays
reproducible. It is no longer the default, and the paragraph that used to be here — "we
do **not** reimplement OSC, that would mean replacing the position actuators with torque
actuators and re-tuning gains" — is now obsolete: that is exactly what was done, and it
needed no gain tuning at all, because OSC's own `kp=150` replaces the position gains
rather than working alongside them. Per control tick:

1. un-normalize the delta: `dpos = a[0:3] * 0.05` m, `drot = a[3:6] * 0.5` rad
2. add it to the **current measured** `grip_site` pose → target pose. Re-basing on the
   achieved pose every tick is what makes this a delta controller and stops error
   accumulating: `target_mat = R(drot) @ cur_mat`, composed in the **world** frame,
   matching what `_orientation_error` works in and what robosuite's OSC does by default
3. **clamp the target's z to the table floor** (see below) — before IK, never after
4. one damped-least-squares IK solve for that target (same solver as
   `phase4_collect_demos.solve_ik`), run on a throwaway `MjData` so the live sim is
   untouched: `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e` with λ = 0.15, `dq` clipped to ±0.3, `q` clipped to
   joint limits, ≤200 iterations, converged at ‖e‖ < 1e-4. Non-convergence is what the
   `IK unreached` counter reports
5. write the joint solution to `data.ctrl[0:7]`, gripper to `data.ctrl[7]`
6. step `decimation` physics steps (25 at LIBERO's 20 Hz)

Keeps the existing actuators and scene. ~~`--control-mode` is reserved for adding a real
OSC path later~~ — **built, see above.**

### The table floor (`--min-clearance`)

> **Obsolete in `--control-mode osc`, where it defaults to OFF.** Everything below is a
> description of a workaround for missing compliance, and OSC has compliance natively
> (−1.11 mm at 35.4 N, coming to rest *above* the surface). Both "known problems" listed
> at the end of this section — that it does not fully work, and that it may cost more
> than it buys — are resolved by not needing it. Kept accurate for the `ik` path, and
> still passable in osc mode if you want a paired comparison under identical constraints.

Route A tracks a commanded **pose** with a stiff position servo; OSC commands **forces**
and yields on contact. So nothing in this pipeline stops the model asking for a target
below the table — IK solves it happily and the actuators drive there, pressing instead of
settling. Measured on the stock hand: **2.9 mm** of real penetration by the finger mesh
under ~70 N.

`--min-clearance` (default **0.013**) is the metres the *commanded* target is held above
the table top. It is applied to the target before IK, so the solver never sees an
impossible request — clamping the solution afterwards would leave the orientation solved
for a position the arm is no longer at.

13 mm is measured, not chosen: `grip_site` is a point between the fingers, and the actual
hardware hangs below it by 6.1–12.1 mm depending on wrist angle (over 80 control ticks of
`stockhand_02`).

**Two known problems with it, both live:**

- **It does not fully work.** Penetration drops 2.9 mm → 0.3 mm, not to zero. The clamp
  bounds the *target*; the position servo then sags ~5 mm below whatever it is told
  (`stockhand_03`: achieved `eef_z` min −0.1037 against a −0.0990 floor). Sizing it from
  geometry alone ignored that lag — covering it means ~18 mm.
- **It may be costing more than it buys.** In `stockhand_03` three chunks hit `10/10`
  clamped — the model asked for a below-floor descent on *every* action and every one was
  refused. That run touched the ball zero times, against 32%/38% gripper closes and one
  successful lift in the two unclamped runs before it. One run, and the policy samples
  (`PROGRESS.md` §18), so this is a suspicion with a mechanism, not a result.

Pass a **negative** `--min-clearance` to switch it off and restore LIBERO's unclamped
control law. **`0` does not switch it off** — it floors the target at the table top, which
still clamps *and* still lets the hand through, because `grip_site` is held at the surface
while the hardware hangs 6–12 mm below it. That mistake made the control arm of the first
A/B sweep useless; "off" has to be its own value, not a boundary case of the number.

Deciding between clamped and unclamped wants a paired comparison, not another single run.

## Two things that are NOT verified, and how to test them

These are the known risks in this port. Both are cheap to resolve empirically.

**1. Image orientation.** `LiberoProcessorStep` flips raw LIBERO frames 180°
(`env_processor.py:58-59`), commenting that it "accounts for the HuggingFaceVLA/libero
camera orientation convention". robosuite renders bottom-up (OpenGL), so the training
frames are raw-robosuite-rotated-180°. `mujoco.Renderer` already returns top-down images,
so it is genuinely unclear whether we should match by flipping or by not flipping.
`--image-flip {none,180,vertical}` exists to try all three — run each and compare. Do this
before drawing any conclusion about the checkpoint.

**1b. Which render size.** `--render-size` defaults to 256, LIBERO's native size. At the
matched camera that puts a 40 mm ball at 16.0 px; 378 gives 23.7 px. Matching LIBERO's
size vs. keeping more detail are in tension and only a rollout will say which wins.

**2. ~~Gripper state units.~~ RESOLVED — the scene now mounts the stock Franka hand.**
The faithful fix described here was taken: `scene_libero_hand.xml` uses
`panda_libero_hand.xml`, which is upstream menagerie's Panda + stock Franka hand, so
`robot0_gripper_qpos` is read straight off the same two prismatic joints LIBERO reports,
in the same metres. No conversion, no stand-in.

Two sign traps came with the swap, both now handled in `libero_closed_loop.py` and
verified against the compiled model:

- **`actuator8` runs the other way.** For the stock hand `ctrl = 255` is **open** and `0`
  is closed (the home keyframe pairs `ctrl=255` with `qpos=0.04 0.04`). The 2F-85's
  `fingers_actuator` was the reverse. Keeping the old polarity would have closed the
  gripper every time the model asked for an open.
- **The reported pair is `[+x, −x]`, not `[+x, +x]`.** robosuite mirrors its fingers with
  opposite joint *ranges* (`finger_joint1 ∈ [0, 0.04]`, `finger_joint2 ∈ [−0.04, 0]`);
  menagerie mirrors the same hardware with a body quaternion and keeps both in
  `[0, 0.04]`. The second element is negated on the way out. Every LIBERO run before this
  change reported both positive — one of the two gripper state channels was wrong across
  its entire range.

## Delta actions change two things we relied on

Carried over from the DROID loop, both now wrong by default:

- **Expired-action dropping is unsafe.** With *absolute* joint targets, skipping a stale
  action loses nothing — the next one already encodes the pose. With *deltas*, skipping
  loses displacement. `--replan-at` is therefore pinned to sequential here; the drop path
  would have to *sum* skipped deltas instead of discarding them.
- **Holding a delta for `decimation` steps is not the same as holding an absolute
  target.** We re-solve IK to an absolute joint target each tick, so within a tick the
  semantics are the same as the DROID loop — but that is a property of Route A, not of
  the action space. A real OSC path would integrate differently.

## Scene: `scene_libero_hand.xml`

Uses `mujoco_menagerie/franka_emika_panda/scene_libero_hand.xml`, **not**
`scene_pick_place.xml`. Same objects; `external_cam` is re-framed to LIBERO's `agentview`
(0.707 m standoff, `fovy 45`, rigid `xyaxes` mount instead of `mode="targetbody"`).

It has to sit next to `scene.xml` rather than in `libero/` — the Panda XML declares
`meshdir="assets"` relative to the *top-level* file's directory, so a scene file here
would break every mesh load. The code stays decoupled; the scene can't be.

`scene_pick_place.xml` and `panda.xml` are deliberately untouched: the DROID loop and the
50-demo dataset were both built through them, and `panda.xml` is where the Robotiq 2F-85
lives. `scene_libero.xml` (the 2F-85 version of this scene) is also kept, so the earlier
runs stay reproducible.

### `panda_libero_hand.xml` — the gripper swap

Upstream menagerie's `panda.xml` (recoverable as `git show HEAD:panda.xml` — the working
copy has been modified) is the Panda with its **stock Franka hand**, which is what
robosuite and therefore LIBERO use. `panda_libero_hand.xml` is that file verbatim plus
three additions, all taken from robosuite 1.5.2 rather than derived:

| addition | source | why |
|---|---|---|
| `grip_site` | `panda_gripper.xml` `right_gripper`→`eef`→`grip_site` | LIBERO's `robot0_eef`, **position and frame** — the state reports `eef_axisangle` and the IK solves for orientation, so the −90° z rotation matters, not just the point |
| `eye_in_hand` | `robots/panda/robot.xml:224`, copied verbatim | LIBERO's `robot0_eye_in_hand`, the second image the checkpoint trained on |
| pad `friction`/`solref` | `panda_gripper.xml` pad geoms | `2 0.05 0.0001` / `0.01 0.5` — double the sliding friction and a deliberately soft pad; both matter for a sphere grasp, which is friction-only |

Verified against robosuite's own FK (its `robot.xml` loaded directly into MuJoCo):
`right_hand` relative to `link0` agrees to **0.5 mm**, and the finger opening axis lands
on the eef frame's x in both, as robosuite has it.

**One deliberate 9.5 mm departure.** `grip_site` is at `z=0.1065`, not robosuite's
literal `0.097`. The two finger meshes differ — robosuite's pad centre is at `0.0934`
with `grip_site` `+3.6 mm` ahead of it; menagerie's is at `0.1029`. Copying `0.097`
verbatim would put the site 5.9 mm *behind* our pads where LIBERO's sits 3.6 mm *ahead*
of its own. What the checkpoint learned is the relationship, so that is what is
reproduced (`0.1029 + 0.0036`). Change the one number if you want the literal value.

### Correction to `docs/PHASE5_PLAN.md` Tier B

The plan says `wrist_cam` "re-aims every frame", "pinning the target near center", and
that this "deleted the best cue for the last centimetre." That is **not** what the XML
did. The camera and its target `grasp_target` were both children of the same `base` body,
so the view direction in the gripper frame was constant — measured across three arm
configurations the forward vector was bit-identical at `(0, 0.86946, 0.49401)`. It never
re-aimed. And a rigidly-bolted real wrist camera would *also* hold a fixed point of the
gripper at a fixed image position; the drift that carries servo information is the
*ball's* drift relative to the gripper, which we already have.

This is now moot for the LIBERO path either way: `wrist_cam` / `wrist_cam_rigid` were
**our own design** (22 cm side-mount, `fovy 56.7`) and matched nothing the model was
trained on. The LIBERO scene uses robosuite's `eye_in_hand` instead. The old cameras stay
in `panda.xml` for the DROID path.

## Usage

```bash
modal deploy libero/libero_modal.py                       # serve MolmoAct2-LIBERO (L4)

# default: native OSC on scene_libero_osc.xml, table clamp off
uv run python libero/libero_closed_loop.py --dry-run --server-url <url>/act
uv run python libero/libero_closed_loop.py --chunks 8 --server-url <url>/act

# the old Route A path, for reproducing pre-2026-07-28 runs
uv run python libero/libero_closed_loop.py --control-mode ik --chunks 8 --server-url <url>/act

# controller verification -- CPU only, free, run after any change to the OSC path
MUJOCO_GL=egl uv run python libero/tools/verify_osc.py
```

`--model-path` defaults to whichever scene matches `--control-mode`; pass it only to
override.

Artifacts land under `assets/<model>/<fine-tune>/logs/<run_id>.jsonl` and
`.../images/<run_id>/`, with `image` / `wrist_image` camera names instead of
`external_cam` / `wrist_cam`. `--model` / `--fine-tune` set the two directory levels; leave
them off and they are derived from the server's `/health` checkpoint. See the root
`README.md`, "Run artifacts".

### Two things to expect when watching a run

**The arm is stationary most of the time, and that is correct.** Replanning is sequential
(`--replan-at 0`, see below), so one chunk is 10 actions × 25 steps × 2 ms = **0.5 s of
motion**, followed by a ~3 s round trip during which the sim is *frozen*. The arm moves
for roughly 15% of the wall clock. Add ~40–60 s on the first chunk if the Modal container
has scaled down and has to reload 10 GB. `PROGRESS.md` §2: transport is ~4× the inference
cost, and encoding the frames before POSTing is the fix nobody has done yet.

**One run is one sample.** MolmoAct2's action expert is flow-matching — it *samples*, so
the policy is not a deterministic function of the observation. Two runs from an identical
start state diverged completely (§18). Score success rate over several runs; do not read a
single rollout as the model's behaviour. Several conclusions in `PROGRESS.md` that were
drawn from single runs are flagged there for this reason.

### GPU

`libero_modal.py` runs on an **L4** (24 GB, ~$0.80/hr). The checkpoint is 5B params ≈
10 GB at bf16, so the A100-40GB it started on was ~4× oversized. T4 is cheaper but is
Turing — no bf16 at all, so it fails rather than running slow. Measured after the switch:
server dt ~730 ms, round trip 2.8–3.8 s, i.e. no latency penalty. Next rungs if needed:
`A10` (24 GB, $1.10/hr), `L40S` (48 GB, $1.95/hr).
