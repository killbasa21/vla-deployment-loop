"""Collect scripted expert demonstrations of the green-box pick-and-place task, in the
**LIBERO** convention, for LoRA + action-expert fine-tuning of MolmoAct2-LIBERO.

Not to be confused with `droid/phase4_collect_demos.py` at the repo root. That one records the
**DROID** convention (8-D absolute joint targets, 15 Hz, `scene_pick_place.xml`) for the
DROID checkpoint. This one records what the LIBERO checkpoint speaks:

                          droid/phase4_collect_demos.py      this file
    action                8-D absolute joint pos       7-D delta eef pose, [-1, 1]
    state                 [q1..q7, gripper_rad]        [eef_pos(3), axisangle(3),
                                                        gripper_qpos(2)], LIBERO frame
    rate                  15 Hz                        20 Hz (robosuite control_freq)
    cameras               external_cam, wrist_cam      image, wrist_image
    dataset               LeRobot v2.1 + mp4           LeRobot v3.0 + inline PNG
    scene                 scene_pick_place.xml         --control-mode, see below

THE ONE DESIGN RULE
-------------------
Every recorded action is produced by, and executed through, the SAME `libero_closed_loop`
function that will consume the fine-tuned model's output at inference. Nothing here
re-implements the controller. `--control-mode` selects which one, on both sides:

    osc (default)  L.apply_action_osc + scene_libero_osc.xml
                   robosuite's OSC_POSE port, joint torques. What the client defaults to
                   since 2026-07-28, and what LIBERO itself uses.
    ik             L.apply_action     + scene_libero_hand.xml
                   Route A differential IK into position servos. What a1-a4 were
                   collected on; kept so those datasets stay reproducible.

That is the entire point. The diagnostic in PROGRESS.md sec.20 scored 3/3 on a real LIBERO
task through robosuite's OSC, which proved the checkpoint and our serving are both fine
and located the failure in our environment. The leading explanation was controller
mismatch: the policy learned OSC's compliant transfer function and we drove a stiff
position servo through IK. A fine-tune can absorb that mismatch, but ONLY if the actions it
is trained on are the actions that produce the recorded motion in our controller. So the
expert here is a Cartesian reference trajectory, and the label at each tick is the
normalised delta from the arm's ACTUAL pose to that reference -- closed-loop, not a
replayed open-loop plan. Tracking lag, controller error and (in ik mode) the table clamp
all end up inside the labels, which is where they have to be for the model to learn to
compensate for them.

**A DATASET IS THEREFORE CONTROLLER-SPECIFIC.** An osc dataset and an ik dataset have
identical schemas and different label VALUES: the two plants realise ~12.3% and ~72% of a
commanded per-tick displacement respectively, so the corrective deltas differ by ~6x.
Training on one and serving under the other reintroduces exactly the mismatch above. Every
plant-calibrated constant in this file (NOISE_SIGMA_POS, RETREAT_BACKOFF*, RECOVER_KICK*,
and the dwell durations in `waypoints`) was measured on the ik plant and must be
re-measured, not carried over -- see docs/SERVO_DROOP.md sec.4.3 for how that sweep is run.

THE THREE COHORTS
-----------------
  reach   (20)  Nominal. Randomised box position and bin layout, LIBERO's reset pose.
  noise   (20)  DART-style noise injection. Gaussian noise is added to the EXECUTED
                action, but the recorded label is the clean expert action AT THE STATE
                THE NOISE PRODUCED. This is what teaches recovery: the model sees
                off-trajectory states paired with the action that corrects them, which
                plain behaviour cloning never provides and which is the direct remedy for
                the compounding drift seen in sec.17-19. Recording the noisy action
                instead would just teach the model to be noisy.
  recover (10)  Chosen here, not requested. Larger start-pose jitter plus one hard
                disturbance at a random tick during the approach, then the expert drives
                back and completes the task. Same motivation as `noise` but with rarer,
                bigger excursions -- noise covers the small-perturbation regime densely,
                this covers the tail, and our observed failures are tail failures.

Usage:
    uv run python libero/fine_tune/collect_finetune_data.py --selftest
    uv run python libero/fine_tune/collect_finetune_data.py --out libero/fine_tune/a1
    uv run python libero/fine_tune/collect_finetune_data.py --out libero/fine_tune/a2 \
        --control-hz 10 --reach 4 --noise 4 --recover 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import libero.libero_closed_loop as L  # noqa: E402
from lerobot_v30_writer import LeRobotV30Writer  # noqa: E402

INSTRUCTION = L.INSTRUCTION

# Gripper command in the LIBERO action space. -1 = open, +1 = closed.
OPEN, CLOSED = L.GRIPPER_ACTION_OPEN, L.GRIPPER_ACTION_CLOSED

# Half-extent of the 40 mm pick cube. Taken from libero_closed_loop rather than restated,
# because the collector's resting height and the scorer's lift threshold have to move
# together -- they were separate 0.02 literals until 2026-08-03 and nothing would have
# caught them diverging.
BOX_HALF = L.BOX_HALF
BOX_REST_Z = L.TABLE_TOP_Z + BOX_HALF

# Bin anchor slots, read off scene_libero_hand.xml. Which colour sits at which slot is
# shuffled per episode so the model has to LOOK for the green one rather than memorise a
# fixed coordinate -- the same argument droid/phase4_collect_demos.py makes.
# Imported, not redeclared: libero_closed_loop.py --randomize-bins draws from the same
# three slots, and two copies of this list is how the collector and the evaluation end up
# sampling different distributions without anything erroring.
BIN_SLOTS = L.BIN_SLOTS
BIN_SLOT_JITTER = L.BIN_SLOT_JITTER
BIN_NAMES = L.BIN_NAMES

# Noise cohort. Sigma is in ACTION units, so 0.15 is 7.5 mm of translation per tick
# (0.15 * DELTA_POS_SCALE) and 0.04 is 0.02 rad of rotation. Sized to knock the arm
# meaningfully off the reference without making the task unachievable -- the released
# LIBERO actions have a per-channel std around 0.33, so this is a visible but minority
# perturbation. The gripper channel is deliberately NOT perturbed: flipping it mid-grasp
# does not produce a recoverable state, it produces a dropped box.
#
# 2026-07-28: 0.15 -> 0.08 (and rot 0.04 -> 0.021, holding the same 0.267 ratio).
#
# Sigma is a perturbation on the COMMANDED action, but what actually knocks the arm off
# the reference is the perturbation that gets REALISED, and the kp x2 / kd x0.7 plant
# realises 72% of a command where the old one realised 33%. So the identical sigma now
# delivers well over twice the physical disturbance per tick, and it compounds over all
# 174 ticks. Measured: at 0.15 on the new plant, 7 of 10 noise slots were DROPPED after
# exhausting 8 attempts each.
#
# Calibrated by running 10 noise episodes per sigma on the new plant:
#
#   sigma   placed   q01 of dx     P(slot filled | 8 attempts)
#   0.05     5/10      -0.052            99.6%
#   0.07     5/10      -0.085            99.6%
#   0.08    (chosen, interpolates)
#   0.09     3/10      -0.116            94%
#   0.11     0/10      -0.148            ~0
#   0.15     1/10      -0.215            <1%     <- a1's value
#
# The trade is real: a larger sigma buys a more two-sided dx distribution, which is the
# entire point of the cohort, but past ~0.09 it stops producing keepable episodes. 0.08
# sits just inside the knee, and --max-attempts-per-episode is raised to 12 to make
# filling every slot near-certain rather than merely likely.
NOISE_SIGMA_POS = 0.08
NOISE_SIGMA_ROT = 0.021

# --- drift cohort (2026-08-03, for the box datasets) --------------------------------
# A third way of perturbing, between `noise` (every tick, so the arm is never on the
# reference) and `recover` (a single-tick impulse, so the excursion arrives faster than
# anything physical would produce it).
#
# `drift` picks a few WINDOWS of consecutive ticks and applies the noise-cohort Gaussian
# inside them only. Outside a window the executed action is the clean expert action, so
# the arm is genuinely tracking the reference and the model sees what on-trajectory looks
# like -- which the `noise` cohort never shows it. Inside a window the perturbation
# COMPOUNDS over 8-20 ticks into a several-cm excursion, which one tick of the same sigma
# cannot do (0.47 * 0.05 m * 0.123 realised = 2.9 mm for a single tick).
#
# The label is the clean expert action throughout, exactly as in `noise` -- see sec.4.1.
# The gripper channel is not perturbed, for the same reason it is not perturbed there.
DRIFT_WINDOWS = (3, 3)          # inclusive: how many noise bursts per episode
DRIFT_WINDOW_TICKS = (8, 20)    # inclusive: length of each burst, in control ticks
# Minimum clear ticks between two windows. Windows are placed at random, so without this
# three draws can overlap and merge into one longer burst -- the episode would then be
# labelled "3 drifts" while containing one. At ~14 ticks mean length in a ~410-tick
# episode two draws collide maybe 10% of the time, so this is not a hot path; the gap also
# guarantees the arm gets back onto the reference between excursions, which is the part
# that makes them separate recovery examples rather than one long one.
DRIFT_MIN_GAP = 15

# --- decoy cohort ------------------------------------------------------------------
# The deterministic counterpart of `drift`, and the one the box datasets use. Inside a
# window the EXECUTED action points at a decoy target offset from the reference, so the arm
# visibly moves the wrong way; the LABEL still points at the true reference, so every frame
# of the excursion is annotated "you are off, come back".
#
# The distinction that matters: putting the wrong move in `waypoints` instead would make it
# part of the reference, and the label would then say "move wrong" -- the policy would learn
# to detour and correct, reproducing both halves, because both were labelled as correct.
# Same argument as sec.4.1 makes for the noise cohort. The wrong move belongs in the
# execution and nowhere else.
#
# 35 mm, not more: the corrective label peaks at DECOY_OFFSET / DELTA_POS_SCALE once the
# arm has reached the decoy, i.e. 0.70 at LIBERO's 0.05 m/unit. At 50 mm that is exactly
# 1.0 and the label clips at the moment it carries the most information.
DECOY_OFFSET = 0.035


def _sample_windows(rng, track):
    """Non-overlapping perturbation windows, all ending before the box is released.

    Windows stop at the RELEASE. Everything after it -- the gripper opening, the rise clear
    of the bin, the return to the start pose -- happens with the box already placed, so a
    perturbation there pushes an arm that has nothing left to do and produces recovery
    labels for a state the task is already finished in.

    The cutoff is derived from the track rather than from a fixed episode fraction: the
    release is waypoint 11, and how many ticks precede it depends on the box and bin
    positions, --motion-speed and --no-backoff. Any hardcoded fraction would go stale the
    moment one of those changed. The last tick at exactly GRIPPER_ACTION_CLOSED is the tick
    before the open ramp begins.

    Each window is rejection-sampled against the ones already placed so they stay
    DRIFT_MIN_GAP apart -- see DRIFT_MIN_GAP for why merging them is not the same thing as
    having them. If a window cannot be placed, fewer are returned than requested, which is
    visible in the per-episode log line rather than silently merged.
    """
    closed = np.flatnonzero(
        np.array([g for _, g in track]) >= L.GRIPPER_ACTION_CLOSED - 1e-9)
    end = int(closed[-1]) + 1 if closed.size else len(track)
    spans = []
    for _ in range(int(rng.integers(DRIFT_WINDOWS[0], DRIFT_WINDOWS[1] + 1))):
        for _try in range(50):
            length = int(rng.integers(DRIFT_WINDOW_TICKS[0], DRIFT_WINDOW_TICKS[1] + 1))
            start = int(rng.integers(0, max(1, end - length)))
            if all(start >= b + DRIFT_MIN_GAP or start + length + DRIFT_MIN_GAP <= a
                   for a, b in spans):
                break
        else:
            continue
        spans.append((start, min(start + length, end)))
    return sorted(spans)

# ---- OSC-plant defaults -----------------------------------------------------------
# Everything above this line was measured on the ik plant. OSC is a different plant and
# needs its own numbers; carrying the ik ones over is the mistake this file's header warns
# about. Both of these are overridable with --speed-scale / --noise-sigma-pos.
#
# OSC_SPEED_SCALE: the ik-era segment durations clip the labels under OSC. Label magnitude
# is lag/DELTA_POS_SCALE with lag ~= v*dt/realised, so label ~= 8.1*v at 12.3% realised and
# 20 Hz -- anything above 0.123 m/s saturates. The retreat runs 0.12 m in 0.45 s
# (0.27 m/s, label 2.2) and the transport ~0.15 m/s (label 1.2). Measured on the unscaled
# timings under OSC: dx q01 pinned at exactly -1.000 in EVERY cohort, against a4's -0.648
# and released LIBERO's -0.679. 2.5x puts the worst segment at 0.108 m/s -> label ~0.88,
# just inside the bound, at the cost of a 27 s episode instead of 10.8 s.
OSC_SPEED_SCALE = 2.5
# OSC_NOISE_SIGMA_POS: the ik value of 0.08 does nothing here. Swept on the OSC plant at
# 8 noise episodes per sigma: 0.08 -> 8/8 placed with 0 rejected attempts, 0.20 -> 8/8 with
# 0 rejected. A cohort that never fails is a cohort that is not perturbing anything, so
# both are far below the knee that 0.08 sat on for the ik plant. The ratio argument agrees:
# holding the REALISED disturbance fixed across 72% -> 12.3% scales sigma by ~5.9x, i.e.
# 0.08 -> ~0.47.
OSC_NOISE_SIGMA_POS = 0.47
OSC_NOISE_SIGMA_ROT = 0.125   # holds the ik plant's 0.267 rot/pos ratio

# --- Distance retiming (--motion-speed) ---------------------------------------------
# A waypoint whose target is within this of the previous one is a DWELL, and keeps its
# hand-set duration: it is there to let the plant settle or the gripper actuate, so its
# length is set by physics, not by how far the arm has to travel. 1 mm is well under the
# smallest real motion (the 30 mm backoff lift).
DWELL_DIST = 1e-3
# Floor on a retimed motion segment. Below ~5 ticks at 20 Hz the smoothstep no longer has
# room to accelerate and decelerate, and the reference is effectively a step.
MIN_MOTION_SECS = 0.25
# Fraction of the label ceiling the default target speed aims at. Smoothstep PEAKS at 1.5x
# its mean, so the mean budget is v_max/1.5, and the arm needs headroom above the nominal
# geometry this was computed on -- a box at the near edge of BOX_SAMPLE_X and a bin at
# the far slot is a longer transport than the nominal 0.252 m.
MOTION_SPEED_HEADROOM = 0.90
# The plant constant everything above keys off: OSC realises this fraction of a commanded
# per-tick displacement (verify_osc.py; 12.3% is the analytic step response of a critically
# damped wn = sqrt(150) system at 50 ms, to within 0.3 points).
OSC_REALISED_FRACTION = 0.123
SMOOTHSTEP_PEAK = 1.5   # a smoothstep segment's peak velocity over its mean

# Recovery cohort.
#
# a1's version was ONE kick of 0.85 placed in the first 45% of the episode, and it was too
# easy: 10/10 episodes were kept on the first attempt, which means the reference tracker
# absorbed the disturbance without the cohort teaching much. A single kick during the
# approach has 100+ ticks left to wash out and an empty gripper, so nothing is at stake.
#
# a3 kicks TWICE, and the second one is the point: it lands mid-transport with the box
# actually in the hand, which is where the observed closed-loop failures live (PROGRESS
# sec.18 -- grasp achieved, retention not, run degenerates from chunk 5). A loaded kick can
# genuinely drop the box, and rejection sampling then throws that episode away, so this
# costs attempts -- which is the honest sign that the disturbance is doing work now.
# The loaded kick is smaller (0.6) than the free one deliberately: at 0.85 with a box in
# the gripper almost every episode fails and the cohort collects nothing.
RECOVER_START_JITTER = 0.09   # radians, per joint, on top of LIBERO's reset pose
RECOVER_KICK = 0.85           # free-arm disturbance during the approach, near full scale
RECOVER_KICK_LOADED = 0.60    # smaller: fired mid-transport, with the box in the gripper
# Fractions of the episode. Re-derived for a4's longer trajectory (10.8 s vs 7.6 s): the
# gripper closes at 3.8 s and opens again at 9.2 s, i.e. the arm is LOADED over fraction
# 0.35-0.85, so the free-arm window has to end before 0.35 and the loaded one has to sit
# inside it. a3's (0.08, 0.40) would now fire the "free-arm" kick onto a closed gripper.
RECOVER_KICK_WINDOWS = ((0.06, 0.33), (0.45, 0.78))

# Retreat segments -- the a4 change, and the reason a4 exists.
#
# a3's dx channel is ONE-SIDED: q01 = -0.072 against the released dataset's -0.679, i.e.
# over the whole dataset there is essentially no label that says "move back in -x". That is
# exactly the correction the closed-loop failures need (README sec.6.5, sec.6.6): the arm
# overshoots past the box and never comes back. Noise sigma cannot buy this -- raising it
# past ~0.09 simply stops episodes from succeeding (sec.4.3) -- so the -x motion has to come
# from the EXPERT trajectory itself, where it is labelled unconditionally and survives
# rejection sampling by construction.
#
# Two segments are added:
#   (a) a back-off-and-re-approach inserted between the hover and the descent. The arm
#       arrives above the box, withdraws toward the base, then comes back in. This is the
#       recovery manoeuvre we want the policy to have, demonstrated in the exact visual
#       context (box centred, gripper open) where it will need it.
#   (b) a return to the episode's own start pose after the release, replacing a3's "rise
#       20 cm and stop". Bins sit at x = 0.56 or 0.80 and the reset pose at x ~ 0.45, so
#       this is -0.11 to -0.35 m of travel in -x, and it is what a real demonstration would
#       end with anyway.
#
# Sizing: BACKOFF / BACKOFF_SECS is a mean speed of 0.27 m/s, and smoothstep peaks at 1.5x
# the mean, so ~0.4 m/s -> 0.02 m/tick -> 0.4 action units before servo lag, against a
# standing droop bias of about +0.10. Comfortably two-sided, comfortably inside the +-1
# clip. Do not slow this down to be gentle: sec.3.2 shows the label ratio is dt/tau and is
# INDEPENDENT of reference speed, so a slower retreat produces a smaller label, not a
# cleaner one.
RETREAT_BACKOFF = 0.12        # m, withdrawn along -x from the hover pose
RETREAT_BACKOFF_LIFT = 0.03   # m, and slightly up, so the withdrawal is not a table graze
RETREAT_BACKOFF_SECS = 0.45   # each way


def bin_layout(model, bin_ids, rng, mode="shuffle", scene_xy=None):
    """Place the three bins for one episode and return green's (x, y).

    Two modes, and the choice is about what the fine-tune is FOR:

    "shuffle" (a1-a5) permutes the three bins across the three slots with +-2 cm of
    jitter, so the policy has to resolve "green" by colour rather than by memorised
    position. That is the general skill -- but the closed loop NEVER randomises bins
    (libero_closed_loop.py has --randomize-box and no bin equivalent), so every
    evaluation runs the one layout the scene XML declares, green at (0.56, 0.25). In a5
    that layout drew only 6 of 30 episodes; the other 24 taught reaching toward bins that
    are not there at inference.

    "scene" pins the bins to the scene's own positions, i.e. exactly what will be
    evaluated, and keeps the jitter off. At 30-50 episodes this is usually the better
    trade: 5x the density on the configuration under test, at the cost of a policy that
    has memorised where green is instead of learning to look. Use "shuffle" if the point
    is colour grounding, "scene" if the point is a working demo.

    `scene_xy` is the model's ORIGINAL green-bin (x, y), captured before any episode
    moved it -- see main(). Reading it live would return the last episode's placement.
    """
    if mode == "scene":
        for name, (x, y) in zip(BIN_NAMES, scene_xy):
            model.body_pos[bin_ids[name]] = [x, y, L.TABLE_TOP_Z]
        return tuple(scene_xy[BIN_NAMES.index("green_bin")])

    order = rng.permutation(len(BIN_SLOTS))
    green_xy = None
    for name, slot_idx in zip(BIN_NAMES, order):
        x, y = BIN_SLOTS[slot_idx]
        jx, jy = rng.uniform(-BIN_SLOT_JITTER, BIN_SLOT_JITTER, size=2)
        model.body_pos[bin_ids[name]] = [x + jx, y + jy, L.TABLE_TOP_Z]
        if name == "green_bin":
            green_xy = (x + jx, y + jy)
    return green_xy


def reset_episode(model, data, rng, box_xy, arm_jitter=0.0, osc=None):
    """LIBERO's reset pose, plus this episode's box and bin layout, settled.

    `osc` is the OSCController in --control-mode osc, None in ik mode. It changes two
    things, both mirroring `L.build_sim`:
      * `data.ctrl[ARM]` is TORQUE, not a joint target, so writing `q` into it would
        command a constant few N.m instead of a pose.
      * "hold still" has to be ACTIVELY COMMANDED. With ctrl = 0 the arm is unpowered and
        falls, so the settling loop below has to run the controller every step."""
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home != -1:
        mujoco.mj_resetDataKeyframe(model, data, home)

    q = L.LIBERO_INIT_QPOS.copy()
    if arm_jitter > 0:
        lower, upper = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
        q = np.clip(q + rng.normal(0.0, arm_jitter, size=7), lower, upper)
    data.qpos[:7] = q
    data.qvel[:] = 0.0   # a previous episode's velocities must not leak into this reset
    if osc is None:
        data.ctrl[L.ARM] = q
    else:
        data.ctrl[L.ARM] = 0.0
    data.ctrl[7] = L.GRIPPER_CTRL_MAX  # open

    for name, pos in [(L.TARGET_BODY, (box_xy[0], box_xy[1], BOX_REST_Z)),
                      ("red_box", (0.56, -0.28, BOX_REST_Z))]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            continue
        adr = model.jnt_qposadr[model.body_jntadr[bid]]
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)
    if osc is not None:
        # Park the goal and the nullspace reference on the pose we just wrote, so the
        # settling loop holds station instead of driving somewhere.
        osc.reset(data)
    for _ in range(200):
        if osc is not None:
            data.ctrl[L.ARM] = osc.compute_torque(model, data)
        mujoco.mj_step(model, data)
    if osc is not None:
        # Re-park on the SETTLED pose, so "goal == where we actually are" is exactly true
        # at the moment the first observation of the episode is taken.
        osc.reset(data)
        # ...but pin the NULLSPACE reference to LIBERO's unjittered reset posture, which is
        # what it will be at inference: build_sim always resets from LIBERO_INIT_QPOS, never
        # from a jittered pose. Without this the recover cohort (arm_jitter > 0) would have
        # osc.reset capture the JITTERED joints as the posture the nullspace PD pulls the
        # elbow toward -- a controller configuration that occurs in 10 of 30 training
        # episodes and never at serving time. Only the redundant DOF is affected, so eef
        # labels barely move, but it is a plant difference with no reason to exist.
        osc.initial_joint = L.LIBERO_INIT_QPOS.copy()


def waypoints(box_xy, bin_xy, start_pos, backoff=True):
    """The Cartesian reference trajectory, as (grip_site target, gripper, seconds).

    `start_pos` is the arm's ACTUAL settled pose at reset (jittered, for the recover
    cohort), used as the target of the closing retreat so the episode ends where it began.

    `backoff=False` drops segments 1a/1b/1c, the withdraw-and-re-approach inserted between
    the hover and the descent. Those exist to put -x motion in the labels (see
    RETREAT_BACKOFF), and dropping them makes the approach a single clean descent at the
    cost of thinning the -x distribution. The closing return to `start_pos` (segment 13) is
    NOT dropped with it and is then the only -x source left, which is why it stays: a3 had
    neither and its dx q01 was -0.072 against released LIBERO's -0.679.

    Heights are relative to the box and the table, both of which moved on 2026-07-28 when
    the table was corrected upward by 100 mm -- so they are written in terms of
    BOX_REST_Z / TABLE_TOP_Z rather than as literals.

    The grasp target is the box CENTRE, not some offset above it. That only became the
    right answer once `grip_site` was moved to robosuite's literal 0.097: the site is now
    robosuite's own grasp point, so aiming it at the object centre is exactly what LIBERO's
    demonstrations do. Under the old 0.1065 site the same command would have driven the
    hand 9.5 mm deeper.

    Several segments are pure dwells -- same pose as the one before, non-zero duration.
    They are not padding. The arm is a position servo with roughly 330 ms of settling
    (PHASE5_PLAN corrections), so a waypoint reached in the plan is NOT a waypoint reached
    by the hardware. Without a dwell the gripper opens while the arm is still 1-2 cm short
    of the bin and travelling, and the box is thrown rather than placed -- which is
    exactly what the first calibration run did: 12/15 grasps but only 5/15 placements."""
    bx, by = box_xy
    gx, gy = bin_xy
    box = np.array([bx, by, BOX_REST_Z])
    over_bin = np.array([gx, gy, L.TABLE_TOP_Z])
    hover = box + [0, 0, 0.10]
    backoff_pos = hover + [-RETREAT_BACKOFF, 0, RETREAT_BACKOFF_LIFT]
    retreat = [
        (backoff_pos, OPEN, RETREAT_BACKOFF_SECS),  # 1a withdraw toward the base -- see
        (hover, OPEN, RETREAT_BACKOFF_SECS),        # 1b RETREAT_BACKOFF: the -x labels
        (hover, OPEN, 0.2),                  # 1c dwell: let the re-approach settle
    ] if backoff else []
    return [
        (hover, OPEN, 1.0),                  # 1  hover above the box, gripper open
        *retreat,
        (box, OPEN, 0.8),                   # 2  descend onto it
        (box, OPEN, 0.3),                   # 3  dwell: let the servo actually arrive
        (box, CLOSED, 0.6),                 # 4  close, arm holding station
        (box, CLOSED, 0.4),                 # 5  dwell: let the grasp seat before loading it
        (box + [0, 0, 0.15], CLOSED, 1.0),  # 6  lift clear of the table
        (over_bin + [0, 0, 0.20], CLOSED, 2.0),  # 7  transport
        (over_bin + [0, 0, 0.20], CLOSED, 0.4),  # 8  dwell: kill the lateral overshoot
        (over_bin + [0, 0, 0.07], CLOSED, 0.8),  # 9  lower to just above the rim
        (over_bin + [0, 0, 0.07], CLOSED, 0.3),  # 10 dwell before letting go
        (over_bin + [0, 0, 0.07], OPEN, 0.5),    # 11 release
        (over_bin + [0, 0, 0.20], OPEN, 0.6),    # 12 rise clear of the bin
        (np.asarray(start_pos, dtype=float), OPEN, 1.0),  # 13 return to the start pose
    ]


def reference_track(model, data, site_id, box_xy, bin_xy, control_hz, speed_scale=1.0,
                    motion_speed=None, backoff=True):
    """Expand the waypoints into one (target_pos, gripper) per control tick.

    `speed_scale` multiplies every segment DURATION, so >1 makes the reference slower. It
    exists because the label magnitude is set by the reference VELOCITY and the plant's
    tracking lag, not by the geometry:

        label = lag / DELTA_POS_SCALE,  lag ~= v * dt / realised_fraction

    OSC realises 12.3% of a commanded per-tick displacement where the retuned position
    servo realised 72%, so the SAME reference demands ~6x the corrective delta and the
    label clips at the +-1 action bound. Measured on the ik-era waypoint timings under OSC:
    dx q01 pinned at exactly -1.000 in every cohort, against a4's -0.648 and released
    LIBERO's -0.679. Saturated labels are not a scaling nuisance -- they destroy the
    gradient's direction information, since every clipped tick reports "full scale" no
    matter how far off the reference actually is.

    Slowing the reference is the correct lever rather than shrinking the waypoints: the
    geometry (where the box is, how high to lift) is task-defined, the timing is not.
    Solving label <= 1 for v gives v <= DELTA_POS_SCALE * realised / dt = 0.123 m/s, which
    the ik-era timings exceed on the retreat (0.27 m/s) and the transport (~0.15 m/s).

    Seeded from the arm's ACTUAL pose, so the first segment starts wherever the (possibly
    jittered) reset left it.

    Interpolation is a SMOOTHSTEP (3a^2 - 2a^3), not linear. Linear segments meet at a
    velocity discontinuity: the reference jumps from full speed to zero (or to a new
    direction) in one tick, the arm follows with a jerk, and the box is shaken out of a
    friction grasp. That is what the first two calibration runs were doing -- the box was
    lifted to exactly the commanded height every time and then dropped 0.10-0.12 m short of
    the bin, mid-transport. Smoothstep has zero derivative at both ends of every segment,
    so the reference accelerates and decelerates instead of stepping."""
    dt = 1.0 / control_hz
    track = []
    prev_pos = data.site_xpos[site_id].copy()
    prev_grip = OPEN
    for target, grip, dur in waypoints(box_xy, bin_xy, prev_pos, backoff=backoff):
        dist = float(np.linalg.norm(np.asarray(target, dtype=float) - prev_pos))
        if motion_speed is not None and dist > DWELL_DIST:
            # Retime by DISTANCE instead of using the hand-set duration. The hand-set
            # timings are uneven against the ceiling -- measured on the nominal geometry,
            # the retreat and the closing return run at 0.275-0.285 m/s mean while the
            # descent and the transport crawl at 0.125-0.163, so the same trajectory both
            # clips labels AND wastes ticks. One target speed fixes both ends: the fast
            # segments slow just enough to stop clipping, the slow ones speed up.
            #
            # The floor is not cosmetic. A short segment retimed to a couple of ticks is a
            # near-step in the reference, which is the velocity discontinuity smoothstep
            # exists to avoid (below) -- and that discontinuity is what shook the box out
            # of the grasp in the early calibration runs.
            dur = max(MIN_MOTION_SECS, dist / motion_speed)
        n = max(1, int(round(dur * speed_scale / dt)))
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3.0 - 2.0 * a)
            track.append(((1 - a) * prev_pos + a * np.asarray(target, dtype=float),
                          (1 - a) * prev_grip + a * grip))
        prev_pos = np.asarray(target, dtype=float)
        prev_grip = grip
    return track


def expert_action(data, site_id, target_pos, target_mat, grip):
    """The label: normalised delta from the arm's CURRENT pose to the reference pose.

    Exactly inverts what `L.apply_action` does with it -- delta / scale, clipped to the
    action range -- so feeding this action back through the controller commands the
    reference. The rotation error is taken in the world frame because that is the frame
    `apply_action` composes its rotation delta in."""
    cur_pos = data.site_xpos[site_id].copy()
    cur_mat = data.site_xmat[site_id].copy()
    dpos = (np.asarray(target_pos) - cur_pos) / L.DELTA_POS_SCALE
    drot = L._orientation_error(target_mat, cur_mat) / L.DELTA_ROT_SCALE
    a = np.empty(7, dtype=np.float64)
    a[0:3] = np.clip(dpos, -1.0, 1.0)
    a[3:6] = np.clip(drot, -1.0, 1.0)
    a[6] = grip
    return a


def run_episode(model, data, scratch, renderer, site_id, finger_qposadr, box_body,
                rng, cohort, control_hz, decimation, box_xy, bin_xy, osc=None,
                speed_scale=1.0, motion_speed=None, backoff=True):
    """Execute one demonstration. Returns (frames, states, actions, info)."""
    arm_jitter = RECOVER_START_JITTER if cohort == "recover" else 0.0
    reset_episode(model, data, rng, box_xy, arm_jitter=arm_jitter, osc=osc)

    # Hold the reset orientation throughout: measured against a live LIBERO env, our
    # grip_site at LIBERO's reset joints has axis-angle (3.140, 0, -0.089) against
    # LIBERO's own (3.141, 0.002, -0.090). Top-down, and already in distribution.
    target_mat = data.site_xmat[site_id].copy()

    track = reference_track(model, data, site_id, box_xy, bin_xy, control_hz,
                            speed_scale=speed_scale, motion_speed=motion_speed,
                            backoff=backoff)

    # Recovery cohort: two shoves, one free-armed during the approach and one loaded
    # during the transport. See RECOVER_KICK_WINDOWS.
    kicks = {}
    if cohort == "recover":
        for (lo, hi), mag in zip(RECOVER_KICK_WINDOWS, (RECOVER_KICK, RECOVER_KICK_LOADED)):
            t = int(rng.integers(max(1, int(lo * len(track))), max(2, int(hi * len(track)))))
            kicks[t] = mag

    # Perturbation windows, shared by the `drift` and `decoy` cohorts -- they differ only
    # in WHAT is executed inside a window, not in where the windows go.
    drift_ticks, drift_spans, decoy_offsets = set(), [], {}
    if cohort in ("drift", "decoy"):
        spans = _sample_windows(rng, track)
        drift_spans = spans
        if cohort == "drift":
            for a, b in spans:
                drift_ticks.update(range(a, b))
        else:
            # One offset direction per window, held for the whole window, so the excursion
            # accumulates in a straight line instead of averaging back to zero. Magnitude
            # is fixed: DECOY_OFFSET / DELTA_POS_SCALE is the label the corrective action
            # reaches once the arm has actually arrived at the decoy, so it sets how large
            # a correction the cohort teaches -- and it must stay under 1.0 or the label
            # clips at exactly the moment the recovery signal is strongest.
            for a, b in spans:
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                for t in range(a, b):
                    decoy_offsets[t] = DECOY_OFFSET * d

    frames = {"image": [], "wrist_image": []}
    states, actions = [], []
    clamped_ticks = 0
    saturated_steps = 0   # OSC's failure signal, the analogue of ik mode's "IK unreached"
    # Per-PHYSICS-STEP torque bookkeeping. An episode-wide FRACTION is not enough: three
    # saturated steps out of 70k is 0.004%, which passes any sane fraction gate, and three
    # CONSECUTIVE saturated steps on one joint is a controller that lost authority for 6 ms.
    # These two catch the localised burst the fraction averages away.
    peak_torque_frac = 0.0      # max over every step and joint of |tau| / limit, pre-clip
    max_sat_run = 0             # longest unbroken run of steps with any joint saturated
    sat_run = 0
    peak_torque_tick = -1       # which control tick the peak landed on, for the log
    box_adr = model.jnt_qposadr[model.body_jntadr[box_body]]
    max_box_z = -np.inf

    for tick, (target_pos, grip) in enumerate(track):
        # Observation BEFORE the action, matching how the policy is queried at inference.
        main_img, wrist_img = L.render_cameras(renderer, data, "none")
        frames["image"].append(main_img)
        frames["wrist_image"].append(wrist_img)
        states.append(L.read_state(model, data, site_id, finger_qposadr))

        action = expert_action(data, site_id, target_pos, target_mat, grip)
        actions.append(action.astype(np.float32))

        executed = action.copy()
        if tick in decoy_offsets:
            # Drive at the DECOY. `action` -- already appended above -- stays the delta to
            # the TRUE reference, so the label is the correction for wherever this puts us.
            executed = expert_action(data, site_id, target_pos + decoy_offsets[tick],
                                     target_mat, grip)
        elif cohort == "noise" or tick in drift_ticks:
            executed[0:3] += rng.normal(0.0, NOISE_SIGMA_POS, size=3)
            executed[3:6] += rng.normal(0.0, NOISE_SIGMA_ROT, size=3)
        elif tick in kicks:
            direction = rng.normal(size=3)
            executed[0:3] += kicks[tick] * direction / np.linalg.norm(direction)
        executed = np.clip(executed, -1.0, 1.0)

        # THE design rule (see this file's header): the label is executed through the same
        # function that will consume the model's output at inference, so servo lag and
        # controller error land inside the labels. That makes the controller choice part of
        # the dataset -- an OSC dataset and an IK dataset are not interchangeable.
        if osc is None:
            _, _, was_clamped = L.apply_action(model, data, scratch, executed, site_id)
        else:
            _, _, was_clamped = L.apply_action_osc(osc, data, executed)
        clamped_ticks += int(was_clamped)
        for _ in range(decimation):
            # Under OSC the torque depends on the live state, so it is recomputed EVERY
            # physics step, not once per control tick -- robosuite does the same
            # (environments/base.py calls the controller each sim step, with set_goal
            # firing only on the first).
            if osc is not None:
                data.ctrl[L.ARM] = osc.compute_torque(model, data)
                # JOINT-steps, not steps-with-any-saturation, so this is directly
                # comparable with the run-log figure libero_closed_loop.py reports
                # (denominator: ticks * decimation * 7). Counting them differently under
                # the same name would silently break the one comparison worth making --
                # "did the demos need torques the policy also needs".
                saturated_steps += int(osc.last_saturated)
                if osc.last_peak_frac > peak_torque_frac:
                    peak_torque_frac = osc.last_peak_frac
                    peak_torque_tick = tick
                sat_run = sat_run + 1 if osc.last_saturated else 0
                max_sat_run = max(max_sat_run, sat_run)
            mujoco.mj_step(model, data)
        max_box_z = max(max_box_z, float(data.qpos[box_adr + 2]))

    box = data.qpos[box_adr:box_adr + 3].copy()
    lifted = max_box_z > BOX_REST_Z + 0.05
    placed = (abs(box[0] - bin_xy[0]) < 0.05 and abs(box[1] - bin_xy[1]) < 0.05
              and box[2] < L.TABLE_TOP_Z + 0.06)
    info = {
        "lifted": bool(lifted),
        "placed": bool(placed),
        "box_final": box.tolist(),
        "box_max_z": max_box_z,
        "table_clamped_ticks": clamped_ticks,
        "osc_saturated_steps": saturated_steps,
        "ticks": len(track),
        "kick_ticks": sorted(int(t) for t in kicks),
        "drift_spans": [[int(a), int(b)] for a, b in drift_spans],
        "drift_ticks": len(drift_ticks),
        # Fraction of joint-steps whose commanded torque hit the actuator ctrlrange. Not
        # merely diagnostic: --max-saturated-frac rejects on it. A demonstration that only
        # succeeded by asking for torque the arm cannot deliver is teaching the policy to
        # ask for it too, and the policy has no such escape hatch at inference.
        "saturated_frac": (saturated_steps / (len(track) * decimation * 7)
                           if osc is not None and len(track) else 0.0),
        "saturated_steps": saturated_steps,
        "peak_torque_frac": peak_torque_frac,
        "peak_torque_tick": peak_torque_tick,
        "max_sat_run": max_sat_run,
        "box_xy": [float(box_xy[0]), float(box_xy[1])],
        "bin_xy": [float(bin_xy[0]), float(bin_xy[1])],
    }
    return frames, np.array(states), np.array(actions), info


def selftest(model, data, site_id, osc=None):
    """Print the frame checks that the 2026-07-28 scene corrections were made against.
    Cheap regression guard: if someone re-edits the scene, these move.

    `osc` must be the controller for whichever --control-mode is active: the settled pose
    differs between the two plants (the position servo sags under gravity, OSC does not),
    so running this guard against the wrong one reports numbers no episode will start from."""
    # Go through reset_episode rather than setting qpos[:7] by hand, so this guard reports
    # the state an episode actually starts from (objects placed, arm settled) rather than a
    # near-identical but not identical hand-rolled one. Measured: both paths converge by 50
    # steps and agree to <0.1 mm here, so this is about the guard being the real thing, not
    # about a bug it was hiding.
    #
    # What this number is: `0.2728` against LIBERO's 0.2733 is the PURE FORWARD KINEMATICS
    # of LIBERO_INIT_QPOS. Whether the arm actually HOLDS that pose is plant-dependent, and
    # that is the whole reason this guard runs through reset_episode:
    #   osc  -- holds it. Measured 0.2728, i.e. the full 0.5 mm match, because a torque
    #           controller with gravity compensation has no standing sag.
    #   ik   -- does not. The position servo sags: 0.2680 on menagerie's stock gains
    #           (4.84 mm, so a 5.3 mm error against LIBERO, not 0.5 mm), 0.2704 on the
    #           kp x2 / kd x0.7 gains (2.44 mm).
    # So the number printed below depends on --control-mode, and only the osc one should be
    # read as "we match LIBERO".
    reset_episode(model, data, np.random.default_rng(0), (0.56, 0.0), osc=osc)
    state = L.read_state(model, data, site_id, _finger_adr(model))
    tg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
    top = model.geom_size[tg][2] + data.geom_xpos[tg][2]
    print("LIBERO ground truth was measured from a live OffScreenRenderEnv:\n")
    print(f"  eef_pos      {np.round(state[:3], 4)}   LIBERO (-0.1485, 0, 1.1733)")
    print(f"  axisangle    {np.round(state[3:6], 4)}   LIBERO ( 3.1408, 0.0018, -0.0899)")
    print(f"  gripper_qpos {np.round(state[6:8], 4)}   LIBERO ( 0.0208, -0.0208)")
    print("      ^ expected to differ. LIBERO's Panda gripper resets to init_qpos "
          "0.020833,\n        i.e. half open, and opens to 0.04 once the first action "
          "(-1) lands. Ours\n        is commanded open at reset and has settled to 0.04 "
          "before the first frame.\n        Same range, same sign convention, a "
          "one-or-two-tick difference at t=0.")
    print(f"  table top    {top + L.LIBERO_ORIGIN_OFFSET[2]:.4f}          LIBERO 0.9000")
    print(f"  eef above table {state[2] - (top + L.LIBERO_ORIGIN_OFFSET[2]):.4f}   "
          f"LIBERO 0.2733")


def _finger_adr(model):
    return [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ("finger_joint1", "finger_joint2")]


def main():
    # Declared up front: the --noise-sigma-* help strings below interpolate these, and
    # Python rejects `global X` that appears after any use of X in the same function.
    global NOISE_SIGMA_POS, NOISE_SIGMA_ROT

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="libero/fine_tune/a1")
    p.add_argument("--reach", type=int, default=20)
    p.add_argument("--noise", type=int, default=20)
    p.add_argument("--recover", type=int, default=10)
    p.add_argument(
        "--drift", type=int, default=0,
        help=(
            f"episodes in the `drift` cohort: {DRIFT_WINDOWS[0]}-{DRIFT_WINDOWS[1]} random "
            f"WINDOWS of {DRIFT_WINDOW_TICKS[0]}-{DRIFT_WINDOW_TICKS[1]} ticks each, with "
            "the noise-cohort Gaussian applied inside them and the clean expert action "
            "outside. Between `noise` (perturbed every tick, so the arm is never on the "
            "reference) and `recover` (a single-tick impulse). Uses --noise-sigma-pos/rot"
        ),
    )
    p.add_argument(
        "--decoy", type=int, default=0,
        help=(
            f"episodes in the `decoy` cohort: same windows as --drift, but inside them the "
            f"EXECUTED action points at a target {DECOY_OFFSET * 1000:.0f} mm off the "
            "reference in a fixed random direction, so the arm deliberately moves wrong and "
            "then recovers. The LABEL always points at the true reference. Deterministic "
            "where --drift is stochastic. Do NOT get this by editing `waypoints` instead -- "
            "that puts the wrong move in the reference and labels it as correct"
        ),
    )
    p.add_argument(
        "--no-backoff", action="store_true",
        help=(
            "drop the withdraw-and-re-approach segments (1a/1b/1c) between the hover and "
            "the descent, making the approach one clean descent. Those segments exist to "
            "put -x motion in the labels (see RETREAT_BACKOFF); the closing return to the "
            "start pose is kept and becomes the only -x source, so check dx q01 against "
            "released LIBERO's -0.679 before trusting a dataset collected with this"
        ),
    )
    p.add_argument(
        "--max-saturated-frac", type=float, default=None, metavar="F",
        help=(
            "reject an episode whose commanded torque hit the actuator ctrlrange on more "
            "than this fraction of joint-steps (denominator ticks * decimation * 7). OSC "
            "mode only. Off by default, which is how a1-a7 were collected: saturation was "
            "recorded in the episode metadata but never gated on. Measured at "
            "--delta-pos-scale 0.05 with clean cohorts: 0.000%%. 0.002 is a sane setting"
        ),
    )
    p.add_argument(
        "--max-torque-frac", type=float, default=None, metavar="F",
        help=(
            "reject an episode in which ANY joint at ANY physics step demanded more than "
            "this fraction of its actuator limit, measured BEFORE the clip. This is the "
            "every-point check: --max-saturated-frac is an episode average and a burst of "
            "a few steps in ~70k disappears into it. Measured at --delta-pos-scale 0.05: "
            "peak 0.73 of limit (j2, holding the arm against gravity), and the decoy "
            "windows peak LOWER than the rest of the episode. 0.95 leaves real headroom "
            "while still catching anything approaching the rail"
        ),
    )
    p.add_argument(
        "--max-sat-run", type=int, default=None, metavar="N",
        help=(
            "reject an episode with more than N CONSECUTIVE physics steps in which any "
            "joint was saturated. Catches a joint that lost authority for a stretch, which "
            "neither the average nor a single-step peak distinguishes from isolated "
            "clipping. 0 means no run at all is tolerated"
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-size", type=int, default=L.RENDER_HEIGHT)
    p.add_argument("--control-hz", type=float, default=L.CONTROL_HZ)
    p.add_argument(
        "--control-mode", choices=("osc", "ik"), default="osc",
        help=(
            "which controller EXECUTES the demonstrations, and therefore which controller "
            "the labels are written against. Must match what the closed loop will run at "
            "inference -- see this file's header. 'osc' (default, and the client's default "
            "too) is the robosuite OSC_POSE port writing joint torques; 'ik' is the "
            "original Route A differential IK into position servos. The scene follows "
            "automatically unless --model-path overrides it"
        ),
    )
    p.add_argument(
        "--model-path", default=None,
        help="scene XML; defaults to the one matching --control-mode (L.SCENE_FOR_MODE)",
    )
    p.add_argument(
        "--min-clearance", type=float, default=None,
        help=(
            "metres the commanded eef target is held above the table top. Default depends "
            "on --control-mode: OFF for osc, because OSC commands forces and yields on "
            "contact the way robosuite does, so the Route A clamp is not needed and would "
            "only teach a constraint inference will not have; "
            f"{L.DEFAULT_MIN_CLEARANCE} for ik, where position actuators would otherwise "
            "drive the hand through the surface. Negative disables it (0 does NOT)"
        ),
    )
    p.add_argument("--max-attempts-per-episode", type=int, default=8,
                   help="rejection sampling budget; an episode that never lifts the box "
                        "is resampled rather than kept")
    p.add_argument("--keep-unplaced", action="store_true",
                   help="accept episodes that lift the box but miss the bin. Off by "
                        "default: a demonstration that fails the task is a demonstration "
                        "of failing the task")
    p.add_argument(
        "--noise-sigma-pos", type=float, default=None,
        help=(
            f"stddev of the Gaussian added to the EXECUTED translation action in the noise "
            f"cohort (default {NOISE_SIGMA_POS}). This is plant-dependent and must be "
            "recalibrated whenever the controller changes: sigma perturbs the COMMANDED "
            "action, but what knocks the arm off the reference is the fraction that gets "
            "REALISED -- 72%% per tick for the retuned position servo a3/a4 were collected "
            "on, 12.3%% for OSC. Carrying a sigma across that change silently changes the "
            "physical disturbance by ~6x. See docs/SERVO_DROOP.md sec.4.3 for the sweep "
            "that set 0.08"
        ),
    )
    p.add_argument(
        "--speed-scale", type=float, default=None,
        help=(
            "multiplier on every reference-segment DURATION (>1 = slower reference). "
            "Plant-dependent, like the sigmas: the label is the tracking lag over "
            "DELTA_POS_SCALE, and lag scales with reference velocity over the plant's "
            "realised fraction. Default 1.0 for ik (the timings were tuned there) and "
            f"{OSC_SPEED_SCALE} for osc, whose 12.3%%/tick makes the ik-era timings clip "
            "the labels at the +-1 bound. See reference_track's docstring"
        ),
    )
    p.add_argument("--noise-sigma-rot", type=float, default=None,
                   help=f"rotation counterpart of --noise-sigma-pos (default {NOISE_SIGMA_ROT})")
    p.add_argument(
        "--delta-pos-scale", type=float, default=L.DELTA_POS_SCALE,
        help=(
            "metres per unit action -- the OTHER end of the same constraint --speed-scale "
            "addresses, and usually the better one. The label ceiling is "
            "v <= DELTA_POS_SCALE * realised / dt, so raising the scale raises the arm's "
            "top speed instead of lowering the expert's. a5 took the --speed-scale route "
            "and produced 27 s / 539-tick episodes that STILL clip dx on 3.07%% of frames; "
            f"--delta-pos-scale {L.FINE_TUNE_DELTA_POS_SCALE} clips 0.93%% at the original "
            "10.8 s. SmolVLA normalises actions MEAN_STD from the "
            "dataset's own stats, so the unit itself is invisible to training. THE SERVING "
            "CLIENT MUST PASS THE SAME VALUE (libero_closed_loop.py --delta-pos-scale); "
            f"default {L.DELTA_POS_SCALE} is LIBERO's own, for stock-checkpoint compatibility"
        ),
    )
    p.add_argument(
        "--motion-speed", type=float, default=None, metavar="M_PER_S",
        help=(
            "retime every MOTION segment of the reference to this mean speed, instead of "
            "using its hand-set duration. Dwells (targets under 1 mm apart) keep theirs -- "
            "they are settling and gripper-actuation time, set by physics not distance. "
            "The hand-set timings are uneven against the label ceiling: on the nominal "
            "geometry the retreat and closing return run at 0.275-0.285 m/s while the "
            "descent and transport crawl at 0.125-0.163, so one target speed both stops "
            "the clipping and reclaims the wasted ticks. Pass 'auto' semantics by omitting "
            f"it in osc mode, which uses {MOTION_SPEED_HEADROOM:g} x the ceiling "
            "(DELTA_POS_SCALE * 0.123 * control_hz / 1.5, the 1.5 being smoothstep's "
            "peak-over-mean). 0 disables retiming and restores the hand-set durations"
        ),
    )
    p.add_argument(
        "--bin-layout", choices=("shuffle", "scene"), default="shuffle",
        help=(
            "where the three bins go each episode. 'shuffle' (default, what a1-a5 used) "
            "permutes them across three slots with jitter, so green has to be found by "
            "colour. 'scene' pins them to the scene XML's own positions -- which is the "
            "ONLY layout libero_closed_loop.py ever evaluates, since it has no bin "
            "randomisation. In a5 the evaluated layout drew 6 of 30 episodes. See "
            "bin_layout's docstring for the trade"
        ),
    )
    p.add_argument("--verbose", action="store_true",
                   help="print why each rejected attempt was rejected")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.model_path is None:
        args.model_path = L.SCENE_FOR_MODE[args.control_mode]

    # Plant-dependent defaults. These MUST key off --control-mode rather than sitting as
    # bare module constants: an ik-calibrated sigma under OSC delivers ~1/6 the physical
    # disturbance, so the noise cohort silently degenerates into a second reach cohort --
    # near-100% first-attempt keeps, which reads as success while teaching nothing.
    L.set_delta_pos_scale(args.delta_pos_scale)

    # Same ceiling that sets --speed-scale, expressed as a speed rather than a multiplier.
    # OSC only: the ik plant realises 72% per tick and was never the binding case.
    #
    # RESOLVED BEFORE --speed-scale as of 2026-08-03, because the speed default has to know
    # whether the retiming is already doing the work. See the block below.
    if args.motion_speed is None and args.control_mode == "osc":
        args.motion_speed = (MOTION_SPEED_HEADROOM * args.delta_pos_scale
                             * OSC_REALISED_FRACTION * args.control_hz / SMOOTHSTEP_PEAK)
    if args.motion_speed is not None and args.motion_speed <= 0:
        args.motion_speed = None   # explicit 0 = keep the hand-set durations

    # The two knobs solve the same clipping problem, so the speed default has to know
    # which one is already doing the work. OSC_SPEED_SCALE was measured at LIBERO's
    # 0.05 m/unit; the ceiling is linear in the scale, so a larger scale needs
    # proportionally less slowdown, and at 0.125 it needs none.
    #
    # FIXED 2026-08-03. `reference_track` applies BOTH knobs -- it retimes the segment to
    # `dist / motion_speed` and then multiplies by `speed_scale` -- so the ceiling-derived
    # default below was double-counting whenever the retiming was also active. It went
    # unnoticed because the two were never simultaneously non-trivial in any collected
    # dataset: a5 is scale 0.05 (speed_scale 2.5) but predates --motion-speed entirely,
    # while a6/a7 are scale 0.10-0.20, where `max(1.0, ...)` pins speed_scale at 1.0. It
    # bites exactly at LIBERO's own 0.05 with the retiming on, which is the setting this
    # project standardised on: 2.5x on top of an already ceiling-respecting retime, i.e.
    # ~4x the intended episode length for no benefit -- and a5's own measurement is that
    # the slowdown did not reduce clipping anyway (README sec.8.1, dx q01 pinned at -1.000).
    #
    # So: --motion-speed already respects the ceiling by construction. Do not slow down on
    # top of it. The ceiling-derived default survives only for --motion-speed 0, where the
    # hand-set durations are back and nothing else is bounding the reference velocity.
    if args.speed_scale is None:
        if args.control_mode != "osc" or args.motion_speed is not None:
            args.speed_scale = 1.0
        else:
            args.speed_scale = max(
                1.0, OSC_SPEED_SCALE * L.DELTA_POS_SCALE_LIBERO / args.delta_pos_scale)
    # Same coupling for the sigmas, and for the same reason: they are in ACTION units, so
    # the physical disturbance one unit delivers is proportional to the scale. 0.47 at
    # 0.05 m/unit is 0.188 at 0.125, and leaving it at 0.47 would triple the kick the
    # sweep was calibrated to (README sec.4.3) and simply stop episodes succeeding.

    sigma_gain = L.DELTA_POS_SCALE_LIBERO / args.delta_pos_scale
    if args.noise_sigma_pos is None and args.control_mode == "osc":
        args.noise_sigma_pos = OSC_NOISE_SIGMA_POS * sigma_gain
    if args.noise_sigma_rot is None and args.control_mode == "osc":
        args.noise_sigma_rot = OSC_NOISE_SIGMA_ROT * sigma_gain

    if args.noise_sigma_pos is not None:
        NOISE_SIGMA_POS = args.noise_sigma_pos
    if args.noise_sigma_rot is not None:
        NOISE_SIGMA_ROT = args.noise_sigma_rot

    # The table clamp is a Route A workaround: it exists because position actuators track a
    # commanded pose and will press through the surface. OSC yields on contact, so under it
    # the clamp is both unnecessary and actively harmful -- every clamped tick is a label
    # teaching a constraint that inference will not impose. Mutating the module global is
    # how libero_closed_loop.py itself does it; apply_action/apply_action_osc read it.
    if args.min_clearance is None:
        args.min_clearance = -1.0 if args.control_mode == "osc" else L.DEFAULT_MIN_CLEARANCE
    L.EEF_MIN_Z = (-np.inf if args.min_clearance < 0
                   else L.TABLE_TOP_Z + args.min_clearance)

    model = mujoco.MjModel.from_xml_path(args.model_path)
    # The same guard L.build_sim applies. This file builds its own model rather than going
    # through build_sim (it needs per-episode resets, not one settled scene), so without
    # this it would happily write IK joint targets into <motor> actuators -- reinterpreting
    # N.m as radians and producing a structurally perfect dataset of physical nonsense.
    if (args.control_mode == "osc") != L.is_torque_actuated(model):
        raise SystemExit(
            f"--control-mode {args.control_mode} needs "
            f"{'torque <motor>' if args.control_mode == 'osc' else 'position servo'} arm "
            f"actuators, but {args.model_path} has the other kind. Use "
            f"{L.SCENE_FOR_MODE[args.control_mode]}, or pass the matching --control-mode."
        )

    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    finger_qposadr = _finger_adr(model)
    box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, L.TARGET_BODY)
    bin_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in BIN_NAMES}
    # Captured BEFORE any episode moves them; --bin-layout scene restores these.
    scene_bin_xy = [model.body_pos[bin_ids[n]][:2].copy() for n in BIN_NAMES]

    osc = L.OSCController(model, site_id) if args.control_mode == "osc" else None

    if args.selftest:
        selftest(model, data, site_id, osc=osc)
        return

    decimation = int(round((1.0 / model.opt.timestep) / args.control_hz))
    renderer = L.make_renderer(model, args.render_size)
    rng = np.random.default_rng(args.seed)

    writer = LeRobotV30Writer(
        root=args.out, fps=args.control_hz,
        image_shape=(args.render_size, args.render_size, 3),
        state_dim=8, action_dim=7, robot_type="panda",
    )

    print(f"scene       : {args.model_path}")
    print(f"controller  : {args.control_mode}"
          + ("  (robosuite OSC_POSE port, joint torques)" if osc is not None
             else "  (Route A differential IK -> position servos)"))
    print(f"control     : {args.control_hz:g} Hz  ({decimation} physics steps/action)")
    print(f"action space: 7-D delta eef pose, [-1,1], {L.DELTA_POS_SCALE} m / "
          f"{L.DELTA_ROT_SCALE} rad per unit")
    print("table floor : DISABLED" if not np.isfinite(L.EEF_MIN_Z)
          else f"table floor : eef target clamped to {L.EEF_MIN_Z:+.4f} "
               f"({args.min_clearance * 1000:.0f} mm above the top)")
    print(f"noise sigma : pos {NOISE_SIGMA_POS}  rot {NOISE_SIGMA_ROT}")
    print("motion speed: " + ("hand-set segment durations" if args.motion_speed is None
                              else f"{args.motion_speed:.3f} m/s mean, retimed by distance "
                                   f"(peak {args.motion_speed * SMOOTHSTEP_PEAK:.3f} vs "
                                   f"ceiling {args.delta_pos_scale * OSC_REALISED_FRACTION * args.control_hz:.3f})"))
    print(f"bin layout  : {args.bin_layout}" + (
        f"  (green pinned at {np.round(scene_bin_xy[BIN_NAMES.index('green_bin')], 3)}, "
        "the layout the closed loop evaluates)" if args.bin_layout == "scene" else
        "  (3 slots permuted; the closed loop needs --randomize-bins to match)"))
    print(f"approach    : " + ("hover -> descend (backoff removed)" if args.no_backoff
                               else "hover -> withdraw -> re-approach -> descend"))
    _gates = [g for g in (
        None if args.max_saturated_frac is None
        else f"frac > {args.max_saturated_frac:.3%} of joint-steps",
        None if args.max_torque_frac is None
        else f"any step > {args.max_torque_frac:.0%} of a joint's limit",
        None if args.max_sat_run is None
        else f"> {args.max_sat_run} consecutive saturated steps",
    ) if g]
    print("torque gates: " + ("OFF -- recorded, never rejected on" if not _gates
                              else "reject if " + "; or ".join(_gates)))
    print(f"out         : {args.out}\n")

    plan = ([("reach", i) for i in range(args.reach)]
            + [("noise", i) for i in range(args.noise)]
            + [("drift", i) for i in range(args.drift)]
            + [("decoy", i) for i in range(args.decoy)]
            + [("recover", i) for i in range(args.recover)])

    t0 = time.time()
    rejected, missing = 0, []
    for cohort, idx in plan:
        # Rejection sampling. A demonstration that fails the task is a demonstration of
        # failing the task, so failed attempts are DISCARDED, never written -- including
        # in the noise and recover cohorts, where the perturbation is meant to produce a
        # recoverable excursion, not an unrecoverable one. If the budget runs out the slot
        # is left empty and reported, rather than filled with a bad episode.
        kept = None
        for attempt in range(args.max_attempts_per_episode):
            box_xy = (rng.uniform(*L.BOX_SAMPLE_X), rng.uniform(*L.BOX_SAMPLE_Y))
            bin_xy = bin_layout(model, bin_ids, rng, mode=args.bin_layout,
                                scene_xy=scene_bin_xy)
            frames, states, actions, info = run_episode(
                model, data, scratch, renderer, site_id, finger_qposadr, box_body,
                rng, cohort, args.control_hz, decimation, box_xy, bin_xy, osc=osc,
                speed_scale=args.speed_scale, motion_speed=args.motion_speed,
                backoff=not args.no_backoff)
            # Three torque gates, deliberately not one. The fraction catches an episode
            # that fights the limits throughout; the peak catches a single over-limit
            # demand anywhere in ~70k joint-steps, which any fraction gate averages into
            # nothing; the run length catches a joint that lost authority for a stretch
            # rather than for one isolated step.
            over_torque = (
                (args.max_saturated_frac is not None
                 and info["saturated_frac"] > args.max_saturated_frac)
                or (args.max_torque_frac is not None
                    and info["peak_torque_frac"] > args.max_torque_frac)
                or (args.max_sat_run is not None
                    and info["max_sat_run"] > args.max_sat_run))
            if (info["lifted"] and (info["placed"] or args.keep_unplaced)
                    and not over_torque):
                kept = (frames, states, actions, info, attempt + 1)
                break
            rejected += 1
            if args.verbose:
                print(f"    reject {cohort}_{idx:02d} try {attempt + 1}: "
                      f"lifted={info['lifted']} placed={info['placed']} "
                      f"sat={info['saturated_frac']:.4%} "
                      f"peak_tau={info['peak_torque_frac']:.2f}@t{info['peak_torque_tick']} "
                      f"run={info['max_sat_run']}"
                      f"{' OVER-TORQUE' if over_torque else ''} "
                      f"box_max_z={info['box_max_z']:+.4f} "
                      f"box_final={np.round(info['box_final'], 3)} "
                      f"bin={np.round(info['bin_xy'], 3)} clamped={info['table_clamped_ticks']}")
        name = f"{cohort}_{idx:02d}"
        if kept is None:
            missing.append(name)
            print(f"  {name:<12} DROPPED after {args.max_attempts_per_episode} attempts")
            continue
        frames, states, actions, info, attempts = kept
        writer.add_episode(frames, states, actions, task=INSTRUCTION,
                           extra={"name": name, "cohort": cohort, "attempts": attempts,
                                  **info})
        print(f"  {name:<12} ticks={info['ticks']:>3}  lifted={info['lifted']:<5} "
              f"placed={info['placed']:<5} clamped={info['table_clamped_ticks']:>3}  "
              f"sat={info['saturated_frac']:.4%}  peak_tau={info['peak_torque_frac']:.2f}"
              f"  run={info['max_sat_run']}  attempts={attempts}"
              + (f"  drift={info['drift_spans']}" if info["drift_spans"] else ""))

    summary = writer.finalize()
    if missing:
        print(f"\nWARNING: {len(missing)} slots left empty: {', '.join(missing)}")
    renderer.close()
    print(f"\n{summary['episodes']} episodes, {summary['frames']} frames, "
          f"{rejected} rejected attempts, {time.time() - t0:.0f}s -> {args.out}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
