"""Collect scripted expert demonstrations of the green-ball pick-and-place task, in the
**LIBERO** convention, for LoRA + action-expert fine-tuning of MolmoAct2-LIBERO.

Not to be confused with `phase4_collect_demos.py` at the repo root. That one records the
**DROID** convention (8-D absolute joint targets, 15 Hz, `scene_pick_place.xml`) for the
DROID checkpoint. This one records what the LIBERO checkpoint speaks:

                          phase4_collect_demos.py      this file
    action                8-D absolute joint pos       7-D delta eef pose, [-1, 1]
    state                 [q1..q7, gripper_rad]        [eef_pos(3), axisangle(3),
                                                        gripper_qpos(2)], LIBERO frame
    rate                  15 Hz                        20 Hz (robosuite control_freq)
    cameras               external_cam, wrist_cam      image, wrist_image
    dataset               LeRobot v2.1 + mp4           LeRobot v3.0 + inline PNG
    scene                 scene_pick_place.xml         scene_libero_hand.xml

THE ONE DESIGN RULE
-------------------
Every recorded action is produced by, and executed through, `libero_closed_loop`'s own
`apply_action` -- the exact function that will consume the fine-tuned model's output at
inference. Nothing here re-implements the controller.

That is the entire point. The diagnostic in PROGRESS.md sec.20 scored 3/3 on a real LIBERO
task through robosuite's OSC, which proved the checkpoint and our serving are both fine
and located the failure in our environment. The leading explanation is controller
mismatch: the policy learned OSC's compliant transfer function and we drive a stiff
position servo through IK ("Route A"). A fine-tune can absorb that mismatch, but ONLY if
the actions it is trained on are the actions that produce the recorded motion in our
controller. So the expert here is a Cartesian reference trajectory, and the label at each
tick is the normalised delta from the arm's ACTUAL pose to that reference -- closed-loop,
not a replayed open-loop plan. Servo lag, IK error and the table clamp all end up inside
the labels, which is where they have to be for the model to learn to compensate for them.

THE THREE COHORTS
-----------------
  reach   (20)  Nominal. Randomised ball position and bin layout, LIBERO's reset pose.
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

BALL_RADIUS = 0.02
BALL_REST_Z = L.TABLE_TOP_Z + BALL_RADIUS

# Bin anchor slots, read off scene_libero_hand.xml. Which colour sits at which slot is
# shuffled per episode so the model has to LOOK for the green one rather than memorise a
# fixed coordinate -- the same argument phase4_collect_demos.py makes.
BIN_SLOTS = [(0.56, 0.25), (0.56, -0.25), (0.80, 0.0)]
BIN_SLOT_JITTER = 0.02
BIN_NAMES = ("green_bin", "blue_bin", "yellow_bin")

# Noise cohort. Sigma is in ACTION units, so 0.15 is 7.5 mm of translation per tick
# (0.15 * DELTA_POS_SCALE) and 0.04 is 0.02 rad of rotation. Sized to knock the arm
# meaningfully off the reference without making the task unachievable -- the released
# LIBERO actions have a per-channel std around 0.33, so this is a visible but minority
# perturbation. The gripper channel is deliberately NOT perturbed: flipping it mid-grasp
# does not produce a recoverable state, it produces a dropped ball.
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

# Recovery cohort.
#
# a1's version was ONE kick of 0.85 placed in the first 45% of the episode, and it was too
# easy: 10/10 episodes were kept on the first attempt, which means the reference tracker
# absorbed the disturbance without the cohort teaching much. A single kick during the
# approach has 100+ ticks left to wash out and an empty gripper, so nothing is at stake.
#
# a3 kicks TWICE, and the second one is the point: it lands mid-transport with the ball
# actually in the hand, which is where the observed closed-loop failures live (PROGRESS
# sec.18 -- grasp achieved, retention not, run degenerates from chunk 5). A loaded kick can
# genuinely drop the ball, and rejection sampling then throws that episode away, so this
# costs attempts -- which is the honest sign that the disturbance is doing work now.
# The loaded kick is smaller (0.6) than the free one deliberately: at 0.85 with a ball in
# the gripper almost every episode fails and the cohort collects nothing.
RECOVER_START_JITTER = 0.09   # radians, per joint, on top of LIBERO's reset pose
RECOVER_KICK = 0.85           # free-arm disturbance during the approach, near full scale
RECOVER_KICK_LOADED = 0.60    # smaller: fired mid-transport, with the ball in the gripper
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
# overshoots past the ball and never comes back. Noise sigma cannot buy this -- raising it
# past ~0.09 simply stops episodes from succeeding (sec.4.3) -- so the -x motion has to come
# from the EXPERT trajectory itself, where it is labelled unconditionally and survives
# rejection sampling by construction.
#
# Two segments are added:
#   (a) a back-off-and-re-approach inserted between the hover and the descent. The arm
#       arrives above the ball, withdraws toward the base, then comes back in. This is the
#       recovery manoeuvre we want the policy to have, demonstrated in the exact visual
#       context (ball centred, gripper open) where it will need it.
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


def bin_layout(model, bin_ids, rng):
    """Shuffle the three bins across the three slots and return green's (x, y)."""
    order = rng.permutation(len(BIN_SLOTS))
    green_xy = None
    for name, slot_idx in zip(BIN_NAMES, order):
        x, y = BIN_SLOTS[slot_idx]
        jx, jy = rng.uniform(-BIN_SLOT_JITTER, BIN_SLOT_JITTER, size=2)
        model.body_pos[bin_ids[name]] = [x + jx, y + jy, L.TABLE_TOP_Z]
        if name == "green_bin":
            green_xy = (x + jx, y + jy)
    return green_xy


def reset_episode(model, data, rng, ball_xy, arm_jitter=0.0):
    """LIBERO's reset pose, plus this episode's ball and bin layout, settled."""
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home != -1:
        mujoco.mj_resetDataKeyframe(model, data, home)

    q = L.LIBERO_INIT_QPOS.copy()
    if arm_jitter > 0:
        lower, upper = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
        q = np.clip(q + rng.normal(0.0, arm_jitter, size=7), lower, upper)
    data.qpos[:7] = q
    data.ctrl[L.ARM] = q
    data.ctrl[7] = L.GRIPPER_CTRL_MAX  # open

    for name, pos in [("green_ball", (ball_xy[0], ball_xy[1], BALL_REST_Z)),
                      ("red_box", (0.56, -0.28, BALL_REST_Z))]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            continue
        adr = model.jnt_qposadr[model.body_jntadr[bid]]
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)


def waypoints(ball_xy, bin_xy, start_pos):
    """The Cartesian reference trajectory, as (grip_site target, gripper, seconds).

    `start_pos` is the arm's ACTUAL settled pose at reset (jittered, for the recover
    cohort), used as the target of the closing retreat so the episode ends where it began.

    Heights are relative to the ball and the table, both of which moved on 2026-07-28 when
    the table was corrected upward by 100 mm -- so they are written in terms of
    BALL_REST_Z / TABLE_TOP_Z rather than as literals.

    The grasp target is the ball CENTRE, not some offset above it. That only became the
    right answer once `grip_site` was moved to robosuite's literal 0.097: the site is now
    robosuite's own grasp point, so aiming it at the object centre is exactly what LIBERO's
    demonstrations do. Under the old 0.1065 site the same command would have driven the
    hand 9.5 mm deeper.

    Several segments are pure dwells -- same pose as the one before, non-zero duration.
    They are not padding. The arm is a position servo with roughly 330 ms of settling
    (PHASE5_PLAN corrections), so a waypoint reached in the plan is NOT a waypoint reached
    by the hardware. Without a dwell the gripper opens while the arm is still 1-2 cm short
    of the bin and travelling, and the ball is thrown rather than placed -- which is
    exactly what the first calibration run did: 12/15 grasps but only 5/15 placements."""
    bx, by = ball_xy
    gx, gy = bin_xy
    ball = np.array([bx, by, BALL_REST_Z])
    over_bin = np.array([gx, gy, L.TABLE_TOP_Z])
    hover = ball + [0, 0, 0.10]
    backoff = hover + [-RETREAT_BACKOFF, 0, RETREAT_BACKOFF_LIFT]
    return [
        (hover, OPEN, 1.0),                  # 1  hover above the ball, gripper open
        (backoff, OPEN, RETREAT_BACKOFF_SECS),  # 1a withdraw toward the base -- see
        (hover, OPEN, RETREAT_BACKOFF_SECS),    # 1b RETREAT_BACKOFF: the -x labels
        (hover, OPEN, 0.2),                  # 1c dwell: let the re-approach settle
        (ball, OPEN, 0.8),                   # 2  descend onto it
        (ball, OPEN, 0.3),                   # 3  dwell: let the servo actually arrive
        (ball, CLOSED, 0.6),                 # 4  close, arm holding station
        (ball, CLOSED, 0.4),                 # 5  dwell: let the grasp seat before loading it
        (ball + [0, 0, 0.15], CLOSED, 1.0),  # 6  lift clear of the table
        (over_bin + [0, 0, 0.20], CLOSED, 2.0),  # 7  transport
        (over_bin + [0, 0, 0.20], CLOSED, 0.4),  # 8  dwell: kill the lateral overshoot
        (over_bin + [0, 0, 0.07], CLOSED, 0.8),  # 9  lower to just above the rim
        (over_bin + [0, 0, 0.07], CLOSED, 0.3),  # 10 dwell before letting go
        (over_bin + [0, 0, 0.07], OPEN, 0.5),    # 11 release
        (over_bin + [0, 0, 0.20], OPEN, 0.6),    # 12 rise clear of the bin
        (np.asarray(start_pos, dtype=float), OPEN, 1.0),  # 13 return to the start pose
    ]


def reference_track(model, data, site_id, ball_xy, bin_xy, control_hz):
    """Expand the waypoints into one (target_pos, gripper) per control tick.

    Seeded from the arm's ACTUAL pose, so the first segment starts wherever the (possibly
    jittered) reset left it.

    Interpolation is a SMOOTHSTEP (3a^2 - 2a^3), not linear. Linear segments meet at a
    velocity discontinuity: the reference jumps from full speed to zero (or to a new
    direction) in one tick, the arm follows with a jerk, and the ball is shaken out of a
    friction grasp. That is what the first two calibration runs were doing -- the ball was
    lifted to exactly the commanded height every time and then dropped 0.10-0.12 m short of
    the bin, mid-transport. Smoothstep has zero derivative at both ends of every segment,
    so the reference accelerates and decelerates instead of stepping."""
    dt = 1.0 / control_hz
    track = []
    prev_pos = data.site_xpos[site_id].copy()
    prev_grip = OPEN
    for target, grip, dur in waypoints(ball_xy, bin_xy, prev_pos):
        n = max(1, int(round(dur / dt)))
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


def run_episode(model, data, scratch, renderer, site_id, finger_qposadr, ball_body,
                rng, cohort, control_hz, decimation, ball_xy, bin_xy):
    """Execute one demonstration. Returns (frames, states, actions, info)."""
    arm_jitter = RECOVER_START_JITTER if cohort == "recover" else 0.0
    reset_episode(model, data, rng, ball_xy, arm_jitter=arm_jitter)

    # Hold the reset orientation throughout: measured against a live LIBERO env, our
    # grip_site at LIBERO's reset joints has axis-angle (3.140, 0, -0.089) against
    # LIBERO's own (3.141, 0.002, -0.090). Top-down, and already in distribution.
    target_mat = data.site_xmat[site_id].copy()

    track = reference_track(model, data, site_id, ball_xy, bin_xy, control_hz)

    # Recovery cohort: two shoves, one free-armed during the approach and one loaded
    # during the transport. See RECOVER_KICK_WINDOWS.
    kicks = {}
    if cohort == "recover":
        for (lo, hi), mag in zip(RECOVER_KICK_WINDOWS, (RECOVER_KICK, RECOVER_KICK_LOADED)):
            t = int(rng.integers(max(1, int(lo * len(track))), max(2, int(hi * len(track)))))
            kicks[t] = mag

    frames = {"image": [], "wrist_image": []}
    states, actions = [], []
    clamped_ticks = 0
    ball_adr = model.jnt_qposadr[model.body_jntadr[ball_body]]
    max_ball_z = -np.inf

    for tick, (target_pos, grip) in enumerate(track):
        # Observation BEFORE the action, matching how the policy is queried at inference.
        main_img, wrist_img = L.render_cameras(renderer, data, "none")
        frames["image"].append(main_img)
        frames["wrist_image"].append(wrist_img)
        states.append(L.read_state(model, data, site_id, finger_qposadr))

        action = expert_action(data, site_id, target_pos, target_mat, grip)
        actions.append(action.astype(np.float32))

        executed = action.copy()
        if cohort == "noise":
            executed[0:3] += rng.normal(0.0, NOISE_SIGMA_POS, size=3)
            executed[3:6] += rng.normal(0.0, NOISE_SIGMA_ROT, size=3)
        elif tick in kicks:
            direction = rng.normal(size=3)
            executed[0:3] += kicks[tick] * direction / np.linalg.norm(direction)
        executed = np.clip(executed, -1.0, 1.0)

        _, _, was_clamped = L.apply_action(model, data, scratch, executed, site_id)
        clamped_ticks += int(was_clamped)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        max_ball_z = max(max_ball_z, float(data.qpos[ball_adr + 2]))

    ball = data.qpos[ball_adr:ball_adr + 3].copy()
    lifted = max_ball_z > BALL_REST_Z + 0.05
    placed = (abs(ball[0] - bin_xy[0]) < 0.05 and abs(ball[1] - bin_xy[1]) < 0.05
              and ball[2] < L.TABLE_TOP_Z + 0.06)
    info = {
        "lifted": bool(lifted),
        "placed": bool(placed),
        "ball_final": ball.tolist(),
        "ball_max_z": max_ball_z,
        "table_clamped_ticks": clamped_ticks,
        "ticks": len(track),
        "kick_ticks": sorted(int(t) for t in kicks),
        "ball_xy": [float(ball_xy[0]), float(ball_xy[1])],
        "bin_xy": [float(bin_xy[0]), float(bin_xy[1])],
    }
    return frames, np.array(states), np.array(actions), info


def selftest(model, data, site_id):
    """Print the frame checks that the 2026-07-28 scene corrections were made against.
    Cheap regression guard: if someone re-edits the scene, these move."""
    # Go through reset_episode rather than setting qpos[:7] by hand, so this guard reports
    # the state an episode actually starts from (objects placed, arm settled) rather than a
    # near-identical but not identical hand-rolled one. Measured: both paths converge by 50
    # steps and agree to <0.1 mm here, so this is about the guard being the real thing, not
    # about a bug it was hiding.
    #
    # What this number is NOT: `0.2728` appears in libero/README.md and fine_tune/README.md
    # as "ours" against LIBERO's 0.2733, i.e. a claimed 0.5 mm match. 0.2728 is the PURE
    # FORWARD KINEMATICS of LIBERO_INIT_QPOS with no dynamics. The settled pose the arm
    # actually holds is lower, because the position servo sags under gravity: 0.2680 on
    # menagerie's stock gains (4.84 mm of sag, so a 5.3 mm error against LIBERO, not
    # 0.5 mm), and 0.2704 on the kp x2 / kd x0.7 gains this file now runs (2.44 mm).
    reset_episode(model, data, np.random.default_rng(0), (0.56, 0.0))
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="libero/fine_tune/a1")
    p.add_argument("--reach", type=int, default=20)
    p.add_argument("--noise", type=int, default=20)
    p.add_argument("--recover", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-size", type=int, default=L.RENDER_HEIGHT)
    p.add_argument("--control-hz", type=float, default=L.CONTROL_HZ)
    p.add_argument("--model-path", default=L.DEFAULT_MODEL_PATH)
    p.add_argument("--max-attempts-per-episode", type=int, default=8,
                   help="rejection sampling budget; an episode that never lifts the ball "
                        "is resampled rather than kept")
    p.add_argument("--keep-unplaced", action="store_true",
                   help="accept episodes that lift the ball but miss the bin. Off by "
                        "default: a demonstration that fails the task is a demonstration "
                        "of failing the task")
    p.add_argument("--verbose", action="store_true",
                   help="print why each rejected attempt was rejected")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    finger_qposadr = _finger_adr(model)
    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_ball")
    bin_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in BIN_NAMES}

    if args.selftest:
        selftest(model, data, site_id)
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
    print(f"control     : {args.control_hz:g} Hz  ({decimation} physics steps/action)")
    print(f"action space: 7-D delta eef pose, [-1,1], {L.DELTA_POS_SCALE} m / "
          f"{L.DELTA_ROT_SCALE} rad per unit")
    print(f"table floor : eef target clamped to {L.EEF_MIN_Z:+.4f} "
          f"({L.DEFAULT_MIN_CLEARANCE * 1000:.0f} mm above the top)")
    print(f"out         : {args.out}\n")

    plan = ([("reach", i) for i in range(args.reach)]
            + [("noise", i) for i in range(args.noise)]
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
            ball_xy = (rng.uniform(*L.BALL_SAMPLE_X), rng.uniform(*L.BALL_SAMPLE_Y))
            bin_xy = bin_layout(model, bin_ids, rng)
            frames, states, actions, info = run_episode(
                model, data, scratch, renderer, site_id, finger_qposadr, ball_body,
                rng, cohort, args.control_hz, decimation, ball_xy, bin_xy)
            if info["lifted"] and (info["placed"] or args.keep_unplaced):
                kept = (frames, states, actions, info, attempt + 1)
                break
            rejected += 1
            if args.verbose:
                print(f"    reject {cohort}_{idx:02d} try {attempt + 1}: "
                      f"lifted={info['lifted']} placed={info['placed']} "
                      f"ball_max_z={info['ball_max_z']:+.4f} "
                      f"ball_final={np.round(info['ball_final'], 3)} "
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
              f"attempts={attempts}")

    summary = writer.finalize()
    if missing:
        print(f"\nWARNING: {len(missing)} slots left empty: {', '.join(missing)}")
    renderer.close()
    print(f"\n{summary['episodes']} episodes, {summary['frames']} frames, "
          f"{rejected} rejected attempts, {time.time() - t0:.0f}s -> {args.out}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
