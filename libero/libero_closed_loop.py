"""Closed-loop MolmoAct2-LIBERO control of the MuJoCo Panda pick-and-place scene.

Self-contained sibling of the project root's `phase3_closed_loop.py`. Deliberately a
copy rather than an import (see libero/README.md): the two speak different action
spaces, and keeping them separate means experimenting here can't break the DROID path.

The LIBERO checkpoint is pretrained on robosuite/MuJoCo renders of a Franka Panda, which
is what we actually have -- unlike MolmoAct2-DROID, pretrained on real-camera footage of
an FR3. That removes both the photoreal-vs-flat-shaded gap and the link-length mismatch.
What it costs is the action space: LIBERO is 7-D *delta end-effector pose* in [-1, 1],
not 8-D absolute joint positions.

Every outer iteration:
  1. Render both cameras (LIBERO names them `image` = agentview, `wrist_image` =
     eye-in-hand) at 256x256.
  2. Read proprioception as LIBERO's own 8-D state:
     [eef_pos(3), eef_axisangle(3), gripper_qpos(2)].
  3. POST {images, instruction, state} to /act -> action chunk, shape (N, 7).
  4. For each action: convert the normalized delta to a Cartesian target pose, solve IK
     for it (Route A -- see below), write joint targets to data.ctrl, and step physics
     `decimation` times to advance one 20 Hz control tick.

ROUTE A (differential IK), and why not real OSC
-----------------------------------------------
LIBERO drives its Panda through an operational-space controller that turns a Cartesian
delta into joint torques. Reimplementing that here would mean swapping the scene's
position actuators for torque actuators and re-tuning gains from scratch -- discarding
the characterization behind panda.xml's kp=4500 / kd=450 / forcerange +/-87. Instead we
keep the position actuators and, per tick, re-solve IK for the requested Cartesian
target and command the resulting joint pose. Same damped-least-squares solver as
phase4_collect_demos.solve_ik.

This is an approximation of OSC, not OSC. It tracks the commanded *pose* rather than
reproducing OSC's compliance, so contact-rich behavior will differ. Good enough to find
out whether the checkpoint understands the scene at all, which is the current question.

Usage:
    uv run python libero/libero_closed_loop.py --dry-run --server-url <url>/act
    uv run python libero/libero_closed_loop.py --chunks 20 --server-url <url>/act
    uv run python libero/libero_closed_loop.py --chunks 20 --image-flip 180 --server-url <url>/act
"""

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import json_numpy
import mujoco
import mujoco.viewer
import numpy as np
import requests
from PIL import Image

json_numpy.patch()

# LIBERO-framed scene: identical objects to scene_pick_place.xml, but external_cam is
# re-framed to match LIBERO's `agentview` (0.707 m standoff, fovy 45, rigid mount). It has
# to live next to scene.xml rather than in libero/ -- panda.xml's meshdir="assets" is
# resolved against the TOP-LEVEL file's directory, so a copy here would break every mesh
# load. See that file's header. scene_pick_place.xml is left alone so the DROID loop and
# the already-collected dataset are unaffected.
#
# This matters more than it sounds. At scene_pick_place.xml's framing a 40 mm ball is
# 4.9 px at 256x256 (7.3 px at 378) -- below one ViT patch, i.e. not localizable at all.
# Matched to LIBERO it is 16.0 px (analytic and measured agree exactly). That 3.3x is the
# single largest divergence from the pretraining distribution we found.
#
# scene_libero_hand.xml, not scene_libero.xml: same scene, but with the STOCK FRANKA HAND
# instead of the Robotiq 2F-85. LIBERO was pretrained on the stock hand, and the 2F-85
# differed from it in four things that all sit inside the model's own observation/action
# space -- reported gripper_qpos units, the eef-site convention, pad friction, and which
# end of the ctrl range means "open". See panda_libero_hand.xml's header.
DEFAULT_MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene_libero_hand.xml"
SERVER_URL = "http://localhost:8000/act"
INSTRUCTION = "pick up the green ball and put it in the green container"

# Our scene's camera names, mapped onto LIBERO's two. Both are now robosuite's own
# cameras copied verbatim rather than anything we designed: `eye_in_hand` replaces the
# 22 cm side-mount at fovy 56.7 that earlier runs used, which matched nothing.
MAIN_CAMERA = "external_cam"        # <- LIBERO's `agentview` / `observation.images.image`
WRIST_CAMERA = "eye_in_hand"        # <- LIBERO's `robot0_eye_in_hand` / `.wrist_image`

# LIBERO's own render size (lerobot/envs/libero.py:109-110). Matching it avoids an extra
# resample (MolmoAct2's processor resizes to 378 internally either way) and matches the
# effective detail level of the training frames. Overridable with --render-size: at the
# matched camera 256 already puts the ball at 16 px, but 378 gives 23.7 px if you want
# the headroom, at the cost of a larger payload and a mismatch with LIBERO's native size.
RENDER_HEIGHT = 256
RENDER_WIDTH = 256

# --- LIBERO action space -------------------------------------------------
# 7-D, [-1, 1]: [dx, dy, dz, drx, dry, drz, gripper]  (lerobot/envs/libero.py:82-87).
# robosuite's OSC_POSE maps that onto a physical delta via
# output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]. So a full-scale action is 5 cm of
# translation or 0.5 rad of rotation -- at 20 Hz that is ~1 m/s, i.e. a large move, not
# a nudge. Getting these two numbers wrong scales every motion the model asks for.
DELTA_POS_SCALE = 0.05   # metres per unit action
DELTA_ROT_SCALE = 0.5    # radians per unit action (axis-angle, world frame)

# Gripper action: the no-op action is [0,0,0,0,0,0,-1], so -1 = open, +1 = closed.
GRIPPER_ACTION_OPEN = -1.0
GRIPPER_ACTION_CLOSED = 1.0

# robosuite's default control frequency. NOTE this differs from DROID's 15 Hz -- with
# the scene's 0.002 s timestep that makes decimation 25, not 33.
CONTROL_HZ = 20.0

# Informational: LIBERO's action_horizon (data_mixtures.py:333). The server tells us the
# real chunk length; this is only used to sanity-check what comes back.
EXPECTED_ACTION_HORIZON = 10

ARM = slice(0, 7)

# --- Gripper -------------------------------------------------------------
# RESOLVED: the scene now mounts the stock Franka hand, so `robot0_gripper_qpos` is no
# longer approximated -- it is read straight off the same two prismatic finger joints
# LIBERO reports, in the same metres. The old Robotiq 2F-85 knuckle-radians -> metres
# linear stand-in is gone.
#
# Two sign/direction traps survive the swap, both verified against the compiled model:
#
#  1. SENSE. `actuator8` drives the finger pair through the "split" tendon with
#     ctrlrange 0-255, and the home keyframe pairs ctrl=255 with qpos=0.04 0.04. So for
#     the stock hand 255 = OPEN, 0 = CLOSED -- the exact OPPOSITE of the 2F-85's
#     fingers_actuator, where 0 was open. Sending the old polarity would have commanded
#     a close every time the model asked for an open.
#
#  2. REPORTED SIGN. robosuite mirrors its two fingers with opposite joint RANGES
#     (panda_gripper.xml: finger_joint1 in [0, 0.04], finger_joint2 in [-0.04, 0]), so
#     `robot0_gripper_qpos` is [+x, -x]. Menagerie mirrors the same hardware with a body
#     quaternion instead and keeps both joints in [0, 0.04], giving [+x, +x]. The second
#     element therefore has to be negated on the way out. Every LIBERO run before this
#     change reported [+x, +x] -- one of the two gripper state channels was wrong across
#     its whole range, in the channel that decides grasping.
GRIPPER_CTRL_MAX = 255.0           # actuator8 ctrl range; 255 = OPEN for the stock hand

# --- LIBERO world-frame alignment ---------------------------------------
# The model conditions on ABSOLUTE eef_pos, so the numbers we report have to live in
# LIBERO's coordinate frame, not ours. Read out of robosuite 1.5.2:
#     link0 world pos = (-0.56, 0, 0.912)   (Panda on a 0.912 m RethinkMount pedestal)
#     table top       = z = 0.800           (TableArena table_offset)
# i.e. LIBERO's robot base sits 0.112 m ABOVE the table and works downward onto it.
#
# scene_libero.xml cannot move link0 (panda.xml pins it at the origin, and editing that
# would change the DROID path and invalidate the collected dataset), so it puts the world
# origin AT the robot base and lowers everything else by 0.912 instead. Relative geometry
# -- all the physics and the arm's posture depend on -- is identical; absolute coordinates
# differ by a constant (+0.56, 0, -0.912). Adding this offset back is a change of origin,
# nothing more.
#
# CORRECTED 2026-07-28 against a live LIBERO env (x was -0.56, from reading robosuite's
# source rather than measuring it; link0 is actually at x = -0.6). Verified after the
# matching scene fixes: at LIBERO's reset joints our grip_site is (0.4515, 0, 0.2608) ->
# (-0.1485, 0, 1.1728) in LIBERO's frame, against LIBERO's own (-0.1485, 0, 1.1733).
#
# WHICH LIBERO FRAME: there isn't one. Measured link0 z is 0.912 in libero_spatial and
# libero_goal, 0.0 in libero_object, 0.42 in libero_10 -- the suites do not share a world
# origin, which is why the released dataset's eef z spans 0.008..1.366 (q01 0.519, q50
# 0.719, q99 1.043). What IS invariant across all four suites is the pose relative to the
# robot: eef at reset is (0.4515, 0, 0.2613) from link0 in every one of them, and the
# table top is link0 - 0.012. So the model cannot be keying on absolute z, and this offset
# only has to land somewhere sane in that spread. It reproduces the spatial/goal frame.
#
# This applies ONLY to the state. Actions are deltas, so a constant origin shift leaves
# them unchanged -- do not offset them.
LIBERO_ORIGIN_OFFSET = np.array([-0.6, 0.0, 0.912])

# LIBERO's OWN reset joint positions, read out of a live env (robot0_joint_pos right
# after env.reset(), identical in all four suites). This replaces robosuite's generic
# Panda.init_qpos, which is what used to be here and is NOT what LIBERO resets to: it put
# the eef 0.201 m above the table where LIBERO's puts it 0.273 m up, a 72 mm head start on
# being out of distribution before the first action.
LIBERO_INIT_QPOS = np.array([0.0, -0.16103739, 0.0, -2.44459747, 0.0, 2.2267522, 0.78539816])

# Table top in THIS scene's frame (scene_libero*.xml: body at z=-0.462, half-height 0.45).
# Was -0.112 until 2026-07-28. A live LIBERO env puts its table top 0.012 m below link0,
# not 0.112 m: our work surface was 100 mm too low, so every descent the policy commanded
# stopped short of the height it expected and it kept pressing. See the scene file header.
TABLE_TOP_Z = -0.012

# --- Table floor on the commanded eef ------------------------------------
# Route A solves IK for whatever target it is handed and writes joint positions, so it
# will happily command a pose below the table surface; MuJoCo then resists with contact
# rather than the arm settling compliantly the way robosuite's force-based OSC would.
# Measured on the stock hand: 2.9 mm of real penetration by the finger MESH (not the
# pads). panda_libero_hand.xml stiffens that contact; this clamps the target so the
# situation does not arise in the first place. Both, because "not at all" needs both.
#
# 16 mm is measured, not guessed, and was re-measured on 2026-07-28 after grip_site moved
# from 0.1065 to robosuite's 0.097: the site now rides 15.5 mm above the lowest point of
# the hand (it used to be 6.1-12.1 mm, and the whole 9.5 mm difference is that move). A
# 16 mm floor therefore leaves the hardware ~0.5 mm clear of the surface at the limit.
#
# It still does not cost grasp height. The ball centre is 20 mm above the table, so a
# 16 mm floor on the site permits descending 4 mm PAST the ball's equator -- grip_site is
# now robosuite's own grasp point, so aiming it straight at the object centre, which is
# what LIBERO's demonstrations do, is inside the allowed range.
#
# It is still a departure from LIBERO's control law -- the model asks for a pose and we
# decline part of it. Pass a NEGATIVE --min-clearance to switch it off entirely.
#
# Note that 0 does NOT switch it off: it puts the floor at the table top, which still
# clamps, and still lets the hand through -- grip_site is held at the surface while the
# hardware hangs 6-12 mm BELOW it. That mistake made the control arm of the first A/B
# sweep useless. "Off" has to be its own value, not a boundary case of the number.
DEFAULT_MIN_CLEARANCE = 0.016

# Set from --min-clearance in main(); apply_action reads it. -inf = clamp disabled.
EEF_MIN_Z = TABLE_TOP_Z + DEFAULT_MIN_CLEARANCE


def make_renderer(model, size=None):
    h = w = size if size else RENDER_HEIGHT
    return mujoco.Renderer(model, height=h, width=w)


def apply_image_flip(img, mode):
    """LIBERO's processor rotates raw frames 180 degrees (env_processor.py:58-59),
    commenting that it matches "the HuggingFaceVLA/libero camera orientation
    convention". robosuite renders bottom-up (OpenGL), so the TRAINING frames are
    raw-robosuite-rotated-180. mujoco.Renderer already returns top-down frames, so
    whether we should match by flipping or by leaving alone is genuinely ambiguous --
    hence a flag rather than a hardcoded choice. Try all three before concluding
    anything about the checkpoint; a 180-degree error looks exactly like a model that
    cannot see."""
    if mode == "none":
        return img
    if mode == "180":
        return img[::-1, ::-1]
    if mode == "vertical":
        return img[::-1]
    raise ValueError(f"unknown image flip mode: {mode}")


def render_cameras(renderer, data, flip="none"):
    """LIBERO's two cameras: `image` (agentview) and `wrist_image` (eye-in-hand), which
    in our scene are MAIN_CAMERA / WRIST_CAMERA."""
    renderer.update_scene(data, camera=MAIN_CAMERA)
    main = apply_image_flip(renderer.render().copy(), flip)
    renderer.update_scene(data, camera=WRIST_CAMERA)
    wrist = apply_image_flip(renderer.render().copy(), flip)
    return main, wrist


def _quat_wxyz_to_xyzw(q):
    """MuJoCo stores quaternions (w,x,y,z); robosuite/LIBERO use (x,y,z,w)
    (env_processor.py:120). Silent 90-degree-ish state corruption if skipped."""
    return np.array([q[1], q[2], q[3], q[0]])


def quat2axisangle(quat_xyzw):
    """(x,y,z,w) quaternion -> axis*angle 3-vector. Mirrors
    LiberoProcessorStep._quat2axisangle (env_processor.py:114-150) including its
    degenerate-case handling: when w -> +/-1 the axis is undefined and the rotation is
    ~zero, so return zeros rather than dividing by ~0."""
    w = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    den = np.sqrt(max(0.0, 1.0 - w * w))
    if den < 1e-10:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    axis = np.asarray(quat_xyzw[:3], dtype=np.float64) / den
    return axis * angle


def read_state(model, data, site_id, finger_qposadr):
    """LIBERO's 8-D state: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
    (env_processor.py:68-75).

    `site_id` is `grip_site`, which reproduces robosuite's `robot0_eef` in both position
    and frame -- see panda_libero_hand.xml. `finger_qposadr` is the (finger_joint1,
    finger_joint2) qpos addresses.

    eef_pos is shifted into LIBERO's world frame (see LIBERO_ORIGIN_OFFSET) because the
    model conditions on absolute position. Without it every z we report is ~0.8 m below
    anything in the training distribution, and the observed response was the model
    commanding descent until the arm bottomed out through the floor."""
    eef_pos = data.site_xpos[site_id].copy() + LIBERO_ORIGIN_OFFSET

    quat_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(quat_wxyz, data.site_xmat[site_id])
    eef_axisangle = quat2axisangle(_quat_wxyz_to_xyzw(quat_wxyz))

    # The stock hand's own joints, in LIBERO's own metres -- no conversion. The second
    # element is negated to match robosuite's mirrored joint range; see trap 2 above.
    q1 = float(data.qpos[finger_qposadr[0]])
    q2 = float(data.qpos[finger_qposadr[1]])
    gripper_qpos = np.array([q1, -q2])

    return np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float32)


def _orientation_error(target_mat, current_mat):
    """Rotation vector (axis*angle, world frame) taking current -> target. Both inputs
    are flat length-9 rotation matrices (data.site_xmat form). Copied verbatim from
    phase4_collect_demos.py so both paths solve identical IK."""
    cur_q = np.zeros(4)
    tgt_q = np.zeros(4)
    mujoco.mju_mat2Quat(cur_q, current_mat)
    mujoco.mju_mat2Quat(tgt_q, target_mat)
    neg_cur = np.zeros(4)
    mujoco.mju_negQuat(neg_cur, cur_q)
    err_q = np.zeros(4)
    mujoco.mju_mulQuat(err_q, tgt_q, neg_cur)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, err_q, 1.0)
    return vel


def solve_ik(model, scratch, site_id, target_pos, target_mat, q_seed,
             iters=200, tol=1e-4, damping=0.15, step_clip=0.3):
    """Damped-least-squares IK for (target_pos, target_mat) at `site_id`, run on a
    throwaway MjData so the live sim is never disturbed. Returns (q[7], reached).
    Copied verbatim from phase4_collect_demos.solve_ik."""
    lower = model.jnt_range[:7, 0]
    upper = model.jnt_range[:7, 1]
    scratch.qpos[:7] = np.clip(q_seed, lower, upper)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    reached = False
    for _ in range(iters):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        err = np.concatenate([
            target_pos - scratch.site_xpos[site_id],
            _orientation_error(target_mat, scratch.site_xmat[site_id]),
        ])
        if np.linalg.norm(err) < tol:
            reached = True
            break
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        dq = np.clip(dq, -step_clip, step_clip)
        scratch.qpos[:7] = np.clip(scratch.qpos[:7] + dq, lower, upper)
    return scratch.qpos[:7].copy(), reached


def axisangle_to_mat(rotvec):
    """axis*angle 3-vector -> flat length-9 rotation matrix, via MuJoCo so the
    conventions match everything else here."""
    angle = float(np.linalg.norm(rotvec))
    quat = np.zeros(4)
    if angle < 1e-12:
        quat[:] = [1.0, 0.0, 0.0, 0.0]
    else:
        mujoco.mju_axisAngle2Quat(quat, np.asarray(rotvec) / angle, angle)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat


def apply_action(model, data, scratch, action, site_id):
    """ROUTE A: normalized 7-D delta EE pose -> absolute joint targets in data.ctrl.

    action is (7,): [dx, dy, dz, drx, dry, drz, gripper], each in [-1, 1].

    The delta is taken relative to the arm's CURRENT pose, which is what makes this a
    delta controller: the same action applied twice moves twice as far. That is the
    property that makes dropping stale actions unsafe here (unlike the DROID loop's
    absolute joint targets) -- see --replan-at.

    Returns (ik_reached, delta_pos_norm, clamped) for logging: a persistently unreached
    IK means the model is asking for poses this arm can't hold, which is a different
    failure from it asking for the wrong ones; `clamped` counts ticks where the table
    floor had to cut the requested descent."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)

    dpos = a[0:3] * DELTA_POS_SCALE
    drot = a[3:6] * DELTA_ROT_SCALE

    cur_pos = data.site_xpos[site_id].copy()
    cur_mat = data.site_xmat[site_id].copy()

    target_pos = cur_pos + dpos

    # Table floor: never ask IK for a pose that would put the hand through the surface.
    # Clamped on the TARGET, before IK, so the solver never sees an impossible request --
    # clamping the solution afterwards would leave the orientation solved for a pose the
    # position is no longer at. See EEF_MIN_Z.
    clamped = target_pos[2] < EEF_MIN_Z
    if clamped:
        target_pos[2] = EEF_MIN_Z

    # Compose the rotation delta in the WORLD frame: target = R(drot) @ current, matching
    # the frame _orientation_error works in. (robosuite's OSC also treats the rotation
    # delta as a world-frame axis-angle by default.)
    delta_mat = axisangle_to_mat(drot).reshape(3, 3)
    target_mat = (delta_mat @ cur_mat.reshape(3, 3)).reshape(9)

    q_target, reached = solve_ik(
        model, scratch, site_id, target_pos, target_mat, data.qpos[:7].copy()
    )
    data.ctrl[ARM] = q_target

    # Gripper: action [-1, 1] with -1 = open. actuator8 is 255 = OPEN, 0 = CLOSED for the
    # stock Franka hand (the 2F-85 ran the other way round -- see trap 1 above), so the
    # fraction is inverted on the way to ctrl.
    closed_frac = (a[6] - GRIPPER_ACTION_OPEN) / (GRIPPER_ACTION_CLOSED - GRIPPER_ACTION_OPEN)
    data.ctrl[7] = (1.0 - float(np.clip(closed_frac, 0.0, 1.0))) * GRIPPER_CTRL_MAX

    return reached, float(np.linalg.norm(dpos)), clamped


# Which payload keys carry the two camera frames. This is NOT cosmetic -- it decides
# which server can read the request at all:
#
#   "droid"  -> host_server_droid.py (libero_modal.py, serving the released checkpoint)
#              reads payload["external_cam"] / payload["wrist_cam"] by name and forwards
#              them positionally as images=[first, second]. Nothing DROID-specific
#              survives past the key names.
#   "libero" -> scripts/serve_policy.py (libero_modal_finetuned.py, serving our
#              fine-tune) looks up IMAGE_KEY_PRESETS["libero"] = ["image", "wrist_image"]
#              and turns the key it found into the feature name
#              `observation.images.<key>` fed to the model. Send external_cam to that
#              server and it finds no images at all.
#
# Both servers ignore keys they do not know, so sending all four would work -- at the
# cost of doubling a payload that is already ~4x the inference cost (PROGRESS sec.2).
PAYLOAD_KEY_SETS = {
    "droid": ("external_cam", "wrist_cam"),
    "libero": ("image", "wrist_image"),
}


def query_server(main_img, wrist_img, state, server_url, send_wrist=True, timeout=30,
                 payload_keys="droid"):
    """POST one observation and return (actions, server_dt_ms, request_bytes)."""
    main_key, wrist_key = PAYLOAD_KEY_SETS[payload_keys]
    payload = {
        main_key: main_img,
        wrist_key: wrist_img if send_wrist else np.zeros_like(wrist_img),
        "instruction": INSTRUCTION,
        "state": state,
    }
    body_str = json_numpy.dumps(payload)
    resp = requests.post(
        server_url,
        data=body_str,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    actions = np.asarray(body["actions"], dtype=np.float32)
    dt_ms = body.get("dt_ms", body.get("latency_ms"))
    if dt_ms is None:
        dt_ms = 1000 * resp.elapsed.total_seconds()
    return actions, float(dt_ms), len(body_str)


def capture_and_submit(executor, renderer, data, model, site_id, finger_qposadr,
                       tracked_body_id, server_url, send_wrist, timeout, flip, tick=0,
                       payload_keys="droid"):
    """Render + read state on the main thread (mujoco.Renderer's GL context is not
    thread-safe), hand the HTTP call to the worker. Returns
    (future, submit_time, obs_snapshot, tick)."""
    main_img, wrist_img = render_cameras(renderer, data, flip)
    state = read_state(model, data, site_id, finger_qposadr)
    obs_snapshot = {
        "eef_pos": state[:3].tolist(),
        "eef_axisangle": state[3:6].tolist(),
        "gripper_qpos": state[6:8].tolist(),
        "arm_qpos": data.qpos[:7].tolist(),
        "tracked_object_xpos": (
            data.xpos[tracked_body_id].tolist() if tracked_body_id is not None else None
        ),
        "ctrl_before": data.ctrl.tolist(),
    }
    t0 = time.time()
    future = executor.submit(
        query_server, main_img, wrist_img, state, server_url,
        send_wrist=send_wrist, timeout=timeout, payload_keys=payload_keys,
    )
    return future, t0, obs_snapshot, tick


# Freejoint objects now rest ON THE TABLE (top at TABLE_TOP_Z), not on the floor. Sphere
# and box centres sit one half-extent above the surface. Same fix as phase3/phase4 is still
# needed: the "home" keyframe predates these freejoints, so mj_resetDataKeyframe zero-pads
# their qpos to the world origin instead of their intended resting pose.
FREE_BODY_INITIAL_POSES = {
    "green_ball": (0.56, 0.0, TABLE_TOP_Z + 0.02),
    "red_box": (0.56, -0.28, TABLE_TOP_Z + 0.02),
}
# NOTE both follow TABLE_TOP_Z, so raising the table on 2026-07-28 carried them with it.
# Ball sampling box, kept on the table surface and around its centre (0.56, 0) so the
# ball stays inside the 0.8x0.8 m top and within comfortable reach. Deliberately tighter
# than phase4_collect_demos.py's floor-scene box, which was anchored to a different frame.
BALL_SAMPLE_X = (0.46, 0.66)
BALL_SAMPLE_Y = (-0.12, 0.12)
TRACKED_OBJECT_CANDIDATES = ("green_ball", "red_box")


def build_sim(model_path, randomize_ball=False, rng=None):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home != -1:
        mujoco.mj_resetDataKeyframe(model, data, home)

    # Start from LIBERO's own reset posture rather than the scene's home keyframe. Both
    # qpos AND ctrl must be set: ctrl is the position actuators' target, so leaving it at
    # the keyframe's value would just drag the arm back to the old pose on the first step.
    data.qpos[:7] = LIBERO_INIT_QPOS
    data.ctrl[ARM] = LIBERO_INIT_QPOS
    data.ctrl[7] = GRIPPER_CTRL_MAX  # gripper open (stock hand: 255 = open)

    for name, pos in FREE_BODY_INITIAL_POSES.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            continue
        jid = model.body_jntadr[bid]
        if jid == -1:
            continue
        adr = model.jnt_qposadr[jid]
        xyz = np.array(pos, dtype=float)
        if name == "green_ball" and randomize_ball and rng is not None:
            xyz[0] = rng.uniform(*BALL_SAMPLE_X)
            xyz[1] = rng.uniform(*BALL_SAMPLE_Y)
        data.qpos[adr:adr + 3] = xyz
        data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)
    # Let contacts settle so objects rest on the table rather than being reported
    # mid-drop in the first observation the model sees.
    for _ in range(200):
        mujoco.mj_step(model, data)
    return model, data


def make_key_callback(model, viewer_ref):
    """Let the passive viewer be switched onto the two cameras the model is actually
    being fed, so you can see its input rather than only a free orbit view.

      1 = MAIN_CAMERA (agentview)   2 = WRIST_CAMERA (eye-in-hand)   0 = free/orbit

    This was missing in the first version of this file, which is why the camera views
    were not inspectable at all -- the viewer only ever showed the free camera."""
    main_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, MAIN_CAMERA)
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)

    def key_callback(keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            return
        viewer = viewer_ref.get("viewer")
        if viewer is None:
            return
        if ch == "1" and main_id != -1:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = main_id
            print(f"  viewer -> {MAIN_CAMERA}")
        elif ch == "2" and wrist_id != -1:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = wrist_id
            print(f"  viewer -> {WRIST_CAMERA}")
        elif ch == "0":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            print("  viewer -> free/orbit")

    return key_callback


class _NullViewer:
    """Stand-in for mujoco.viewer's context manager when --no-view is passed."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_running(self):
        return True

    def sync(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default=SERVER_URL,
                        help=f"/act endpoint (default: {SERVER_URL})")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--chunks", type=int, default=10,
                        help="action chunks to run; <=0 runs until killed")
    parser.add_argument("--dry-run", action="store_true",
                        help="round-trip the server and log it, but never touch the sim")
    parser.add_argument("--no-view", action="store_true", help="headless")
    parser.add_argument("--no-wrist-to-model", action="store_true",
                        help="send a blank wrist frame (schema still requires the key)")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--randomize-ball", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--payload-keys", choices=tuple(PAYLOAD_KEY_SETS), default="droid",
        help=(
            "which keys carry the camera frames in the /act request. `droid` "
            "(external_cam/wrist_cam) for a host_server_droid.py deployment, i.e. "
            "libero_modal.py serving the released checkpoint. `libero` "
            "(image/wrist_image) for a scripts/serve_policy.py deployment, i.e. "
            "libero_modal_finetuned.py serving our fine-tune -- that server derives the "
            "model's feature names from these keys and sees NO images if they are wrong"
        ),
    )
    parser.add_argument(
        "--image-flip", choices=("none", "180", "vertical"), default="none",
        help=(
            "orientation correction applied to both frames before sending. RESOLVED "
            "2026-07-28: leave it at `none`. LIBERO's processor rotates raw robosuite "
            "frames 180 degrees, which is what made this look ambiguous, but decoding "
            "the released dataset shows its STORED frames are already upright -- so "
            "upright is what the model consumes, and mujoco.Renderer already returns "
            "upright. The processor's 180 exists because robosuite renders bottom-up "
            "and we do not. Kept as a flag only because it is cheap to re-test"
        ),
    )
    parser.add_argument(
        "--render-size", type=int, default=RENDER_HEIGHT,
        help=(
            f"square render size for both cameras (default {RENDER_HEIGHT}, LIBERO's own). "
            "At the matched agentview a 40 mm ball is 16.0 px here and 23.7 px at 378; "
            "378 buys detail but diverges from LIBERO's native size and grows the payload"
        ),
    )
    parser.add_argument(
        "--min-clearance", type=float, default=DEFAULT_MIN_CLEARANCE,
        help=(
            f"metres the commanded eef target is kept above the table top "
            f"(default {DEFAULT_MIN_CLEARANCE}). Route A commands joint positions, so "
            "without this it will drive the hand through the surface and press rather "
            "than settle. 13 mm is measured: grip_site rides 6-12 mm above the lowest "
            "point of the hand. Pass a NEGATIVE value to disable and restore LIBERO's "
            "unclamped control law -- 0 does NOT disable it, it floors the target at the "
            "table top, which still clamps and still lets the hand through"
        ),
    )
    parser.add_argument(
        "--decimation", type=int, default=None,
        help=(
            f"physics steps per action. Default: derived from the scene timestep and "
            f"{CONTROL_HZ:g} Hz = 25 for a 0.002 s timestep. NOTE this differs from the "
            "DROID loop's 33, because LIBERO runs at 20 Hz not 15"
        ),
    )
    parser.add_argument(
        "--replan-at", type=int, default=0,
        help=(
            "actions to execute before firing the next /act request. 0 (default) = only "
            "after the whole chunk, i.e. sequential, sim frozen while waiting, zero "
            "staleness. UNLIKE the DROID loop, a non-zero value is NOT safe here: these "
            "are DELTA actions, so actions expired in flight cannot simply be dropped "
            "(dropping loses displacement) -- they would have to be summed. Left as a "
            "flag only so the sequential default is explicit; using it will warn"
        ),
    )
    parser.add_argument("--query-interval", type=float, default=0.0,
                        help="minimum seconds between queries; 0 disables throttling")
    args = parser.parse_args()

    send_wrist = not args.no_wrist_to_model
    rng = np.random.default_rng(args.seed) if args.randomize_ball else None
    model, data = build_sim(args.model_path, randomize_ball=args.randomize_ball, rng=rng)

    decimation = (
        args.decimation if args.decimation is not None
        else int(round((1.0 / model.opt.timestep) / CONTROL_HZ))
    )
    if decimation < 1:
        parser.error("--decimation must be >= 1")

    global EEF_MIN_Z
    # Negative = off. 0 is NOT off -- it floors the target at the table top, which still
    # clamps and still puts the hand through the surface. See EEF_MIN_Z.
    EEF_MIN_Z = (
        -np.inf if args.min_clearance < 0 else TABLE_TOP_Z + args.min_clearance
    )

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    if site_id == -1:
        parser.error("scene has no 'grip_site' site to use as the end-effector frame")
    finger_qposadr = []
    for jname in ("finger_joint1", "finger_joint2"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1:
            parser.error(f"scene has no '{jname}' -- is this the stock Franka hand?")
        finger_qposadr.append(model.jnt_qposadr[jid])
    scratch = mujoco.MjData(model)

    tracked_body_id = None
    for name in TRACKED_OBJECT_CANDIDATES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            tracked_body_id = bid
            break

    print(f"checkpoint convention: LIBERO (7-D delta EE pose, [-1,1], norm_tag=libero)")
    print(
        f"sim timestep {model.opt.timestep * 1000:.1f} ms -> {decimation} steps/action "
        f"({decimation * model.opt.timestep * 1000:.1f} ms, "
        f"{1.0 / (decimation * model.opt.timestep):.2f} Hz control)"
    )
    print(f"delta scale: {DELTA_POS_SCALE} m, {DELTA_ROT_SCALE} rad per unit action")
    print(
        "table floor: DISABLED (negative --min-clearance)"
        if args.min_clearance < 0
        else f"table floor: eef target clamped to z >= {EEF_MIN_Z:+.4f} "
             f"({args.min_clearance * 1000:.0f} mm above the table top)"
    )
    print(f"image flip: {args.image_flip}  (UNVERIFIED -- see --image-flip help)")
    if args.replan_at > 0:
        print(
            "  WARNING: --replan-at > 0 with DELTA actions. Expired actions are dropped, "
            "which LOSES displacement rather than being a no-op. Results are not "
            "trustworthy; use 0 unless you have implemented delta summing."
        )
    print("replan: after the full chunk (sequential)" if args.replan_at <= 0
          else f"replan: after {args.replan_at} action(s)")

    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    assets = Path(args.assets_dir)
    logs_dir = assets / "logs"
    images_dir = assets / "images" / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_id}.jsonl"
    print(f"run_id: {run_id}\n  log:    {log_path}\n  images: {images_dir}/")

    executor = None if args.dry_run else ThreadPoolExecutor(max_workers=1)
    pending = None
    last_query_time = None
    tick = 0
    frame_idx = 0

    # viewer_ref lets key_callback reach the viewer object, which doesn't exist yet at the
    # point launch_passive needs the callback.
    viewer_ref = {}
    viewer_cm = (
        _NullViewer() if args.no_view
        else mujoco.viewer.launch_passive(
            model, data, key_callback=make_key_callback(model, viewer_ref)
        )
    )

    with open(log_path, "w") as log_f, viewer_cm as viewer:
        viewer_ref["viewer"] = viewer
        if not args.no_view:
            print(f"  viewer camera: press 1={MAIN_CAMERA}, 2={WRIST_CAMERA}, 0=free/orbit")
        renderer = make_renderer(model, args.render_size)

        if executor is not None:
            pending = capture_and_submit(
                executor, renderer, data, model, site_id, finger_qposadr,
                tracked_body_id, args.server_url, send_wrist, args.request_timeout,
                args.image_flip, tick=tick, payload_keys=args.payload_keys,
            )

        chunk_iter = itertools.count() if args.chunks <= 0 else range(args.chunks)
        for chunk_i in chunk_iter:
            if not viewer.is_running():
                print("Viewer window closed, stopping.")
                break

            if args.dry_run:
                state = read_state(model, data, site_id, finger_qposadr)
                env_state_before = {
                    "eef_pos": state[:3].tolist(),
                    "eef_axisangle": state[3:6].tolist(),
                    "gripper_qpos": state[6:8].tolist(),
                    "arm_qpos": data.qpos[:7].tolist(),
                    "tracked_object_xpos": (
                        data.xpos[tracked_body_id].tolist()
                        if tracked_body_id is not None else None
                    ),
                    "ctrl_before": data.ctrl.tolist(),
                }
                main_img, wrist_img = render_cameras(renderer, data, args.image_flip)
                t0 = time.time()
                actions, dt_ms, request_bytes = query_server(
                    main_img, wrist_img, state, args.server_url,
                    send_wrist=send_wrist, timeout=args.request_timeout,
                    payload_keys=args.payload_keys,
                )
                round_trip_ms = 1000 * (time.time() - t0)
                snapshot_tick = tick
            else:
                future, t0, env_state_before, snapshot_tick = pending
                actions, dt_ms, request_bytes = future.result()
                round_trip_ms = 1000 * (time.time() - t0)

            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(
                    f"expected (N, 7) delta-EE actions from a LIBERO checkpoint, got "
                    f"{actions.shape}. A (N, 8) chunk means the server is still serving "
                    f"MolmoAct2-DROID -- check libero_modal.py is what you deployed."
                )
            if actions.shape[0] != EXPECTED_ACTION_HORIZON:
                print(
                    f"  note: chunk length {actions.shape[0]} != LIBERO's documented "
                    f"action_horizon {EXPECTED_ACTION_HORIZON}"
                )

            stale = 0 if args.dry_run else max(0, tick - snapshot_tick)
            if stale:
                # Only reachable with --replan-at > 0, already warned about at startup.
                if stale >= len(actions):
                    stale = len(actions) - 1
                actions = actions[stale:]

            print(
                f"chunk {chunk_i}: got actions{actions.shape} "
                f"(server dt={dt_ms:.1f}ms, round trip={round_trip_ms:.1f}ms"
                + (f", DROPPED {stale} expired -- displacement lost" if stale else "")
                + ")"
            )

            log_entry = {
                "chunk": chunk_i,
                "timestamp": time.time(),
                "convention": "libero_delta_eef",
                "env_state": env_state_before,
                "model_output": {
                    "actions": actions.tolist(),
                    "server_dt_ms": dt_ms,
                },
                "timing": {
                    "decimation": decimation,
                    "control_hz": 1.0 / (decimation * model.opt.timestep),
                    "replan_at": args.replan_at,
                    "staleness_ticks": stale,
                    "actions_executed": len(actions),
                },
                "network": {
                    "url": args.server_url,
                    "request_bytes": request_bytes,
                    "round_trip_ms": round_trip_ms,
                    "wrist_cam_sent_to_model": send_wrist,
                    "image_flip": args.image_flip,
                },
            }

            if args.dry_run:
                print("  value range:", actions.min(), actions.max())
                print("  first action:", actions[0])
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.flush()
                continue

            fire_after = (
                len(actions) if args.replan_at <= 0
                else min(args.replan_at, len(actions))
            )
            ik_failures = 0
            clamped_ticks = 0
            cmd_dpos = []

            for i, action in enumerate(actions):
                reached, dpos_norm, was_clamped = apply_action(
                    model, data, scratch, action, site_id
                )
                clamped_ticks += int(was_clamped)  # int(): numpy bool serializes as a dict
                if not reached:
                    ik_failures += 1
                cmd_dpos.append(dpos_norm)

                for _ in range(decimation):
                    mujoco.mj_step(model, data)
                    viewer.sync()

                main_frame, wrist_frame = render_cameras(renderer, data, args.image_flip)
                ts_ms = int(time.time() * 1000)
                Image.fromarray(main_frame).save(images_dir / f"camera_{ts_ms}_image.png")
                Image.fromarray(wrist_frame).save(images_dir / f"camera_{ts_ms}_wrist_image.png")
                frame_idx += 1
                tick += 1

                if (i + 1) == fire_after and (args.chunks <= 0 or chunk_i + 1 < args.chunks):
                    if args.query_interval > 0 and last_query_time is not None:
                        remaining = args.query_interval - (time.time() - last_query_time)
                        if remaining > 0:
                            time.sleep(remaining)
                    last_query_time = time.time()
                    pending = capture_and_submit(
                        executor, renderer, data, model, site_id, finger_qposadr,
                        tracked_body_id, args.server_url, send_wrist,
                        args.request_timeout, args.image_flip, tick=tick,
                        payload_keys=args.payload_keys,
                    )

            eef = data.site_xpos[site_id]
            print(
                f"  eef after chunk {chunk_i}: {np.array2string(eef, precision=3)}  "
                f"mean |dpos| cmd = {np.mean(cmd_dpos) * 1000:.1f} mm"
                + (f"  IK unreached on {ik_failures}/{len(actions)}" if ik_failures else "")
                + (f"  table-clamped {clamped_ticks}/{len(actions)}" if clamped_ticks else "")
            )

            log_entry["env_state"]["ctrl_after"] = data.ctrl.tolist()
            log_entry["env_state"]["eef_pos_after"] = eef.tolist()
            log_entry["env_state"]["tracked_object_xpos_after"] = (
                data.xpos[tracked_body_id].tolist() if tracked_body_id is not None else None
            )
            log_entry["timing"]["ik_unreached"] = ik_failures
            log_entry["timing"]["table_clamped"] = clamped_ticks
            log_entry["timing"]["mean_cmd_dpos_m"] = float(np.mean(cmd_dpos))
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

        # Same deterministic teardown order as phase3_closed_loop.py: close the
        # offscreen renderer's GL context before the passive viewer's GLFW teardown,
        # otherwise MuJoCo can abort with "free(): invalid pointer".
        renderer.close()

    if executor is not None:
        executor.shutdown(wait=True)
    if not args.dry_run:
        print(f"Saved {frame_idx} frame pairs to {images_dir}/")


if __name__ == "__main__":
    main()
