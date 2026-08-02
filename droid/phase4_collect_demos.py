"""
Phase 4: collect classical (IK-scripted) expert demonstrations of the green-ball
pick-and-place task, recorded in the MolmoAct2-DROID observation/action convention
and written out as a LeRobot v2.1 dataset for (eventual) VLA fine-tuning.

There is no teleoperation and no policy server in the loop here. The "expert" is a
hand-written waypoint script (pre-grasp -> descend -> close -> lift -> transport ->
lower -> release -> retreat) whose Cartesian waypoints are turned into arm joint
targets by a numerical Jacobian (damped-least-squares) inverse-kinematics solver run
against the loaded MuJoCo model. The IK is used purely as a *planner*: it solves each
waypoint on a throwaway copy of the sim state to get a target joint configuration, and
the real sim is then driven toward those configurations by the Panda's position
actuators, so the actual grasp happens through real contact physics (the fingers close
on the ball, friction holds it) rather than by teleporting qpos.

Every control tick (~15 Hz, matching DROID) we record, *before* stepping:
  - external_cam + wrist_cam RGB frames (same two cameras phase3 uses),
  - state (8,) = [q1..q7, gripper_rad]  -- the DROID server's own state schema,
  - action (8,) = [q1..q7 target, gripper_target_rad]  -- absolute joint targets, the
    same thing MolmoAct2-DROID's /act endpoint returns, so a model trained on this data
    speaks the same wire format droid/phase3_closed_loop.py already consumes.

Only episodes where the ball ends up resting inside the green container are kept.

Usage:
    # render 1 episode to an mp4, no dataset
    uv run python droid/phase4_collect_demos.py --preview
    # collect 50 successful demos -> dataset
    uv run python droid/phase4_collect_demos.py --episodes 50
    uv run python droid/phase4_collect_demos.py --episodes 50 --out data/green_ball_pick
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene_pick_place.xml"
INSTRUCTION = "pick up the green ball and put it in the green container"

# Gripper conventions. The Panda mounts a Robotiq 2F-85 (see panda.xml). Its
# fingers_actuator takes ctrl in 0-255 (0=open, 255=closed). MolmoAct2-DROID's gripper
# axis is the raw radian position of the driver knuckle joint, 0=open .. ~0.8=closed
# (see droid/phase3_closed_loop.py's long note). We record state/action gripper values in that
# DROID radian space and convert to the 0-255 actuator command when driving the sim.
GRIPPER_CTRL_MAX = 255.0
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_RAD = 0.8

CONTROL_HZ = 15.0  # DROID's control rate; sim runs at 500 Hz so decimation = 33

ARM = slice(0, 7)  # first 7 qpos / qvel / ctrl entries are the arm joints

# The 3 bins have no joint (see scene_pick_place.xml's own comment: rigidly welded to
# the world), so their world pose is set once at compile time from the XML's <body pos>
# and would otherwise never change episode-to-episode. LoRA fine-tuning needs the model
# to learn to LOCATE the green bin by looking at the scene, not memorize one fixed
# coordinate for it -- so each episode we shuffle which bin (green/blue/yellow) sits at
# which of these 3 anchor spots (the scene's original default positions) by overwriting
# model.body_pos directly; since these bodies have no joint, that's sufficient to move
# them (no qpos/freejoint involved) as long as mj_forward runs afterward.
BIN_SLOT_POSITIONS = [(0.5, 0.25, 0.0), (0.5, -0.25, 0.0), (0.7, 0.0, 0.0)]
BIN_SLOT_JITTER = 0.02  # meters, uniform +/- jitter added to each slot so the model
                         # can't memorize the 3 exact slot coordinates either
BIN_NAMES = ("green_bin", "blue_bin", "yellow_bin")


def randomize_bin_layout(model, bin_ids, rng):
    """Shuffle BIN_NAMES across BIN_SLOT_POSITIONS (each slot jittered) and write the
    result into model.body_pos. Returns the new green_bin position (the placement
    target for this episode) -- caller must mj_forward (or call reset_scene, which
    already does) before rendering/IK so data.xpos picks up the change."""
    order = rng.permutation(len(BIN_SLOT_POSITIONS))
    green_pos = None
    for name, slot_idx in zip(BIN_NAMES, order):
        slot = np.array(BIN_SLOT_POSITIONS[slot_idx])
        jitter = rng.uniform(-BIN_SLOT_JITTER, BIN_SLOT_JITTER, size=2)
        pos = slot + [jitter[0], jitter[1], 0.0]
        model.body_pos[bin_ids[name]] = pos
        if name == "green_bin":
            green_pos = pos
    return green_pos


def gripper_rad_to_ctrl(rad):
    """DROID-space gripper radians (0=open..0.8=closed) -> fingers_actuator 0-255."""
    frac = np.clip(rad / GRIPPER_CLOSED_RAD, 0.0, 1.0)
    return frac * GRIPPER_CTRL_MAX


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def freejoint_qpos_adr(model, body_name):
    """qpos start index of body_name's freejoint (7 entries: xyz + wxyz quat)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jid = model.body_jntadr[bid]
    return model.jnt_qposadr[jid]


def build_sim(model_path=MODEL_PATH):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data


def reset_scene(model, data, ball_xy, rng=None, arm_jitter=0.0):
    """Reset to the home keyframe and (re)place the green ball. The 'home' keyframe
    predates the freejoint props, so mj_resetDataKeyframe zero-pads their qpos to the
    origin -- we set the ball's freejoint pose by hand (same fix phase3 uses). red_box
    is parked off to the side as a static-ish distractor at its own default spot.

    arm_jitter (radians): if >0 and rng is given, perturb the 7 arm joints around the
    home pose so demos don't all start from the identical configuration -- start-state
    diversity a behavior-cloned policy needs to not overfit to one fixed opening pose."""
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)

    if arm_jitter > 0 and rng is not None:
        lower, upper = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
        data.qpos[:7] = np.clip(
            data.qpos[:7] + rng.normal(0.0, arm_jitter, size=7), lower, upper)
        data.ctrl[ARM] = data.qpos[:7]  # hold the jittered pose, don't snap back to home

    # green ball: rest on the floor (center = radius above z=0) at the requested xy.
    ball_adr = freejoint_qpos_adr(model, "green_ball")
    data.qpos[ball_adr : ball_adr + 3] = [ball_xy[0], ball_xy[1], 0.02]
    data.qpos[ball_adr + 3 : ball_adr + 7] = [1, 0, 0, 0]

    # red_box distractor: keep it in the scene but out of the ball's grasp corridor.
    box_adr = freejoint_qpos_adr(model, "red_box")
    data.qpos[box_adr : box_adr + 3] = [0.5, -0.28, 0.02]
    data.qpos[box_adr + 3 : box_adr + 7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Numerical inverse kinematics (Jacobian damped least squares)
# ---------------------------------------------------------------------------

def _orientation_error(target_mat, current_mat):
    """Rotation vector (axis*angle, world frame) that rotates current -> target.
    Both inputs are flat length-9 rotation matrices (data.site_xmat form)."""
    cur_q = np.zeros(4)
    tgt_q = np.zeros(4)
    mujoco.mju_mat2Quat(cur_q, current_mat)
    mujoco.mju_mat2Quat(tgt_q, target_mat)
    neg_cur = np.zeros(4)
    mujoco.mju_negQuat(neg_cur, cur_q)
    err_q = np.zeros(4)
    mujoco.mju_mulQuat(err_q, tgt_q, neg_cur)  # target * inv(current)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, err_q, 1.0)
    return vel


def solve_ik(model, scratch, site_id, target_pos, target_mat, q_seed,
             iters=200, tol=1e-4, damping=0.15, step_clip=0.3):
    """Solve for the 7 arm joint angles that put `site_id` at (target_pos, target_mat).

    Runs entirely on `scratch` (a throwaway MjData) seeded from q_seed, so it never
    disturbs the live sim. Returns (q_solution[7], reached_bool). Damped least squares
    keeps it stable near singularities; per-iteration steps are clipped so a large
    initial error can't fling the seed across a joint flip."""
    lower = model.jnt_range[:7, 0]
    upper = model.jnt_range[:7, 1]
    scratch.qpos[:7] = np.clip(q_seed, lower, upper)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    reached = False
    for _ in range(iters):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)  # site_xpos/xmat + jacobians need com frames
        err = np.concatenate([
            target_pos - scratch.site_xpos[site_id],
            _orientation_error(target_mat, scratch.site_xmat[site_id]),
        ])
        if np.linalg.norm(err) < tol:
            reached = True
            break
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])  # 6x7
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        dq = np.clip(dq, -step_clip, step_clip)
        scratch.qpos[:7] = np.clip(scratch.qpos[:7] + dq, lower, upper)
    return scratch.qpos[:7].copy(), reached


# ---------------------------------------------------------------------------
# Waypoint script -> per-tick joint + gripper targets
# ---------------------------------------------------------------------------

def plan_episode(model, data, site_id, home_mat, ball_xy, bin_pos):
    """Turn the pick-and-place waypoint script into a list of (q_target[7],
    gripper_rad) control setpoints, one per control tick, by solving IK at each
    waypoint and interpolating joint space between them.

    home_mat is the pinch-site orientation at the home keyframe (gripper pointing
    straight down); every waypoint reuses it, so the grasp stays top-down throughout."""
    ball = np.array([ball_xy[0], ball_xy[1], 0.02])
    bin_top = np.array([bin_pos[0], bin_pos[1], 0.0])

    O, C = GRIPPER_OPEN_RAD, GRIPPER_CLOSED_RAD
    # (target pinch position, gripper_rad, seconds for this segment)
    waypoints = [
        (ball + [0, 0, 0.14], O, 1.2),   # 1 pre-grasp: hover above ball, open
        (ball + [0, 0, 0.005], O, 1.0),  # 2 descend onto ball, still open
        (ball + [0, 0, 0.005], C, 0.8),  # 3 close gripper (arm holds still)
        (ball + [0, 0, 0.24], C, 1.0),   # 4 lift straight up
        (bin_top + [0, 0, 0.26], C, 1.4),# 5 transport over the container
        (bin_top + [0, 0, 0.12], C, 1.0),# 6 lower into the container
        (bin_top + [0, 0, 0.12], O, 0.7),# 7 release
        (bin_top + [0, 0, 0.26], O, 1.0),# 8 retreat up
    ]

    scratch = mujoco.MjData(model)
    q_seed = data.qpos[:7].copy()
    # Solve a joint config for each waypoint, chaining seeds so consecutive solutions
    # stay on the same IK branch (no sudden elbow flips between waypoints).
    q_targets = []
    for pos, _grip, _dur in waypoints:
        q_sol, ok = solve_ik(model, scratch, site_id, pos, home_mat, q_seed)
        if not ok:
            return None  # unreachable waypoint -> abandon this layout
        q_targets.append(q_sol)
        q_seed = q_sol

    dt_ctrl = 1.0 / CONTROL_HZ
    setpoints = []  # (q_target[7], gripper_rad) per control tick
    q_prev = data.qpos[:7].copy()
    grip_prev = O
    for (pos, grip, dur), q_next in zip(waypoints, q_targets):
        n = max(1, int(round(dur / dt_ctrl)))
        for k in range(1, n + 1):
            a = k / n
            q_interp = (1 - a) * q_prev + a * q_next
            grip_interp = (1 - a) * grip_prev + a * grip
            setpoints.append((q_interp, grip_interp))
        q_prev = q_next
        grip_prev = grip
    return setpoints


# ---------------------------------------------------------------------------
# Episode execution + recording
# ---------------------------------------------------------------------------

def read_state(model, data, driver_qposadr):
    """(8,) DROID state: 7 arm joint angles + the Robotiq driver knuckle angle
    (0=open..0.8=closed radians), matching phase3_closed_loop.read_state."""
    return np.concatenate([
        data.qpos[:7],
        [data.qpos[driver_qposadr]],
    ]).astype(np.float32)


def run_episode(model, data, renderer, site_id, home_mat, driver_qposadr,
                ball_xy, bin_pos, decimation, cam_names, record=True):
    """Plan + execute one episode. Returns (frames_dict, states, actions, success)
    where frames_dict maps camera name -> list of (H,W,3) uint8 arrays. When
    record=False, images are still rendered (needed for the preview mp4) but the
    caller decides what to do with them."""
    setpoints = plan_episode(model, data, site_id, home_mat, ball_xy, bin_pos)
    if setpoints is None:
        return None, None, None, False

    frames = {c: [] for c in cam_names}
    states, actions = [], []
    ball_adr = freejoint_qpos_adr(model, "green_ball")

    for q_target, grip_rad in setpoints:
        # Observation captured BEFORE applying this tick's action.
        for c in cam_names:
            renderer.update_scene(data, camera=c)
            frames[c].append(renderer.render().copy())
        states.append(read_state(model, data, driver_qposadr))
        # Action = absolute joint targets + gripper target, DROID convention.
        actions.append(np.concatenate([q_target, [grip_rad]]).astype(np.float32))

        # Apply and step the sim `decimation` times to advance one control tick.
        data.ctrl[ARM] = q_target
        data.ctrl[7] = gripper_rad_to_ctrl(grip_rad)
        for _ in range(decimation):
            mujoco.mj_step(model, data)

    # Success: ball resting inside the green container footprint (10x10cm, rim ~4cm).
    ball_pos = data.qpos[ball_adr : ball_adr + 3]
    dx, dy = ball_pos[0] - bin_pos[0], ball_pos[1] - bin_pos[1]
    success = abs(dx) < 0.05 and abs(dy) < 0.05 and ball_pos[2] < 0.06
    return frames, np.array(states), np.array(actions), success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="number of SUCCESSFUL demos to collect")
    parser.add_argument("--out", default="data/green_ball_pick",
                        help="output LeRobot dataset directory")
    parser.add_argument("--preview", action="store_true",
                        help="render a single episode to an mp4 and exit (no dataset)")
    parser.add_argument("--res", type=int, default=256,
                        help="square render resolution for recorded camera frames")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="cap total episode attempts (default: 4x --episodes)")
    parser.add_argument("--start-jitter", type=float, default=0.05,
                        help="std (radians) of per-episode random perturbation to the "
                             "arm's home start pose (0 = always start at home)")
    args = parser.parse_args()

    model, data = build_sim()
    renderer = mujoco.Renderer(model, height=args.res, width=args.res)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
    driver_qposadr = model.jnt_qposadr[driver_id]
    bin_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for name in BIN_NAMES}
    decimation = int(round((1.0 / model.opt.timestep) / CONTROL_HZ))
    cam_names = ("external_cam", "wrist_cam")

    # Capture the gripper-pointing-down orientation from the home keyframe once.
    reset_scene(model, data, ball_xy=(0.5, 0.0))
    home_mat = data.site_xmat[site_id].copy()

    rng = np.random.default_rng(args.seed)

    def sample_ball_xy():
        # Reachable workspace patch in front of the arm, kept clear of the bins.
        return (rng.uniform(0.40, 0.58), rng.uniform(-0.16, 0.14))

    if args.preview:
        import imageio.v2 as imageio
        ball_xy = sample_ball_xy()
        bin_pos = randomize_bin_layout(model, bin_ids, rng)
        reset_scene(model, data, ball_xy, rng=rng, arm_jitter=args.start_jitter)
        frames, states, actions, success = run_episode(
            model, data, renderer, site_id, home_mat, driver_qposadr,
            ball_xy, bin_pos, decimation, cam_names)
        out = Path("assets/phase4_preview.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        if frames is None:
            print("IK could not solve the sampled layout; try another --seed.")
            return
        # Side-by-side external | wrist for the preview.
        combined = [np.concatenate([e, w], axis=1)
                    for e, w in zip(frames["external_cam"], frames["wrist_cam"])]
        imageio.mimsave(out, combined, fps=CONTROL_HZ)
        print(f"ball_xy={np.round(ball_xy,3)}  success={success}  "
              f"steps={len(states)}  ball_final_z={data.qpos[freejoint_qpos_adr(model,'green_ball')+2]:.3f}")
        print(f"wrote {out}  (external | wrist, {len(combined)} frames @ {CONTROL_HZ} fps)")
        renderer.close()
        return

    # Full collection -> LeRobot dataset.
    from lerobot_writer import LeRobotDatasetWriter

    writer = LeRobotDatasetWriter(
        root=args.out,
        fps=CONTROL_HZ,
        cameras=cam_names,
        image_shape=(args.res, args.res, 3),
        state_dim=8,
        action_dim=8,
        robot_type="franka_panda_robotiq2f85",
    )

    max_attempts = args.max_attempts or (4 * args.episodes)
    kept, attempts = 0, 0
    t0 = time.time()
    while kept < args.episodes and attempts < max_attempts:
        attempts += 1
        ball_xy = sample_ball_xy()
        bin_pos = randomize_bin_layout(model, bin_ids, rng)
        reset_scene(model, data, ball_xy, rng=rng, arm_jitter=args.start_jitter)
        frames, states, actions, success = run_episode(
            model, data, renderer, site_id, home_mat, driver_qposadr,
            ball_xy, bin_pos, decimation, cam_names)
        if not success:
            continue
        writer.add_episode(frames, states, actions, task=INSTRUCTION)
        kept += 1
        print(f"[{kept}/{args.episodes}] attempt {attempts}  "
              f"ball_xy={np.round(ball_xy,3)}  bin_pos={np.round(bin_pos,3)}  steps={len(states)}")

    writer.finalize()
    renderer.close()
    dt = time.time() - t0
    print(f"\nDone: {kept} demos from {attempts} attempts "
          f"({100*kept/attempts:.0f}% success) in {dt:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
