"""Operational Space Control (OSC_POSE), ported from robosuite 1.4.0 to raw MuJoCo.

WHY THIS EXISTS
---------------
LIBERO drives its Panda with robosuite's `OSC_POSE`: a force-based controller that turns
a normalised Cartesian delta into JOINT TORQUES. `libero_closed_loop.py`'s original
"Route A" instead solved damped-least-squares IK and wrote JOINT ANGLES into position
actuators. Route A was always known to be a stand-in, and three of this project's
long-running problems are artefacts of it rather than of the policy:

  1. No compliance -- a position servo drives to its target through contact
     (PROGRESS sec.19: 2.9 mm of penetration under ~70 N). `--min-clearance` exists only
     to fake the yielding a force controller gets for free.
  2. Servo droop -- an overdamped position servo never arrives, so the collector's
     `(target - current)` label carries a standing sag. The whole of docs/SERVO_DROOP.md is about
     that bug. A torque controller has no joint setpoint to lag behind.
  3. `IK unreached` -- a hard failure with no analogue in OSC, where the commanded wrench
     simply falls off near a singularity.

This module is a PORT, not a reimplementation from the paper. Every constant and every
line of the math is traceable to robosuite 1.4.0 (the version LIBERO pins -- 1.5.2 has a
restructured controller with `input_ref_frame`, which 1.4.0 does not have and which we
therefore must not introduce):

    robosuite/controllers/osc.py                     OSC.run_controller, OSC.set_goal
    robosuite/controllers/config/osc_pose.json       kp=150, damping_ratio=1,
                                                     uncouple_pos_ori=true,
                                                     output_max=[0.05]*3+[0.5]*3
    robosuite/utils/control_utils.py                 opspace_matrices, nullspace_torques,
                                                     orientation_error,
                                                     set_goal_position/orientation
    robosuite/controllers/base_controller.py         torque_compensation = qfrc_bias

THE TIMING DETAIL THAT IS EASY TO GET WRONG
-------------------------------------------
robosuite does NOT compute a torque once per policy action and hold it. In
`environments/base.py:454` it loops `control_timestep / model_timestep` times, calling
`_pre_action` INSIDE the loop with `policy_step=True` only on the first iteration. So:

    set_goal()        <- once per 20 Hz action, from the eef pose at that instant
    run_controller()  <- EVERY physics step (500 Hz), toward that fixed goal

Holding a torque constant across a 25-step tick would be a completely different (and
unstable) controller: the PD term would not see the velocity it is meant to damp. Hence
the two-method split below, and hence `libero_closed_loop.py`'s stepping loop had to be
restructured to call `compute_torque()` inside the decimation loop.

WHAT IS DELIBERATELY NOT PORTED
-------------------------------
- Interpolators. `osc_pose.json` sets `"interpolation": null`, so LIBERO does not use
  them and the `interpolator_pos`/`interpolator_ori` branches are dead code there too.
- `impedance_mode` other than "fixed". LIBERO's config pins "fixed", so kp/kd are not
  part of the action space and the action stays 7-D.
- `position_limits` / `orientation_limits`. Both null in LIBERO's config.
Adding any of these would be inventing a difference from the training plant.
"""

import mujoco
import numpy as np

# robosuite/controllers/config/osc_pose.json, verbatim. These are the numbers LIBERO's
# demonstrations were generated under; they are not tuning knobs to sweep casually.
DEFAULT_KP = 150.0
DEFAULT_DAMPING_RATIO = 1.0     # kd = 2*sqrt(kp)*damping_ratio -> critically damped
UNCOUPLE_POS_ORI = True         # decouple via lambda_pos/lambda_ori instead of lambda_full
NULLSPACE_JOINT_KP = 10.0       # control_utils.nullspace_torques default joint_kp

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))


def orientation_error(desired, current):
    """robosuite's `control_utils.orientation_error`, verbatim.

    NOTE this is NOT the same function as `libero_closed_loop._orientation_error`, which
    goes through quaternions (`mju_quat2Vel`). That one is used to drive IK to
    convergence, where only the direction and the zero matter. Here the magnitude is
    multiplied by kp to make a torque, so the two are not interchangeable and this one
    must be robosuite's. Both take (3,3) rotation matrices; this returns axis*angle.
    """
    rc1, rc2, rc3 = current[:, 0], current[:, 1], current[:, 2]
    rd1, rd2, rd3 = desired[:, 0], desired[:, 1], desired[:, 2]
    return 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))


def opspace_matrices(mass_matrix, J_full, J_pos, J_ori):
    """robosuite's `control_utils.opspace_matrices`, verbatim.

    Lambda = (J M^-1 J^T)^-1 is the task-space ("operational space") inertia: it is what
    turns a desired Cartesian acceleration into a wrench that accounts for how heavy the
    arm feels in each direction at this configuration. `pinv` rather than `inv` is
    robosuite's own choice and it matters -- at a singularity lambda_full_inv is rank
    deficient, and pinv zeroes those directions instead of blowing up. That is precisely
    the graceful degradation Route A's IK could not do.
    """
    mass_matrix_inv = np.linalg.inv(mass_matrix)
    lambda_full = np.linalg.pinv(J_full @ mass_matrix_inv @ J_full.T)
    lambda_pos = np.linalg.pinv(J_pos @ mass_matrix_inv @ J_pos.T)
    lambda_ori = np.linalg.pinv(J_ori @ mass_matrix_inv @ J_ori.T)
    # Dynamically-consistent generalised inverse, and the nullspace projector built on it.
    Jbar = mass_matrix_inv @ J_full.T @ lambda_full
    nullspace_matrix = np.eye(J_full.shape[-1]) - Jbar @ J_full
    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


def nullspace_torques(mass_matrix, nullspace_matrix, initial_joint, joint_pos, joint_vel,
                      joint_kp=NULLSPACE_JOINT_KP):
    """robosuite's `control_utils.nullspace_torques`, verbatim.

    The Panda has 7 joints controlling a 6-DOF task, so one DOF is redundant: a whole
    curve of elbow positions achieves the same eef pose. Left unconstrained the elbow
    drifts. This projects a joint-space PD pull toward `initial_joint` through the
    nullspace, so it steers the elbow WITHOUT disturbing the eef.
    """
    joint_kv = 2 * np.sqrt(joint_kp)
    pose_torques = mass_matrix @ (joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel)
    return nullspace_matrix.T @ pose_torques


def axisangle_to_mat(rotvec):
    """axis*angle 3-vector -> (3,3) rotation matrix, via MuJoCo so the convention matches
    the rest of this project. Equivalent to robosuite's
    `quat2mat(axisangle2quat(delta))` in `set_goal_orientation`."""
    angle = float(np.linalg.norm(rotvec))
    quat = np.zeros(4)
    if angle < 1e-12:
        quat[:] = [1.0, 0.0, 0.0, 0.0]
    else:
        mujoco.mju_axisAngle2Quat(quat, np.asarray(rotvec, dtype=float) / angle, angle)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


class OSCController:
    """Stateful OSC_POSE controller over a MuJoCo model with TORQUE actuators.

    Requires a scene whose arm actuators are `<motor>` (scene_libero_osc.xml). Pointing
    it at scene_libero_hand.xml would write torques into position servos, i.e. interpret
    N.m as radians -- `libero_closed_loop.py` checks the actuator type and refuses.

    Usage, per 20 Hz control tick:

        osc.set_goal(data, action)             # once, from the pose at this instant
        for _ in range(decimation):            # 25 physics steps
            data.ctrl[0:7] = osc.compute_torque(model, data)
            mujoco.mj_step(model, data)
    """

    def __init__(self, model, site_id, kp=DEFAULT_KP, damping_ratio=DEFAULT_DAMPING_RATIO,
                 uncouple_pos_ori=UNCOUPLE_POS_ORI, nullspace_kp=NULLSPACE_JOINT_KP):
        self.site_id = site_id
        self.kp = np.full(6, float(kp))
        self.kd = 2 * np.sqrt(self.kp) * float(damping_ratio)
        self.uncoupling = bool(uncouple_pos_ori)
        self.nullspace_kp = float(nullspace_kp)

        # Resolve the arm's qpos/dof indices by NAME rather than assuming 0..6. They do
        # happen to be 0..6 in this scene, but the scene also contains free-joint objects
        # (green_box, red_box) and the two finger joints, so nv is 21, not 7 -- slicing
        # a 21x21 mass matrix with [:7] would be right by luck and wrong the moment a
        # body is added above the robot in the tree.
        self.qpos_idx = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in ARM_JOINT_NAMES], dtype=int)
        self.dof_idx = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in ARM_JOINT_NAMES], dtype=int)

        self.torque_limits = model.actuator_ctrlrange[:7].copy()
        # Per-joint magnitude bound, for reporting headroom as a fraction. Taken as the
        # larger of |lo| and |hi| rather than assuming symmetry -- it IS symmetric in this
        # scene (+-80 on j1-5, +-12 on j6-7), but an asymmetric ctrlrange would otherwise
        # silently report >100% headroom on the wider side.
        self._limit_mag = np.maximum(np.abs(self.torque_limits[:, 0]),
                                     np.abs(self.torque_limits[:, 1]))

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._M_full = np.zeros((model.nv, model.nv))

        self.goal_pos = None
        self.goal_mat = None
        self.initial_joint = None
        # Diagnostics, read by the closed loop for logging.
        self.last_saturated = 0
        self.last_torque = np.zeros(7)
        # Largest |torque| / limit over the 7 joints at the last physics step, measured
        # BEFORE the clip. A saturation COUNT only says a joint hit the rail; this says how
        # hard it was pushing on it, and it is the only signal that distinguishes "0.99 of
        # the limit, fine" from "3x the limit, the controller is asking for motion this arm
        # cannot make". >= 1.0 exactly when last_saturated > 0.
        self.last_peak_frac = 0.0

    # -- lifecycle ---------------------------------------------------------

    def reset(self, data):
        """Capture the nullspace reference posture and park the goal on the current pose.

        Call after the scene is reset and settled. robosuite does the equivalent in
        `Controller.update_initial_joints` at env reset, and the nullspace reference
        being LIBERO's reset posture (rather than, say, zeros) is part of the plant.

        Parking the goal on the CURRENT pose matters for the settle loop: with torque
        actuators, `ctrl = 0` means the arm falls under gravity. A zero-delta goal makes
        the controller hold station instead, which is exactly what robosuite does between
        policy steps.
        """
        self.initial_joint = data.qpos[self.qpos_idx].copy()
        self.goal_pos = data.site_xpos[self.site_id].copy()
        self.goal_mat = data.site_xmat[self.site_id].reshape(3, 3).copy()

    def set_goal(self, data, dpos, drot):
        """Once per control tick. `dpos` (m) and `drot` (rad, axis-angle) are already
        un-normalised -- the caller applies DELTA_POS_SCALE / DELTA_ROT_SCALE, so this
        module stays agnostic about the action encoding.

        Both are composed against the CURRENT MEASURED pose, which is what makes this a
        delta controller and stops error accumulating (robosuite `set_goal_position`:
        `goal = current + delta`; `set_goal_orientation`: `goal = R(delta) @ current`,
        i.e. world-frame pre-multiplication).
        """
        cur_pos = data.site_xpos[self.site_id].copy()
        cur_mat = data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.goal_pos = cur_pos + np.asarray(dpos, dtype=float)
        self.goal_mat = axisangle_to_mat(drot) @ cur_mat
        return self.goal_pos, self.goal_mat

    def clamp_goal_z(self, min_z):
        """Optional table floor on the GOAL, for parity with Route A's --min-clearance.

        Off by default in OSC mode and it should stay off: the clamp only ever existed to
        fake compliance, which this controller has natively. Kept so the two control
        modes can be compared under identical constraints if wanted.
        """
        if self.goal_pos is not None and self.goal_pos[2] < min_z:
            self.goal_pos[2] = min_z
            return True
        return False

    # -- per physics step --------------------------------------------------

    def compute_torque(self, model, data):
        """robosuite's `OSC.run_controller`, evaluated at the current state. Returns the
        7 arm torques to write into `data.ctrl[0:7]`. Call EVERY physics step."""
        if self.goal_pos is None:
            raise RuntimeError("OSCController.reset() or .set_goal() must be called first")

        cur_pos = data.site_xpos[self.site_id]
        cur_mat = data.site_xmat[self.site_id].reshape(3, 3)

        # Site Jacobian, restricted to the arm's dofs.
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.site_id)
        J_pos = self._jacp[:, self.dof_idx]
        J_ori = self._jacr[:, self.dof_idx]
        J_full = np.vstack([J_pos, J_ori])

        joint_pos = data.qpos[self.qpos_idx]
        joint_vel = data.qvel[self.dof_idx]

        # Site velocity. robosuite reads `sim.data.get_site_xvelp/xvelr`; here it is
        # derived as J @ qvel over the arm dofs, which is the same quantity (the site
        # hangs off the arm chain only -- the finger and free-body columns of the site
        # Jacobian are zero). Verified against mj_objectVelocity in
        # tools/verify_osc.py, rather than trusted: mj_objectVelocity returns
        # (rot, lin) ordering, which is the opposite of the (lin, rot) convention used
        # here, and that swap is invisible in the numbers until the arm rotates.
        ee_pos_vel = J_pos @ joint_vel
        ee_ori_vel = J_ori @ joint_vel

        # Cartesian PD -> desired wrench.
        position_error = self.goal_pos - cur_pos
        ori_error = orientation_error(self.goal_mat, cur_mat)
        desired_force = position_error * self.kp[0:3] + (-ee_pos_vel) * self.kd[0:3]
        desired_torque = ori_error * self.kp[3:6] + (-ee_ori_vel) * self.kd[3:6]

        # Task-space inertia + nullspace projector.
        # Signature is (model, data, dst) in this MuJoCo version -- older bindings took
        # (model, dst, qM), which is the form robosuite's mujoco-py era used.
        mujoco.mj_fullM(model, data, self._M_full)
        mass_matrix = self._M_full[np.ix_(self.dof_idx, self.dof_idx)]
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            mass_matrix, J_full, J_pos, J_ori)

        if self.uncoupling:
            decoupled_wrench = np.concatenate(
                [lambda_pos @ desired_force, lambda_ori @ desired_torque])
        else:
            decoupled_wrench = lambda_full @ np.concatenate([desired_force, desired_torque])

        # Gamma = J^T F + gravity/Coriolis compensation + nullspace term.
        # `qfrc_bias` is robosuite's `torque_compensation` exactly (base_controller.py:230):
        # it is the Coriolis + centrifugal + gravity vector, so adding it makes the arm
        # weightless from the controller's point of view and the PD acts on the task alone.
        torques = J_full.T @ decoupled_wrench + data.qfrc_bias[self.dof_idx]
        torques = torques + nullspace_torques(
            mass_matrix, nullspace_matrix, self.initial_joint, joint_pos, joint_vel,
            joint_kp=self.nullspace_kp)

        # Headroom BEFORE the clip, so an over-limit demand is visible as its true size
        # rather than flattened to exactly the limit. Recorded every physics step; the
        # collector maxes it over the episode and rejects on it (--max-torque-frac).
        self.last_peak_frac = float(np.max(np.abs(torques) / self._limit_mag))

        clipped = np.clip(torques, self.torque_limits[:, 0], self.torque_limits[:, 1])
        # MuJoCo would clip these anyway (ctrllimited="true"), but doing it here lets the
        # closed loop LOG saturation. A persistently saturated joint means the model is
        # asking for motion this arm cannot produce -- the OSC-mode analogue of Route A's
        # `IK unreached`, and the thing to watch if a run stalls.
        self.last_saturated = int(np.sum(np.abs(clipped - torques) > 1e-9))
        self.last_torque = clipped
        return clipped
