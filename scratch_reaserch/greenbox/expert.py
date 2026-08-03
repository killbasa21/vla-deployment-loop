"""Scripted expert for `put the green box in the green container`.

Reads privileged sim state (box pose, target tray pose) and emits actions in the
*same* 7-D OSC_POSE space the learned policy will use, so demonstrations and
rollouts go through one identical control path. Nothing here is available to the
policy -- it only ever sees images plus the 9-D proprio vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import robosuite.utils.transform_utils as T

from greenbox import task_spec as spec

# OSC_POSE `output_max`: an action of 1.0 commands this much per control step.
POS_SCALE = 0.05  # metres
ROT_SCALE = 0.50  # radians

# Gripper convention for the Panda gripper: positive closes.
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# Top-down grasp: gripper z axis points down, x forward.
R_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

PHASES = (
    "align",
    "descend",
    "grasp",
    "lift",
    "transport",
    "lower",
    "release",
    "retreat",
    "done",
)


@dataclass
class ExpertConfig:
    kp_pos: float = 1.4
    kp_rot: float = 1.2
    hover_height: float = 0.12  # above the box, during ALIGN
    carry_height: float = 0.20  # above the table top, during LIFT/TRANSPORT
    release_height: float = 0.05  # above the tray rim, during LOWER
    grasp_z_offset: float = 0.0  # grip site height relative to box centre
    pos_tol: float = 0.008
    fine_pos_tol: float = 0.004
    rot_tol: float = 0.08
    settle_steps: int = 10  # steps spent holding still in GRASP / RELEASE
    max_phase_steps: int = 90
    action_noise: float = 0.0  # std of gaussian noise added to the action
    waypoint_noise: float = 0.0  # std of per-episode jitter on the waypoints
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


class ScriptedExpert:
    """Phase machine. Call `reset(env)` then `act(env)` once per control step."""

    def __init__(self, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()
        self.phase = "align"
        self.phase_step = 0
        self._jitter = np.zeros(3)
        self._hover = self.cfg.hover_height
        self._grasp_yaw = 0.0

    # ------------------------------------------------------------------ reset

    def reset(self, env) -> None:
        c = self.cfg
        self.phase = "align"
        self.phase_step = 0
        n = c.waypoint_noise
        self._jitter = c.rng.normal(0.0, n, size=3) if n > 0 else np.zeros(3)
        self._hover = c.hover_height + (c.rng.normal(0.0, n) if n > 0 else 0.0)
        self._grasp_yaw = self._box_yaw(env)

    # ------------------------------------------------------------- kinematics

    @staticmethod
    def _box_pos(env) -> np.ndarray:
        return np.array(env.sim.data.body_xpos[env.box_body_id])

    @staticmethod
    def _box_yaw(env) -> float:
        quat_wxyz = np.array(env.sim.data.body_xquat[env.box_body_id])
        mat = T.quat2mat(quat_wxyz[[1, 2, 3, 0]])
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        # A square cube is symmetric every 90 deg; pick the equivalent yaw
        # closest to zero so the wrist never has to swing more than 45 deg.
        return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

    @staticmethod
    def _target_pos(env) -> np.ndarray:
        return np.array(env.sim.data.body_xpos[env.container_body_ids[env.target_slot]])

    @staticmethod
    def _eef_pos(env) -> np.ndarray:
        return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])

    @staticmethod
    def _eef_mat(env) -> np.ndarray:
        return np.array(env.sim.data.site_xmat[env.robots[0].eef_site_id]).reshape(3, 3)

    def _desired_mat(self) -> np.ndarray:
        cy, sy = np.cos(self._grasp_yaw), np.sin(self._grasp_yaw)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return rz @ R_DOWN

    # ------------------------------------------------------------- waypointing

    def waypoint(self, env) -> tuple[np.ndarray, float]:
        """Return (target eef position, gripper command) for the current phase."""
        c = self.cfg
        box = self._box_pos(env)
        tray = self._target_pos(env)
        table_z = env.table_offset[2]
        rim_z = tray[2] + spec.CONTAINER_HALF_SIZE[2]
        j = self._jitter

        grasp_xy = box[:2] + j[:2]
        grasp_z = box[2] + c.grasp_z_offset

        if self.phase == "align":
            return np.array([*grasp_xy, box[2] + self._hover]), GRIPPER_OPEN
        if self.phase == "descend":
            return np.array([*grasp_xy, grasp_z]), GRIPPER_OPEN
        if self.phase == "grasp":
            return np.array([*grasp_xy, grasp_z]), GRIPPER_CLOSE
        if self.phase == "lift":
            return np.array([*grasp_xy, table_z + c.carry_height]), GRIPPER_CLOSE
        if self.phase == "transport":
            return np.array([*tray[:2], table_z + c.carry_height]), GRIPPER_CLOSE
        if self.phase == "lower":
            return np.array([*tray[:2], rim_z + c.release_height]), GRIPPER_CLOSE
        if self.phase == "release":
            return np.array([*tray[:2], rim_z + c.release_height]), GRIPPER_OPEN
        # retreat / done
        return np.array([*tray[:2], table_z + c.carry_height + 0.05]), GRIPPER_OPEN

    # ------------------------------------------------------------------ acting

    def act(self, env) -> np.ndarray:
        c = self.cfg
        target, grip = self.waypoint(env)

        pos_err = target - self._eef_pos(env)
        cur_q = T.mat2quat(self._eef_mat(env))
        des_q = T.mat2quat(self._desired_mat())
        err_q = T.quat_multiply(des_q, T.quat_inverse(cur_q))
        # q and -q are the same rotation but quat2axisangle only unwraps one of
        # them onto the short arc; without this a -0.8 rad error comes back as
        # 5.5 rad, saturates the rotation command and drags the arm off target.
        if err_q[3] < 0:
            err_q = -err_q
        rot_err = T.quat2axisangle(err_q)

        action = np.zeros(spec.ACTION_DIM)
        action[:3] = np.clip(c.kp_pos * pos_err / POS_SCALE, -1.0, 1.0)
        action[3:6] = np.clip(c.kp_rot * rot_err / ROT_SCALE, -1.0, 1.0)
        action[6] = grip

        if c.action_noise > 0 and self.phase not in ("grasp", "release"):
            action[:6] += c.rng.normal(0.0, c.action_noise, size=6)
            action = np.clip(action, -1.0, 1.0)

        self._advance(pos_err, rot_err)
        return action.astype(np.float32)

    def _advance(self, pos_err: np.ndarray, rot_err: np.ndarray) -> None:
        c = self.cfg
        self.phase_step += 1
        pos_ok = np.linalg.norm(pos_err) < (
            c.fine_pos_tol if self.phase in ("descend", "lower") else c.pos_tol
        )
        rot_ok = np.linalg.norm(rot_err) < c.rot_tol

        if self.phase in ("grasp", "release"):
            done = self.phase_step >= c.settle_steps
        elif self.phase in ("align", "descend"):
            done = (pos_ok and rot_ok) or self.phase_step >= c.max_phase_steps
        elif self.phase == "retreat":
            done = pos_ok or self.phase_step >= c.settle_steps
        else:
            done = pos_ok or self.phase_step >= c.max_phase_steps

        if done and self.phase != "done":
            self.phase = PHASES[PHASES.index(self.phase) + 1]
            self.phase_step = 0

    @property
    def finished(self) -> bool:
        return self.phase == "done"
