"""Per-episode metrics.

Success alone is a one-bit signal that says nothing about *where* a policy broke.
This records the whole chain so a failure can be located:

    reach -> grasp -> lift -> transport -> lower -> release -> settle

Stage flags (each is a prerequisite of the next, so the first `False` in the
chain is the failure point):

  reached      eef came within `reach_tol` of the box, with the wrist within
               `rad_tol` of a valid top-down grasp
  grasped      gripper pads in contact with the box on both sides
  lifted       box rose `lift_tol` above its start height while still grasped
  transported  box passed over the target tray footprint while above its rim
  released     gripper let go of the box *over the target tray* -- it was dropped
               in. Opening the fingers anywhere else is not a release, it is a
               drop, and scores False; `release_pos_err` records how far off it
               was and `release_height` how far above the rim it let go
  placed       box ended inside the target tray, below the rim, not held
  complete     placed AND released AND the box had settled (near-zero velocity)

Grounding flags -- these separate "cannot control" from "cannot see":

  placed_wrong  box ended inside one of the two distractor trays
  nearest_tray  which tray the box ended closest to

Continuous measures. For each, `closest` is the best value reached at any point
in the episode and `final` is the value at the last step. Reported across
episodes as mean / median / p90, so "expected" in the statistical sense is the
mean column.

  grasp_pos     ‖eef - box‖, metres
  grasp_rad     angle between the wrist and the nearest valid top-down grasp
                orientation, radians
  place_pos     ‖box_xy - target_tray_xy‖, metres
  lift_height   box height above its starting height, metres

`grasp_pos_at_close` / `grasp_rad_at_close` are sampled at the step the gripper
first commanded a close -- the pose that actually decided whether the grasp took,
as opposed to the best pose ever visited.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import robosuite.utils.transform_utils as T

from greenbox import task_spec as spec
from greenbox.expert import R_DOWN

REACH_TOL = 0.03  # m
RAD_TOL = 0.30  # rad
LIFT_TOL = 0.05  # m
SETTLE_VEL = 0.02  # m/s
GRIPPER_CLOSE_THRESHOLD = 0.0  # action[6] > this counts as a close command


def grasp_orientation_error(env) -> float:
    """Angle to the nearest valid top-down grasp, exploiting the cube's 90 deg
    symmetry so a wrist rotated a quarter turn is not scored as a 90 deg error."""
    eef_mat = np.array(env.sim.data.site_xmat[env.robots[0].eef_site_id]).reshape(3, 3)
    box_q = np.array(env.sim.data.body_xquat[env.box_body_id])[[1, 2, 3, 0]]
    box_mat = T.quat2mat(box_q)
    box_yaw = np.arctan2(box_mat[1, 0], box_mat[0, 0])

    best = np.pi
    cur_q = T.mat2quat(eef_mat)
    for k in range(4):
        yaw = box_yaw + k * np.pi / 2
        cy, sy = np.cos(yaw), np.sin(yaw)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        des_q = T.mat2quat(rz @ R_DOWN)
        err_q = T.quat_multiply(des_q, T.quat_inverse(cur_q))
        if err_q[3] < 0:
            err_q = -err_q
        best = min(best, float(np.linalg.norm(T.quat2axisangle(err_q))))
    return best


@dataclass
class EpisodeMetrics:
    episode: int = 0
    seed: int = 0
    target_slot: str = ""
    slot_colors: dict = field(default_factory=dict)
    steps: int = 0
    timeout: bool = False

    reached: bool = False
    grasped: bool = False
    lifted: bool = False
    transported: bool = False
    released: bool = False
    placed: bool = False
    complete: bool = False
    placed_wrong: bool = False
    nearest_tray: str = ""

    grasp_pos_closest: float = np.inf
    grasp_rad_closest: float = np.inf
    grasp_pos_at_close: float = float("nan")
    grasp_rad_at_close: float = float("nan")
    place_pos_closest: float = np.inf
    place_pos_final: float = float("nan")
    lift_height_max: float = 0.0
    release_pos_err: float = float("nan")  # box_xy vs target tray at let-go
    release_height: float = float("nan")  # box height above the rim at let-go

    def as_dict(self) -> dict:
        return asdict(self)


class EpisodeTracker:
    """Call `start(env)` after reset, `step(env, action)` after every env.step,
    then `finish(env)` to get the record."""

    def __init__(self):
        self.m = EpisodeMetrics()
        self._box_z0 = 0.0
        self._closed = False
        self._was_holding = False
        self._let_go = False

    def start(self, env, episode: int = 0, seed: int = 0) -> None:
        self.m = EpisodeMetrics(
            episode=episode,
            seed=seed,
            target_slot=env.target_slot,
            slot_colors=dict(env.slot_colors),
        )
        self._box_z0 = float(env.sim.data.body_xpos[env.box_body_id][2])
        self._closed = False
        self._was_holding = False
        self._let_go = False

    def step(self, env, action) -> None:
        m = self.m
        m.steps += 1

        box = np.array(env.sim.data.body_xpos[env.box_body_id])
        eef = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])
        tray = np.array(env.sim.data.body_xpos[env.container_body_ids[env.target_slot]])

        pos_err = float(np.linalg.norm(eef - box))
        rad_err = grasp_orientation_error(env)
        m.grasp_pos_closest = min(m.grasp_pos_closest, pos_err)
        m.grasp_rad_closest = min(m.grasp_rad_closest, rad_err)
        if pos_err < REACH_TOL and rad_err < RAD_TOL:
            m.reached = True

        # The decisive pose: the first step on which a close was commanded.
        if not self._closed and float(action[6]) > GRIPPER_CLOSE_THRESHOLD:
            self._closed = True
            m.grasp_pos_at_close = pos_err
            m.grasp_rad_at_close = rad_err

        holding = env._gripper_holds_box()
        if holding:
            m.grasped = True

        # The let-go event: the fingers opening after having held the box. Use the
        # *last* such event, not the first -- contact can break transiently during
        # transport and re-form, and scoring that as the drop understates a policy
        # that goes on to place the box correctly.
        rim_z = tray[2] + spec.CONTAINER_HALF_SIZE[2]
        if self._was_holding and not holding:
            self._let_go = True
            m.release_pos_err = float(np.linalg.norm(box[:2] - tray[:2]))
            m.release_height = float(box[2] - rim_z)
            m.released = m.release_pos_err < spec.CONTAINER_INNER_XY
        self._was_holding = holding

        lift = float(box[2] - self._box_z0)
        m.lift_height_max = max(m.lift_height_max, lift)
        if holding and lift > LIFT_TOL:
            m.lifted = True

        place_err = float(np.linalg.norm(box[:2] - tray[:2]))
        m.place_pos_closest = min(m.place_pos_closest, place_err)

        over_tray = place_err < spec.CONTAINER_INNER_XY and box[2] > rim_z
        if m.lifted and over_tray:
            m.transported = True

    def finish(self, env) -> EpisodeMetrics:
        m = self.m
        box = np.array(env.sim.data.body_xpos[env.box_body_id])
        tray = np.array(env.sim.data.body_xpos[env.container_body_ids[env.target_slot]])
        m.place_pos_final = float(np.linalg.norm(box[:2] - tray[:2]))

        dists = {
            slot: float(np.linalg.norm(box[:2] - np.array(
                env.sim.data.body_xpos[bid])[:2]))
            for slot, bid in env.container_body_ids.items()
        }
        m.nearest_tray = min(dists, key=dists.get)

        # `released` is set at the let-go event in step(); if the episode ends with
        # the box still held there was no release at all.
        m.placed = bool(env._check_success())
        m.placed_wrong = bool(
            not m.placed
            and m.released
            and any(
                slot != env.target_slot and self._inside(env, slot, box)
                for slot in env.container_body_ids
            )
        )
        vel = float(np.linalg.norm(env.sim.data.get_body_xvelp(env.box.root_body)))
        m.complete = bool(m.placed and m.released and vel < SETTLE_VEL)
        return m

    @staticmethod
    def _inside(env, slot: str, box: np.ndarray) -> bool:
        pos = np.array(env.sim.data.body_xpos[env.container_body_ids[slot]])
        rim_z = pos[2] + spec.CONTAINER_HALF_SIZE[2]
        return bool(
            np.all(np.abs(box[:2] - pos[:2]) < spec.CONTAINER_INNER_XY)
            and box[2] < rim_z
            and box[2] > pos[2] - spec.CONTAINER_HALF_SIZE[2]
        )
