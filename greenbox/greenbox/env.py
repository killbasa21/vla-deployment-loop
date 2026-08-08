"""The pick-and-place environment.

Franka Panda, one green box, three open-topped containers at fixed left / right /
top slots whose colours are permuted every episode. Task: put the green box in
the green container.

Action space is robosuite `OSC_POSE` with `control_delta=True`: 7-D, first six in
[-1, 1] scaled to +-5 cm / +-0.5 rad per control step, last is the gripper
(-1 open, +1 close), at 20 Hz. That is deliberately the same convention the
LIBERO datasets use, so a policy pretrained there is not being asked to learn a
new action space.
"""

from __future__ import annotations

import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from greenbox import task_spec as spec
from greenbox.objects import ContainerObject


class GreenBoxPickPlace(SingleArmEnv):
    """`put the green box in the green container`."""

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=spec.CONTROL_FREQ,
        horizon=400,
        ignore_done=False,
        hard_reset=False,
        camera_names=tuple(spec.CAMERAS),
        camera_heights=spec.IMAGE_SIZE,
        camera_widths=spec.IMAGE_SIZE,
        camera_depths=False,
        **kwargs,
    ):
        self.table_full_size = spec.TABLE_FULL_SIZE
        self.table_offset = np.array(spec.TABLE_OFFSET)
        self.table_friction = (1.0, 5e-3, 1e-4)

        # Set by _reset_internal(), read by the expert and the success check.
        self.slot_colors: dict[str, str] = {}
        self.target_slot: str = "top"

        if controller_configs is None:
            controller_configs = suite.load_controller_config(default_controller="OSC_POSE")

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=list(camera_names),
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            **kwargs,
        )

    # ------------------------------------------------------------------ model

    def _load_model(self):
        super()._load_model()

        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        )

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self.box = BoxObject(
            name="green_box",
            size=[spec.BOX_HALF_SIZE] * 3,
            rgba=spec.BOX_COLOR,
            density=500.0,
            friction=[1.0, 0.005, 0.0001],
        )

        self.containers = {
            slot: ContainerObject(name=f"container_{slot}") for slot in spec.CONTAINER_SLOTS
        }

        self.placement_initializer = UniformRandomSampler(
            name="BoxSampler",
            mujoco_objects=[self.box],
            x_range=[
                spec.BOX_SAMPLE_CENTER[0] - spec.BOX_SAMPLE_HALF_RANGE,
                spec.BOX_SAMPLE_CENTER[0] + spec.BOX_SAMPLE_HALF_RANGE,
            ],
            y_range=[
                spec.BOX_SAMPLE_CENTER[1] - spec.BOX_SAMPLE_HALF_RANGE,
                spec.BOX_SAMPLE_CENTER[1] + spec.BOX_SAMPLE_HALF_RANGE,
            ],
            rotation=(-np.pi / 4, np.pi / 4),
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[self.box, *self.containers.values()],
        )

    def _setup_references(self):
        super()._setup_references()
        self.box_body_id = self.sim.model.body_name2id(self.box.root_body)
        self.container_body_ids = {
            slot: self.sim.model.body_name2id(obj.root_body)
            for slot, obj in self.containers.items()
        }
        # `duplicate_collision_geoms=True` gives each geom a `_vis` twin, and every
        # name carries the object's prefix, so match on the prefix rather than on
        # the names handed to CompositeObject.
        self.container_geom_ids = {
            slot: [
                gid
                for gid, name in enumerate(self.sim.model.geom_names)
                if name is not None and name.startswith(f"{obj.naming_prefix}")
            ]
            for slot, obj in self.containers.items()
        }

    # ------------------------------------------------------------------ reset

    def _reset_internal(self):
        super()._reset_internal()

        rng = np.random  # robosuite seeds np.random via env.seed()
        gen = np.random.default_rng(rng.randint(0, 2**31 - 1))

        # Containers are static: position is a model edit, not a qpos write.
        for slot, (dx, dy) in spec.CONTAINER_SLOTS.items():
            pos = self.table_offset + np.array([dx, dy, spec.CONTAINER_HALF_SIZE[2]])
            self.sim.model.body_pos[self.container_body_ids[slot]] = pos

        self.slot_colors = spec.sample_container_colors(gen)
        self.target_slot = next(
            s for s, c in self.slot_colors.items() if c == spec.TARGET_COLOR
        )
        for slot, color in self.slot_colors.items():
            rgba = np.array(spec.COLORS[color])
            for gid in self.container_geom_ids[slot]:
                self.sim.model.geom_rgba[gid] = rgba

        if not self.deterministic_reset:
            for obj_pos, obj_quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

        self.sim.forward()

    # ----------------------------------------------------------- observations

    def _setup_observables(self):
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        @sensor(modality=modality)
        def box_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.box_body_id])

        @sensor(modality=modality)
        def box_quat(obs_cache):
            return np.array(self.sim.data.body_xquat[self.box_body_id])[[1, 2, 3, 0]]

        @sensor(modality=modality)
        def target_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.container_body_ids[self.target_slot]])

        @sensor(modality=modality)
        def box_to_eef_pos(obs_cache):
            if f"{pf}eef_pos" not in obs_cache or "box_pos" not in obs_cache:
                return np.zeros(3)
            return obs_cache["box_pos"] - obs_cache[f"{pf}eef_pos"]

        for s in (box_pos, box_quat, target_pos, box_to_eef_pos):
            observables[s.__name__] = Observable(
                name=s.__name__, sensor=s, sampling_rate=self.control_freq
            )
        return observables

    def policy_state(self) -> np.ndarray:
        """The 9-D vector a policy sees: eef pos, eef quat (xyzw), finger qpos."""
        pf = self.robots[0].robot_model.naming_prefix
        eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        eef_quat = self._eef_quat_xyzw()
        gripper_qpos = np.array(
            [self.sim.data.qpos[x] for x in self.robots[0]._ref_gripper_joint_pos_indexes]
        )
        del pf
        return np.concatenate([eef_pos, eef_quat, gripper_qpos]).astype(np.float32)

    def _eef_quat_xyzw(self) -> np.ndarray:
        from robosuite.utils.transform_utils import mat2quat

        mat = np.array(self.sim.data.site_xmat[self.robots[0].eef_site_id]).reshape(3, 3)
        return mat2quat(mat)  # robosuite returns xyzw

    # ----------------------------------------------------------- task outcome

    def _check_success(self) -> bool:
        box = np.array(self.sim.data.body_xpos[self.box_body_id])
        cpos = np.array(self.sim.data.body_xpos[self.container_body_ids[self.target_slot]])
        inside_xy = np.all(np.abs(box[:2] - cpos[:2]) < spec.CONTAINER_INNER_XY)
        rim_z = cpos[2] + spec.CONTAINER_HALF_SIZE[2]
        below_rim = box[2] < rim_z
        resting = box[2] > cpos[2] - spec.CONTAINER_HALF_SIZE[2]
        released = not self._gripper_holds_box()
        return bool(inside_xy and below_rim and resting and released)

    def _gripper_holds_box(self) -> bool:
        return self.check_contact(self.robots[0].gripper, self.box)

    def reward(self, action=None) -> float:
        return 1.0 if self._check_success() else 0.0
