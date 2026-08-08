"""Procedural scene objects."""

from __future__ import annotations

import numpy as np
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial  # noqa: F401  (re-export point)

from greenbox import task_spec as spec


class ContainerObject(CompositeObject):
    """An open-topped tray: one base plate plus four walls.

    Built as a `CompositeObject` rather than loaded from an XML file so that the
    five geoms live in a single body whose `rgba` can be rewritten at runtime
    (`sim.model.geom_rgba`) without rebuilding the model. `joints=None` makes it
    static -- it is welded to the world, and its position is set by writing
    `sim.model.body_pos`, which is also a runtime edit.
    """

    def __init__(self, name: str, rgba=(0.5, 0.5, 0.5, 1.0)):
        hx, hy, hz = spec.CONTAINER_HALF_SIZE
        w = spec.CONTAINER_WALL
        base_h = 0.005

        geom_types, geom_sizes, geom_locations, geom_names = [], [], [], []

        def add(name_, size, loc):
            geom_types.append("box")
            geom_sizes.append(np.array(size))
            geom_locations.append(np.array(loc))
            geom_names.append(name_)

        # locations are relative to the centre of the bounding box
        add("base", [hx, hy, base_h], [0.0, 0.0, -hz + base_h])
        add("wall_px", [w, hy, hz], [hx - w, 0.0, 0.0])
        add("wall_nx", [w, hy, hz], [-hx + w, 0.0, 0.0])
        add("wall_py", [hx, w, hz], [0.0, hy - w, 0.0])
        add("wall_ny", [hx, w, hz], [0.0, -hy + w, 0.0])

        super().__init__(
            name=name,
            total_size=np.array([hx, hy, hz]),
            geom_types=geom_types,
            geom_sizes=geom_sizes,
            geom_locations=geom_locations,
            geom_names=geom_names,
            geom_rgbas=[np.array(rgba)] * len(geom_types),
            geom_frictions=[np.array([1.0, 0.005, 0.0001])] * len(geom_types),
            density=1000.0,
            locations_relative_to_center=True,
            joints=None,  # static
            obj_types="all",
            duplicate_collision_geoms=True,
        )
