"""Single source of truth for everything a policy is conditioned on.

Collection, training and serving all import from here. A mismatch between the
instruction string used at collection time and the one used at serving time does
not raise -- it just conditions the policy on a prompt it never saw -- so there
is exactly one copy of it, here.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- instruction

INSTRUCTION = "put the green box in the green container"

# ---------------------------------------------------------------- observations

# Camera names as they appear in robosuite observations, and the LeRobot feature
# key each one is stored under.
CAMERAS = {
    "agentview": "observation.images.agentview",
    "robot0_eye_in_hand": "observation.images.wrist",
}
IMAGE_SIZE = 256  # square, rendered and stored at this resolution

# 9-D proprioceptive state, LIBERO layout: eef pos (3) + eef quat xyzw (4) +
# gripper finger qpos (2).
STATE_DIM = 9
STATE_KEY = "observation.state"

# 7-D action: OSC_POSE delta [dx, dy, dz, drx, dry, drz] in [-1, 1] plus gripper
# in {-1 open, +1 close}. 20 Hz.
ACTION_DIM = 7
ACTION_KEY = "action"
CONTROL_FREQ = 20

# ---------------------------------------------------------------- scene layout

# All positions are XY offsets from the centre of the table top.
TABLE_FULL_SIZE = (0.8, 0.8, 0.05)
TABLE_OFFSET = (0.0, 0.0, 0.80)

# The three container slots. Names are as seen in the `agentview` image, which
# looks at the table from the front: the robot is at the top of the frame, so
# "top" is the slot nearest the robot base (-x) and +x renders downward.
CONTAINER_SLOTS = {
    "top": (-0.20, 0.00),
    "left": (0.06, -0.19),
    "right": (0.06, 0.19),
}

CONTAINER_HALF_SIZE = (0.065, 0.065, 0.028)  # outer bounding half-extents
CONTAINER_WALL = 0.005  # wall half-thickness
CONTAINER_INNER_XY = CONTAINER_HALF_SIZE[0] - 2 * CONTAINER_WALL  # 0.055

# The green box is sampled uniformly from a square in the middle of the table.
BOX_HALF_SIZE = 0.02  # 4 cm cube
BOX_SAMPLE_CENTER = (0.08, 0.0)
BOX_SAMPLE_HALF_RANGE = 0.055  # -> 12 cm x 12 cm square

# Distractor / target colours. The target is always green; the other two slots
# get red and blue in a random order, so which slot is the target changes every
# episode and has to be read off the image.
COLORS = {
    "green": (0.10, 0.65, 0.20, 1.0),
    "red": (0.75, 0.12, 0.12, 1.0),
    "blue": (0.12, 0.25, 0.80, 1.0),
}
TARGET_COLOR = "green"
BOX_COLOR = (0.10, 0.75, 0.20, 1.0)


def sample_container_colors(rng: np.random.Generator) -> dict[str, str]:
    """Assign one colour per slot; `TARGET_COLOR` lands on exactly one slot."""
    names = list(CONTAINER_SLOTS)
    colors = list(COLORS)
    rng.shuffle(colors)
    return dict(zip(names, colors))
