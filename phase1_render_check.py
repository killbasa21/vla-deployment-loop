"""Render a camera view to a PNG so we can visually check scene layout.

Usage: uv run python phase1_render_check.py [camera_name] [output_path.png]
Both arguments are optional -- defaults to the external_cam and render_check.png.
"""

import sys

import mujoco
from PIL import Image

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene_pick_place.xml"

# sys.argv is the list of command-line arguments; argv[0] is always the script's own
# path, so argv[1]/argv[2] are the first two arguments *we* pass in. The `if len(...)`
# guards let you run the script with zero, one, or two arguments and fall back to
# sensible defaults instead of crashing with an IndexError.
CAMERA = sys.argv[1] if len(sys.argv) > 1 else "external_cam"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "render_check.png"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home_id)

# The "home" keyframe was authored in panda.xml before red_box existed, so MuJoCo
# zero-padded its qpos to match our larger model (9 panda DOFs + 7 box freejoint DOFs
# = 16) -- that padding put the box's freejoint qpos at (0,0,0), not the (0.5, 0, 0.02)
# we declared as its body "pos". So we set the box's pose by hand here instead of
# trusting the keyframe for it.
#
# To do that we need the box's qpos *array index*, which takes 2 lookups because
# MuJoCo's data model is: body -> (which joint(s) it has) -> (where that joint's
# numbers live in the flat qpos array). mj_name2id gets us the body's index; from
# there body_jntadr tells us the index of its first joint (a freejoint counts as
# exactly one joint, just one that owns 7 qpos slots instead of 1); jnt_qposadr
# then tells us where in `data.qpos` that joint's numbers begin.
box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
box_jnt_id = model.body_jntadr[box_body_id]
qpos_adr = model.jnt_qposadr[box_jnt_id]

# A freejoint's 7 qpos values are always [x, y, z, qw, qx, qy, qz] -- 3 for position,
# 4 for orientation as a unit quaternion. [1, 0, 0, 0] is the identity quaternion
# (no rotation), since we don't care what angle the box starts at.
data.qpos[qpos_adr : qpos_adr + 3] = [0.5, 0, 0.02]  # x, y, z
data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]  # qw, qx, qy, qz

# mj_forward recomputes every derived quantity (body/geom world positions, contact
# points, etc.) from the current qpos, WITHOUT advancing time or touching qpos itself
# -- unlike mj_step, which also integrates physics forward by one timestep. We want the
# former here: we just hand-edited qpos and need the renderer to see the consequences
# of that edit, not to simulate anything moving.
mujoco.mj_forward(model, data)

# mujoco.Renderer is the *offscreen* rendering path (no window, unlike mujoco.viewer) --
# what you'd use for saving images/video, or for feeding pixels to a vision model.
renderer = mujoco.Renderer(model, height=480, width=640)

# update_scene renders from a specific camera's point of view, by name (matching the
# `name="..."` attribute in our <camera> tags in scene_pick_place.xml/panda.xml).
renderer.update_scene(data, camera=CAMERA)

# .render() returns a plain (height, width, 3) uint8 numpy array of RGB pixel values --
# exactly the format PIL's Image.fromarray expects, and also exactly the format you'd
# hand to a vision-language-action model later (an "image" input is just this array).
pixels = renderer.render()

Image.fromarray(pixels).save(OUT_PATH)
print(f"Saved {OUT_PATH} ({pixels.shape[1]}x{pixels.shape[0]}) from camera '{CAMERA}'")
