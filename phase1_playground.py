"""
Phase 1 playground: drag box/bin positions around interactively, print them out.

Controls in the viewer window:
  - Double-click a body (box or a bin) to select it.
  - Ctrl + left-drag on a selected/hovered body -> translate it (this is the "drag and
    drop" part). Gravity is on, same as the real task scene, so once you let go the
    object falls/settles naturally against the floor or other objects -- it won't
    float mid-air. Drag it low enough to land where you want.
  - Ctrl + right-drag -> rotate it, if you also want to experiment with orientation.
  - Press "P" any time to print every movable object's current position to the
    terminal, already formatted as an XML pos="..." attribute ready to paste into
    scene_pick_place.xml.
  - Close the window to exit.
"""

import time

import glfw
import mujoco
import mujoco.viewer

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene_playground.xml"

# Every body we made draggable (freejoint in the XML) -- kept as an explicit list here
# so print_positions() below knows which bodies to report on.
MOVABLE_BODIES = ["red_box", "green_bin", "blue_bin", "yellow_bin"]

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home_id)

# Same fix as phase1_render_check.py: the "home" keyframe predates these bodies, so its
# qpos got zero-padded for all of them. Reset each one to the position we actually want
# to start from (matching scene_pick_place.xml), the same way we did for red_box there.
INITIAL_POSITIONS = {
    "red_box": (0.5, 0, 0.02),
    "green_bin": (0.5, 0.25, 0),
    "blue_bin": (0.5, -0.25, 0),
    "yellow_bin": (0.7, 0, 0),
}
for name, (x, y, z) in INITIAL_POSITIONS.items():
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    jnt_id = model.body_jntadr[body_id]
    qpos_adr = model.jnt_qposadr[jnt_id]
    data.qpos[qpos_adr : qpos_adr + 3] = [x, y, z]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)


def print_positions():
    print("--- current positions ---")
    for name in MOVABLE_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        # data.xpos is the body's *computed* world position (kept up to date by
        # mj_forward/mj_step every frame) -- more direct here than re-deriving it from
        # qpos ourselves, since mj_forward has already done that work for us.
        x, y, z = data.xpos[body_id]
        print(f'  {name}: pos="{x:.3f} {y:.3f} {z:.3f}"')


# key_callback lets us hook into the viewer's own keyboard handling. MuJoCo's viewer
# calls this function with a GLFW key code every time a key is pressed; we only care
# about one key (P), so everything else is ignored.
def key_callback(keycode):
    if keycode == glfw.KEY_P:
        print_positions()


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print("Drag the box/bins around (Ctrl+left-drag after double-click to select).")
    print("Press P to print current positions. Close the window to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
