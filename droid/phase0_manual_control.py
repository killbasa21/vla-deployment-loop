"""
Phase 0, step 2: manual control via the viewer's own UI, no scripted motion.

Unlike droid/phase0_hello_panda.py, this script never touches data.ctrl itself after
the initial reset -- it just holds the home pose and steps physics, so YOU can
drive the robot instead:

  - Open the right UI panel (Tab / Shift+Tab if it's not already visible) and find
    the "Control" tab: one slider per actuator (actuator1..7 = arm joints,
    actuator8 = gripper, 0=open/255=closed). Dragging a slider sets data.ctrl
    for that actuator directly -- nothing in this script will overwrite it.
  - Or skip sliders entirely: Ctrl+left-drag on the arm/end-effector to shove it
    with a mouse-applied force, Ctrl+right-drag to apply a torque.
  - Space pauses/unpauses physics if you want to freeze a pose and inspect it.

Close the window to exit.
"""

import time

import mujoco
import mujoco.viewer

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home_id)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # No data.ctrl writes here -- whatever ctrl currently holds (from the
        # keyframe, or from you moving a Control-tab slider) is what gets applied.
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
