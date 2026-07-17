"""
Phase 0, step 1: load the Franka Panda, watch it move.

Run it, then in the viewer window: left-drag to orbit, scroll to zoom, right-drag to pan,
Ctrl+left-drag to apply a force to a body (try dragging the end effector).
"""

import time

import mujoco          # the physics engine's Python bindings (loading models, stepping physics)
import mujoco.viewer   # a separate module: the interactive 3D viewer window
import numpy as np     # just for np.sin() below, to generate a smooth oscillation

# Path to the scene file. scene.xml is a small XML wrapper that <include>s panda.xml
# (the arm+gripper itself) and additionally adds a floor plane, lighting, and a skybox --
# things panda.xml doesn't define because they're scene-specific, not robot-specific.
MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene.xml"

# --- Loading the model -------------------------------------------------------------
#
# mujoco.MjModel.from_xml_path() parses the XML and builds "model": a description of
# everything that does NOT change while the simulation runs -- how many bodies/joints/
# actuators exist, their masses, their gear ratios, their control ranges, etc. Think of
# it as the *blueprint*.
model = mujoco.MjModel.from_xml_path(MODEL_PATH)

# mujoco.MjData allocates the *mutable* simulation state that DOES change every physics
# step: joint positions/velocities, actuator control signals, contact forces, etc.
# You always need exactly one MjData per MjModel you're simulating.
# Analogy: model = the class definition, data = an instance of it.
data = mujoco.MjData(model)

# --- Setting a sane starting pose ---------------------------------------------------
#
# If we didn't do this, every joint and actuator would start at 0, which for this robot
# is actually an invalid/awkward pose (e.g. joint4's allowed range is -3.07 to -0.07, so
# 0 is out of range and MuJoCo would clamp it oddly). The Panda's XML author anticipated
# this and baked in a named <keyframe> called "home" with a good starting pose -- we just
# need to look it up by name and load it.

# mj_name2id looks up the *integer index* of an object given its type and name. Almost
# everything in MuJoCo (bodies, joints, actuators, keyframes...) is stored in flat arrays
# internally and referenced by these integer IDs, so name-based lookup is a convenience
# layer on top -- you'll see this "name2id" pattern reused constantly.
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

# This copies the keyframe's saved qpos (joint positions) AND ctrl (actuator targets)
# into our `data` object, resetting the simulation to that pose.
mujoco.mj_resetDataKeyframe(model, data, home_id)

# --- Inspecting actuators -------------------------------------------------------------
#
# data.ctrl is a plain numpy array, one entry per actuator, in the order they were
# declared in the XML. There's no name attached to each entry at runtime -- so we print
# the names here (looked up from `model`, which does track names) purely so *we* know
# ctrl[0] means "actuator1" etc. when reading/writing this array by index below.
#
# model.nu = "number of actuators" (the standard MuJoCo naming: nu, nq, nv, ... are
# short for "number of {u=controls, q=positions, v=velocities}").
print("Actuator names:", [model.actuator(i).name for i in range(model.nu)])

# .copy() matters here: data.ctrl is a live view into MuJoCo's internal buffer, so if we
# just wrote `home_ctrl = data.ctrl` (no copy), home_ctrl would keep changing underneath
# us every time we later mutate data.ctrl in the loop below, instead of staying frozen
# at the home values.
home_ctrl = data.ctrl.copy()
print("Home ctrl:", home_ctrl)

# --- The simulation loop --------------------------------------------------------------
#
# launch_passive() opens the interactive 3D window but does NOT step physics for you or
# block your code -- "passive" means the viewer is passive/along-for-the-ride, and *we*
# drive the simulation loop ourselves. (The alternative, mujoco.viewer.launch(), takes
# over the loop entirely and is meant for just eyeballing a static model, not for a
# script that computes its own control signals every step like this one.)
#
# The `with` block ensures the viewer window is cleanly closed when we exit the loop.
with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()

    # viewer.is_running() goes False once you close the window, so closing it is how
    # you stop this script (instead of Ctrl+C in the terminal).
    while viewer.is_running():
        t = time.time() - t0  # seconds elapsed since we started, used as our "clock"

        # Reset every actuator to its home target, then override just the two we want
        # to move. Doing a full reset each iteration is a cheap way to guarantee every
        # OTHER joint just holds its home position steady, rather than drifting.
        data.ctrl[:] = home_ctrl

        # data.ctrl[1] is "actuator2" (0-indexed), which drives joint2, the shoulder.
        # We add a sine wave on top of its home value: amplitude 0.4 radians, and
        # 0.5 rad/s inside the sine controls how *fast* it oscillates (angular frequency).
        data.ctrl[1] = home_ctrl[1] + 0.4 * np.sin(0.5 * t)

        # data.ctrl[3] is "actuator4", driving joint4, the elbow. Same idea, but phase-
        # shifted by pi/2 (90 degrees) so the elbow and shoulder aren't moving in lockstep
        # -- purely to make the motion look less robotic/uniform, no deeper meaning.
        data.ctrl[3] = home_ctrl[3] + 0.4 * np.sin(0.5 * t + np.pi / 2)

        # data.ctrl[7] is "actuator8", the gripper. Unlike the arm joints (which are
        # continuous PD position targets), this one's XML declares ctrlrange="0 255"
        # mapped onto the physical finger-opening tendon: 0 = fully open, 255 = fully
        # closed. We just flip it fully open/closed every ~10s (2*pi / 0.3 ≈ 21s per
        # full cycle) based on the sign of a slower sine wave -- no gradual gripping here,
        # just enough to see the fingers move.
        data.ctrl[7] = 255

        # mj_step is the actual physics update: given the current data.ctrl, it computes
        # forces (including each actuator's internal PD controller pulling qpos toward
        # ctrl), integrates one timestep forward, and writes the new qpos/qvel back into
        # `data`. This is the single most important function call in all of MuJoCo --
        # everything else in this script exists to set up ctrl before this call, or to
        # observe/render the results after it.
        mujoco.mj_step(model, data)

        # mj_step only updates `data`; it does NOT redraw the window. viewer.sync()
        # pushes the latest data into the render window so you actually see the motion.
        viewer.sync()

        # model.opt.timestep is the simulation timestep declared in the XML (how much
        # simulated time one mj_step advances, e.g. 0.002s = 2ms). Sleeping for that long
        # keeps our *wall-clock* loop roughly in sync with real time, so the arm appears
        # to move at realistic speed instead of as fast as the CPU can churn through
        # mj_step calls.
        time.sleep(model.opt.timestep)
