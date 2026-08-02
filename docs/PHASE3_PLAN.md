# Phase 3 — Wire the closed-loop client

## Context
Phase 1 built the local simulation: `mujoco_menagerie/franka_emika_panda/scene_pick_place.xml` (Panda arm, red box, green/blue/yellow bins) with two working cameras (`external_cam`, `wrist_cam`), verified by rendering and visually inspecting both. Phase 2 stood up MolmoAct2's DROID checkpoint as a FastAPI server (`/act` endpoint, port 8000) on a rented vast.ai RTX 5090 instance, verified with a synthetic HTTP round trip.

Phase 3 connects the two: a Python script running locally that, every control step, renders both cameras, reads the arm's proprioception, sends all of it to the remote `/act` endpoint, and applies the returned action back into the sim — the actual closed loop the whole project has been building toward.

## Before writing any loop code: resolve the units question
MolmoAct2-DROID's action/state format is documented as `(8,) = [q1..q7, gripper]`, but the **gripper value's units** (0-1 normalized? raw jaw width in meters? something else?) aren't nailed down from public docs alone. Don't guess this — once `allenai/molmoact2` is cloned (Phase 2), read the actual client example (`examples/droid/*client*.py` or similar) to see exactly how their own reference client encodes/decodes the gripper value before writing our conversion code. This mirrors the Phase 1 lesson (render-and-check beat guessing coordinates) — here the equivalent is "read their client code and check" rather than assume a convention.

Also confirm from that same client code: the exact `json_numpy`-encoded request schema (key names for images/instruction/state), and whether DRIOD expects joint angles in the same order/sign convention our `qpos[0:7]` already uses (Panda joint order should match, since both are describing the same physical robot, but verify rather than assume).

## Step-by-step

**1. Observation capture function**
- Render `external_cam` and `wrist_cam` via `mujoco.Renderer` (same pattern as `droid/phase1_render_check.py`), returning two `(H, W, 3)` uint8 arrays.
- Read proprioception: `data.qpos[0:7]` for the 7 arm joint angles, plus the gripper's current opening — pull this from `data.ctrl[7]` (our own last commanded gripper target) or from the finger joints' actual `qpos`, converted into whatever units Phase 3's step 0 determined MolmoAct2 expects.

**2. Client request/response**
- Reuse MolmoAct2's own client-side encoding helper if the repo exposes one, instead of hand-rolling the HTTP/`json_numpy` payload — less surface area for a units/schema mismatch.
- POST `{images: [...], instruction: "pick up the red box and put it in the green container", state: [...]}` to `http://<server-host>:8000/act` (via SSH tunnel or vast.ai's exposed port from Phase 2).
- Response is an action chunk: an array of shape `(N, 8)`, N somewhere in 10-30 per MolmoAct2's docs.

**3. Applying the chunk**
- For each of the N actions in the chunk: set `data.ctrl[0:7]` from the returned joint angles, convert the returned gripper value into our actuator8's `0-255` range (this conversion is the other concrete thing to verify empirically — command a known value, render, and visually confirm the gripper actually opens/closes the amount expected, the same way Phase 1 verified camera placement by rendering rather than trusting the math blind).
- Step physics (`mj_step`) once per action in the chunk, re-rendering each step if you want to save video, before requesting the next chunk.

**4. Loop and logging**
- Outer loop: capture observation → request chunk → apply chunk → repeat, until some step limit or manual stop.
- Save rendered frames (e.g. append `external_cam` frames to a list, write out as a short video or image sequence at the end) so success/failure can be judged visually per episode — no formal pass/fail test needed for this learning exercise.

## Key files
- `mujoco_menagerie/franka_emika_panda/scene_pick_place.xml` — the scene (already built)
- `droid/phase1_render_check.py` — reference pattern for offscreen rendering + the keyframe-padding fix for freejoint bodies (same fix will be needed here since we're reading/writing `data.qpos` directly again)
- MolmoAct2 repo's `examples/droid/` client code — reference for exact request schema and unit conventions (read before writing our own client, per the "resolve the units question" section above)

## Verification
- First, a dry run with the loop calling the server but NOT applying the response (just print the returned action chunk's shape and value ranges) — confirms the request/response plumbing works before touching the sim.
- Then enable applying actions, run one full episode, and inspect the saved frames: did the arm move toward the box, close the gripper near it, and move toward the green bin? Full task success isn't the bar for the first attempt — plausible, directed motion is enough to confirms the loop is wired correctly; tuning (Phase 4) is where actual task success gets chased.
