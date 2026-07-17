# End-to-end VLA pick-and-place in MuJoCo with MolmoAct2 (learning project)

## Context
The goal is educational: get hands-on intuition for the full VLA (vision-language-action) deployment pipeline — sim ↔ camera ↔ policy server ↔ robot control loop — by building a pick-and-place demo rather than reading about one. The task: instruct a simulated arm to pick up a red box and place it into a green container (with distractor blue/other containers on the table), driven by Ai2's **MolmoAct2** running on a rented vast.ai GPU, with **MuJoCo** as the simulator. The user does not have a MuJoCo/robotics background yet, so the plan is structured as a sequence of small, testable milestones rather than one big build.

Research done this session (grounded via web search, not assumption):
- MolmoAct2 (github.com/allenai/molmoact2) publicly ships two ready-made embodiment checkpoints/servers: **DROID** (single Franka arm, 2 cameras, 8-dim action = 7 joint angles + gripper) and **Bimanual YAM** (14-dim). It also ships a `FastAPI` inference server per embodiment (`examples/`, `/act` endpoint, `json_numpy` encoding) plus a matching HTTP client pattern — i.e. the exact "local robot, remote GPU brain" split we want is already the officially supported deployment shape.
- Inference is fast enough for closed-loop use (180ms–1.3s per call on an H100) and the model outputs **action chunks** (10–30 steps executed open-loop before the next inference call), so a network round-trip per chunk (not per sim step) is very tolerable — this is what makes a rented remote GPU practical instead of a hard blocker.
- bfloat16 inference needs <16GB VRAM for either checkpoint (fp32 needs 26–88GB) — an RTX 5090 (32GB) comfortably covers this with headroom to spare, even in fp32 for the YAM checkpoint; no need to rent an H100.
- MuJoCo Menagerie ships a high-fidelity **Franka Emika Panda** XML model (7-DoF + 2-finger gripper) — this matches the DROID checkpoint's action space exactly, which is why we're using this arm and checkpoint pairing instead of building a custom embodiment (which would require collecting data / fine-tuning — out of scope for a learning pass).

## Architecture
```
 ┌─────────────── your machine (local) ───────────────┐        ┌──── vast.ai rented GPU instance ────┐
 │  MuJoCo sim (Franka Panda + table + red box +       │  HTTP  │  MolmoAct2 DROID checkpoint          │
 │  green/blue/... containers)                         │ ─────► │  FastAPI server (examples/droid),    │
 │  - renders 2 camera views each control step          │  POST  │  port 8000, /act endpoint             │
 │  - reads joint qpos + gripper state (proprioception) │ /act   │  input: {images, instruction, state}  │
 │  - applies returned action chunk via position        │ ◄───── │  output: action chunk (8,) x N steps  │
 │    actuators, steps physics, repeats                 │  JSON  │                                        │
 └──────────────────────────────────────────────────────┘        └────────────────────────────────────────┘
```
Fixed instruction for this exercise: `"pick up the red box and put it in the green container"`.

## Phased build plan

**Phase 0 — MuJoCo fundamentals (no policy yet)**
- Install `mujoco` Python bindings + viewer locally; clone MuJoCo Menagerie, load `franka_emika_panda` standalone, get comfortable with `qpos`/`qvel`/`ctrl`, position actuators, and the passive/interactive viewer.
- Milestone: manually drive the arm's joints/gripper via a small script and watch it move in the viewer.

**Phase 1 — Build the scene**
- Compose a scene XML: import the Panda model, add a table (plane/box geom), a red box (free joint, graspable size ~4–5cm), and 3 colored bins (green target + blue + one more distractor) as static containers.
- Add 2 cameras approximating the DROID setup (one external/overhead, one wrist-mounted on the Panda's end-effector body) since MolmoAct2-DROID expects 2 image inputs.
- Milestone: render both camera views to image files/window; confirm red box and bins are visible and distinguishable.

**Phase 2 — Stand up the MolmoAct2 server on vast.ai**
- Rent a vast.ai instance with an RTX 5090 (32GB VRAM), CUDA 12.1+, Python 3.11. Blackwell is newer than what MolmoAct2's pinned PyTorch/CUDA build may officially target, so confirm the `pyproject.toml`'s torch/CUDA version has Blackwell (sm_120) wheels available before renting — if not, install a matching nightly/CUDA-12.4+ torch build instead of the pinned one.
- Clone `allenai/molmoact2`, install with `uv` per its `pyproject.toml`, run the DROID `examples/` FastAPI server (port 8000) in bfloat16.
- Expose the port (SSH tunnel or vast.ai port mapping) and smoke-test with a raw `curl`/small script sending a dummy image+instruction+state payload, confirming an action chunk comes back.
- Milestone: successful `/act` round trip from your local machine to the rented GPU with synthetic (non-sim) test data.

**Phase 3 — Wire the closed-loop client**
- Write the control loop: read proprioception (7 joint angles + gripper) from MuJoCo, render both cameras, POST `{images, instruction, state}` to the server, receive the action chunk, apply each step of the chunk as position-actuator targets while stepping physics, then re-render and re-query for the next chunk.
- Log episode frames/video so you can visually judge grasp/placement success.
- Milestone: first full end-to-end attempt at the pick-and-place instruction, success or not.

**Phase 4 — Iterate**
- Tune camera placement/FOV to better match DROID's training distribution, check action scaling/units match what the position actuators expect, adjust re-planning cadence (query every chunk vs. more/less frequently), and try a few box/bin layout variations.
- Stretch: swap in the SO-100/SO-101 embodiment or try the Bimanual YAM checkpoint once the single-arm loop works end to end.

## Key references
- `allenai/molmoact2` GitHub repo — checkpoints, `examples/` FastAPI servers, client pattern, `pyproject.toml` deps.
- MuJoCo Menagerie — `franka_emika_panda` model directory.
- `mujoco` Python package docs — `qpos`/`ctrl`/renderer/viewer APIs.

## Verification
- Phase 0/1: visually confirm in the MuJoCo viewer/rendered frames that the arm moves as commanded and the scene (box + bins) looks correct.
- Phase 2: verify with a direct HTTP call (curl or a short script) that the rented server returns a well-formed action array before wiring it into the sim loop.
- Phase 3/4: judge success qualitatively per episode (did the red box end up in the green container?) via saved video/frames; no formal test suite needed for a learning exercise like this.
