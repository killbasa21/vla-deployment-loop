"""
Phase 3: closed-loop MolmoAct2-DROID control of the MuJoCo Panda pick-and-place scene.

Every outer iteration:
  1. Render both cameras (external_cam, wrist_cam).
  2. Read proprioception: 7 arm joint angles + a gripper-opening proxy, packed into
     MolmoAct2's (8,) state format.
  3. POST {images, instruction, state} to the remote /act endpoint (direct external port,
     no SSH tunnel) -> get back an action chunk, shape (N, 8).
  4. Apply each of the N actions in turn: set data.ctrl, step physics once, re-render and
     save the external_cam frame so the episode can be inspected afterward.

Usage:
    uv run python phase3_closed_loop.py --dry-run          # just check the server round trip
    uv run python phase3_closed_loop.py --chunks 5          # actually run the sim, 5 chunks
"""

import argparse
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import json_numpy
import mujoco
import mujoco.viewer
import numpy as np
import requests
from PIL import Image

json_numpy.patch()

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene_pick_place.xml"
SERVER_URL = "http://1.193.138.57:34011/act"
INSTRUCTION = "pick up the red box and put it in the green container"

# MolmoAct2's own image processor resizes every input image to 378x378 internally
# (see processor_config.json). We used to render much smaller than that (128x128)
# specifically to keep request payloads under a size threshold where the old vast.ai
# instance's China-routed network path silently stalled (~500KB-1.8MB). Now serving
# from Modal, that constraint is gone -- rendering at 128 and letting the processor
# upscale to 378 was throwing away real detail (box edges, gripper geometry) that a
# native 378x378 render preserves, so match the model's own target size exactly:
# no upload-bandwidth waste (it would still resize down from anything larger) and no
# avoidable quality loss from resizing up from something smaller.
RENDER_HEIGHT = 378
RENDER_WIDTH = 378

# fingers_actuator's ctrl range (0-255) maps linearly onto right_driver_joint/
# left_driver_joint's 0-0.8 radian range -- see the comment in panda.xml (copied from
# 2f85.xml): "scale = 0.8 * 100 / 255".
GRIPPER_CTRL_MAX = 255.0

# MolmoAct2-DROID's gripper axis is the *raw radian position of a Robotiq 2F-85 knuckle
# joint* on the FR3 arm it was trained on (confirmed by reading sim_eval/inference/
# common.py's droid_state_adapter and DroidClient.action_adapter=None -- no
# normalization at all, unlike the YAM checkpoint's [0,1] convention). panda.xml now
# mounts an actual Robotiq 2F-85 (mujoco_menagerie/robotiq_2f85) in place of the stock
# Franka hand specifically so this is no longer a cross-gripper proxy conversion --
# right_driver_joint's own range is 0-0.8 radians, the same units and (confirmed by
# rendering ctrl=0 vs ctrl=255 and visually comparing finger spacing) the same
# direction DROID's real gripper uses: 0 = open, 0.8 = closed. That direction also
# matches Robotiq's own documented convention for this register (0=open, max=closed),
# so this number is a real physical constant now, not a guess.
ROBOTIQ_KNUCKLE_CLOSED_MAX = 0.8  # radians; 0 = open, 0.8 = closed


def render_cameras(renderer, data):
    renderer.update_scene(data, camera="external_cam")
    ext = renderer.render()
    renderer.update_scene(data, camera="wrist_cam")
    wrist = renderer.render()
    return ext, wrist


def read_state(model, data):
    """7 arm joint angles + the real Robotiq knuckle angle, packed into MolmoAct2's
    (8,) state format. right_driver_joint's qpos is already in the exact units DROID
    expects (0=open..0.8=closed radians) since this is now the same gripper hardware --
    no proxy/rescale needed."""
    arm = data.qpos[0:7].copy()
    driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
    driver_qposadr = model.jnt_qposadr[driver_id]
    gripper_rad = data.qpos[driver_qposadr]
    return np.concatenate([arm, [gripper_rad]]).astype(np.float32)


def apply_action(data, action):
    """action is (8,): 7 arm joint targets (radians) + 1 DROID-space gripper value
    (0=open..0.8=closed radians, see ROBOTIQ_KNUCKLE_CLOSED_MAX)."""
    data.ctrl[0:7] = action[0:7]
    gripper_frac = np.clip(action[7] / ROBOTIQ_KNUCKLE_CLOSED_MAX, 0.0, 1.0)
    data.ctrl[7] = gripper_frac * GRIPPER_CTRL_MAX


def query_server(ext_img, wrist_img, state, server_url, send_wrist=True, timeout=30):
    """send_wrist=False still fills the required wrist_cam key (the server's schema
    demands it) but with a blank image, so the model effectively gets no real wrist
    camera information -- lets us compare behavior with/without that input."""
    payload = {
        "external_cam": ext_img,
        "wrist_cam": wrist_img if send_wrist else np.zeros_like(wrist_img),
        "instruction": INSTRUCTION,
        "state": state,
    }
    body_str = json_numpy.dumps(payload)
    resp = requests.post(
        server_url,
        data=body_str,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    actions = np.asarray(body["actions"], dtype=np.float32)
    return actions, body["dt_ms"], len(body_str)


def capture_and_submit(executor, renderer, data, model, box_body_id, server_url, send_wrist, timeout):
    """Render + read state *now* (must happen on the main thread -- mujoco.Renderer's
    GL context isn't safe to touch from a worker thread) and hand the actual HTTP
    call off to the background thread pool. Returns (future, submit_time, obs_snapshot)
    so the caller can log exactly what observation this request was based on."""
    ext_img, wrist_img = render_cameras(renderer, data)
    state = read_state(model, data)
    obs_snapshot = {
        "arm_qpos": state[:7].tolist(),
        "gripper_proxy": float(state[7]),
        "red_box_xpos": data.xpos[box_body_id].tolist(),
        "ctrl_before": data.ctrl.tolist(),
    }
    t0 = time.time()
    future = executor.submit(
        query_server, ext_img, wrist_img, state, server_url,
        send_wrist=send_wrist, timeout=timeout,
    )
    return future, t0, obs_snapshot


def build_sim():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    # Same keyframe zero-padding fix as phase1_render_check.py: the "home" keyframe
    # predates red_box, so its qpos got zero-padded for it. Set the box's pose by hand.
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
    box_jnt_id = model.body_jntadr[box_body_id]
    qpos_adr = model.jnt_qposadr[box_jnt_id]
    data.qpos[qpos_adr : qpos_adr + 3] = [0.5, 0, 0.02]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    return model, data


class _NullViewer:
    """Stand-in for mujoco.viewer's context manager when --no-view is passed."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def is_running(self):
        return True

    def sync(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="query the server and print the response, but never touch data.ctrl",
    )
    parser.add_argument(
        "--chunks", type=int, default=5,
        help="how many action chunks to request (0 or negative = run indefinitely, "
             "until the viewer window is closed or the process is killed)",
    )
    parser.add_argument("--save-dir", default="phase3_frames", help="where to save external_cam frames")
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="disable the live local viewer window (useful for background/headless runs)",
    )
    parser.add_argument(
        "--log-file",
        default="phase3_run_log.jsonl",
        help="JSON-lines file recording env state, model output, and network info per chunk",
    )
    parser.add_argument(
        "--no-wrist-to-model",
        action="store_true",
        help=(
            "still render the wrist camera (saved to <save-dir>/wrist for visualization) "
            "but send a blank image in its place to the model, instead of the real feed"
        ),
    )
    parser.add_argument(
        "--server-url",
        default=SERVER_URL,
        help=(
            "override the /act endpoint (default: the vast.ai instance in SERVER_URL above). "
            "Point this at a Modal deployment's URL (see phase3_modal.py) to run against that instead."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30,
        help=(
            "HTTP read timeout in seconds (default: 30). A cold Modal container's first "
            "request pays for checkpoint download + model load + warmup, which can take "
            "several minutes -- bump this for that first call."
        ),
    )
    args = parser.parse_args()
    send_wrist = not args.no_wrist_to_model

    model, data = build_sim()
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")

    save_dir = Path(args.save_dir)
    wrist_save_dir = save_dir / "wrist"
    if not args.dry_run:
        save_dir.mkdir(exist_ok=True)
        wrist_save_dir.mkdir(exist_ok=True)

    log_f = open(args.log_file, "w")

    # The live viewer runs locally (it's just a GLFW window on this machine, same as
    # phase0/phase1) and is independent of the offscreen `renderer` above, which renders
    # the camera images we actually send to the server. Both can point at the same
    # `data` at once with no conflict -- viewer.sync() just redraws from whatever qpos
    # currently holds.
    viewer_cm = _NullViewer() if args.no_view else mujoco.viewer.launch_passive(model, data)

    frame_idx = 0
    # Only the non-dry-run path pipelines requests: dry-run never advances the
    # sim, so there's no "apply chunk" work to overlap the next request behind --
    # it stays a simple serial loop, unchanged from before.
    executor = None if args.dry_run else ThreadPoolExecutor(max_workers=1)
    pending = None

    with viewer_cm as viewer:
        if not args.no_view:
            # Viewer-only visualization: draws camera icons/frustums as a UI overlay in
            # the interactive window. Purely cosmetic for us to see camera placement --
            # it is NOT part of the model geometry, so mujoco.Renderer (the offscreen
            # renderer producing the images we actually send to MolmoAct2) never sees
            # it. Zero risk of this affecting the model's input.
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True

        if executor is not None:
            pending = capture_and_submit(
                executor, renderer, data, model, box_body_id,
                args.server_url, send_wrist, args.request_timeout,
            )

        chunk_iter = itertools.count() if args.chunks <= 0 else range(args.chunks)
        for chunk_i in chunk_iter:
            if not viewer.is_running():
                print("Viewer window closed, stopping.")
                break

            if args.dry_run:
                # Environment state *before* this chunk's (never-applied) actions --
                # exactly what read_state() packs into the request, plus a couple of
                # extra quantities (box position, full ctrl) useful for offline
                # analysis but not sent to the server.
                state = read_state(model, data)
                env_state_before = {
                    "arm_qpos": state[:7].tolist(),
                    "gripper_proxy": float(state[7]),
                    "red_box_xpos": data.xpos[box_body_id].tolist(),
                    "ctrl_before": data.ctrl.tolist(),
                }
                ext_img, wrist_img = render_cameras(renderer, data)
                t0 = time.time()
                actions, dt_ms, request_bytes = query_server(
                    ext_img, wrist_img, state, args.server_url,
                    send_wrist=send_wrist, timeout=args.request_timeout,
                )
                round_trip_ms = 1000 * (time.time() - t0)
            else:
                # `pending` was submitted either before the loop (chunk 0) or after
                # the first action of the *previous* chunk (see below) -- by the time
                # we get here it's often already finished, since its ~2s network +
                # inference round trip has been running in the background while we
                # stepped through the rest of the previous chunk's actions.
                future, t0, env_state_before = pending
                actions, dt_ms, request_bytes = future.result()
                round_trip_ms = 1000 * (time.time() - t0)

            print(
                f"chunk {chunk_i}: got actions{actions.shape} "
                f"(server dt={dt_ms:.1f}ms, round trip={round_trip_ms:.1f}ms)"
            )

            log_entry = {
                "chunk": chunk_i,
                "timestamp": time.time(),
                "env_state": env_state_before,
                "model_output": {
                    "actions": actions.tolist(),
                    "server_dt_ms": dt_ms,
                },
                "network": {
                    "url": args.server_url,
                    "request_bytes": request_bytes,
                    "round_trip_ms": round_trip_ms,
                    "wrist_cam_sent_to_model": send_wrist,
                },
            }

            if args.dry_run:
                print("  value range:", actions.min(), actions.max())
                print("  first action:", actions[0])
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.flush()
                continue

            for i, action in enumerate(actions):
                apply_action(data, action)
                mujoco.mj_step(model, data)
                viewer.sync()
                ext_frame, wrist_frame = render_cameras(renderer, data)
                Image.fromarray(ext_frame).save(save_dir / f"frame_{frame_idx:04d}.png")
                Image.fromarray(wrist_frame).save(wrist_save_dir / f"frame_{frame_idx:04d}.png")
                frame_idx += 1

                # Fire the next chunk's request right after the first action, using
                # the state as of *now* (one step fresher than the state this
                # chunk's own request used) -- gives the background request the
                # rest of this chunk's local stepping/rendering time to complete in.
                if i == 0 and (args.chunks <= 0 or chunk_i + 1 < args.chunks):
                    pending = capture_and_submit(
                        executor, renderer, data, model, box_body_id,
                        args.server_url, send_wrist, args.request_timeout,
                    )

            print(f"  ctrl after chunk {chunk_i}: {np.array2string(data.ctrl, precision=3, suppress_small=True)}")

            log_entry["env_state"]["ctrl_after"] = data.ctrl.tolist()
            log_entry["env_state"]["red_box_xpos_after"] = data.xpos[box_body_id].tolist()
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

    if executor is not None:
        executor.shutdown(wait=False)
    log_f.close()

    if not args.dry_run:
        print(f"Saved {frame_idx} frames to {save_dir}/")


if __name__ == "__main__":
    main()
