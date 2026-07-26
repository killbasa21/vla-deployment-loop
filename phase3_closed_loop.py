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
import os
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

DEFAULT_MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene_pick_place.xml"
# DROID-matching alternative: FR3 (not Panda) + Robotiq 2F-85, DROID's real rest pose, a table,
# and droid_put_everything_in_box.py's own object layout (lego duplo / tennis ball / open box)
# instead of this project's own red-box-and-bins task. See mujoco_menagerie/franka_fr3/
# fr3_robotiq.xml and scene_droid.xml for how each DROID spec (camera FOVs, gripper hardware,
# rest keyframe, box dimensions) was carried over vs. where it had to be re-derived (camera
# placement, table/pedestal geometry -- ManiSkill's own TableSceneBuilder mesh isn't vendored
# here) and verified by offscreen rendering instead.
DROID_MODEL_PATH = "mujoco_menagerie/franka_fr3/scene_droid.xml"
SERVER_URL = "http://1.193.138.57:34011/act"
INSTRUCTION = "pick up the green ball and put it in the green container"

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
    dt_ms = body.get("dt_ms", body.get("latency_ms"))
    if dt_ms is None:
        dt_ms = 1000 * resp.elapsed.total_seconds()
    return actions, float(dt_ms), len(body_str)


def throttle_query_rate(last_query_time, interval_s):
    """Block until at least `interval_s` seconds have passed since last_query_time (a
    time.time() value, or None on the very first call -- nothing to wait out yet). Call
    right before each new image-capture-and-/act-call, on both the pipelined path
    (capture_and_submit) and the sequential --dry-run path, so a fresh request only ever
    fires once per interval regardless of how fast the previous round trip actually was
    (previously: back-to-back, paced only by network/inference latency, ~3-6s in practice).
    Returns the new last_query_time to pass into the next call."""
    if last_query_time is not None and interval_s > 0:
        remaining = interval_s - (time.time() - last_query_time)
        if remaining > 0:
            time.sleep(remaining)
    return time.time()


def capture_and_submit(executor, renderer, data, model, tracked_body_id, server_url, send_wrist, timeout):
    """Render + read state *now* (must happen on the main thread -- mujoco.Renderer's
    GL context isn't safe to touch from a worker thread) and hand the actual HTTP
    call off to the background thread pool. Returns (future, submit_time, obs_snapshot)
    so the caller can log exactly what observation this request was based on."""
    ext_img, wrist_img = render_cameras(renderer, data)
    state = read_state(model, data)
    obs_snapshot = {
        "arm_qpos": state[:7].tolist(),
        "gripper_proxy": float(state[7]),
        "tracked_object_xpos": data.xpos[tracked_body_id].tolist() if tracked_body_id is not None else None,
        "ctrl_before": data.ctrl.tolist(),
    }
    t0 = time.time()
    future = executor.submit(
        query_server, ext_img, wrist_img, state, server_url,
        send_wrist=send_wrist, timeout=timeout,
    )
    return future, t0, obs_snapshot


# Every scene's graspable props are freejoint bodies whose world pose lives in qpos -- but the
# "home" keyframe in each scene's XML predates them (see phase1_render_check.py's original
# comment on red_box), so mj_resetDataKeyframe zero-pads their qpos to the origin instead of
# their intended resting pose. Same fix, generalized across scenes: look up whichever of these
# bodies the loaded model actually has (scene_pick_place.xml has red_box; scene_droid.xml has
# lego_duplo + tennis_ball) and set each one's freejoint qpos by hand. Positions match each
# scene's own XML comments for why these numbers -- EXCEPT red_box, see below.
FREE_BODY_INITIAL_POSES = {
    # NOT the XML's own default body pos ("0.5 0 0.02", that was phase3's original red-box-task
    # spot). This is where phase4_collect_demos.py's reset_scene() actually parks red_box as a
    # static distractor on every single training episode -- match that exactly, since that's the
    # position the green_ball_pick fine-tune's images actually show it at.
    "red_box": ((0.5, -0.28, 0.02), (1, 0, 0, 0)),
    "green_ball": ((0.45, -0.1, 0.02), (1, 0, 0, 0)),
    "lego_duplo": ((0.45, 0.20, 0.4095), (1, 0, 0, 0)),
    "tennis_ball": ((0.45, -0.20, 0.433), (1, 0, 0, 0)),
}

# green_ball's training-time xy distribution -- phase4_collect_demos.py's sample_ball_xy().
# --randomize-ball draws from this same range so the sim replicates the training distribution
# instead of always using FREE_BODY_INITIAL_POSES' single fixed spot.
GREEN_BALL_XY_RANGE = ((0.40, 0.58), (-0.16, 0.14))

# Where --hide-red-box parks red_box: far outside both cameras' frustums and below the floor,
# so it doesn't just sit off to the side still visible in a wide shot. Physics-inert once here
# (nothing else reaches into this region), so it's effectively "removed" without recompiling
# the XML to drop the body.
RED_BOX_HIDDEN_POSE = ((5.0, 5.0, -5.0), (1, 0, 0, 0))

# The lora_train fine-tune (data/lora_adapter/) additionally randomizes which of the 3 bins
# (green/blue/yellow) sits at which of these 3 anchor slots, per episode -- see
# phase4_collect_demos.py's randomize_bin_layout(). scene_pick_place.xml's own default body_pos
# for these three bodies is exactly this identity assignment (green->slot0, blue->slot1,
# yellow->slot2, zero jitter) -- i.e. ONE single point in the training distribution. Evaluating
# the LoRA checkpoint without also randomizing bins would only ever test that one layout, not
# the visual-grounding generalization this fine-tune was meant to fix. --randomize-bins draws a
# fresh permutation + jitter the same way training did.
BIN_SLOT_POSITIONS = [(0.5, 0.25, 0.0), (0.5, -0.25, 0.0), (0.7, 0.0, 0.0)]
BIN_SLOT_JITTER = 0.02
BIN_NAMES = ("green_bin", "blue_bin", "yellow_bin")

# Which of FREE_BODY_INITIAL_POSES' bodies (in priority order) gets its pose logged/returned as
# this run's single "tracked object" -- mirrors the old red_box-only tracking, just extended to
# pick the first match present in whichever scene got loaded rather than hardcoding red_box.
# green_ball first: it's the actual Phase 4 fine-tune target (see phase4_collect_demos.py),
# scene_pick_place.xml has both it and red_box (a distractor there), so it must outrank red_box.
TRACKED_OBJECT_CANDIDATES = ["green_ball", "red_box", "lego_duplo"]


def build_sim(model_path, randomize_ball=False, hide_red_box=False, randomize_bins=False, rng=None):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    poses = dict(FREE_BODY_INITIAL_POSES)
    if randomize_ball:
        rng = rng or np.random.default_rng()
        (x_lo, x_hi), (y_lo, y_hi) = GREEN_BALL_XY_RANGE
        ball_xy = (rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi))
        poses["green_ball"] = ((ball_xy[0], ball_xy[1], 0.02), (1, 0, 0, 0))
    if hide_red_box:
        poses["red_box"] = RED_BOX_HIDDEN_POSE

    for name, (pos, quat) in poses.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            continue
        jnt_id = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr : qpos_adr + 3] = pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

    if randomize_bins:
        rng = rng or np.random.default_rng()
        order = rng.permutation(len(BIN_SLOT_POSITIONS))
        for name, slot_idx in zip(BIN_NAMES, order):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                continue
            slot = np.array(BIN_SLOT_POSITIONS[slot_idx])
            jitter = rng.uniform(-BIN_SLOT_JITTER, BIN_SLOT_JITTER, size=2)
            model.body_pos[body_id] = slot + [jitter[0], jitter[1], 0.0]

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
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="root dir for per-run debug artifacts: <assets-dir>/logs/<run-id>.jsonl "
             "and <assets-dir>/images/<run-id>/ (see CLAUDE.md)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="identifier for this run's logs/images subtree (default: timestamp_pid, "
             "so concurrent runs never collide)",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="disable the live local viewer window (useful for background/headless runs)",
    )
    parser.add_argument(
        "--no-wrist-to-model",
        action="store_true",
        help=(
            "still render the wrist camera (saved to <assets-dir>/images/<run-id>/ for "
            "visualization) but send a blank image in its place to the model, instead of "
            "the real feed"
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
    parser.add_argument(
        "--query-interval",
        type=float, default=10.0,
        help=(
            "minimum seconds between successive model queries (image capture + /act call), "
            "default 10 -- even if the previous round trip finished faster, the next capture "
            "waits out the rest of this interval first. 0 disables throttling (fire "
            "back-to-back, paced only by network/inference latency)."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            f"path to the MuJoCo scene XML to load (default: {DEFAULT_MODEL_PATH}, this "
            f"project's own Panda pick-and-place task). Pass '{DROID_MODEL_PATH}' (or its "
            "shorthand 'droid') for the FR3+Robotiq/table/DROID-object scene instead -- see "
            "that path's own XML comments for what it does and doesn't carry over from DROID's "
            "real rig spec."
        ),
    )
    parser.add_argument(
        "--randomize-ball",
        action="store_true",
        help=(
            "place green_ball at a random xy drawn from the same range "
            "phase4_collect_demos.py's sample_ball_xy() used for training "
            f"({GREEN_BALL_XY_RANGE}), instead of its single fixed default spot"
        ),
    )
    parser.add_argument(
        "--hide-red-box",
        action="store_true",
        help=(
            "teleport red_box out of both camera frustums instead of its training-matching "
            "distractor spot -- an ablation run with no red box in view at all"
        ),
    )
    parser.add_argument(
        "--randomize-bins",
        action="store_true",
        help=(
            "shuffle green/blue/yellow bin positions across the same 3 anchor slots (with "
            "jitter) phase4_collect_demos.py's randomize_bin_layout() used for the lora_train "
            "fine-tune, instead of scene_pick_place.xml's single fixed default layout"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int, default=None,
        help="seed for --randomize-ball/--randomize-bins' rng, for reproducibility",
    )
    args = parser.parse_args()
    send_wrist = not args.no_wrist_to_model
    model_path = DROID_MODEL_PATH if args.model_path == "droid" else args.model_path
    rng = (
        np.random.default_rng(args.seed)
        if (args.randomize_ball or args.randomize_bins) else None
    )

    model, data = build_sim(
        model_path,
        randomize_ball=args.randomize_ball,
        hide_red_box=args.hide_red_box,
        randomize_bins=args.randomize_bins,
        rng=rng,
    )
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    tracked_body_id = None
    for name in TRACKED_OBJECT_CANDIDATES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            tracked_body_id = body_id
            break

    # Every run gets its own subtree under assets/ (see CLAUDE.md), keyed by run_id, so
    # concurrent/successive runs never clobber each other's logs or frames and the two
    # can be cross-referenced by run_id while a run is still in progress.
    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    assets_dir = Path(args.assets_dir)
    log_dir = assets_dir / "logs"
    images_dir = assets_dir / "images" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{run_id}.jsonl"
    print(f"run_id: {run_id}")
    print(f"  log:    {log_path}")
    print(f"  images: {images_dir}/" if not args.dry_run else "  images: (skipped, --dry-run)")
    log_f = open(log_path, "w")

    # Key bindings to jump the *live viewer's* own camera to look through external_cam or
    # wrist_cam directly -- i.e. see exactly what gets captured/sent to the model, not just
    # the frustum icon overlay (mjVIS_CAMERA, set below) marking where they are. '0' returns
    # to the normal free/orbit camera. (The GLFW UI's own Camera dropdown, under the
    # right-hand "Rendering" panel -- on by default, see launch_passive's show_right_ui --
    # already lists both by name and does the same thing; this is just a faster shortcut.)
    # References `viewer_cm` before it's assigned below: fine, Python closures resolve names
    # at *call* time, and this callback is never invoked until launch_passive's window
    # actually exists and viewer_cm is bound.
    def key_callback(keycode):
        key = chr(keycode) if 0 <= keycode < 256 else ""
        if key == "0":
            viewer_cm.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            return
        cam_name = {"1": "external_cam", "2": "wrist_cam"}.get(key)
        if cam_name is None:
            return
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id != -1:
            viewer_cm.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer_cm.cam.fixedcamid = cam_id

    # The live viewer runs locally (it's just a GLFW window on this machine, same as
    # phase0/phase1) and is independent of the offscreen `renderer` above, which renders
    # the camera images we actually send to the server. Both can point at the same
    # `data` at once with no conflict -- viewer.sync() just redraws from whatever qpos
    # currently holds.
    viewer_cm = (
        _NullViewer() if args.no_view
        else mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    )
    if not args.no_view:
        print("  viewer camera: press 1=external_cam, 2=wrist_cam, 0=free/orbit")

    frame_idx = 0
    # Only the non-dry-run path pipelines requests: dry-run never advances the
    # sim, so there's no "apply chunk" work to overlap the next request behind --
    # it stays a simple serial loop, unchanged from before.
    executor = None if args.dry_run else ThreadPoolExecutor(max_workers=1)
    pending = None
    last_query_time = None  # see throttle_query_rate()

    with viewer_cm as viewer:
        if not args.no_view:
            # Viewer-only visualization: draws camera icons/frustums as a UI overlay in
            # the interactive window. Purely cosmetic for us to see camera placement --
            # it is NOT part of the model geometry, so mujoco.Renderer (the offscreen
            # renderer producing the images we actually send to MolmoAct2) never sees
            # it. Zero risk of this affecting the model's input.
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True

        if executor is not None:
            last_query_time = throttle_query_rate(last_query_time, args.query_interval)
            pending = capture_and_submit(
                executor, renderer, data, model, tracked_body_id,
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
                    "tracked_object_xpos": data.xpos[tracked_body_id].tolist() if tracked_body_id is not None else None,
                    "ctrl_before": data.ctrl.tolist(),
                }
                ext_img, wrist_img = render_cameras(renderer, data)
                last_query_time = throttle_query_rate(last_query_time, args.query_interval)
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
                # Millisecond epoch timestamp, shared by both cameras' filenames for this
                # step -- lets you pair external/wrist frames and match them back to a
                # log_entry's "timestamp" field while a run is still in progress.
                ts_ms = int(time.time() * 1000)
                Image.fromarray(ext_frame).save(images_dir / f"camera_{ts_ms}_external_cam.png")
                Image.fromarray(wrist_frame).save(images_dir / f"camera_{ts_ms}_wrist_cam.png")
                frame_idx += 1

                # Fire the next chunk's request right after the first action, using
                # the state as of *now* (one step fresher than the state this
                # chunk's own request used) -- gives the background request the
                # rest of this chunk's local stepping/rendering time to complete in.
                if i == 0 and (args.chunks <= 0 or chunk_i + 1 < args.chunks):
                    last_query_time = throttle_query_rate(last_query_time, args.query_interval)
                    pending = capture_and_submit(
                        executor, renderer, data, model, tracked_body_id,
                        args.server_url, send_wrist, args.request_timeout,
                    )

            print(f"  ctrl after chunk {chunk_i}: {np.array2string(data.ctrl, precision=3, suppress_small=True)}")

            log_entry["env_state"]["ctrl_after"] = data.ctrl.tolist()
            log_entry["env_state"]["tracked_object_xpos_after"] = (
                data.xpos[tracked_body_id].tolist() if tracked_body_id is not None else None
            )
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

        # Explicit, deterministic teardown order: close the offscreen renderer's own GL
        # context *before* viewer_cm's own __exit__ tears down the passive viewer's GLFW
        # window (which happens right after this, once we leave the `with` block). Without
        # this, both GL contexts get released during interpreter shutdown in whatever order
        # Python's GC happens to finalize them, which reliably crashes with glibc's
        # "free(): invalid pointer" (SIGABRT) after a --no-view-less run's last chunk --
        # harmless in that all data/frames are already flushed to disk by that point, but
        # noisy and worth avoiding. renderer is constructed unconditionally (used by the
        # dry-run path too, for the request payload), so close it unconditionally as well.
        renderer.close()

    if executor is not None:
        executor.shutdown(wait=False)
    log_f.close()

    if not args.dry_run:
        print(f"Saved {frame_idx} frame pairs to {images_dir}/")


if __name__ == "__main__":
    main()
