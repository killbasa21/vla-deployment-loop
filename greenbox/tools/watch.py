"""Watch an episode play out in the real MuJoCo passive viewer, in real time.

Two windows:

  * the MuJoCo viewer itself, driven straight off robosuite's underlying
    `MjModel` / `MjData`, so it is the same interactive viewer as any other
    MuJoCo scene: drag to orbit, space to pause, `[`/`]` to cycle cameras.
    Overlaid on it are the controller's own internals -- a sphere at the current
    waypoint and an arrow for the commanded translation.
  * an OpenCV HUD showing exactly what the policy sees (both camera streams at
    the resolution it is trained on) plus live bars for the 7 action dimensions.

The action source is pluggable, so the expert and a served checkpoint are
watched through the identical path:

    uv run python tools/watch.py                       # scripted expert
    uv run python tools/watch.py --policy server \
        --server-url http://localhost:8000             # a served checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen cameras; viewer uses GLFW
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np

from greenbox import task_spec as spec
from greenbox.env import GreenBoxPickPlace

ACTION_LABELS = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]


from greenbox.policies import ExpertSource, ServerSource  # noqa: F401


# ------------------------------------------------------------------ HUD window


def draw_hud(cv2, obs, action, frames, headless, lines):
    tiles = []
    for cam in spec.CAMERAS:
        img = obs[f"{cam}_image"][::-1]
        tiles.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    strip = np.concatenate(tiles, axis=1)
    strip = cv2.resize(strip, (strip.shape[1] * 2, strip.shape[0] * 2),
                       interpolation=cv2.INTER_NEAREST)

    panel_h = 30 * len(ACTION_LABELS) + 20 + 26 * len(lines)
    panel = np.full((panel_h, strip.shape[1], 3), 24, np.uint8)

    mid = strip.shape[1] // 2
    for i, (label, val) in enumerate(zip(ACTION_LABELS, action)):
        y = 22 + 30 * i
        cv2.line(panel, (mid, y - 10), (mid, y + 10), (70, 70, 70), 1)
        w = int(abs(float(val)) * (mid - 90))
        x0, x1 = (mid, mid + w) if val >= 0 else (mid - w, mid)
        color = (90, 200, 90) if val >= 0 else (90, 130, 240)
        cv2.rectangle(panel, (x0, y - 8), (x1, y + 8), color, -1)
        cv2.putText(panel, f"{label:>4} {float(val):+.2f}", (8, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    for i, text in enumerate(lines):
        cv2.putText(panel, text, (8, 30 * len(ACTION_LABELS) + 24 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

    canvas = np.concatenate([strip, panel], axis=0)
    if frames is not None:
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    if headless:
        return 0
    cv2.imshow("policy view", canvas)
    return cv2.waitKey(1) & 0xFF


# ---------------------------------------------------------------------- markers


def set_markers(viewer, eef, waypoint, action):
    scn = viewer.user_scn
    scn.ngeom = 0
    if waypoint is not None:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.012, 0, 0]), np.asarray(waypoint, float),
            np.eye(3).flatten(), np.array([1.0, 0.35, 0.1, 0.85], np.float32),
        )
        scn.ngeom += 1
    tip = np.asarray(eef, float) + np.asarray(action[:3], float) * 0.05 * 3.0
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
        np.array([0.2, 0.9, 0.3, 0.9], np.float32),
    )
    mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 0.006,
                         np.asarray(eef, float), tip)
    scn.ngeom += 1


# ------------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["expert", "server"], default="expert")
    p.add_argument("--server-url", default="http://localhost:8000")
    p.add_argument("--chunk-reuse", type=int, default=0,
                   help="actions consumed per server call; 0 = the whole chunk")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-noise", type=float, default=0.0)
    p.add_argument("--waypoint-noise", type=float, default=0.0)
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--no-viewer", action="store_true",
                   help="skip the MuJoCo window (for recording on a headless box)")
    p.add_argument("--record", default=None,
                   help="write the HUD frames to this mp4")
    args = p.parse_args()

    cv2 = None
    if not args.no_hud:
        try:
            import cv2  # noqa: F811
        except ImportError:
            print("opencv not installed -- running without the HUD "
                  "(uv add opencv-python)")

    np.random.seed(args.seed)
    env = GreenBoxPickPlace(use_camera_obs=cv2 is not None, horizon=args.horizon,
                            ignore_done=True)
    source = (
        ExpertSource(args.seed, args.action_noise, args.waypoint_noise)
        if args.policy == "expert"
        else ServerSource(args.server_url, args.chunk_reuse)
    )

    model, data = env.sim.model._model, env.sim.data._data
    dt = 1.0 / spec.CONTROL_FREQ / max(args.speed, 1e-6)

    frames = [] if args.record else None

    import contextlib

    class _NullViewer:
        """Stand-in so the loop body is identical with and without a window."""

        def is_running(self):
            return True

        def sync(self):
            pass

        user_scn = None

    viewer_ctx = (
        contextlib.nullcontext(_NullViewer())
        if args.no_viewer
        else mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                          show_right_ui=False)
    )

    results = []
    with viewer_ctx as viewer:
        for ep in range(args.episodes):
            obs = env.reset()
            source.reset(env)
            action = np.zeros(spec.ACTION_DIM)
            for t in range(args.horizon):
                if not viewer.is_running():
                    return
                tic = time.time()
                action = np.asarray(source.act(env, obs), dtype=np.float32)
                obs, _, _, _ = env.step(action)

                eef = env.sim.data.site_xpos[env.robots[0].eef_site_id]
                if viewer.user_scn is not None:
                    set_markers(viewer, eef, source.waypoint(env), action)
                viewer.sync()

                if cv2 is not None:
                    key = draw_hud(cv2, obs, action, frames, args.no_viewer, [
                        f"ep {ep}  t {t:3d}  {source.name}  {source.status}",
                        f"target slot: {env.target_slot}   colors: "
                        + " ".join(f"{k}={v}" for k, v in env.slot_colors.items()),
                        f'"{spec.INSTRUCTION}"',
                        f"success: {env._check_success()}",
                    ])
                    if key == ord("q"):
                        return
                    if key == ord("n"):
                        break

                if source.finished:
                    break
                time.sleep(max(0.0, dt - (time.time() - tic)))

            ok = env._check_success()
            results.append(ok)
            print(f"ep{ep}: success={ok} steps={t + 1} target={env.target_slot}",
                  flush=True)
            time.sleep(0.4)

    if cv2 is not None and not args.no_viewer:
        cv2.destroyAllWindows()

    if frames:
        import imageio.v3 as iio

        iio.imwrite(args.record, np.stack(frames), fps=spec.CONTROL_FREQ)
        print(f"wrote {args.record} ({len(frames)} frames)")
    if results:
        print(f"success {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
