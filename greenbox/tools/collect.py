"""Collect scripted-expert demonstrations.

Rejection-sampled: an episode is written only if it ends in success, so the
dataset contains no failure modes for the policy to imitate.

Each episode is one `.npz`:
    agentview, wrist   object arrays of JPEG-encoded frames, one per step
    state              (T, 9) float32
    action             (T, 7) float32
    meta               json blob (target slot, slot colours, seed, instruction)

JPEG rather than raw arrays because raw is ~60x larger and the upload to Modal
is the slow part of the loop. Frames are encoded at the resolution the policy
trains on, so nothing is resized twice.

    uv run python tools/collect.py --episodes 300 --out data/demos
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from greenbox import task_spec as spec
from greenbox.env import GreenBoxPickPlace
from greenbox.policies import ExpertSource


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    iio.imwrite(buf, img, extension=".jpg", quality=quality)
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300, help="successful episodes to keep")
    p.add_argument("--max-attempts", type=int, default=0, help="0 = episodes * 3")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-noise", type=float, default=0.05)
    p.add_argument("--waypoint-noise", type=float, default=0.004)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--out", default="data/demos")
    args = p.parse_args()

    max_attempts = args.max_attempts or args.episodes * 3
    os.makedirs(args.out, exist_ok=True)

    np.random.seed(args.seed)
    env = GreenBoxPickPlace(use_camera_obs=True, horizon=args.horizon, ignore_done=True)
    source = ExpertSource(args.seed, args.action_noise, args.waypoint_noise)

    kept, attempts, frames, t0 = 0, 0, 0, time.time()
    bar = tqdm(total=args.episodes, desc="demos")
    while kept < args.episodes and attempts < max_attempts:
        attempts += 1
        obs = env.reset()
        source.reset(env)
        agentview, wrist, states, actions = [], [], [], []

        for _ in range(args.horizon):
            states.append(env.policy_state())
            agentview.append(obs["agentview_image"][::-1])
            wrist.append(obs["robot0_eye_in_hand_image"][::-1])
            action = np.asarray(source.act(env, obs), dtype=np.float32)
            actions.append(action)
            obs, _, _, _ = env.step(action)
            if source.finished:
                break

        if not env._check_success():
            continue

        path = os.path.join(args.out, f"ep_{kept:05d}.npz")
        np.savez_compressed(
            path,
            agentview=np.array([encode_jpeg(f, args.jpeg_quality) for f in agentview],
                               dtype=object),
            wrist=np.array([encode_jpeg(f, args.jpeg_quality) for f in wrist],
                           dtype=object),
            state=np.asarray(states, dtype=np.float32),
            action=np.asarray(actions, dtype=np.float32),
            meta=json.dumps({
                "target_slot": env.target_slot,
                "slot_colors": env.slot_colors,
                "instruction": spec.INSTRUCTION,
                "attempt": attempts,
                "seed": args.seed,
                "steps": len(states),
            }),
        )
        kept += 1
        frames += len(states)
        bar.update(1)
    bar.close()

    size_mb = sum(os.path.getsize(os.path.join(args.out, f))
                  for f in os.listdir(args.out)) / 1e6
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({
            "episodes": kept,
            "attempts": attempts,
            "frames": frames,
            "instruction": spec.INSTRUCTION,
            "state_dim": spec.STATE_DIM,
            "action_dim": spec.ACTION_DIM,
            "control_freq": spec.CONTROL_FREQ,
            "image_size": spec.IMAGE_SIZE,
            "cameras": list(spec.CAMERAS.values()),
            "action_noise": args.action_noise,
            "waypoint_noise": args.waypoint_noise,
        }, fh, indent=2)

    print(f"kept {kept}/{attempts} attempts ({kept / max(attempts, 1):.0%} success), "
          f"{frames} frames, {size_mb:.0f} MB, {time.time() - t0:.0f}s")
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
