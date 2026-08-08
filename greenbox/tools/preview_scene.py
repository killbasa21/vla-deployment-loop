"""Render a few resets of the scene to PNGs so the layout can be eyeballed."""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v3 as iio
import numpy as np

from greenbox.env import GreenBoxPickPlace
from greenbox import task_spec as spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="assets/preview")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--cameras", nargs="*", default=["agentview", "robot0_eye_in_hand", "frontview", "birdview"])
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = GreenBoxPickPlace(
        camera_names=args.cameras,
        camera_heights=args.size,
        camera_widths=args.size,
        use_camera_obs=True,
    )
    np.random.seed(args.seed)

    for ep in range(args.episodes):
        obs = env.reset()
        print(f"ep{ep}: target={env.target_slot!r} colors={env.slot_colors} "
              f"box={np.round(obs['box_pos'], 3)}")
        tiles = []
        for cam in args.cameras:
            img = obs[f"{cam}_image"][::-1]  # robosuite renders bottom-up
            tiles.append(img)
        strip = np.concatenate(tiles, axis=1)
        iio.imwrite(f"{args.out}/ep{ep}.png", strip)
    print(f"wrote {args.episodes} frames to {args.out}/  (cameras L->R: {args.cameras})")


if __name__ == "__main__":
    main()
