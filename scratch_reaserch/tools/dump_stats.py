"""Compute normalization statistics for state and action from expert rollouts.

A policy cannot be served without them: `state_proj` and `action_out_proj` work
in normalized units, and the stock checkpoint ships stats for a different robot.
Runs with cameras off, so a few hundred episodes take seconds.

    uv run python tools/dump_stats.py --episodes 200 --out assets/stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from greenbox import task_spec as spec
from greenbox.env import GreenBoxPickPlace
from greenbox.policies import ExpertSource


def stat_block(arr: np.ndarray) -> dict:
    return {
        "mean": arr.mean(0).tolist(),
        "std": (arr.std(0) + 1e-6).tolist(),
        "min": arr.min(0).tolist(),
        "max": arr.max(0).tolist(),
        "count": [int(len(arr))],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--action-noise", type=float, default=0.05)
    p.add_argument("--waypoint-noise", type=float, default=0.004)
    p.add_argument("--out", default="assets/stats.json")
    args = p.parse_args()

    np.random.seed(args.seed)
    env = GreenBoxPickPlace(use_camera_obs=False, horizon=args.horizon, ignore_done=True)
    source = ExpertSource(args.seed, args.action_noise, args.waypoint_noise)

    states, actions, kept = [], [], 0
    for ep in range(args.episodes):
        obs = env.reset()
        source.reset(env)
        ep_s, ep_a = [], []
        for _ in range(args.horizon):
            ep_s.append(env.policy_state())
            a = np.asarray(source.act(env, obs), dtype=np.float32)
            ep_a.append(a)
            obs, _, _, _ = env.step(a)
            if source.finished:
                break
        if env._check_success():  # only successful demos define the distribution
            states.extend(ep_s)
            actions.extend(ep_a)
            kept += 1

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)

    stats = {
        spec.STATE_KEY: stat_block(states),
        spec.ACTION_KEY: stat_block(actions),
    }
    # Images are handled by the VLM's own preprocessing; supply neutral stats so
    # an IDENTITY-or-MEAN_STD mapping both behave sanely.
    for key in spec.CAMERAS.values():
        stats[key] = {
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.5]], [[0.5]], [[0.5]]],
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "count": [int(len(states))],
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(stats, fh, indent=2)

    np.set_printoptions(precision=3, suppress=True)
    print(f"kept {kept}/{args.episodes} successful episodes, {len(states)} frames")
    print(f"state mean {np.round(states.mean(0), 3)}")
    print(f"state std  {np.round(states.std(0), 3)}")
    print(f"action mean {np.round(actions.mean(0), 3)}")
    print(f"action std  {np.round(actions.std(0), 3)}")
    print(f"action q01  {np.round(np.percentile(actions, 1, axis=0), 3)}")
    print(f"action q99  {np.round(np.percentile(actions, 99, axis=0), 3)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
