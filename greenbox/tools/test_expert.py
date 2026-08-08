"""Headless success-rate check for the scripted expert."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from greenbox.env import GreenBoxPickPlace
from greenbox.expert import ExpertConfig, ScriptedExpert


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--action-noise", type=float, default=0.0)
    p.add_argument("--waypoint-noise", type=float, default=0.0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)
    env = GreenBoxPickPlace(use_camera_obs=False, horizon=args.horizon, ignore_done=True)
    cfg = ExpertConfig(
        action_noise=args.action_noise,
        waypoint_noise=args.waypoint_noise,
        rng=np.random.default_rng(args.seed),
    )
    expert = ScriptedExpert(cfg)

    successes, lengths, stalls = 0, [], Counter()
    for ep in range(args.episodes):
        env.reset()
        expert.reset(env)
        for t in range(args.horizon):
            env.step(expert.act(env))
            if expert.finished:
                break
        ok = env._check_success()
        successes += ok
        lengths.append(t + 1)
        if not ok:
            stalls[expert.phase] += 1
        if args.verbose:
            print(f"ep{ep:03d} {'OK ' if ok else 'FAIL'} steps={t + 1:3d} "
                  f"end_phase={expert.phase} target={env.target_slot}")

    print(f"\nsuccess {successes}/{args.episodes} = {successes / args.episodes:.0%}")
    print(f"mean length {np.mean(lengths):.0f} steps  (max {max(lengths)})")
    if stalls:
        print(f"failures ended in phase: {dict(stalls)}")


if __name__ == "__main__":
    main()
