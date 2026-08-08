"""Score an action source over N episodes and print the metric breakdown.

    uv run python tools/score.py --policy expert   --episodes 50
    uv run python tools/score.py --policy random   --episodes 20
    uv run python tools/score.py --policy server   --episodes 20 \
        --server-url https://<modal-app>.modal.run

Writes one JSON object per episode to `--out`, so runs can be re-aggregated or
compared later without re-simulating.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from greenbox import task_spec as spec
from greenbox.env import GreenBoxPickPlace
from greenbox.metrics import EpisodeTracker
from greenbox.policies import make_source

STAGES = ["reached", "grasped", "lifted", "transported", "released", "placed", "complete"]
CONTINUOUS = [
    ("grasp_pos_closest", "m"),
    ("grasp_pos_at_close", "m"),
    ("grasp_rad_closest", "rad"),
    ("grasp_rad_at_close", "rad"),
    ("place_pos_closest", "m"),
    ("place_pos_final", "m"),
    ("lift_height_max", "m"),
]


def summarize(records: list[dict]) -> str:
    n = len(records)
    out = [f"\nepisodes: {n}    instruction: {spec.INSTRUCTION!r}", ""]

    out.append("stage completion (each stage requires the one above it)")
    out.append(f"  {'stage':<14}{'count':>7}{'rate':>8}")
    for stage in STAGES:
        c = sum(bool(r[stage]) for r in records)
        out.append(f"  {stage:<14}{c:>7}{c / n:>8.0%}")

    wrong = sum(bool(r["placed_wrong"]) for r in records)
    out.append("")
    out.append("colour grounding")
    out.append(f"  {'placed in target tray':<26}{sum(bool(r['placed']) for r in records):>5}"
               f"{sum(bool(r['placed']) for r in records) / n:>8.0%}")
    out.append(f"  {'placed in a distractor':<26}{wrong:>5}{wrong / n:>8.0%}")
    near = Counter(r["nearest_tray"] for r in records)
    hit = Counter(r["nearest_tray"] for r in records if r["nearest_tray"] == r["target_slot"])
    out.append(f"  final box nearest tray: {dict(near)}")
    out.append(f"  ...of which the target: {sum(hit.values())}/{n}")
    by_slot = Counter(r["target_slot"] for r in records)
    ok_by_slot = Counter(r["target_slot"] for r in records if r["placed"])
    out.append("  success by target slot: "
               + ", ".join(f"{s}={ok_by_slot[s]}/{by_slot[s]}" for s in sorted(by_slot)))

    out.append("")
    out.append("continuous measures        mean   median      p90      min")
    for key, unit in CONTINUOUS:
        vals = [r[key] for r in records if np.isfinite(r[key])]
        if not vals:
            out.append(f"  {key:<22} {'--':>8}")
            continue
        p90 = float(np.percentile(vals, 90))
        out.append(
            f"  {key:<22}{statistics.mean(vals):>8.4f}{statistics.median(vals):>9.4f}"
            f"{p90:>9.4f}{min(vals):>9.4f}  {unit}"
        )

    steps = [r["steps"] for r in records]
    tos = sum(bool(r["timeout"]) for r in records)
    out.append("")
    out.append(f"steps: mean {statistics.mean(steps):.0f}  max {max(steps)}  "
               f"timeouts {tos}/{n}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["expert", "random", "server"], default="expert")
    p.add_argument("--server-url", default="http://localhost:8000")
    p.add_argument("--chunk-reuse", type=int, default=0)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-noise", type=float, default=0.0)
    p.add_argument("--waypoint-noise", type=float, default=0.0)
    p.add_argument("--out", default=None, help="JSONL path (default assets/scores/<tag>.jsonl)")
    p.add_argument("--tag", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    tag = args.tag or f"{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_path = args.out or f"assets/scores/{tag}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.random.seed(args.seed)
    needs_images = args.policy == "server"
    env = GreenBoxPickPlace(use_camera_obs=needs_images, horizon=args.horizon,
                            ignore_done=True)
    source = make_source(args)
    tracker = EpisodeTracker()

    records = []
    with open(out_path, "w") as fh:
        for ep in range(args.episodes):
            obs = env.reset()
            source.reset(env)
            tracker.start(env, episode=ep, seed=args.seed)
            t = 0
            for t in range(args.horizon):
                action = np.asarray(source.act(env, obs), dtype=np.float32)
                obs, _, _, _ = env.step(action)
                tracker.step(env, action)
                if source.finished:
                    break
            m = tracker.finish(env)
            m.timeout = (t + 1) >= args.horizon and not m.placed
            rec = m.as_dict()
            rec["policy"] = source.name
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if not args.quiet:
                chain = "".join("1" if rec[s] else "0" for s in STAGES)
                print(f"ep{ep:03d} {chain} steps={rec['steps']:3d} "
                      f"target={rec['target_slot']:<5} "
                      f"grasp={rec['grasp_pos_closest']:.3f} "
                      f"place={rec['place_pos_final']:.3f}")

    print(summarize(records))
    print(f"\nwrote {out_path}   (stage bits in order: {'/'.join(STAGES)})")


if __name__ == "__main__":
    main()
