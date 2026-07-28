"""Score closed-loop runs from their log files, several at a time.

WHY THIS EXISTS
---------------
`libero_closed_loop.py` writes one JSON object per action chunk and leaves interpretation
to the reader, so every evaluation so far has been done by eye. That does not survive the
one rule this project keeps re-learning: **the action expert is flow-matching, so it
samples.** One rollout is one draw, two runs from an identical start state have diverged
completely (PROGRESS sec.18), and README sec.9's baseline is a RATE -- 0/3 placements,
1/3 grasp-and-lift -- not a description of a run.

So this reads N logs and reports the rate, using the same success test the demo collector
uses to accept an episode (`collect_finetune_data.run_episode`): the ball must have been
lifted 50 mm off the table at some point, and must end inside the green bin's footprint.

It also reports the two diagnostics README sec.9 used to characterise HOW it fails --
closest lateral approach to the ball, and what fraction of actions command the gripper
shut -- plus the per-channel action statistics, which is where an overfitted checkpoint
shows itself: FINE_TUNE_LEARNINGS sec.1 predicts the rotation channels collapsing toward
zero std first.

Usage:
    uv run python libero/score_runs.py assets/logs/ft_step200_*.jsonl
    uv run python libero/score_runs.py --json assets/logs/*.jsonl
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import libero.libero_closed_loop as L  # noqa: E402

BIN_RADIUS = 0.05          # same tolerance the collector accepts a placement at
LIFT_HEIGHT = 0.05         # ball must clear the table by this to count as lifted
BALL_RADIUS = 0.02
CHANNELS = ("dx", "dy", "dz", "drx", "dry", "drz", "grip")


def green_bin_xy(model_path):
    """Read the green bin's position out of the scene, rather than hardcoding it -- the
    scene has already moved 100 mm once (README sec.1.1 / fine_tune README sec.1.1)."""
    model = mujoco.MjModel.from_xml_path(model_path)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_bin")
    if bid == -1:
        raise SystemExit("no green_bin body in the scene")
    return model.body_pos[bid][:2].copy()


def score_log(path, bin_xy):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not entries:
        return None

    ball, eef, actions = [], [], []
    for e in entries:
        env = e.get("env_state", {})
        for key in ("tracked_object_xpos", "tracked_object_xpos_after"):
            if env.get(key):
                ball.append(np.asarray(env[key], dtype=float))
        for key in ("eef_pos", "eef_pos_after"):
            if env.get(key):
                eef.append(np.asarray(env[key], dtype=float))
        chunk = e.get("model_output", {}).get("actions")
        if chunk:
            actions.append(np.asarray(chunk, dtype=float))

    if not ball:
        raise SystemExit(f"{path}: no tracked_object_xpos in the log; nothing to score")
    ball = np.stack(ball)
    actions = np.concatenate(actions) if actions else np.zeros((0, 7))

    lifted = bool(ball[:, 2].max() > L.TABLE_TOP_Z + BALL_RADIUS + LIFT_HEIGHT)
    final = ball[-1]
    placed = bool(abs(final[0] - bin_xy[0]) < BIN_RADIUS
                  and abs(final[1] - bin_xy[1]) < BIN_RADIUS
                  and final[2] < L.TABLE_TOP_Z + 0.06)

    # Closest the hand ever got to the ball in the table plane. eef_pos in the log is in
    # LIBERO's frame (README sec.9.1 -- mixing the two frames is a mistake already made
    # once), so shift the ball into that frame rather than the other way round.
    lateral = None
    if eef:
        eef = np.stack(eef)
        n = min(len(eef), len(ball))
        ball_libero = ball[:n, :2] + L.LIBERO_ORIGIN_OFFSET[:2]
        lateral = float(np.linalg.norm(eef[:n, :2] - ball_libero, axis=1).min())

    return {
        "run": Path(path).stem,
        "chunks": len(entries),
        "lifted": lifted,
        "placed": placed,
        "best_lateral_mm": None if lateral is None else round(1000 * lateral, 1),
        "ball_max_z_mm": round(1000 * (ball[:, 2].max() - L.TABLE_TOP_Z), 1),
        "gripper_close_pct": (round(100 * float(np.mean(actions[:, 6] > 0)), 1)
                              if len(actions) else None),
        "action_std": {c: round(float(actions[:, i].std()), 4) for i, c in enumerate(CHANNELS)}
        if len(actions) else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--model-path", default=L.DEFAULT_MODEL_PATH)
    p.add_argument("--json", action="store_true", help="dump the per-run dicts as JSON")
    args = p.parse_args()

    bin_xy = green_bin_xy(args.model_path)
    results = [r for r in (score_log(f, bin_xy) for f in args.logs) if r]
    if not results:
        raise SystemExit("nothing to score")

    print(f"green bin at ({bin_xy[0]:.3f}, {bin_xy[1]:.3f}), "
          f"tolerance +-{BIN_RADIUS * 1000:.0f} mm\n")
    print(f"{'run':<28} {'chunks':>6} {'lift':>5} {'place':>6} {'lateral':>9} "
          f"{'ballz':>7} {'grip%':>6}")
    for r in results:
        print(f"{r['run']:<28} {r['chunks']:>6} {str(r['lifted']):>5} "
              f"{str(r['placed']):>6} "
              f"{'-' if r['best_lateral_mm'] is None else r['best_lateral_mm']:>9} "
              f"{r['ball_max_z_mm']:>7} "
              f"{'-' if r['gripper_close_pct'] is None else r['gripper_close_pct']:>6}")

    n = len(results)
    print(f"\nplacements     {sum(r['placed'] for r in results)}/{n}")
    print(f"grasp-and-lift {sum(r['lifted'] for r in results)}/{n}")
    print("baseline, stock checkpoint on this scene (README sec.9): 0/3 placed, 1/3 lifted")

    stds = [r["action_std"] for r in results if r["action_std"]]
    if stds:
        print("\nmean per-channel action std across runs "
              "(rotation collapsing toward 0 = overfitted, FINE_TUNE_LEARNINGS sec.1):")
        print("  " + "  ".join(
            f"{c}={np.mean([s[c] for s in stds]):.4f}" for c in CHANNELS))
        print("  a4 labels for reference: drx~0.009 dry~0.009 drz~0.010; "
              "released LIBERO: 0.039 / 0.063 / 0.078")

    if args.json:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
