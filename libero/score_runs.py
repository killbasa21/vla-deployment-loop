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
uses to accept an episode (`collect_finetune_data.run_episode`): the target must have been
lifted 50 mm off the table at some point, and must end inside the green bin's footprint --
plus a RELEASE check (see `released` in score_log), because the footprint test alone scored an
object still held in the gripper over the bin as a successful placement.

It also reports the two diagnostics README sec.9 used to characterise HOW it fails --
closest lateral approach to the target, and what fraction of actions command the gripper
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
LIFT_HEIGHT = 0.05         # target must clear the table by this to count as lifted
HALF_EXTENT = 0.02         # fallback only; runs since 2026-08-03 log their own
BIN_FLOOR_MARGIN = 0.04    # placed-height slack above an object resting in the bin
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

    # Prefer the position the RUN recorded over the one in the scene XML. `--randomize-bins`
    # shuffles the three bins across BIN_SLOTS at build time by mutating model.body_pos, so
    # for those runs the XML's green bin is simply a different bin's slot -- scoring a
    # placement against it is wrong by up to 500 mm and fails silently, as a policy that
    # looked like it never placed anything.
    #
    # Logs written before 2026-08-01 have no such field. Those runs are scoreable ONLY if
    # they were not randomised; there is no way to recover the layout afterwards.
    # Completeness. A log is written incrementally and flushed every entry, so reading one
    # mid-run is easy to do by accident -- and a run cut off before the release scores as
    # `released=False`, i.e. it looks like a policy failure rather than an unfinished run.
    # `chunks_requested` is absent in logs written before 2026-08-01, and 0 means the run was
    # asked to go until killed; neither supports a completeness claim.
    requested = next((e.get("chunks_requested") for e in entries
                      if e.get("chunks_requested")), None)
    complete = None if not requested else len(entries) >= requested

    logged = next((e["green_bin_xy"] for e in entries if e.get("green_bin_xy")), None)
    if logged is not None:
        bin_xy = np.asarray(logged, dtype=float)
    stale_layout = logged is None

    # THE TWO EEF KEYS ARE IN DIFFERENT FRAMES. libero_closed_loop.py:379 writes
    # `eef_pos = site_xpos + LIBERO_ORIGIN_OFFSET` (LIBERO frame, what the model is shown),
    # while line 1271 writes `eef_pos_after = site_xpos` raw (world frame). They differ by
    # (-0.6, 0, 0.912). Appending both into one array -- which this file used to do --
    # produces a sequence that teleports 1.6 m every step.
    #
    # It did not corrupt `best_lateral_mm`, but only by luck: the world-frame entries sit
    # ~0.6 m from the LIBERO-frame obj and so can never win the .min(). Any metric that
    # sums or differences the array rather than minimising over it WAS wrong (a path length
    # of 120 m in a 1 m workspace). Keep them separate and use each in its own frame.
    obj, eef_libero, eef_world, actions = [], [], [], []
    for e in entries:
        env = e.get("env_state", {})
        for key in ("tracked_object_xpos", "tracked_object_xpos_after"):
            if env.get(key):
                obj.append(np.asarray(env[key], dtype=float))
        if env.get("eef_pos"):
            eef_libero.append(np.asarray(env["eef_pos"], dtype=float))
        if env.get("eef_pos_after"):
            eef_world.append(np.asarray(env["eef_pos_after"], dtype=float))
        chunk = e.get("model_output", {}).get("actions")
        if chunk:
            actions.append(np.asarray(chunk, dtype=float))

    if not obj:
        raise SystemExit(f"{path}: no tracked_object_xpos in the log; nothing to score")
    obj = np.stack(obj)
    actions = np.concatenate(actions) if actions else np.zeros((0, 7))

    # Both height tests below measure the object's CENTRE, so both shift with its half-width.
    # A run scored with the wrong one reads as 10 mm more lifted than it was, per 10 mm of
    # geom. Same precedence as green_bin_xy: what the run recorded beats what the constant
    # says.
    #
    # Two key names because the object changed shape on 2026-08-03: `box_half` is the cube's
    # half-extent, `ball_radius` was the sphere's radius, both mean "half the object's width"
    # and both default to 0.02. Old ball logs keep scoring correctly; the key that is present
    # is also the only record in the log of WHICH object a run used, so do not collapse them.
    half_extent = next(
        (e.get("box_half") or e.get("ball_radius")
         for e in entries if e.get("box_half") or e.get("ball_radius")),
        HALF_EXTENT,
    )

    lifted = bool(obj[:, 2].max() > L.TABLE_TOP_Z + half_extent + LIFT_HEIGHT)
    final = obj[-1]

    # RELEASE. Added 2026-08-01 after a run scored `placed` while the obj was still in the
    # gripper: the policy carried it over the bin, held on, and ended with the obj suspended
    # 42 mm up. That is z = TABLE_TOP_Z + 0.054, and the height test below allows anything
    # under TABLE_TOP_Z + 0.06, so it passed. Caught by watching the viewer, NOT by this
    # scorer -- which is the whole reason it is being tightened here.
    #
    # The test is the last commanded gripper action rather than a tighter height threshold:
    # heights would have to be fitted to this scene's bin floor (resting obj measured at
    # z = 0.012 against a held 0.042, only 30 mm apart), whereas "did the policy let go" is
    # what the word `placed` is supposed to mean and reads the same in any scene.
    # CORRECTED 2026-08-04. The last-action test above is necessary but not sufficient: it
    # asks "is the gripper open at the buzzer", not "did the policy let go of the object".
    # b1_000488_seed6 dropped the box in the green bin (final z 24 mm, 33 mm from centre --
    # confirmed in the rendered frames) and then closed its gripper again on empty air
    # before the run ended, which scored released=False and cost a real placement.
    #
    # So also accept a positive->negative transition anywhere after the first close. That
    # does NOT reopen the 2026-08-01 hole this test was added for: in that failure the
    # policy never let go at all, so there is no such transition, and the height test in
    # `placed` independently rejects an object still suspended in the gripper.
    per_chunk_grip_ = [float(np.mean(np.asarray(e["model_output"]["actions"])[:, 6]))
                       for e in entries if e.get("model_output", {}).get("actions")]
    first_close_ = next((i for i, g in enumerate(per_chunk_grip_) if g > 0), None)
    let_go = (first_close_ is not None
              and any(g < 0 for g in per_chunk_grip_[first_close_ + 1:]))
    released = bool(len(actions) and (actions[-1, 6] < 0 or let_go))

    placed = bool(abs(final[0] - bin_xy[0]) < BIN_RADIUS
                  and abs(final[1] - bin_xy[1]) < BIN_RADIUS
                  and final[2] < L.TABLE_TOP_Z + BIN_FLOOR_MARGIN + half_extent
                  and released)

    # --- CONTINUOUS measures -----------------------------------------------------------
    # The booleans above are the score, but at n~7 rollouts a rate cannot separate two
    # checkpoints: 4/7 vs 5/7 is one run. Every conclusion in act/PROGRESS.md sec.7.4-7.5 came
    # from the quantities below instead, and they were computed in throwaway scripts before
    # being moved here -- which meant they were neither reproducible nor visible in --json.
    #
    # `final_dist_mm` is `placed` before it is thresholded: how far the obj actually ended
    # from the bin centre. A run at 52 mm and one at 400 mm are both `placed=False`.
    final_dist_mm = round(1000 * float(np.linalg.norm(final[:2] - bin_xy)), 1)

    # When the policy let go, in chunks. `None` if it never did. This is the axis on which
    # ck10000 and ck30000 differ (mean 20.0 vs 22.4) while their placement rates barely do.
    per_chunk_grip = [float(np.mean(np.asarray(e["model_output"]["actions"])[:, 6]))
                      for e in entries if e.get("model_output", {}).get("actions")]
    reopen_chunk = next((i for i in range(1, len(per_chunk_grip))
                         if per_chunk_grip[i - 1] > 0 and per_chunk_grip[i] < 0), None)
    close_chunk = next((i for i, g in enumerate(per_chunk_grip) if g > 0), None)

    # Scene difficulty, not policy behaviour: how far the obj must be carried TOWARD the
    # robot. Negative = inward carry, which is where every ck10000 failure lived (sec.7.4).
    dx_mm = round(1000 * float(bin_xy[0] - obj[0][0]), 1)

    # Total end-effector path, Euclidean, from the WORLD-frame samples only (one per chunk).
    # Distinguishes a tight trajectory from a wandering one -- but ONLY comparable between
    # runs with the same `released` outcome, since a held run hovers where a released one
    # retreats.
    eef_path_m = None
    if len(eef_world) > 1:
        w = np.stack(eef_world)
        eef_path_m = round(float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()), 2)

    # Closest the hand ever got to the obj in the table plane. eef_pos in the log is in
    # LIBERO's frame (README sec.9.1 -- mixing the two frames is a mistake already made
    # once), so shift the obj into that frame rather than the other way round.
    lateral = None
    if eef_libero:
        el = np.stack(eef_libero)
        # `obj` holds two samples per chunk (before, after) and eef_libero one, so align on
        # the before-samples: obj[0::2] is the same instant eef_pos was recorded at.
        b = obj[0::2][:len(el)]
        n = min(len(el), len(b))
        obj_libero = b[:n, :2] + L.LIBERO_ORIGIN_OFFSET[:2]
        lateral = float(np.linalg.norm(el[:n, :2] - obj_libero, axis=1).min())

    return {
        "run": Path(path).stem,
        "chunks": len(entries),
        "bin_from_log": not stale_layout,
        "complete": complete,
        "chunks_requested": requested,
        "bin_xy": [round(float(v), 3) for v in bin_xy],
        "lifted": lifted,
        "placed": placed,
        "released": released,
        "best_lateral_mm": None if lateral is None else round(1000 * lateral, 1),
        "final_dist_mm": final_dist_mm,
        "close_chunk": close_chunk,
        "reopen_chunk": reopen_chunk,
        "dx_mm": dx_mm,
        "eef_path_m": eef_path_m,
        "obj_max_z_mm": round(1000 * (obj[:, 2].max() - L.TABLE_TOP_Z), 1),
        "gripper_close_pct": (round(100 * float(np.mean(actions[:, 6] > 0)), 1)
                              if len(actions) else None),
        "action_std": {c: round(float(actions[:, i].std()), 4) for i, c in enumerate(CHANNELS)}
        if len(actions) else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+",
                   help="log files, or any directory below assets/ -- a directory is "
                        "searched recursively for *.jsonl, so a whole policy "
                        "(assets/act/act-green-ball_010000) scores in one command")
    p.add_argument("--model-path", default=L.DEFAULT_MODEL_PATH)
    p.add_argument("--json", action="store_true", help="dump the per-run dicts as JSON")
    args = p.parse_args()

    paths = []
    for entry in args.logs:
        path = Path(entry)
        paths.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
    if not paths:
        raise SystemExit("no .jsonl logs found in: " + " ".join(args.logs))

    bin_xy = green_bin_xy(args.model_path)
    results = [r for r in (score_log(f, bin_xy) for f in paths) if r]
    if not results:
        raise SystemExit("nothing to score")

    print(f"green bin (scene XML) at ({bin_xy[0]:.3f}, {bin_xy[1]:.3f}), "
          f"tolerance +-{BIN_RADIUS * 1000:.0f} mm; per-run positions below\n")
    stale = [r["run"] for r in results if not r["bin_from_log"]]
    if stale:
        print("WARNING: no green_bin_xy in these logs, so the scene XML's layout was "
              "assumed:\n  " + ", ".join(stale) +
              "\n  If they were run with --randomize-bins, `place` is MEANINGLESS for "
              "them -- the bin moved and the log did not record where.\n")
    print(f"{'run':<28} {'chunks':>6} {'lift':>5} {'place':>6} {'lateral':>9} "
          f"{'objz':>7} {'rel':>5} {'close':>6} {'reopen':>7} {'finaldist':>10} "
          f"{'dx_mm':>7} {'path':>6}")
    for r in results:
        print(f"{r['run']:<28} {r['chunks']:>6} {str(r['lifted']):>5} "
              f"{str(r['placed']):>6} "
              f"{'-' if r['best_lateral_mm'] is None else r['best_lateral_mm']:>9} "
              f"{r['obj_max_z_mm']:>7} "
              f"{str(r['released']):>5} "
              f"{str(r['close_chunk']):>6} {str(r['reopen_chunk']):>7} "
              f"{r['final_dist_mm']:>10} {r['dx_mm']:>7} "
              f"{'-' if r['eef_path_m'] is None else r['eef_path_m']:>6}"
              f"{'' if r['bin_from_log'] else ' (XML)'}")

    n = len(results)
    partial = [f"{r['run']} ({r['chunks']}/{r['chunks_requested']})"
               for r in results if r["complete"] is False]
    if partial:
        print("\nINCOMPLETE -- fewer chunks than requested; still running, or killed. "
              "Every outcome below is provisional for these:\n  " + ", ".join(partial))

    held = [r["run"] for r in results if not r["released"]]
    if held:
        print("\nNOT RELEASED (obj still in the gripper at the end -- `place` is False "
              "regardless of where it is):\n  " + ", ".join(held))
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
