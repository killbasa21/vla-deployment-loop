"""DIAGNOSTIC: run MolmoAct2-LIBERO on a REAL LIBERO task, in LIBERO's own environment.

Deliberately separate from `libero_closed_loop.py` and shares no code with it. That is
the whole point: this driver uses LIBERO's own `OffScreenRenderEnv` -- robosuite, real
OSC_POSE control, a task from the benchmark the checkpoint was actually trained on -- so
the ONLY things it has in common with the main loop are the served checkpoint and the
wire format.

WHAT QUESTION THIS ANSWERS
--------------------------
Our own scene fails: the model aims well (2.6-7 mm lateral) but the terminal descent and
grasp go wrong, and one lift in ~5 runs (PROGRESS.md sec.17-19). Two hypotheses:

  A. CONTROLLER. The policy learned robosuite OSC's transfer function -- a compliant
     force controller that yields on contact -- and we run a stiff position servo driven
     by IK (Route A). Same action in, different motion out.
  B. TASK. A green ball on a bare table with coloured bins is not one of LIBERO's 130
     tasks, and the instruction string is ours. Zero-shot to an unseen task.

Running a task the checkpoint knows, in the controller it was trained with, separates
them:

  succeeds here  -> serving is correct, checkpoint is fine, OUR ENVIRONMENT is the
                    problem. Hypothesis A. Building real OSC is worth it.
  fails here     -> the problem is upstream of our scene entirely: serving, norm stats,
                    image convention, or state convention. Fix that before anything else.

A third outcome is possible and worth naming in advance: it succeeds here AND our task is
simply too far out of distribution. That would show up as success here with no improvement
from an OSC port later, and it points at fine-tuning rather than at the controller.

OBSERVATION CONTRACT
--------------------
Replicated from the vendored repo rather than guessed, because getting it wrong would
make a failure here meaningless:

  state  = concat(robot0_eef_pos(3), quat2axisangle(robot0_eef_quat)(3),
                  robot0_gripper_qpos(2))                     env_processor.py:66-74
           robosuite's robot0_eef_quat is ALREADY (x,y,z,w), so no reordering here --
           unlike our MuJoCo loop, which gets (w,x,y,z) and must reorder.
  images = rotated 180 degrees, i.e. flipped in BOTH H and W  env_processor.py:59
           (`torch.flip(img, dims=[2,3])` on (B,C,H,W)). This is the convention question
           `--image-flip` exists for in the main loop; here it is not a question, it is
           what the training pipeline does.
  size   = 256x256                                            envs/libero.py:109-110

Usage (needs the separate venv -- hf-libero pins its own robosuite/mujoco and must not
be installed into the project env):

    MUJOCO_GL=egl <venv>/bin/python libero/libero_benchmark_eval.py \
        --server-url <url>/act --suite libero_object --task-id 0 --episodes 3
"""

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import requests

# NOTE: deliberately NOT json_numpy.patch(). The main loop patches `json` globally and
# that is fine there, but inside the hf-libero venv something in the LIBERO/robosuite
# import chain has already installed its own global `json` object_hook (one that returns
# SimpleNamespace). json_numpy composes hooks -- `lambda dct: _hook(object_hook(dct))` --
# so its own hook then gets handed a SimpleNamespace and dies on `"__numpy__" in dct`.
# Encoding and decoding explicitly below keeps this script independent of whatever else
# has patched `json` in the process.


def _np_encode(obj):
    """json_numpy's wire format, written out explicitly."""
    if isinstance(obj, np.ndarray):
        return {
            "__numpy__": base64.b64encode(np.ascontiguousarray(obj).data).decode(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _np_decode(dct):
    if isinstance(dct, dict) and "__numpy__" in dct:
        return np.frombuffer(
            base64.b64decode(dct["__numpy__"]), dtype=np.dtype(dct["dtype"])
        ).reshape(dct["shape"])
    return dct

# LIBERO's own defaults, matching what the checkpoint was trained on.
RENDER_SIZE = 256
CONTROL_HORIZON = 10          # LIBERO action_horizon; the server returns (10, 7)
DEFAULT_MAX_STEPS = 520       # LIBERO's own episode cap for the short suites


def quat2axisangle(quat):
    """(x, y, z, w) -> 3-vector axis-angle. Ported from env_processor.py's _quat2axisangle
    (the torch version), same clipping and same degenerate-case handling."""
    quat = np.asarray(quat, dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(w)) / den


def build_state(raw_obs):
    """LIBERO's 8-D state, exactly as env_processor.py assembles it."""
    return np.concatenate([
        np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64),
        quat2axisangle(raw_obs["robot0_eef_quat"]),
        np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float64),
    ]).astype(np.float32)


def build_images(raw_obs):
    """Both cameras, rotated 180 degrees. robosuite renders bottom-up (OpenGL) and the
    training pipeline flips H and W; `[::-1, ::-1]` is that same rotation on HWC."""
    return (
        np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]),
        np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]),
    )


def query_server(main_img, wrist_img, state, instruction, server_url, timeout):
    """Same schema host_server_droid.py demands -- the keys are positional and are
    forwarded as images=[first, second], so the first must be agentview."""
    payload = {
        "external_cam": main_img,
        "wrist_cam": wrist_img,
        "instruction": instruction,
        "state": state,
    }
    t0 = time.time()
    resp = requests.post(
        server_url,
        data=json.JSONEncoder(default=_np_encode).encode(payload),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    # JSONDecoder directly, not json.loads -- see the note at the top of this file.
    body = json.JSONDecoder(object_hook=_np_decode).decode(resp.text)
    # Response is {"actions": ndarray, "dt_ms": float}, not a bare array -- same shape
    # host_server_droid.py returns to the main loop (libero_closed_loop.py:414-416).
    actions = np.asarray(body["actions"], dtype=np.float64)
    server_dt = body.get("dt_ms", body.get("latency_ms"))
    round_trip = (time.time() - t0) * 1000.0
    return actions, float(server_dt) if server_dt is not None else round_trip


def run_episode(env, instruction, args, ep_idx):
    raw_obs = env.reset()
    if isinstance(raw_obs, tuple):
        raw_obs = raw_obs[0]

    steps = 0
    success = False
    chunk_i = 0
    while steps < args.max_steps:
        main_img, wrist_img = build_images(raw_obs)
        state = build_state(raw_obs)
        actions, dt_ms = query_server(
            main_img, wrist_img, state, instruction, args.server_url, args.request_timeout
        )
        if actions.ndim == 1:
            actions = actions[None, :]

        # Execute the whole chunk before replanning, matching the main loop's sequential
        # mode and LIBERO's own n_action_steps=10.
        for a in actions[: args.action_steps]:
            raw_obs, reward, done, info = env.step(a.tolist())
            steps += 1
            if done or steps >= args.max_steps:
                break
        # robosuite/LIBERO signal task success through the env's own check, not reward.
        success = bool(env.env._check_success())
        print(
            f"    ep{ep_idx} chunk {chunk_i:>3}  step {steps:>4}  "
            f"eef_z {raw_obs['robot0_eef_pos'][2]:+.4f}  "
            f"grip {actions[:args.action_steps, 6].mean():+.2f}  "
            f"server {dt_ms:.0f} ms" + ("   SUCCESS" if success else "")
        )
        chunk_i += 1
        if success or done:
            break
    return success, steps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-url", required=True, help="the deployed /act endpoint")
    p.add_argument("--suite", default="libero_object",
                   help="libero_spatial | libero_object | libero_goal | libero_10 | libero_90")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--action-steps", type=int, default=CONTROL_HORIZON,
                   help="actions executed per chunk before replanning (LIBERO uses 10)")
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--out", default="assets/logs/benchmark_eval.jsonl")
    args = p.parse_args()

    # Imported here, not at module scope: this file is also read as documentation, and
    # the import only works inside the hf-libero venv.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    instruction = task.language
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)

    print(f"suite       : {args.suite}  (task {args.task_id} of {suite.n_tasks})")
    print(f"instruction : {instruction!r}")
    print(f"bddl        : {bddl}")
    print(f"server      : {args.server_url}")
    print(f"episodes    : {args.episodes}, max {args.max_steps} steps each\n")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=RENDER_SIZE, camera_widths=RENDER_SIZE
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out.open("w") as log_f:
        for ep in range(args.episodes):
            print(f"  episode {ep}:")
            ok, steps = run_episode(env, instruction, args, ep)
            results.append(ok)
            print(f"  -> episode {ep}: {'SUCCESS' if ok else 'failure'} after {steps} steps\n")
            log_f.write(json.dumps({
                "suite": args.suite, "task_id": args.task_id, "instruction": instruction,
                "episode": ep, "success": ok, "steps": steps,
            }) + "\n")
            log_f.flush()
    env.close()

    n = len(results)
    print(f"\n=== {args.suite} task {args.task_id}: {sum(results)}/{n} succeeded "
          f"({100.0 * sum(results) / n:.0f}%) ===")
    print("Reading: success here means the checkpoint and our serving are fine and the "
          "problem is our environment (hypothesis A, controller). Failure here means the "
          "problem is upstream of our scene -- serving, norm stats, or conventions.")


if __name__ == "__main__":
    main()
