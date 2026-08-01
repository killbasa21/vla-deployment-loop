"""Train an ACT (Action Chunking Transformer) from scratch on the green-ball demos, on Modal.

Counterpart to `smolvla_libero/smolvla_modal_train.py`, and deliberately its control arm:
same dataset, same scene, same closed loop, same scorer -- a policy with no pretraining and
no language. See `act/README.md` sec.1 for why that comparison is worth the GPU hour.

THREE THINGS DIFFER FROM THE SMOLVLA SCRIPT, AND ALL THREE ARE THE SAME POINT
-----------------------------------------------------------------------------
ACT has no pretrained checkpoint, so every decision that existed to protect a pretrained
checkpoint's conventions inverts:

1. NO DATASET CONVERSION. `convert_dataset.py` exists because smolvla_libero's config
   hardcodes `observation.images.image2` for the wrist camera, and LeRobot resolves feature
   keys by name inside the normalisation layer -- hand it `wrist_image` and it trains with no
   second camera AND NO ERROR. ACT is constructed from the dataset's own metadata and takes
   any `observation.image*` key, so `libero/fine_tune/a7` uploads verbatim.

2. NORMALISATION COMES FROM THE DATASET, and that is correct rather than a trap. There are no
   pretrained statistics to preserve. The hazard here is the opposite one:
   `pin_released_stats.py` overwrites `meta/stats.json` with MolmoAct2's released LIBERO
   numbers (count 273465) for the MolmoAct2 path. Training ACT under those would normalise
   into a distribution the data does not occupy. The preflight below asserts
   `stats.json count == info.json total_frames` and refuses to start otherwise.

3. NO `--policy.path`, hence no LoRA question. `--policy.type=act` builds the model from
   scratch; the only pretrained weights in it are ResNet18's ImageNet initialisation, and
   that backbone IS in the gradient path. That is the whole experiment: if ACT grounds a
   randomised ball where a SmolVLA LoRA does not, the SmolVLA gap is adaptation, not data.

CHUNK SIZE
----------
ACT defaults `chunk_size = n_action_steps = 100`. `a7` episodes are 334 ticks, so a 100-chunk
is a third of the task predicted open-loop from one frame. We train at 50 (what
smolvla_libero uses, so the comparison stays clean) and BAKE `n_action_steps=10` into the
saved config, matching LIBERO's action horizon and every SmolVLA rollout already logged.

Temporal ensembling stays off. It needs `n_action_steps=1` -- one HTTP round trip per control
tick -- against a measured transport cost of ~4x inference (`libero/PROGRESS.md` sec.2).

Usage:
    modal run act/act_modal_train.py::upload                        # push a7 to the volume
    modal run act/act_modal_train.py::main --max-steps 1 --save-freq 1 --exp-name act-smoke
    modal run act/act_modal_train.py::main --max-steps 60000 --save-freq 10000
    modal run act/act_modal_train.py::main --resume        # continue from checkpoints/last

`::main` is not optional -- this module has two local entrypoints, so a bare `modal run`
exits with "Specify a Modal Function or local entrypoint". And give the SMOKE its own
--exp-name: lerobot-train refuses to write into an existing output_dir, so a smoke under the
default name blocks the real run behind a --overwrite it should not need.
"""

import modal

# a7: 60 episodes, 20034 frames, 334 ticks/ep, --delta-pos-scale 0.10, shuffled bins.
#
# The scale is the middle setting from `libero/PROGRESS.md` sec.23.5's speed/precision
# frontier: at std(dx) 0.235 one unit of normalised policy error is 23.5 mm on the table,
# against a5's 15.5 (slow, grasps) and a6's 40.0 (fast, misses). It also doubles the episode
# count, which tests the competing "too little data for grounding" explanation at the same
# time.
#
# THE SERVING CLIENT MUST RUN `--delta-pos-scale 0.10 --randomize-bins` or the rollout is not
# measuring this dataset.
DATASET_DIR = "libero/fine_tune/a7"          # uploaded as-is; no conversion step exists
REMOTE_REPO = "/data/greenbox/green_ball_a7_act"   # its own path -- an overwritten volume
                                                   # repo removes the ability to A/B runs

# Same base layers as smolvla_modal_train.py so the heavy torch pull is a CACHE HIT rather
# than a 15-25 minute rebuild. `[smolvla]` is NOT needed for ACT, but dropping it here would
# fork the image chain and cost that time back on the very first run; it is a few hundred MB
# of already-cached wheels.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .pip_install("hf-transfer>=0.1.8")
    .pip_install("lerobot[smolvla,dataset]", "peft", "fastapi[standard]", "json-numpy",
                 "pillow")
)

hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
lerobot_data = modal.Volume.from_name("molmoact2-lerobot-data", create_if_missing=True)
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)

app = modal.App("act-green-ball-train")


@app.function(
    image=image,
    # L4 (24 GB, ~$0.80/hr). ACT is 51,574,663 learnable parameters at this config (measured,
    # printed by lerobot-train) -- memory is nowhere near the constraint;
    # the reason not to drop to a T4 is Turing's missing bf16 and its weaker throughput, and
    # the reason not to go up is that the likely bottleneck is CPU, not GPU (see cpu= below).
    gpu="L4",
    # PNG-in-parquet: every sample decodes TWO 256x256 images on the CPU. At ACT's step rate
    # that, not the GPU, is the wall -- so buy cores and dataloader workers. Same allocation
    # and same reason as smolvla_modal_train.py.
    cpu=16.0,
    volumes={
        "/cache/huggingface": hf_cache,
        "/data": lerobot_data,
        "/checkpoints": checkpoints,
    },
    timeout=8 * 60 * 60,
)
def train(max_steps: int = 60000, batch_size: int = 16, save_freq: int = 10000,
          lr: float = 1e-5, lr_backbone: float = 1e-5, num_workers: int = 12,
          exp_name: str = "act-green-ball", chunk_size: int = 50,
          n_action_steps: int = 10, use_state: bool = True, seed: int = 0,
          overwrite: bool = False, resume: bool = False):
    import json
    import os
    import subprocess
    import time

    # --- preflight -------------------------------------------------------------------
    # Every check here is one that fails silently or expensively if left to training.
    info_path = f"{REMOTE_REPO}/meta/info.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"{info_path} missing. Upload first:\n"
            f"  modal run act/act_modal_train.py::upload")
    info = json.load(open(info_path))
    frames, episodes = info["total_frames"], info["total_episodes"]
    image_keys = sorted(k for k, v in info["features"].items() if v["dtype"] == "image")
    print(f"dataset : {episodes} episodes, {frames} frames, v{info['codebase_version']}, "
          f"fps {info['fps']}", flush=True)
    print(f"cameras : {image_keys}", flush=True)

    if not image_keys:
        raise SystemExit(
            "dataset declares no image features. ACT's one hard input requirement is at "
            "least one key starting with `observation.image`.")

    # The stats trap, inverted -- see the module docstring point 2. `pin_released_stats.py`
    # leaves a stats.json whose count is MolmoAct2's 273465 rather than this dataset's own.
    stats_path = f"{REMOTE_REPO}/meta/stats.json"
    if not os.path.exists(stats_path):
        raise SystemExit(
            f"{stats_path} missing. lerobot-train builds the normalisers from it and ACT has "
            f"no pretrained statistics to fall back on. If the collector was killed during "
            f"finalize(), rebuild locally with "
            f"`uv run python libero/fine_tune/rebuild_stats.py <dataset>` and re-upload.")
    stats = json.load(open(stats_path))
    stat_count = int(stats["action"]["count"][0])
    if stat_count != frames:
        raise SystemExit(
            f"meta/stats.json says count={stat_count} but info.json says "
            f"total_frames={frames}. These are almost certainly MolmoAct2's released LIBERO "
            f"statistics, pinned by pin_released_stats.py. Correct for the MolmoAct2 "
            f"fine-tune, WRONG here: ACT trains from scratch and must be normalised by this "
            f"dataset's own distribution. Restore meta/stats_measured.json and re-upload.")
    print(f"stats   : dataset's own (count {stat_count} == total_frames)", flush=True)

    if n_action_steps > chunk_size:
        raise SystemExit(
            f"n_action_steps ({n_action_steps}) > chunk_size ({chunk_size}): the policy "
            f"cannot serve more actions than one forward pass produces.")

    epochs = max_steps * batch_size / max(frames, 1)
    print(f"plan    : {max_steps} steps x batch {batch_size} = {epochs:.1f} epochs, "
          f"chunk {chunk_size}, serving horizon {n_action_steps}", flush=True)

    out_dir = f"/checkpoints/act/{exp_name}"

    # lerobot-train refuses a pre-existing output_dir when resume is False, and it checks
    # this in cfg.validate() -- i.e. AFTER the container is up and billing. The 1-step smoke
    # writes to the same default exp_name as the real run, so the real run hits this every
    # time unless something clears it. Fail here instead, ~2 s in, with the fix in the
    # message. (Cost of learning this the other way: one 27 s GPU container.)
    if resume:
        # RESUME. lerobot restores the step counter, optimizer, LR scheduler and the
        # EpisodeAwareSampler's position from `<ckpt>/training_state`, then runs
        # `for _ in range(step, cfg.steps)` -- so it continues to the ORIGINAL cfg.steps
        # baked into the saved train_config.json, not to this call's --max-steps.
        #
        # It resumes from `checkpoints/last`, which is a symlink to the most recent save. To
        # resume from an earlier step, repoint `last` at it first; there is no flag for
        # picking a checkpoint by number.
        #
        # Everything else on the command line is ignored: --config_path makes lerobot read
        # the whole config back from the checkpoint. That is why batch_size and num_workers
        # must not be "changed" here and quietly appear to take effect -- lerobot itself
        # warns that a differing batch size breaks per-rank sample-exactness.
        cfg_path = f"{out_dir}/checkpoints/last/pretrained_model/train_config.json"
        if not os.path.exists(cfg_path):
            raise SystemExit(
                f"--resume given but {cfg_path} does not exist. Nothing to resume from.")
        state = f"{out_dir}/checkpoints/last/training_state"
        if not os.path.isdir(state):
            raise SystemExit(
                f"{state} missing -- that checkpoint holds weights but no optimizer state, "
                f"so resuming would restart the optimizer from scratch at the saved step. "
                f"Refusing: that is not the same run continued.")
        done = json.load(open(f"{out_dir}/checkpoints/last/training_state/training_step.json")
                         )["step"] if os.path.exists(
            f"{out_dir}/checkpoints/last/training_state/training_step.json") else "?"
        print(f"resuming {out_dir} from step {done} (target from saved config)", flush=True)
        cmd = ["lerobot-train", f"--config_path={cfg_path}", "--resume=true"]
        print("Launching:", " ".join(cmd), flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd="/root")
        print(f"lerobot-train returned {proc.returncode} after {time.time() - t0:.0f}s",
              flush=True)
        checkpoints.commit()
        if proc.returncode != 0:
            raise SystemExit(f"resume failed with code {proc.returncode}")
        print(f"Done. Checkpoints at {out_dir}")
        return

    if os.path.isdir(out_dir):
        existing = sorted(os.listdir(f"{out_dir}/checkpoints")) if os.path.isdir(
            f"{out_dir}/checkpoints") else []
        if not overwrite:
            raise SystemExit(
                f"{out_dir} already exists (checkpoints: {existing}) and lerobot-train will "
                f"not write into it. Either pick a new name with --exp-name, or pass "
                f"--overwrite to delete it. NOTE --overwrite is destructive: a scored "
                f"checkpoint you have not copied elsewhere is gone.")
        import shutil
        print(f"--overwrite: deleting {out_dir} (checkpoints: {existing})", flush=True)
        shutil.rmtree(out_dir)
        checkpoints.commit()

    cmd = [
        "lerobot-train",
        # No --policy.path: from scratch. The only pretrained weights are the backbone's
        # ImageNet initialisation, which ACTConfig sets by default.
        "--policy.type=act",
        f"--dataset.repo_id=greenbox/{REMOTE_REPO.rsplit('/', 1)[-1]}",
        f"--dataset.root={REMOTE_REPO}",
        f"--output_dir={out_dir}",
        f"--job_name={exp_name}",
        "--policy.device=cuda",
        f"--batch_size={batch_size}",
        f"--steps={max_steps}",
        f"--save_freq={save_freq}",
        f"--num_workers={num_workers}",
        f"--seed={seed}",
        "--save_checkpoint=true",
        "--wandb.enable=false",
        # lerobot pushes the finished policy to the Hub by default, which 401s in a container
        # with no token -- AFTER training and checkpointing have both succeeded, so the run
        # exits non-zero and looks like a training failure when nothing is wrong.
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={chunk_size}",
        # Baked in so the SAVED config serves the way we evaluate it, rather than depending
        # on the server remembering to slice. See the module docstring.
        f"--policy.n_action_steps={n_action_steps}",
        f"--policy.optimizer_lr={lr}",
        f"--policy.optimizer_lr_backbone={lr_backbone}",
    ]
    # No scheduler flag. Unlike SmolVLA -- where the checkpoint ships decay_steps=30000 and a
    # 3000-step run silently never decays (smolvla_modal_train.py's comment) -- ACT's preset
    # is a constant LR with no schedule to mis-pin. Nothing to pass, and passing
    # `--policy.scheduler_decay_steps` here is an error, not a no-op.

    if not use_state:
        # The ablation from README sec.3.2. On one stereotyped task the cheapest way to drive
        # the L1 loss down is to regress the action from proprioception plus phase and ignore
        # the cameras -- which fits the training set perfectly and emits one averaged
        # trajectory at test time, i.e. the same failure `libero/PROGRESS.md` sec.4 describes
        # for a frozen VLM. Dropping the state forces the policy through the images.
        #
        # NOT the first run. Its signature (tight on the nominal ball, wide on a randomised
        # one) has to be observed on the standard recipe before spending GPU time on the fix.
        #
        # HOW, and why it is this ugly: lerobot has no feature-exclusion flag. Neither
        # DatasetConfig nor PreTrainedConfig carries one -- `input_features` is normally left
        # None and INFERRED from the dataset, which would pick observation.state back up. The
        # only lever is to stop the inference by supplying the dict outright, which then also
        # obliges us to supply output_features (the same code path fills both).
        #
        # UNVERIFIED until this mode's own 1-step smoke passes. If draccus rejects the nested
        # dict on the command line, the fallback is a dataset copy with the column dropped
        # from meta/{info,stats}.json -- more work, no ambiguity.
        img_feats = ", ".join(
            f"{k}: {{type: VISUAL, shape: [3, {info['features'][k]['shape'][0]}, "
            f"{info['features'][k]['shape'][1]}]}}"
            for k in image_keys)
        action_dim = info["features"]["action"]["shape"][0]
        print("ABLATION: observation.state excluded from the policy's inputs", flush=True)
        cmd += [f"--policy.input_features={{{img_feats}}}",
                f"--policy.output_features={{action: {{type: ACTION, "
                f"shape: [{action_dim}]}}}}"]

    print("Launching:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/root")
    dt = time.time() - t0
    print(f"lerobot-train returned {proc.returncode} after {dt:.0f}s "
          f"({dt / max(max_steps, 1):.3f}s per step including startup and saves)", flush=True)
    # Commit BEFORE inspecting the exit code, so a run that trained and checkpointed
    # successfully and then died on some tail step still leaves its weights on the volume.
    checkpoints.commit()
    if proc.returncode != 0:
        raise SystemExit(f"training failed with code {proc.returncode}")

    saved = sorted(os.listdir(f"{out_dir}/checkpoints")) if os.path.isdir(
        f"{out_dir}/checkpoints") else []
    print(f"Done. Checkpoints on volume molmoact2-checkpoints at {out_dir}: {saved}")
    print("Serve one with:")
    print(f"  ACT_CHECKPOINT={out_dir}/checkpoints/{saved[-1] if saved else '<step>'}"
          f"/pretrained_model modal deploy act/act_modal.py")


@app.local_entrypoint()
def upload():
    """Push the dataset onto the volume. Separate from training so a re-run does not
    re-upload, and so the upload can be verified before GPU time is spent."""
    import json
    import pathlib
    import subprocess

    root = pathlib.Path(DATASET_DIR)
    stats = root / "meta" / "stats.json"
    if not stats.exists():
        raise SystemExit(
            f"{stats} missing -- the collector was probably killed inside finalize(). "
            f"Rebuild it from the parquets first:\n"
            f"  uv run python libero/fine_tune/rebuild_stats.py {DATASET_DIR}")
    info = json.loads((root / "meta" / "info.json").read_text())
    count = int(json.loads(stats.read_text())["action"]["count"][0])
    if count != info["total_frames"]:
        raise SystemExit(
            f"stats.json count={count} != total_frames={info['total_frames']} -- pinned "
            f"MolmoAct2 statistics. Restore meta/stats_measured.json before uploading.")
    print(f"{DATASET_DIR}: {info['total_episodes']} episodes, {info['total_frames']} frames")
    print(f"uploading -> molmoact2-lerobot-data:{REMOTE_REPO}")
    subprocess.run(
        ["modal", "volume", "put", "--force", "molmoact2-lerobot-data",
         DATASET_DIR, REMOTE_REPO.replace("/data", "")],
        check=True,
    )


@app.local_entrypoint()
def main(max_steps: int = 60000, batch_size: int = 16, save_freq: int = 10000,
         lr: float = 1e-5, lr_backbone: float = 1e-5, num_workers: int = 12,
         exp_name: str = "act-green-ball", chunk_size: int = 50, n_action_steps: int = 10,
         use_state: bool = True, gpu: str = "L4", seed: int = 0, overwrite: bool = False,
         resume: bool = False):
    train.with_options(gpu=gpu).remote(
        max_steps=max_steps, batch_size=batch_size, save_freq=save_freq, lr=lr,
        lr_backbone=lr_backbone, num_workers=num_workers, exp_name=exp_name,
        chunk_size=chunk_size, n_action_steps=n_action_steps, use_state=use_state, seed=seed,
        overwrite=overwrite, resume=resume,
    )
