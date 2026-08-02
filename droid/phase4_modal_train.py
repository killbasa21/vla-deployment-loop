"""
Phase 4 (Modal variant): fine-tune MolmoAct2-DROID on our IK-scripted green-ball
pick-and-place LeRobot dataset, on a rented Modal GPU -- the training counterpart to
droid/phase3_modal.py (which serves inference on Modal).

This does not reimplement the trainer. It wraps molmoact2/experiments'
`launch_scripts/train_lerobot.py` inside a Modal container and runs it via `torchrun`,
starting from the same `allenai/MolmoAct2-DROID` checkpoint phase3 serves and pointing
it at the `green_ball_pick` mixture we registered in
`molmoact2/experiments/launch_scripts/data_mixtures.py`. The output checkpoint is a
MolmoAct2-DROID-shaped snapshot, so droid/phase3_modal.py can serve it unchanged afterwards.

=========================  PREREQUISITES (do these first)  =====================

1. Populate the lerobot submodule the trainer installs (it ships empty in this repo):
       cd molmoact2 && git submodule update --init lerobot && cd ..

2. Upload the collected dataset to the Modal data Volume (repo_id path must match the
   mixture: greenbox/green_ball_pick under LEROBOT_DATA_ROOT):
       uv run python droid/phase4_collect_demos.py --episodes 300 --out data/green_ball_pick
       modal volume create molmoact2-lerobot-data            # first time only
       modal volume put molmoact2-lerobot-data data/green_ball_pick /greenbox/green_ball_pick

3. (Optional) Weights & Biases logging -- omit to run with wandb disabled:
       modal secret create wandb-secret WANDB_API_KEY=...

===============================  RUN  ==========================================

    modal setup                                              # one-time auth
    # action-expert-only, 1 GPU (cheapest)
    modal run droid/phase4_modal_train.py
    # LoRA on the VLM path + full action expert
    modal run droid/phase4_modal_train.py --mode lora  --gpus 1
    # full fine-tune (8x80GB, per Ai2's own recipe)
    modal run droid/phase4_modal_train.py --mode full  --gpus 8
    modal run droid/phase4_modal_train.py --max-steps 20 --mode ae_only   # tiny smoke run

Checkpoints land on the `molmoact2-checkpoints` Volume under
checkpoints/finetune/<exp_name>/ ; pull them locally with
    modal volume get molmoact2-checkpoints checkpoints/finetune/<exp_name> ./out

NOTE: this is a best-effort scaffold. A real run costs real GPU money and depends on
the two prerequisites above being done; it has not been executed end-to-end from here.
Watch the first build (heavy: torch + the whole experiments/ + lerobot install) and the
first step's logs, and tune --device-batch-size / --gpus to your rented GPU's VRAM.
"""

import modal

# --- Image ------------------------------------------------------------------
# The *training* package needs transformers>=5.3 (experiments/pyproject.toml), which is
# a DIFFERENT pin from the inference server's 4.57 in droid/phase3_modal.py -- so this is a
# separate image, not a reuse of the serving one. torch from the cu121 index to match
# the GPU; everything else comes from experiments' own `.[all]` extra + lerobot's.
image = (
    # Python 3.12: the vendored lerobot fork requires >=3.12 (the inference server in
    # droid/phase3_modal.py can use 3.11, but the training-side lerobot install cannot).
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache/huggingface",
        "LEROBOT_DATA_ROOT": "/data",
        "LEROBOT_VIDEO_BACKEND": "pyav",
        "TOKENIZERS_PARALLELISM": "false",
        # wandb is off by default (no secret required). The trainer still constructs a
        # WandbConfig that resolves ${oc.env:WANDB_PROJECT}/${WANDB_ENTITY}, so those must
        # be set even when disabled. To turn logging ON: create a wandb secret, set
        # secrets=[modal.Secret.from_name("wandb-secret")] on @app.function, and flip
        # WANDB_MODE to "online".
        "WANDB_MODE": "disabled",
        "WANDB_PROJECT": "greenbox-green-ball",
        "WANDB_ENTITY": "greenbox",
    })
    .pip_install("hf-transfer>=0.1.8")
    # Ship the training source (experiments/ + the now-populated lerobot submodule) and
    # install both editable, exactly as experiments/README.md's Setup section does.
    .add_local_dir("molmoact2/experiments", remote_path="/root/experiments", copy=True)
    .run_commands(
        "cd /root/experiments && pip install -e '.[all]'",
        # [async] only (grpcio + matplotlib) -- the training dataloader. We deliberately
        # drop the [libero] extra the README uses: it pulls hf-libero -> robosuite ->
        # egl_probe, which needs cmake + OpenGL headers to compile and is only for LIBERO
        # *simulator* eval. Training on our own LeRobot dataset imports none of it
        # (verified: no libero/robosuite imports in train_lerobot.py or the data path).
        "cd /root/experiments && pip install -e './lerobot[async]'",
    )
    # train_lerobot.py does `import debugpy` unconditionally (only used under --debugger),
    # but it isn't in experiments' deps. Appended as its own layer so the heavy installs
    # above stay cached. Add any other such runtime-missing imports here the same way.
    .pip_install("debugpy")
)

# Volumes: reuse the inference HF cache (shares the ~22GB DROID base checkpoint), plus a
# data Volume (LEROBOT_DATA_ROOT) and a checkpoints Volume for training output.
hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
lerobot_data = modal.Volume.from_name("molmoact2-lerobot-data", create_if_missing=True)
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)

app = modal.App("molmoact2-droid-train")

# Base checkpoint for DROID-embodiment fine-tuning (experiments/README.md checkpoint
# table). Same weights droid/phase3_modal.py serves -- we adapt them, then serve the result.
START_CHECKPOINT = "allenai/MolmoAct2-DROID"
DEFAULT_MIXTURE = "green_ball_pick"  # registered in data_mixtures.py by us


@app.function(
    image=image,
    gpu="H100",  # overridden per-call below via .options(gpu=...)
    volumes={
        "/cache/huggingface": hf_cache,
        "/data": lerobot_data,
        "/checkpoints": checkpoints,
    },
    timeout=24 * 60 * 60,  # training is long-running
)
def train(mode: str = "ae_only", gpus: int = 1, max_steps: int = 50000,
          device_batch_size: int = 1, global_batch_size: int = 8,
          exp_name: str = "green-ball", save_interval: int = 5000,
          log_interval: int = 20, mixture: str = DEFAULT_MIXTURE):
    """Run train_lerobot.py under torchrun inside the container.

    mode:
      ae_only        -- freeze the VLM/vision/connector/embeddings, train only the
                        continuous action expert. Cheapest, fits 1 GPU, and the right call
                        when a new dataset mainly changes control (our case). README
                        "Action-Expert-Only".
      lora           -- LoRA adapters on the VLM path + full (unfrozen) action expert.
                        README "LoRA".
      lora_frozen_ae -- LoRA adapters on the VLM path only, action expert FROZEN
                        (ft_action_expert=false). Isolates whether vision/language
                        grounding (where in the image is the target, what does the
                        instruction refer to) is the failure mode, vs. motion generation
                        -- the frozen-AE checkpoint can't touch how it moves at all, only
                        what it decides to do based on what it sees. Not in the README;
                        added after `lora` (ft_action_expert=true) failed to visually
                        locate the target in sim (see data/lora_adapter/README.md).
      full           -- full fine-tune of everything (wants ~8x80GB). README
                        "Full Fine-Tuning".
    """
    import os
    import subprocess

    # The olmo data pipeline reads these data-root env vars (the README exports them);
    # unset -> KeyError. Our action-expert-only run uses none of these auxiliary corpora
    # (molmo VLM data / depth / spatial), so they just need to point *somewhere* that
    # exists. Set here (not in the image) so changing them never rebuilds the heavy image.
    env = os.environ.copy()
    for var, path in {
        "MOLMO_DATA_DIR": "/tmp/molmo_data",
        "LEROBOT_DEPTH_DATA_ROOT": "/tmp/lerobot_depth",
        "SPATIAL_DATA_HOME": "/tmp/spatial_data",
    }.items():
        env.setdefault(var, path)
        os.makedirs(env[var], exist_ok=True)

    # olmo's checkpoint writer (distributed_checkpointing.py) defaults its thread pool to
    # min(16, cpu_count+4) concurrent shard writers. The lora_smoke run hung indefinitely
    # partway through the first checkpoint save (7/16 shards written, then zero progress
    # for 10+ minutes) -- Modal Volumes are FUSE-backed and don't reliably handle that much
    # write/fsync concurrency. ae_train never hit this because its checkpoint was smaller
    # and its save path never got shard-count-limited. Capping this is a cheap, safe
    # mitigation; revisit if saves are unexpectedly slow instead of hung.
    env.setdefault("OLMO_NUM_THREADS", "4")

    save_folder = f"/checkpoints/checkpoints/finetune/{exp_name}"

    common = [
        "torchrun", "--standalone", f"--nproc-per-node={gpus}",
        "launch_scripts/train_lerobot.py",
        START_CHECKPOINT, mixture,
        f"--wandb.name={exp_name}",
        f"--max_duration={max_steps}",
        f"--device_batch_size={device_batch_size}",
        f"--global_batch_size={global_batch_size}",
        f"--log_interval={log_interval}",
        "--num_workers=4", "--pin_memory=true", "--data.timeout=900",
        f"--save_interval={save_interval}", "--save_num_checkpoints_to_keep=5",
        f"--save_folder={save_folder}",
        "--packing=false", "--dynamic_seq_len=true",
        # Our droid/lerobot_writer.py computes min/max/mean/std but not q01/q99 percentiles, so
        # override the trainer's default norm_mode=q01_q99 (which would raise "requires
        # stats keys ['q01','q99']"). min_max maps each dim to a bounded range from the
        # observed extremes -- fine for our clean scripted data, and the fine-tuned
        # checkpoint saves this norm config so inference stays consistent.
        "--norm_mode=min_max",
    ]

    if mode == "ae_only":
        flags = ["--ft_vlm=false", "--ft_action_expert=true", "--ft_embedding=none",
                 "--lora_enable=false", "--action_expert_learning_rate=5e-5"]
    elif mode == "lora":
        flags = ["--ft_vlm=true", "--ft_action_expert=true", "--ft_embedding=lm_head",
                 "--lora_enable=true", "--lora_rank=64",
                 "--llm_learning_rate=5e-5", "--vit_learning_rate=5e-5",
                 "--connector_learning_rate=5e-5", "--action_expert_learning_rate=5e-5"]
    elif mode == "lora_frozen_ae":
        # Same LoRA/VLM hyperparameters as `lora`, but ft_action_expert=false freezes the
        # action expert entirely (no action_expert_learning_rate -- nothing to step). Only
        # the VLM (via LoRA) and the lm_head embedding adapt.
        flags = ["--ft_vlm=true", "--ft_action_expert=false", "--ft_embedding=lm_head",
                 "--lora_enable=true", "--lora_rank=64",
                 "--llm_learning_rate=5e-5", "--vit_learning_rate=5e-5",
                 "--connector_learning_rate=5e-5"]
    elif mode == "full":
        flags = ["--ft_vlm=true", "--ft_action_expert=true", "--ft_embedding=lm_head",
                 "--lora_enable=false",
                 "--llm_learning_rate=1e-5", "--vit_learning_rate=5e-6",
                 "--connector_learning_rate=5e-6", "--action_expert_learning_rate=5e-5"]
    else:
        raise ValueError(f"unknown mode {mode!r}; use ae_only | lora | lora_frozen_ae | full")

    cmd = common + flags
    print("Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd="/root/experiments", check=True, env=env)

    # Persist the output so `modal volume get` sees it after the container exits.
    checkpoints.commit()
    print(f"Done. Checkpoints under Volume molmoact2-checkpoints at {save_folder}")


@app.function(
    image=image,
    volumes={"/data": lerobot_data},
    timeout=3600,
)
def convert_v21_to_v30(repo_id: str = "greenbox/green_ball_pick"):
    """One-time: our droid/lerobot_writer.py emits LeRobot v2.1, but this repo's lerobot fork
    only loads v3.0 (BackwardCompatibilityError otherwise). Run the fork's own converter
    on the volume copy, in place (--push-to-hub=false). Leaves a `<name>_old` v2.1 backup
    beside it. Idempotent enough: re-running after conversion would fail the v2.1 check,
    so only run this once per uploaded dataset."""
    import subprocess

    root = f"/data/{repo_id}"
    subprocess.run(
        ["python", "-m", "lerobot.datasets.v30.convert_dataset_v21_to_v30",
         f"--repo-id={repo_id}", f"--root={root}", "--push-to-hub=false"],
        cwd="/root/experiments", check=True,
    )
    lerobot_data.commit()
    print(f"Converted {root} to LeRobot v3.0 (v2.1 backup at {root}_old)")


@app.local_entrypoint()
def convert(repo_id: str = "greenbox/green_ball_pick"):
    """`modal run droid/phase4_modal_train.py::convert` -- convert the uploaded v2.1 dataset to
    v3.0 on the volume. Run once before training."""
    convert_v21_to_v30.remote(repo_id=repo_id)


@app.local_entrypoint()
def main(mode: str = "ae_only", gpus: int = 1, max_steps: int = 50000,
         device_batch_size: int = 1, global_batch_size: int = 8,
         save_interval: int = 5000, log_interval: int = 20,
         exp_name: str = "green-ball", mixture: str = DEFAULT_MIXTURE):
    """`modal run droid/phase4_modal_train.py [--mode ...] [--gpus ...] [--max-steps ...]`.

    Selects the GPU count on the fly so `--gpus 8` requests an 8-GPU container for full
    fine-tuning while the default single-GPU run stays cheap."""
    gpu_spec = "H100" if gpus == 1 else f"H100:{gpus}"
    train.with_options(gpu=gpu_spec).remote(
        mode=mode, gpus=gpus, max_steps=max_steps,
        device_batch_size=device_batch_size, global_batch_size=global_batch_size,
        save_interval=save_interval, log_interval=log_interval, exp_name=exp_name,
        mixture=mixture,
    )
