"""Fine-tune `allenai/MolmoAct2-LIBERO` on our green-ball demos, on a rented Modal GPU.

Sibling of the repo root's `droid/phase4_modal_train.py` (which fine-tunes the DROID checkpoint
on the DROID-convention dataset) and of `libero/libero_modal.py` (which serves the stock
LIBERO checkpoint). Kept separate for the same reason those two are: the DROID path is
working and nothing here should be able to break it.

Three things differ from `droid/phase4_modal_train.py`, and only three:

  1. START_CHECKPOINT   -> "allenai/MolmoAct2-LIBERO"
  2. DEFAULT_MIXTURE    -> "libero_green_ball" (registered by us in data_mixtures.py under
                           tag "libero" -- see the comment there; a new tag would fail
                           silently at serving time)
  3. --norm_mode        -> q01_q99, the trainer's own default. phase4 had to override it to
                           min_max because the repo-root v2.1 writer computes no
                           percentiles; `lerobot_v30_writer.py` computes q01/q10/q50/q90/q99,
                           so the default works and we keep the pretrained convention.

WHY LoRA, NOT ACTION-EXPERT-ONLY
--------------------------------
PROGRESS.md sec.4: a 500-step action-expert-only run drove flow loss 0.176 -> 0.01 while
task success stayed at zero. With the VLM frozen the expert can only learn the average
trajectory -- it cannot learn to LOOK for a differently-placed ball. The observed failure
(README sec.9) is a perception/grounding failure: the arm stalls 33-45 mm lateral of the
ball and never closes the gripper. That is upstream of the action expert.

OVERFITTING IS THE REAL RISK HERE, NOT UNDERTRAINING
----------------------------------------------------
~19k frames of ONE task. README sec.6.1 predicts where it breaks first: the rotation
channels, 3-6x narrower than the released distribution, collapsing toward zero. The
defences, in order of how much they matter:

  * Data.        a4 is ~90 episodes, 3x a3, with randomised ball position and bin layout.
                 Local CPU, free -- always the first lever to pull.
  * Steps.       Budget for roughly half an epoch to an epoch. Loss going to ~0 on this
                 dataset means memorisation, not skill.
  * Capacity.    --lora_rank 32, half the trainer's default.
  * Selection.   --save-interval small, then score EACH checkpoint in closed loop and keep
                 the best. The last checkpoint is not automatically the best one.
  * Augmentation. --img_aug stays at the trainer's default "full" (spatial + photometric).

GPU CHOICE, and why a fixed time budget inverts the usual reasoning
-------------------------------------------------------------------
With a FIXED wall-clock window, "cheapest per hour" and "most training" are different
objectives: a cheaper, slower card simply fits fewer optimizer steps into the same window.
We optimise for steps completed inside the window, so: H100. Serving fits on an L4 (24 GB)
because inference is just 5B params at bf16 = ~10 GB, but training adds LoRA optimizer
state, a fully-trained action expert, activations for two 256x256 images per sample, and
the ~10 GB checkpoint load itself. Modal also bills warm idle containers at the GPU rate,
so a fast card that exits sooner is cheaper in total than a slow one that lingers.

Budget backwards from the smoke run: `--max-steps 1 --save-interval 1` reports
seconds/step, then set --max-steps so the run lands inside the window with margin for the
checkpoint save and container start.

=========================  PREREQUISITES  ======================================

    cd molmoact2 && git submodule update --init lerobot && cd ..     # once

    MUJOCO_GL=egl uv run python libero/fine_tune/collect_finetune_data.py \
        --out libero/fine_tune/a4 --reach 40 --noise 30 --recover 20 --seed 0
    uv run python libero/fine_tune/pin_released_stats.py libero/fine_tune/a4
    modal volume create molmoact2-lerobot-data          # first time only
    modal volume put molmoact2-lerobot-data libero/fine_tune/a4 /greenbox/libero_green_ball

No v2.1 -> v3.0 conversion step here: `lerobot_v30_writer.py` already emits v3.0, which is
what this repo's lerobot fork requires.

===============================  RUN  ==========================================

    modal run libero/libero_modal_train.py --max-steps 1 --save-interval 1   # SMOKE
    modal run libero/libero_modal_train.py --max-steps 400 --save-interval 100

    modal volume get molmoact2-checkpoints checkpoints/finetune/<exp_name> ./out
"""

import sys
from pathlib import Path

import modal

# The shared image definitions live at the repo root, and Modal re-imports this
# module inside the container -- where infra/ lands on /root via with_infra().
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.modal_images import molmoact_experiments_image, with_infra

# Same image recipe as droid/phase4_modal_train.py -- the training package pins
# transformers>=5.3, a different pin from the inference server's 4.57, so this cannot
# share phase3/libero_modal's image.
image = with_infra(
    molmoact_experiments_image({
        "LEROBOT_DATA_ROOT": "/data",
        "LEROBOT_VIDEO_BACKEND": "pyav",
        "WANDB_MODE": "disabled",
        "WANDB_PROJECT": "greenbox-libero-green-ball",
        "WANDB_ENTITY": "greenbox",
    })
    .pip_install("debugpy")  # train_lerobot.py imports it unconditionally
)

hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
lerobot_data = modal.Volume.from_name("molmoact2-lerobot-data", create_if_missing=True)
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)

app = modal.App("molmoact2-libero-train")

START_CHECKPOINT = "allenai/MolmoAct2-LIBERO"
DEFAULT_MIXTURE = "libero_green_ball"
REPO_PATH = "/data/greenbox/libero_green_ball"


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/cache/huggingface": hf_cache,
        "/data": lerobot_data,
        "/checkpoints": checkpoints,
    },
    timeout=6 * 60 * 60,
)
def train(mode: str = "lora", gpus: int = 1, max_steps: int = 400,
          device_batch_size: int = 1, global_batch_size: int = 16,
          exp_name: str = "libero-green-ball", save_interval: int = 100,
          log_interval: int = 10, lora_rank: int = 32,
          mixture: str = DEFAULT_MIXTURE):
    """Run train_lerobot.py under torchrun inside the container.

    mode:
      lora           -- LoRA adapters on the VLM path + fully trained action expert. The
                        default and the recommended one: our failure is grounding, which a
                        frozen VLM cannot fix.
      lora_frozen_ae -- LoRA on the VLM only, action expert frozen. Diagnostic: isolates
                        whether perception or motion generation is the bottleneck.
      ae_only        -- action expert only. Kept for comparison; PROGRESS sec.4 shows it
                        drives the loss down without learning the task.
    """
    import json
    import os
    import subprocess
    import time

    env = os.environ.copy()
    # The olmo data pipeline reads these data-root env vars and KeyErrors if unset. We use
    # none of those auxiliary corpora, so they only need to point somewhere that exists.
    for var, path in {
        "MOLMO_DATA_DIR": "/tmp/molmo_data",
        "LEROBOT_DEPTH_DATA_ROOT": "/tmp/lerobot_depth",
        "SPATIAL_DATA_HOME": "/tmp/spatial_data",
    }.items():
        env.setdefault(var, path)
        os.makedirs(env[var], exist_ok=True)

    # phase4's lora_smoke run hung partway through its first checkpoint save (7 of 16
    # shards, then no progress). olmo's distributed checkpoint writer defaults to
    # min(16, cpu_count+4) concurrent shard writers and Modal Volumes are FUSE-backed.
    env.setdefault("OLMO_NUM_THREADS", "4")

    # Fail loudly and immediately if the dataset was never uploaded, rather than after the
    # 10 GB checkpoint load. Also echo what normalisation the data carries: if
    # pin_released_stats.py was not run, action q01 will read about -0.5 instead of the
    # released -0.679, and the run will silently rebuild the normaliser from our data.
    stats_path = f"{REPO_PATH}/meta/stats.json"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"{stats_path} missing. Upload the dataset first:\n"
            f"  modal volume put molmoact2-lerobot-data libero/fine_tune/a4 "
            f"/greenbox/libero_green_ball")
    stats = json.load(open(stats_path))
    print("dataset action q01:", [round(v, 3) for v in stats["action"]["q01"]], flush=True)
    print("dataset action q99:", [round(v, 3) for v in stats["action"]["q99"]], flush=True)
    print("stats count:", stats["action"].get("count"),
          "(273465 => released stats are pinned)", flush=True)
    print("measured backup present:",
          os.path.exists(f"{REPO_PATH}/meta/stats_measured.json"), flush=True)
    info = json.load(open(f"{REPO_PATH}/meta/info.json"))
    frames, episodes = info["total_frames"], info["total_episodes"]
    epochs = max_steps * global_batch_size / max(frames, 1)
    print(f"dataset: {episodes} episodes, {frames} frames, codebase "
          f"{info['codebase_version']}, fps {info['fps']}", flush=True)
    print(f"plan: {max_steps} steps x global batch {global_batch_size} = "
          f"{epochs:.2f} epochs", flush=True)

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
        f"--save_interval={save_interval}", "--save_num_checkpoints_to_keep=-1",
        f"--save_folder={save_folder}",
        "--packing=false", "--dynamic_seq_len=true",
        # The trainer's own default, and the convention the released checkpoint was
        # normalised under. With pin_released_stats.py run on the dataset, the q01/q99 the
        # normaliser is built from are the released ones.
        "--norm_mode=q01_q99",
        # Explicit rather than inherited: spatial + photometric augmentation is one of the
        # few free defences against overfitting 90 episodes, so it should be visible in the
        # command line, not a default someone can flip without noticing.
        "--img_aug=full",
    ]

    if mode == "lora":
        # Action expert at 1e-4, the trainer's own default, rather than phase4's 5e-5. The
        # $5 budget buys ~1200 samples (about 0.06 epochs), so the binding risk is
        # UNDERtraining, not overfitting -- at this step count a conservative LR mostly
        # buys a checkpoint indistinguishable from the base one. The VLM LoRA path stays
        # at 5e-5: that is where a too-large step damages general grounding, which is the
        # one capability we are relying on the base checkpoint for.
        flags = ["--ft_vlm=true", "--ft_action_expert=true", "--ft_embedding=lm_head",
                 "--lora_enable=true", f"--lora_rank={lora_rank}",
                 "--llm_learning_rate=5e-5", "--vit_learning_rate=5e-5",
                 "--connector_learning_rate=5e-5", "--action_expert_learning_rate=1e-4"]
    elif mode == "lora_frozen_ae":
        flags = ["--ft_vlm=true", "--ft_action_expert=false", "--ft_embedding=lm_head",
                 "--lora_enable=true", f"--lora_rank={lora_rank}",
                 "--llm_learning_rate=5e-5", "--vit_learning_rate=5e-5",
                 "--connector_learning_rate=5e-5"]
    elif mode == "ae_only":
        flags = ["--ft_vlm=false", "--ft_action_expert=true", "--ft_embedding=none",
                 "--lora_enable=false", "--action_expert_learning_rate=5e-5"]
    else:
        raise ValueError(f"unknown mode {mode!r}; use lora | lora_frozen_ae | ae_only")

    cmd = common + flags
    print("Launching:", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd="/root/experiments", check=True, env=env)
    dt = time.time() - t0
    print(f"train_lerobot.py returned after {dt:.0f}s "
          f"({dt / max(max_steps, 1):.1f}s per step including startup and save)",
          flush=True)

    checkpoints.commit()
    print(f"Done. Checkpoints on Volume molmoact2-checkpoints at {save_folder}")


@app.local_entrypoint()
def main(mode: str = "lora", gpus: int = 1, max_steps: int = 400,
         device_batch_size: int = 1, global_batch_size: int = 16,
         save_interval: int = 100, log_interval: int = 10, lora_rank: int = 32,
         exp_name: str = "libero-green-ball", mixture: str = DEFAULT_MIXTURE,
         gpu: str = "H100"):
    """`modal run libero/libero_modal_train.py [--max-steps ...] [--gpu ...]`."""
    gpu_spec = gpu if gpus == 1 else f"{gpu}:{gpus}"
    train.with_options(gpu=gpu_spec).remote(
        mode=mode, gpus=gpus, max_steps=max_steps,
        device_batch_size=device_batch_size, global_batch_size=global_batch_size,
        save_interval=save_interval, log_interval=log_interval, lora_rank=lora_rank,
        exp_name=exp_name, mixture=mixture,
    )
