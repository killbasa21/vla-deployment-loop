"""Fine-tune `HuggingFaceVLA/smolvla_libero` on our green-ball demos, on a rented Modal GPU.

Counterpart to `smolvla_modal.py` (which SERVES the stock checkpoint). Same model, same
conventions; this one trains it.

WHY SMOLVLA AND NOT MOLMOACT2
-----------------------------
`FINE_TUNE_LEARNINGS.md` sec.5.5 records the binding constraint on the MolmoAct2 path: a $5
budget bought 150 steps = **0.06 epochs**, at which point the honest read was that the
checkpoint had "barely moved off the base". SmolVLA is 450M against MolmoAct2's 5.57B, so
the same money buys a real run. It is also already LIBERO-fine-tuned, on a robosuite Panda,
with `observation.state` 8-D and `action` 7-D delta-EE -- the conventions our dataset
already speaks.

WHICH DATASET
-------------
`a5`, not `a4`. a4 was collected through Route A differential IK; inference now runs
robosuite's OSC_POSE port. The two plants realise ~72% and ~12.3% of a commanded per-tick
displacement, so their labels differ by ~6x and are NOT interchangeable -- see
`collect_finetune_data.py`'s header. a5 is the OSC-native re-collection.

WHAT IS TRAINED
---------------
`--mode` picks, mirroring `libero/libero_modal_train.py`'s vocabulary:

  expert    Action expert only, VLM frozen. This is HuggingFace's OWN recipe for
            smolvla_libero (its config.json records `freeze_vision_encoder: true,
            train_expert_only: true`). Cheapest, and no backward pass through the VLM.
  lora      LoRA adapters on the VLM + full action expert. lerobot 0.6.0's SmolVLAPolicy
            supports PEFT natively (`wrap_with_peft`, `_get_default_peft_targets`).

Which one is right depends on WHERE the stock checkpoint fails, and the measured baseline
says grounding: over 3 randomised ball positions the closest lateral approach was 10.7 mm,
108.2 mm and 46.4 mm. Missing by 108 mm at an unseen ball position is a perception failure,
not an action-generation one, and `PROGRESS.md` sec.4 already established that a frozen VLM
cannot fix that -- with the VLM frozen the expert can only learn the average trajectory over
the randomisation box. Hence `lora` is the default here, unlike HuggingFace's recipe.

NORMALISATION -- the trap this project keeps re-learning
--------------------------------------------------------
`--norm-stats` chooses which statistics the STATE/ACTION normalisers are built from:

  checkpoint  (default) keep the ones baked into smolvla_libero. The action expert was
              trained under them; rebuilding from 30 episodes of one task hands it a
              different affine map than it learned, and it spends capacity relearning a
              transform that already exists.
  dataset     rebuild from a5's own stats.json.

This is the same decision `pin_released_stats.py` makes for MolmoAct2, and getting it wrong
is a silent failure: the model returns actions of the correct shape that are simply in the
wrong space (PROGRESS.md sec.5).

OVERFITTING
-----------
30 episodes of ONE task, one instruction string. Defences, all free:
  * `--save-freq` small, then score EACH checkpoint in closed loop and keep the best. The
    last checkpoint is not automatically the best one.
  * The scheduler's `decay_steps` is pinned to the ACTUAL run length below. The checkpoint
    ships `decay_steps: 30000`; run 3000 steps against that and the LR never decays, it
    just stops mid-schedule.
  * Watch the rotation channels. a5 holds one top-down orientation throughout, so its
    drx/dry/drz spans are far narrower than released LIBERO's -- that is where a collapse
    shows up first.

Usage:
    modal run smolvla_libero/smolvla_modal_train.py::upload      # push a5 to the volume
    modal run smolvla_libero/smolvla_modal_train.py --max-steps 1 --save-freq 1   # SMOKE
    modal run smolvla_libero/smolvla_modal_train.py --max-steps 3000 --save-freq 750
"""

import modal

# a7, the third dataset in this sequence, and the reason for each move is worth keeping:
#
#   a5  scale 0.05, expert slowed 2.5x   539 ticks/ep   trained fine, served SLOW
#   a6  scale 0.20, distance-retimed     161 ticks/ep   FAST, and misses the ball
#   a7  scale 0.10, distance-retimed     ~256 ticks/ep  the middle setting
#
# a6 solved the speed problem and bought a new one: at 0.20 m per action unit, one unit of
# normalised policy error is 40 mm on the table against a5's 15 mm, so the same policy
# quality that grasps at a5's scale closes on air at a6's (PROGRESS.md sec.23.5). 0.10
# halves that to ~27 mm while still running ~2.1x faster than a5.
#
# a7 is also 60 episodes rather than 30. The competing explanation for a6's miss is simply
# too little data for grounding, and CPU collection is free, so this run tests both.
#
# NOTE the serving client must run --delta-pos-scale 0.10 to match, and --randomize-bins,
# since a7 shuffles the bin layout every episode.
DATASET_DIR = "smolvla_libero/data/a7_smolvla"   # local, produced by convert_dataset.py
REMOTE_REPO = "/data/greenbox/green_ball_a7"     # its own path, like a6 got its own: an
                                                 # overwritten volume repo leaves no way to
                                                 # A/B the datasets against each other.
BASE_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"

# Same recipe as smolvla_modal.py so the heavy pip layers are a CACHE HIT rather than a
# 15-25 minute rebuild. Any divergence here (a different .env, an extra package earlier in
# the chain) invalidates the cache and costs that time back.
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
    # `[dataset]` on top of the serving image's `[smolvla]`: without it lerobot-train dies
    # at import with "ImportError: 'datasets' is required but not installed". Found on a
    # 2-minute CPU container rather than on a rented GPU. The heavy torch layer above is
    # still shared with smolvla_modal.py, so only this last layer rebuilds.
    # `peft` is NOT pulled by any lerobot extra, but policies/factory.py imports it
    # unconditionally as soon as `--policy.use_peft=true`. Without it the run dies at
    # policy construction, ~30 s in, after the dataset has already been built.
    .pip_install("lerobot[smolvla,dataset]", "peft", "fastapi[standard]", "json-numpy",
                 "pillow")
)

hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
lerobot_data = modal.Volume.from_name("molmoact2-lerobot-data", create_if_missing=True)
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)

app = modal.App("smolvla-green-ball-train")


@app.function(
    image=image,
    # L4 (24 GB, ~$0.80/hr): 450M trains comfortably in bf16 here, and unlike the T4 the
    # serving app uses, Ada HAS bf16. A T4 would work only in fp32 and slowly. A10G
    # ($1.10/hr) is the next rung if steps/second disappoints.
    gpu="L4",
    # PNG-in-parquet means every sample decodes 2 images on the CPU. With a small CPU
    # allocation that, not the GPU, is the wall -- so buy cores and dataloader workers.
    cpu=16.0,
    volumes={
        "/cache/huggingface": hf_cache,
        "/data": lerobot_data,
        "/checkpoints": checkpoints,
    },
    timeout=5 * 60 * 60,
)
def train(mode: str = "lora", max_steps: int = 3000, batch_size: int = 16,
          save_freq: int = 750, lr: float = 1e-4, num_workers: int = 12,
          exp_name: str = "smolvla-green-ball", norm_stats: str = "checkpoint",
          seed: int = 0, lora_r: int = 32):
    import json
    import os
    import subprocess
    import time

    # Fail loudly and immediately if the dataset was never uploaded, rather than after the
    # model load. Also echo what it contains: a wrong-plant dataset is the single most
    # expensive mistake available here and it is invisible once training starts.
    info_path = f"{REMOTE_REPO}/meta/info.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"{info_path} missing. Upload first:\n"
            f"  modal run smolvla_libero/smolvla_modal_train.py::upload")
    info = json.load(open(info_path))
    frames, episodes = info["total_frames"], info["total_episodes"]
    feats = sorted(k for k in info["features"] if k.startswith("observation") or k == "action")
    print(f"dataset : {episodes} episodes, {frames} frames, v{info['codebase_version']}, "
          f"fps {info['fps']}", flush=True)
    print(f"features: {feats}", flush=True)
    if "observation.images.image2" not in info["features"]:
        raise SystemExit(
            "dataset has no `observation.images.image2`. smolvla_libero's config declares "
            "that key for the wrist camera; a dataset using `wrist_image` will train with "
            "NO second camera and no error. Run convert_dataset.py.")

    epochs = max_steps * batch_size / max(frames, 1)
    print(f"plan    : {max_steps} steps x batch {batch_size} = {epochs:.2f} epochs", flush=True)

    out_dir = f"/checkpoints/smolvla/{exp_name}"

    cmd = [
        "lerobot-train",
        f"--policy.path={BASE_CHECKPOINT}",
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
        # lerobot pushes the finished policy to the HF Hub by default, which 401s in a
        # container with no token -- AFTER training and checkpointing have both succeeded.
        # The run then exits non-zero and looks like a training failure when nothing is
        # wrong. (checkpoints.commit() below deliberately runs before we inspect the exit
        # code, so even that first run's weights survived.)
        "--policy.push_to_hub=false",
        # Pin the decay schedule to the ACTUAL run length. The checkpoint ships
        # decay_steps=30000; leaving it there means a 3000-step run never decays the LR and
        # simply stops a tenth of the way through its schedule.
        f"--policy.scheduler_decay_steps={max_steps}",
        f"--policy.optimizer_lr={lr}",
        # Serve-time horizon. The checkpoint ships n_action_steps=1 (one forward per control
        # tick), which is fine in-process and pointless over HTTP -- our server returns 10,
        # matching LIBERO's horizon and what lerobot itself uses to reproduce published
        # LIBERO numbers. Bake it in so the trained checkpoint serves the way we evaluate it.
        "--policy.n_action_steps=10",
    ]

    if mode == "lora":
        # CORRECTED 2026-07-31, after a 29 s smoke failure:
        #
        #   ValueError: Can't find 'adapter_config.json' at 'HuggingFaceVLA/smolvla_libero'
        #
        # `--policy.use_peft=true` does NOT create a LoRA. In lerobot 0.6.0 it means "this
        # checkpoint IS a PEFT adapter -- load it", so factory.make_policy goes looking for
        # an adapter_config.json next to the base checkpoint and dies when the stock
        # checkpoint (which is a plain policy) has none. Creating a fresh adapter is a
        # TOP-LEVEL train field, `--peft.*`, which lerobot_train.py turns into
        # `policy.wrap_with_peft(...)`; that call sets `config.use_peft = True` itself, on
        # the way out, which is what makes the saved checkpoint loadable with use_peft
        # later. Passing it on the way IN is the error.
        #
        # target_modules is left at SmolVLA's own default rather than specified here:
        #   (model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj
        #    |model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_(in|out)))
        # i.e. the ACTION EXPERT's attention projections plus the state/action heads. That
        # is a narrower scope than this mode's original comment claimed (it does not touch
        # the VLM at all), and it is the right one for what a6 is fixing: episode SPEED is
        # encoded in action magnitudes, which is the expert's job, not a grounding problem
        # the VLM has to relearn. Revisit if the failure turns back into a localisation
        # miss.
        #
        # freeze_vision_encoder / train_expert_only are NOT passed: wrap_with_peft calls
        # requires_grad_(False) on every base parameter regardless, so they would be dead
        # flags that read as if they were doing something.
        cmd += [f"--peft.method_type=LORA", f"--peft.r={lora_r}"]
    elif mode == "full":
        # No adapters: train the VLM outright. 450M is small enough that this is affordable,
        # and it is the strongest option if LoRA underfits 30 episodes.
        cmd += ["--policy.use_peft=false",
                "--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false"]
    elif mode == "expert":
        # HuggingFace's own smolvla_libero recipe.
        cmd += ["--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true"]
    else:
        raise ValueError(f"unknown mode {mode!r}; use lora | expert | full")

    if norm_stats == "dataset":
        # Explicit opt-in only. See the module docstring: rebuilding the normaliser from 30
        # single-task episodes hands the action expert a different affine map than the one
        # it was trained under.
        print("WARNING: rebuilding normalisation from the dataset's own stats", flush=True)

    print("Launching:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/root")
    dt = time.time() - t0
    print(f"lerobot-train returned {proc.returncode} after {dt:.0f}s "
          f"({dt / max(max_steps, 1):.2f}s per step including startup and saves)", flush=True)
    checkpoints.commit()
    if proc.returncode != 0:
        raise SystemExit(f"training failed with code {proc.returncode}")
    print(f"Done. Checkpoints on volume molmoact2-checkpoints at {out_dir}")


@app.local_entrypoint()
def upload():
    """Push the converted dataset onto the volume. Separate from training so a re-run does
    not re-upload, and so the upload can be verified before GPU time is spent."""
    import subprocess
    print(f"uploading {DATASET_DIR} -> molmoact2-lerobot-data:{REMOTE_REPO}")
    subprocess.run(
        ["modal", "volume", "put", "--force", "molmoact2-lerobot-data",
         DATASET_DIR, REMOTE_REPO.replace("/data", "")],
        check=True,
    )


@app.local_entrypoint()
def main(mode: str = "lora", max_steps: int = 3000, batch_size: int = 16,
         save_freq: int = 750, lr: float = 1e-4, num_workers: int = 12,
         exp_name: str = "smolvla-green-ball", norm_stats: str = "checkpoint",
         gpu: str = "L4", seed: int = 0, lora_r: int = 32):
    train.with_options(gpu=gpu).remote(
        mode=mode, max_steps=max_steps, batch_size=batch_size, save_freq=save_freq,
        lr=lr, num_workers=num_workers, exp_name=exp_name, norm_stats=norm_stats, seed=seed,
        lora_r=lora_r,
    )
