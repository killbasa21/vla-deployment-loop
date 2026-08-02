"""Serve OUR fine-tuned MolmoAct2-LIBERO checkpoint's /act endpoint on a Modal GPU.

Third member of the family, and the differences between them are all that matter:

  libero/libero_modal.py            released allenai/MolmoAct2-LIBERO, HF format,
                                    loaded by examples/droid/host_server_droid.py
  droid/phase3_modal_finetuned.py         our DROID fine-tune, OLMo DCP format,
                                    loaded by experiments/scripts/serve_policy.py
  THIS FILE                         our LIBERO fine-tune, OLMo DCP format, serve_policy.py

Everything below is a consequence of those two axes (which checkpoint, which loader).

1. FORMAT. `libero_modal_train.py` writes a PyTorch Distributed Checkpoint, not a HF
   model directory, so `transformers` cannot load it and `host_server_droid.py` is not an
   option. `scripts/serve_policy.py` rebuilds the model from the shards and serves the
   same `/act` wire protocol.

2. MERGED, NOT RAW, FOR A LoRA RUN. `droid/phase3_modal_finetuned.py` paid for this lesson:
   the serving path builds a plain non-PEFT architecture unconditionally, so a raw
   `stepN` checkpoint of a PEFT-wrapped training model has parameter names
   (`base_layer`/`lora_A`/`lora_B`) that do not line up. Point CHECKPOINT_PATH at
   `stepN-merged` -- the trainer's `merge_and_save_unsharded()` output, with the LoRA
   deltas folded into the base weights. (An `ae_only` run has no PEFT wrapping and loads
   either way, which is exactly why this trap survived to bite the LoRA run.)

3. NORM_TAG = "libero". The trap docs/NEXT_STEPS_FOR_FINE_TUNE.md flags:
   `host_server_droid.py` hardcodes `NORM_TAG = "franka_droid"` at module level, and a
   wrong tag yields garbage actions OF THE CORRECT SHAPE -- a silent failure. This file
   does not go through that module at all, but the same rule applies: the tag must match
   the mixture tag we trained under, which `data_mixtures.py:build_molmoact2_libero_green_ball`
   deliberately keeps as "libero". CONFIRM IT AT /health BEFORE BELIEVING A ROLLOUT.

4. image_keys="libero", i.e. ["image", "wrist_image"]. serve_policy turns the payload key
   it matched into the model feature name `observation.images.<key>`, and the tag metadata
   declares `observation.images.image` / `observation.images.wrist_image`. So the client
   must POST those key names: run `libero_closed_loop.py --payload-keys libero`. Against
   this server the default `droid` keys produce "Missing images payload".

Usage:
    modal deploy libero/libero_modal_finetuned.py
    curl -s <printed-url>/health | python3 -m json.tool     # expect norm_tag "libero"
    uv run python libero/libero_closed_loop.py --payload-keys libero \\
        --server-url <printed-url>/act --request-timeout 600 --chunks 12
"""

import os

import sys
from pathlib import Path

import modal

# The shared image definitions live at the repo root, and Modal re-imports this
# module inside the container -- where infra/ lands on /root via with_infra().
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.modal_images import molmoact_experiments_image, with_infra

# Which checkpoint to serve. Override without editing the file:
#     MOLMOACT2_LIBERO_CKPT=/checkpoints/checkpoints/finetune/libero-green-ball/step200-merged \\
#         modal deploy libero/libero_modal_finetuned.py
# Checkpoint SELECTION is part of the anti-overfitting plan (docs/FINE_TUNE_LEARNINGS.md sec.1):
# the last checkpoint is not automatically the best, so each saved step gets deployed and
# scored, and the env var is how you switch between them without a code edit.
CHECKPOINT_PATH = os.environ.get(
    "MOLMOACT2_LIBERO_CKPT",
    "/checkpoints/checkpoints/finetune/libero-green-ball/step150-merged",
)
NORM_TAG = "libero"

# The env var has to be BAKED INTO THE IMAGE, not just read here. This module is executed
# twice: locally at deploy time (where MOLMOACT2_LIBERO_CKPT is set) and again inside the
# container (where it is not, so the module-level os.environ.get would quietly fall back to
# the default and serve a different checkpoint than the one asked for). Putting the
# resolved value in the image makes the container read the same string. It is its own thin
# layer, so switching checkpoints does not rebuild the heavy pip layers.

# Same image as libero_modal_train.py: serving a DCP checkpoint needs the exact code that
# trained it. Kept byte-identical to the training image's install steps so Modal reuses
# the cached layers instead of rebuilding ~20 minutes of pip.
image = with_infra(
    molmoact_experiments_image({"MOLMOACT2_LIBERO_CKPT": CHECKPOINT_PATH})
)

hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)

app = modal.App("molmoact2-libero-finetuned")


@app.cls(
    image=image,
    # L40S (48 GB, $1.95/hr), NOT the L4 the released checkpoint is served on.
    #
    # MEASURED, after an L4 attempt returned 500s on every /act: this loader puts 20.5 GB
    # of weights on the card, so an L4's 22 GB usable leaves ~1 GB and inference dies with
    # "Tried to allocate 1.49 GiB". libero/README.md's "L4 is plenty, 5B at bf16 is ~10 GB"
    # is true of `host_server_droid.py` serving the released HF checkpoint -- it is NOT
    # true here, because scripts/serve_policy.py rebuilds the model from the OLMo config
    # and exposes no dtype/precision knob, so the weights land in fp32. (phase3_modal_
    # finetuned.py, the same loader, was already on an A100-40GB for this reason.)
    # L40S is the cheapest rung with the headroom; A100-40GB also works and costs more.
    gpu="L40S",
    volumes={"/cache/huggingface": hf_cache, "/checkpoints": checkpoints},
    scaledown_window=300,
    timeout=900,
)
class MolmoActLiberoFinetunedServer:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/root/experiments")
        from scripts.serve_policy import MolmoAct2Server

        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"{CHECKPOINT_PATH} not on the molmoact2-checkpoints Volume. "
                "For a LoRA run the loadable directory is stepN-MERGED, not stepN.")
        print(f"loading {CHECKPOINT_PATH} with norm_tag={NORM_TAG!r}", flush=True)

        self.server = MolmoAct2Server(
            checkpoint=CHECKPOINT_PATH,
            use_hf_ckpt=False,          # OLMo DCP training checkpoint, not an HF export
            device="cuda:0",
            image_keys="libero",        # -> payload keys "image" / "wrist_image"
            inference_action_mode="continuous",   # our action expert is flow-matching
            enable_depth_reasoning=False,         # not trained, and the DROID server is
                                                  # also served with it off
            norm_tag=NORM_TAG,
            enable_inference_cuda_graph=False,    # unsafe under concurrent requests
            verbose=False,
        )

    @modal.asgi_app()
    def serve(self):
        """Publish serve_policy's own FastAPI app (routes /act, /reset, /health)."""
        return self.server.app
