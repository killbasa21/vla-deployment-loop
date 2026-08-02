"""
Phase 3 (fine-tuned Modal variant): serve OUR Phase 4 fine-tuned MolmoAct2 checkpoint.

WHY THIS FILE EXISTS (and how it differs from droid/phase3_modal.py)
-------------------------------------------------------------
`droid/phase3_modal.py` serves the *released* checkpoint `allenai/MolmoAct2-DROID`. That
checkpoint ships in Hugging Face format (a `config.json` + `*.safetensors` weights +
processor files), and the DROID example server (`host_server_droid.py`) loads it with
the normal `transformers`/`AutoModel` machinery.

Our Phase 4 fine-tune produced a DIFFERENT on-disk format. The training code
(`molmoact2/experiments`, run by `droid/phase4_modal_train.py`) is built on Ai2's "OLMo"
training stack, which saves a **PyTorch Distributed Checkpoint (DCP)**:

    step500/
      config.yaml                     # the full model + training config (resolved)
      model_and_optim/
        __0_0.distcp ... __0_15.distcp # 16 sharded tensor files (model + optimizer)
        .metadata                      # index describing how the shards fit together
      train/rank0.pt                   # RNG / step counter (training bookkeeping)

That is a *training* checkpoint, not an HF model, so `transformers` can't load it and
`host_server_droid.py` won't work. Instead we load it through the experiments repo's own
LeRobot policy wrapper, exposed by `molmoact2/experiments/scripts/serve_policy.py` as a
class `MolmoAct2Server` that builds a FastAPI app speaking the SAME `/act` wire protocol.
So the local sim client (`droid/phase3_closed_loop.py`) doesn't need to change on the request
side -- it POSTs the same `{external_cam, wrist_cam, instruction, state}` payload.

(One wire difference on the RESPONSE side: this server reports timing as `latency_ms`,
while `host_server_droid.py` uses `dt_ms`. If `droid/phase3_closed_loop.py` reads `dt_ms`
strictly it needs a small tolerance -- `body.get("dt_ms") or body.get("latency_ms")`.)

MODAL MENTAL MODEL (serverless GPU)
-----------------------------------
Modal runs our code in a container it builds from the `image` spec below, on a rented
GPU, and exposes the FastAPI app to the internet. Key ideas used here:
  - `modal.Image` is built ONCE (cached by content hash) and reused across runs.
  - `@app.cls` + `@modal.enter()` = the expensive setup (loading a 4B model onto the
    GPU) runs ONCE per container start ("cold start"), not once per HTTP request.
  - `@modal.asgi_app()` publishes the FastAPI app at a public URL.
  - Volumes are network storage that persists across containers/machines -- that's where
    the 25 GB checkpoint lives, so we never re-upload it.

Usage:
    modal serve droid/phase3_modal_finetuned.py    # ephemeral dev server, live-reloads on save
    modal deploy droid/phase3_modal_finetuned.py   # persistent deployment, prints a stable URL

Then point the local sim at the printed URL + "/act":
    uv run python droid/phase3_closed_loop.py --model-path droid \
        --server-url https://<workspace>--molmoact2-droid-finetuned-molmoactfinetunedserver-serve.modal.run/act \
        --request-timeout 600 --chunks 5
(The `--request-timeout 600` matters: the very first request after a cold start pays for
model load + warmup, which can take minutes.)
"""

import modal

# WHERE the checkpoint lives INSIDE the container. This is not a local path on your
# laptop -- it's a path on the `molmoact2-checkpoints` Modal Volume, mounted at
# "/checkpoints" by the @app.cls below. `droid/phase4_modal_train.py` wrote this fine-tune as
# .../finetune/lora_train/ -- the LoRA run (VLM adapters + full action expert) on the
# bin-randomized `lora_adapter` dataset, meant to fix ae_train's failure to visually locate
# the target (see data/lora_adapter/README.md).
#
# MUST point at step500-merged, NOT step500. The serving path
# (lerobot/.../modeling_molmoact2.py:_prepare_model) builds a plain, non-PEFT model
# architecture unconditionally -- it has no LoRA/PEFT awareness at all. step500 is the raw
# sharded checkpoint of the PEFT-wrapped *training* model (param names renamed to
# base_layer/lora_A/lora_B by peft's LoRA wrapping), which won't line up with that plain
# architecture's parameter names. step500-merged is trainer.py's merge_and_save_unsharded()
# output: LoRA deltas folded directly into the base weights, saved as a single model.pt
# with the same plain architecture the server builds -- this is the one that's actually
# loadable. (ae_train's step500 worked fine unmodified because that run had
# lora_enable=false -- no PEFT wrapping to begin with, so sharded-vs-merged was a
# non-issue there.)
CHECKPOINT_PATH = "/checkpoints/checkpoints/finetune/lora_train/step500-merged"

# WHICH set of normalization statistics to use. MolmoAct2 trains/serves actions in a
# NORMALIZED space, then un-normalizes back to real joint angles before returning them.
# The stats (per-dimension min/max, since we trained with norm_mode=min_max) are baked
# into the checkpoint's config.yaml under this tag name. It MUST match the mixture tag we
# trained with (`lora_adapter`, registered in experiments' data_mixtures.py); a wrong
# tag makes the server reject the request ("unknown normalization tag").
NORM_TAG = "lora_adapter"

# --- Image ------------------------------------------------------------------
# The container blueprint. This mirrors droid/phase4_modal_train.py's image because SERVING
# the DCP checkpoint needs the exact same code + deps that TRAINED it (the experiments
# package + its lerobot fork know how to reconstruct the model from the shards). Notes:
#   - Python 3.12: the vendored lerobot fork requires >=3.12.
#   - torch from the cu121 index so the CUDA build matches the rented GPU.
#   - `experiments[all]` pulls the OLMo model/training code (~transformers 5.x, heavier
#     than the 4.57 the released-HF server pins -- hence a SEPARATE image from serving).
#   - `lerobot[async]` only -- we deliberately skip the `[libero]` extra the README uses,
#     because it drags in a simulator stack (robosuite -> egl_probe) that needs cmake and
#     is only for LIBERO benchmark eval, which we don't run.
# Caching caveat: because this `.env(...)` differs from droid/phase4_modal_train.py's, the heavy
# pip layers below are NOT shared with that already-built image -- the FIRST deploy of
# this file rebuilds them (~15-25 min). (To make deploys instant you could instead import
# and reuse droid/phase4_modal_train.py's `image` object, at the cost of coupling the files.)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",   # faster HF downloads (base weights, etc.)
            "HF_HOME": "/cache/huggingface",    # HF cache -> the hf_cache Volume below
            "TOKENIZERS_PARALLELISM": "false",  # silence a fork-vs-tokenizer warning
        }
    )
    .pip_install("hf-transfer>=0.1.8")
    # Ship the training/serving source into the image. `copy=True` bakes it into the
    # image layer (needed because the very next step -- pip install -e -- must see it at
    # build time, not just at run time).
    .add_local_dir("molmoact2/experiments", remote_path="/root/experiments", copy=True)
    .run_commands(
        "cd /root/experiments && pip install -e '.[all]'",       # OLMo model + tooling
        "cd /root/experiments && pip install -e './lerobot[async]'",  # LeRobot policy
    )
)

# --- Volumes (persistent network storage) -----------------------------------
# `hf_cache`: shared with droid/phase3_modal.py / droid/phase4_modal_train.py so any Hugging Face
# downloads (e.g. the base tokenizer/processor the checkpoint references) are cached once.
hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
# `checkpoints`: where droid/phase4_modal_train.py saved the fine-tune. We mount it read/serve
# and load CHECKPOINT_PATH from it -- no 25 GB re-upload per deploy.
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)

app = modal.App("molmoact2-droid-finetuned")


@app.cls(
    image=image,
    # A100-40GB is plenty for 4B-param INFERENCE (weights ~8-16 GB + activations); the
    # 25 GB on disk includes optimizer shards we don't load for serving. The base server
    # (droid/phase3_modal.py) uses the same GPU for the same model.
    gpu="A100-40GB",
    volumes={
        "/cache/huggingface": hf_cache,
        "/checkpoints": checkpoints,   # CHECKPOINT_PATH resolves under here
    },
    # Keep a warm container for 5 min after the last request, so back-to-back sim runs
    # don't each re-pay the cold-start model load. After that it scales to zero (no GPU
    # billed while idle).
    scaledown_window=300,
    # Max seconds a single request may run. Generous because the first request after a
    # cold start includes model warmup.
    timeout=900,
)
class MolmoActFinetunedServer:
    @modal.enter()
    def load(self):
        """Runs ONCE when a container starts (before it serves any request). This is the
        expensive part -- reconstruct the 4B model from the DCP shards and move it onto
        the GPU -- so we pay it once per cold start, not per HTTP call."""
        import sys

        # serve_policy.py lives at /root/experiments/scripts/serve_policy.py. Put the repo
        # root on the import path so `scripts.serve_policy` resolves (it further adds its
        # own lerobot/src path internally).
        sys.path.insert(0, "/root/experiments")
        from scripts.serve_policy import MolmoAct2Server

        # Build the policy server. Each argument, and why it's set this way:
        self.server = MolmoAct2Server(
            checkpoint=CHECKPOINT_PATH,
            # False = load a raw OLMo DCP training checkpoint (our case). True would be
            # for a Hugging-Face-exported directory instead.
            use_hf_ckpt=False,
            device="cuda:0",
            # Preset naming the two camera keys this embodiment expects, in order:
            # "droid" -> ["external_cam", "wrist_cam"], matching our dataset and exactly
            # the keys droid/phase3_closed_loop.py sends.
            image_keys="droid",
            # MolmoAct2 can emit actions as a continuous flow-matching trajectory or as
            # discrete tokens. Our checkpoint's action expert is continuous (the released
            # DROID one is too).
            inference_action_mode="continuous",
            # MolmoAct2 can also predict intermediate depth/spatial "reasoning" before the
            # action. We didn't train that (and the DROID server disables it too), so off
            # -- faster, and avoids needing depth supervision we don't have.
            enable_depth_reasoning=False,
            # Which normalization stats to un-normalize actions with (see NORM_TAG above).
            norm_tag=NORM_TAG,
            # CUDA graphs speed up the action expert but are not safe under concurrent
            # requests and add fragility. Off for correctness/simplicity.
            enable_inference_cuda_graph=False,
            verbose=False,
        )

    @modal.asgi_app()
    def serve(self):
        """Publish the FastAPI app `serve_policy` already built (routes: /act, /reset,
        /health) at a public URL. Modal calls this to get the ASGI app to serve."""
        return self.server.app


@app.local_entrypoint()
def smoke_test():
    """`modal run droid/phase3_modal_finetuned.py` -- spin the app up ephemerally and hit
    /health to confirm the model loaded and the endpoint answers, without touching the
    sim. For a persistent deployment use `modal deploy` instead; it prints the same base
    URL. The `/act` endpoint the sim client needs is that base URL + "/act"."""
    import json
    import urllib.request

    # Modal 1.5 removed the `.web_url` attribute in favor of the `.get_web_url()` method.
    # The returned base URL has no path suffix, so we append the route ourselves (the
    # server routes /act, /reset, /health -- see serve_policy.MolmoAct2Server).
    base_url = MolmoActFinetunedServer().serve.get_web_url()
    print(f"endpoint (base):  {base_url}")
    print(f"  /act for client: {base_url.rstrip('/')}/act")
    # 600 s timeout: /health returns fast once loaded, but this request triggers the cold
    # start (model load) if the container isn't already warm.
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=600) as resp:
        print(json.dumps(json.loads(resp.read().decode()), indent=2))
