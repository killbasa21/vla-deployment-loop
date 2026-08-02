"""One definition of every Modal image this project deploys.

WHY THIS FILE EXISTS
--------------------
`torch==2.5.1` + cu121 was pinned independently in five image chains across `libero/`,
`act/` and `smolvla_libero/`, with no mechanism keeping them in sync. Modal caches image
layers by the definition, so two chains that were *meant* to share the multi-GB torch pull
silently stopped sharing it the moment one of them drifted. Defining the chains once makes
the sharing structural instead of a comment asking you to be careful.

THE PINS ARE VALIDATED, NOT ARBITRARY
-------------------------------------
torch 2.5.1 / torchvision 0.20.1 / transformers 4.57.x are what MolmoAct2's own
`pyproject.toml` was validated against. Do not relax them casually, and change them in
this file only -- that is the whole point of it.

TWO PYTHON VERSIONS, DELIBERATELY
---------------------------------
- **3.11** for the MolmoAct2 serving path (`molmoact_serve_image`), matching the vendored
  repo.
- **3.12** for everything LeRobot-based (ACT, SmolVLA) and for the MolmoAct2 *experiments*
  path, which installs `experiments/` and its bundled lerobot editable.

USING THESE
-----------
Modal re-imports the app module **inside the container**, so any file importing from here
must also ship this package into the image. Every helper below is returned WITHOUT that
mount, because `add_local_python_source()` has to be the last layer -- no build step may
follow it. The caller appends it:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from infra.modal_images import lerobot_serve_image, with_infra

    image = with_infra(lerobot_serve_image().env({"ACT_CHECKPOINT": CHECKPOINT}))

`droid/` is NOT wired to this module. That track is retired (see PROGRESS.md sec.2 -- every
one of its evaluations predates the decimation fix), and its images are kept exactly as
they were last deployed so the historical runs stay reproducible.
"""

import modal

# --------------------------------------------------------------------------------------
# Pins. Change here, nowhere else.
# --------------------------------------------------------------------------------------

TORCH_PIN = ("torch==2.5.1", "torchvision==0.20.1")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"

TRANSFORMERS_PIN = "transformers>=4.57,<4.58"

# Env every image wants. HF_HOME points into the mounted cache volume; TOKENIZERS_PARALLELISM
# is off because every one of these images forks dataloader workers.
BASE_ENV = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HOME": "/cache/huggingface",
    "TOKENIZERS_PARALLELISM": "false",
}


def with_infra(image: modal.Image) -> modal.Image:
    """Ship this package into the container. MUST be the final layer of any chain."""
    return image.add_local_python_source("infra")


# --------------------------------------------------------------------------------------
# LeRobot path (ACT, SmolVLA) -- python 3.12
# --------------------------------------------------------------------------------------


def _lerobot_base() -> modal.Image:
    """Everything up to and including the heavy torch layer.

    Both the serving and the training images share this prefix byte-for-byte, which is what
    makes the multi-GB torch pull a cache hit on the second one.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "ffmpeg")
        .pip_install(*TORCH_PIN, extra_index_url=TORCH_INDEX_URL)
        .env(BASE_ENV)
        .pip_install("hf-transfer>=0.1.8")
    )


def lerobot_serve_image() -> modal.Image:
    """Serving image for ACT and SmolVLA.

    `[smolvla]` pulls the SmolVLM2 vision/language deps. Deliberately NOT `[libero]`: that
    extra drags in robosuite -> egl_probe, which needs cmake and an OpenGL toolchain and
    exists only to run the LIBERO *simulator*. Our simulator is local MuJoCo.

    `[smolvla]` is dead weight for ACT, but keeping the layer identical across both servers
    is worth more than the wheels it saves.

    `peft` is pulled by no lerobot extra, yet `policies/factory.py` imports it
    unconditionally as soon as a LoRA checkpoint is loaded. Installed unconditionally so the
    stock and fine-tuned deployments do not build different images -- it is a few hundred KB.
    """
    return _lerobot_base().pip_install(
        "lerobot[smolvla]", "peft", "fastapi[standard]", "json-numpy", "pillow"
    )


def lerobot_train_image() -> modal.Image:
    """Training image for ACT and SmolVLA.

    `[dataset]` on top of the serving extras: without it `lerobot-train` dies at import with
    "ImportError: 'datasets' is required but not installed". Only this last layer rebuilds --
    the torch layer above is shared with `lerobot_serve_image()`.
    """
    return _lerobot_base().pip_install(
        "lerobot[smolvla,dataset]", "peft", "fastapi[standard]", "json-numpy", "pillow"
    )


# --------------------------------------------------------------------------------------
# MolmoAct2 path -- serving on 3.11, experiments/training on 3.12
# --------------------------------------------------------------------------------------


def molmoact_serve_image() -> modal.Image:
    """MolmoAct2 `/act` server, wrapping the vendored `examples/droid` host server.

    Python 3.11 to match the vendored repo. Mounts `molmoact2/examples/droid` from the
    local checkout, so it must be deployed **from the repo root**.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(*TORCH_PIN, extra_index_url=TORCH_INDEX_URL)
        .pip_install(
            TRANSFORMERS_PIN,
            "accelerate>=1.0",
            "safetensors>=0.4",
            "huggingface-hub[cli]>=0.36",
            "hf-transfer>=0.1.8",
            "pillow>=10",
            "numpy>=1.26,<3",
            "einops>=0.7",
            "sentencepiece>=0.2",
            "protobuf>=4.25",
            "fastapi>=0.116",
            "uvicorn[standard]>=0.35",
            "json-numpy>=2.1.0",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/huggingface"})
        .add_local_dir("molmoact2/examples/droid", remote_path="/root/droid_server")
    )


def molmoact_experiments_image(extra_env: dict | None = None) -> modal.Image:
    """MolmoAct2 `experiments/` installed editable -- used for both training and for serving
    a raw OLMo/PyTorch distributed checkpoint via `serve_policy.py`.

    `./lerobot[async]` only. The `[libero]` extra pulls hf-libero -> robosuite -> egl_probe,
    which needs cmake + OpenGL headers and exists only for LIBERO *simulator* eval; training
    on a LeRobot dataset imports none of it.

    Mounts `molmoact2/experiments` with `copy=True` because build steps follow it, so this
    must be deployed **from the repo root**.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "ffmpeg")
        .pip_install(*TORCH_PIN, extra_index_url=TORCH_INDEX_URL)
        .env({**BASE_ENV, **(extra_env or {})})
        .pip_install("hf-transfer>=0.1.8")
        .add_local_dir("molmoact2/experiments", remote_path="/root/experiments", copy=True)
        .run_commands(
            "cd /root/experiments && pip install -e '.[all]'",
            "cd /root/experiments && pip install -e './lerobot[async]'",
        )
    )
