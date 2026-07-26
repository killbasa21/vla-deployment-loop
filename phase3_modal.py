"""
Phase 3 (Modal variant): host MolmoAct2-DROID's /act endpoint on a Modal A100
instead of a manually rented GPU box.

This does not reimplement the server -- it wraps the existing FastAPI app from
molmoact2/examples/droid/host_server_droid.py (Policy + build_app) inside a
Modal container so the wire protocol phase3_closed_loop.py already speaks
(json_numpy POST to /act) is unchanged. Only SERVER_URL in phase3_closed_loop.py
needs to change, to whatever URL `modal deploy` prints.

Checkpoint weights are cached on a Modal Volume mounted at the HF cache path,
so `snapshot_download` only pays the ~22GB download once -- every later
container start (even on a different physical machine) reads it back off
Modal's network storage instead of re-downloading from the Hub.

Usage:
    modal setup                        # one-time auth (opens a browser)
    modal serve phase3_modal.py        # ephemeral dev server, live-reloads on save
    modal deploy phase3_modal.py       # persistent deployment, prints the URL

Then point phase3_closed_loop.py's SERVER_URL (or its --server-url flag) at the
printed URL + "/act". The slug is longer than you'd guess from the class name
alone -- Modal includes both the app name and the @app.cls name in it, so it's
https://<workspace>--molmoact2-droid-molmoactdroidserver-serve.modal.run/act,
not https://<workspace>--molmoact2-droid-serve.modal.run/act. Always confirm
against `modal deploy`'s own printed "Created Web Function URL" line rather
than reconstructing it by hand.
"""

import modal

# --- Image -------------------------------------------------------------
# Same pins as molmoact2/pyproject.toml (CLAUDE.md: "don't relax these pins
# casually -- validated against torch 2.5.1 / transformers 4.57.x").
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.57,<4.58",
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
    # Ship host_server_droid.py (Policy, build_app, the bf16 patches) into the
    # container as-is instead of re-deriving that logic here.
    .add_local_dir(
        "molmoact2/examples/droid",
        remote_path="/root/droid_server",
    )
)

# HF cache Volume: first cold start downloads the ~22GB DROID checkpoint,
# every later container start (any machine) reads it back from here.
hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)

# Only needed if allenai/MolmoAct2-DROID ever requires a gated/authenticated
# download (it doesn't appear to). If snapshot_download() 401s in the logs:
#   modal secret create huggingface-secret HF_TOKEN=hf_...
# and add secrets=[modal.Secret.from_name("huggingface-secret")] to @app.cls below.

app = modal.App("molmoact2-droid")


@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache/huggingface": hf_cache},
    scaledown_window=300,  # keep container warm 5min after last request
    timeout=600,
    # min_containers=1,  # uncomment to keep one instance always warm (billed continuously)
)
class MolmoActDroidServer:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/root/droid_server")
        from host_server_droid import Policy, warmup

        self.policy = Policy(
            repo_id="allenai/MolmoAct2-DROID",
            device="cuda:0",
            dtype=__import__("torch").bfloat16,
            enable_cuda_graph=False,
        )
        warmup(self.policy)

    @modal.asgi_app()
    def serve(self):
        import sys

        sys.path.insert(0, "/root/droid_server")
        from host_server_droid import build_app

        return build_app(self.policy)


@app.local_entrypoint()
def smoke_test():
    """`modal run phase3_modal.py` -- spins up the app ephemerally, prints its
    URL, and hits /healthz. For a persistent deployment use `modal deploy`
    instead; it prints the same URL, which stays stable across redeploys."""
    import urllib.request

    url = MolmoActDroidServer().serve.web_url
    print(f"endpoint: {url}")
    with urllib.request.urlopen(url.replace("/act", "/healthz")) as resp:
        print(resp.read().decode())
