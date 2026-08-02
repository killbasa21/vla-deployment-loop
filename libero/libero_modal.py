"""Serve `allenai/MolmoAct2-LIBERO`'s /act endpoint on a Modal GPU.

Sibling of the project root's `droid/phase3_modal.py`, kept separate (see libero/README.md)
so the DROID deployment stays untouched. Same trick: don't reimplement the server, wrap
the existing FastAPI app from molmoact2/examples/droid/host_server_droid.py.

Two things differ from droid/phase3_modal.py, and only two:

  1. `repo_id` -> "allenai/MolmoAct2-LIBERO"
  2. `NORM_TAG` -> "libero"

(2) is the one that is easy to miss. `host_server_droid.py` hardcodes
`NORM_TAG = "franka_droid"` at module level (line 55) and passes it into
`predict_action`, which uses it to pick which normalization statistics to
un-normalize the predicted actions with. Serving the LIBERO checkpoint with the DROID
tag would load the wrong stats and silently produce garbage actions in the right
shape -- the worst kind of failure. The mixture registers this checkpoint's stats
under the tag "libero" (molmoact2/experiments/launch_scripts/data_mixtures.py:322),
so we overwrite the module attribute after import rather than editing the vendored
file. Verified below by echoing the effective tag back from /health.

Despite the name, `host_server_droid.py` is not DROID-specific: the request schema is
two images + instruction + an (8,) state, and LIBERO's state is also 8-D (eef position
3 + eef axis-angle 3 + gripper qpos 2), so the wire format needs no changes at all.
The payload keys stay "external_cam"/"wrist_cam"; they are positional -- the server
just forwards them as `images=[first, second]` -- so the first is LIBERO's agentview
and the second its eye-in-hand.

Usage:
    modal setup                                  # one-time auth
    modal serve libero/libero_modal.py           # ephemeral dev server
    modal deploy libero/libero_modal.py          # persistent, prints the URL

Point libero_closed_loop.py --server-url at the printed URL + "/act". As with
droid/phase3_modal.py, read the URL off modal's own "Created Web Function URL" line rather
than reconstructing the slug by hand.
"""

import sys
from pathlib import Path

import modal

# The shared image definitions live at the repo root, and Modal re-imports this
# module inside the container -- where infra/ lands on /root via with_infra().
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.modal_images import molmoact_serve_image, with_infra

REPO_ID = "allenai/MolmoAct2-LIBERO"
# Must match the tag this checkpoint's norm stats are registered under. See the module
# docstring -- a wrong tag here fails silently, not loudly.
NORM_TAG = "libero"

# Pins live in infra/modal_images.py -- change them there, not here. Do not relax them
# casually: validated against torch 2.5.1 / transformers 4.57.x.
image = with_infra(molmoact_serve_image())

# Shared with droid/phase3_modal.py on purpose: same HF cache Volume, so the two checkpoints
# sit side by side and neither re-downloads when you switch between them.
hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)

app = modal.App("molmoact2-libero")


@app.cls(
    image=image,
    # L4, not A100-40GB. MolmoAct2-LIBERO is 5B params -> ~10 GB of weights at the
    # bfloat16 we load it in, so 40 GB was ~4x more VRAM than this needs. L4 has 24 GB
    # and is Ada, so bfloat16 is native (T4 is cheaper still but is Turing, which has no
    # bf16 at all -- it would fail, not just run slow).
    #
    # The saving is bigger than the per-hour gap suggests. PROGRESS.md sec.2 measured
    # inference at 1394 ms inside a 6845 ms round trip: transport dominates, so a slower
    # card barely moves total wall clock. And with scaledown_window below, most of what
    # gets billed is a WARM IDLE container, charged at the GPU rate regardless of
    # utilisation. Idle time, not FLOPs, is the bill -- so the cheapest card that fits
    # wins. $0.80/hr vs $2.10/hr.
    #
    # If this OOMs or inference balloons past ~5 s, the next rungs are "A10" (24 GB,
    # $1.10/hr) then "L40S" (48 GB, $1.95/hr). The molmoact2-hf-cache Volume below means
    # switching costs no re-download.
    gpu="L4",
    volumes={"/cache/huggingface": hf_cache},
    scaledown_window=300,  # keep container warm 5 min after last request
    timeout=600,
)
class MolmoActLiberoServer:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/root/droid_server")
        import host_server_droid

        # Retag BEFORE constructing Policy. `Policy.predict` reads the module-level
        # constant at call time, and build_app echoes it from /health, but setting it
        # up front means warmup() also runs against the right stats.
        host_server_droid.NORM_TAG = NORM_TAG
        # Display-only, but retag it too: the GET /act health route echoes the module's
        # own REPO_ID, so without this it truthfully reports norm_tag=libero while
        # claiming repo_id=allenai/MolmoAct2-DROID. The loaded weights come from the
        # repo_id passed to Policy() below, so this was cosmetic -- but a health endpoint
        # that misreports which checkpoint is serving will cost someone an hour.
        host_server_droid.REPO_ID = REPO_ID

        self.policy = host_server_droid.Policy(
            repo_id=REPO_ID,
            device="cuda:0",
            dtype=__import__("torch").bfloat16,
            enable_cuda_graph=False,
        )
        host_server_droid.warmup(self.policy)

    @modal.asgi_app()
    def serve(self):
        import sys

        sys.path.insert(0, "/root/droid_server")
        import host_server_droid

        # Set again: @modal.asgi_app() may run in a different import context than
        # @modal.enter(), and a stale "franka_droid" here would mean /health reports a
        # tag the model isn't actually using.
        host_server_droid.NORM_TAG = NORM_TAG
        host_server_droid.REPO_ID = REPO_ID
        return host_server_droid.build_app(self.policy)


@app.local_entrypoint()
def smoke_test():
    """`modal run libero/libero_modal.py` -- ephemeral spin-up, prints the URL and the
    /health body. Check that the reported norm_tag is "libero", not "franka_droid":
    that is the one-line confirmation the retag above actually took effect."""
    import urllib.request

    url = MolmoActLiberoServer().serve.web_url
    print(f"endpoint: {url}")
    for path in ("/health", "/healthz"):
        try:
            with urllib.request.urlopen(url.rstrip("/") + path, timeout=300) as resp:
                print(f"{path}: {resp.read().decode()}")
                break
        except Exception as exc:  # noqa: BLE001 -- which of the two exists varies
            print(f"{path}: {exc}")
