"""Serve an ACT checkpoint from the Modal volume over the project's `/act` wire protocol.

Counterpart to `smolvla_libero/smolvla_modal.py`, and protocol-identical to it on purpose:
the same sim client, the same logs, the same scorer, so ACT's numbers can be put beside
SmolVLA's without an asterisk.

NO CHANGES ARE NEEDED IN THE SIM CLIENT:

    uv run python libero/libero_closed_loop.py \
        --payload-keys libero --server-url https://<printed-url>/act \
        --delta-pos-scale 0.10 --randomize-bins --randomize-ball \
        --chunks 25 --no-view --run-id act_ck60000_00

`--payload-keys libero` sends `{image, wrist_image, instruction, state}`. The `droid` default
sends `external_cam`/`wrist_cam`, which this server rejects with a 400 rather than silently
running blind. `--delta-pos-scale 0.10 --randomize-bins` must match how `a7` was collected.

WHICH CHECKPOINT
----------------
`ACT_CHECKPOINT`, an absolute path on the `molmoact2-checkpoints` volume, baked into the
image at build time so the container reads the value the deploy was made with:

    ACT_CHECKPOINT=/checkpoints/act/act-green-ball/checkpoints/060000/pretrained_model \
        modal deploy act/act_modal.py

Score EVERY saved checkpoint, not just the last. `libero/PROGRESS.md` sec.23.5 measured
ck3000 at three times the lateral accuracy of ck5000 from the same run.

AND POLL /health BEFORE TRUSTING A ROLLOUT. A `modal deploy` returning in six seconds is not
a cutover: the previous container answers from inside its 300 s scaledown_window. Section
23.5 threw away a run that looked like its best result for exactly this. /health echoes the
checkpoint path.

WHY THE IDENTITY KEY MAP IS STILL CHECKED
-----------------------------------------
SmolVLA needed `wrist_image -> observation.images.image2`, because its config hardcodes that
name and LeRobot resolves feature keys by name inside the normalisation layer -- the wrong
name means no wrist camera and no error. ACT infers its features from the training dataset,
which kept the native `observation.images.wrist_image`, so the map below is the identity.
The startup assertion against `cfg.input_features` stays anyway: it costs nothing and the
failure it catches is silent.

ACT IGNORES THE INSTRUCTION. It has no language encoder; task identity is implicit in the
weights. `instruction` is accepted and dropped, so the client needs no special-casing.
"""

import os

import sys
from pathlib import Path

import modal

# The shared image definitions live at the repo root, and Modal re-imports this
# module inside the container -- where infra/ lands on /root via with_infra().
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.modal_images import lerobot_serve_image, with_infra

# Absolute path on the checkpoints volume, or an HF repo id. No default that points at a real
# run: a stale default is how the wrong checkpoint gets measured.
CHECKPOINT = os.environ.get("ACT_CHECKPOINT", "")

# How many actions one /act call returns. LIBERO's action horizon, what
# libero_closed_loop.py expects, and what every SmolVLA rollout already logged used.
ACTION_HORIZON = 10

# ~80M parameters in fp32 is ~320 MB. A T4 is far more card than this needs, and Turing's
# missing bf16 is irrelevant because we serve in float32 anyway.
GPU = "T4"

# The two camera keys the sim client sends (libero_closed_loop.PAYLOAD_KEY_SETS["libero"])
# mapped onto the feature names the trained policy declares. Identity -- see the docstring.
WIRE_TO_FEATURE = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}

image = with_infra(
    # Serving extras and the torch layer below them are defined once in
    # infra/modal_images.py, shared with act_modal_train.py and the SmolVLA servers so
    # the multi-GB torch pull is a cache hit. The checkpoint is baked in so the container
    # reads the value the deploy was made with, not whatever the env holds at runtime.
    lerobot_serve_image().env({"ACT_CHECKPOINT": CHECKPOINT})
)

hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)

app = modal.App("act-libero")


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/cache/huggingface": hf_cache, "/checkpoints": checkpoints},
    # Keep a container warm for 5 min after the last request so back-to-back rollouts do not
    # each re-pay the model load. Scales to zero after -- no GPU billed idle. NOTE this is
    # also the window that makes a redeploy look instant while the OLD checkpoint is still
    # answering; see the docstring.
    scaledown_window=300,
    timeout=900,
)
class ACTServer:
    @modal.enter()
    def load(self):
        import threading

        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        if not CHECKPOINT:
            raise RuntimeError(
                "ACT_CHECKPOINT is empty. Set it at deploy time, e.g. "
                "ACT_CHECKPOINT=/checkpoints/act/act-green-ball/checkpoints/060000/"
                "pretrained_model modal deploy act/act_modal.py")

        # One policy object serves every request and _predict mutates it (reset, then the
        # forward). Two overlapping requests would interleave those mutations -- a silent
        # wrong-actions failure, not an error. Same lock, same reason, as smolvla_modal.py.
        self._lock = threading.Lock()

        self.torch = torch
        self.device = "cuda"
        # float32. A T4 has no bf16 at all, and at 80M the memory cost is irrelevant.
        self.dtype = torch.float32

        # No PEFT branch here, unlike smolvla_modal.py. An ACT run trains the whole network
        # from scratch, so a checkpoint is always a plain policy directory -- there is no
        # adapter_config.json case to detect.
        self.policy = ACTPolicy.from_pretrained(CHECKPOINT)
        self.policy.to(device=self.device, dtype=self.dtype)
        self.policy.eval()
        self.policy.reset()

        cfg = self.policy.config

        # lerobot 0.6.0 keeps preprocessing OUT of the policy. `pretrained_path=CHECKPOINT`
        # takes the normalisation statistics FROM THE CHECKPOINT -- which, for a from-scratch
        # ACT, are the training dataset's own. Rebuilding them from anywhere else would
        # normalise into a different space and produce the failure `libero/PROGRESS.md`
        # sec.5 calls "garbage actions of the correct shape".
        #
        # The pipeline is shorter than SmolVLA's: no tokenizer step, because ACT has no
        # language input. Rename -> AddBatchDimension -> Device -> Normalizer, and the
        # postprocessor un-normalises back to real units on the CPU. AddBatchDimension means
        # observations must be submitted UNBATCHED.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=CHECKPOINT
        )

        # The serialised processor config carries whatever device it was saved with (usually
        # cpu). Retarget by step TYPE rather than by registry name -- the name is an
        # implementation detail, the type is not.
        moved = 0
        for step in list(self.preprocessor.steps):
            if type(step).__name__ == "DeviceProcessorStep":
                step.device = self.device
                moved += 1

        print("loaded", CHECKPOINT, flush=True)
        print(f"  preprocessor : {[type(s).__name__ for s in self.preprocessor.steps]}", flush=True)
        print(f"  postprocessor: {[type(s).__name__ for s in self.postprocessor.steps]}", flush=True)
        print(f"  device steps retargeted to {self.device}: {moved}", flush=True)
        print("  input_features :", {k: list(v.shape) for k, v in cfg.input_features.items()}, flush=True)
        print("  output_features:", {k: list(v.shape) for k, v in cfg.output_features.items()}, flush=True)
        print("  chunk_size     :", cfg.chunk_size, " n_action_steps:", cfg.n_action_steps, flush=True)
        print("  normalization  :", cfg.normalization_mapping, flush=True)
        print("  temporal_ensemble_coeff:", cfg.temporal_ensemble_coeff, flush=True)

        # Whether the policy was trained WITH proprioception is a property of the checkpoint,
        # not of this file, and it changes what a valid request looks like. Read it off the
        # config rather than guessing -- act_modal_train.py's --use-state=false ablation
        # produces checkpoints where sending a state is meaningless.
        self.wants_state = "observation.state" in cfg.input_features
        print("  uses observation.state:", self.wants_state, flush=True)

        missing = set(WIRE_TO_FEATURE.values()) - set(cfg.input_features)
        if missing:
            raise RuntimeError(
                f"checkpoint does not declare {sorted(missing)}; its input features are "
                f"{sorted(cfg.input_features)}. Sending images under a key the policy does "
                f"not know means it silently sees no image at all.")

        if cfg.temporal_ensemble_coeff is not None:
            # Temporal ensembling averages overlapping chunks and only works when the policy
            # is re-queried every tick (n_action_steps=1). This server returns 10 actions per
            # call by design, so the ensembling would silently never happen.
            raise RuntimeError(
                f"checkpoint sets temporal_ensemble_coeff={cfg.temporal_ensemble_coeff}, "
                f"which requires n_action_steps=1 and one HTTP round trip per control tick. "
                f"This server serves {ACTION_HORIZON} actions per call.")

        if ACTION_HORIZON > cfg.chunk_size:
            raise RuntimeError(
                f"ACTION_HORIZON={ACTION_HORIZON} exceeds the checkpoint's chunk_size="
                f"{cfg.chunk_size}; one forward pass does not produce that many actions.")

    def _predict(self, obs):
        """Return an (ACTION_HORIZON, 7) float32 array for one UNBATCHED observation dict.

        `predict_action_chunk` gives the full chunk_size actions from a single forward pass
        and we keep the leading ACTION_HORIZON. Deliberately NOT `select_action`, which slices
        to config.n_action_steps and maintains an internal queue across calls -- our wire
        protocol never promised successive requests are contiguous.
        """
        import numpy as np

        with self._lock, self.torch.no_grad():
            self.policy.reset()
            batch = self.preprocessor(obs)
            chunk = self.policy.predict_action_chunk(batch)   # (1, chunk_size, 7)
            actions = self.postprocessor(chunk)               # un-normalised, CPU

        return actions[0, :ACTION_HORIZON].float().cpu().numpy().astype(np.float32)

    @modal.asgi_app()
    def serve(self):
        import time

        import json_numpy
        import numpy as np
        from fastapi import FastAPI, Request, Response
        from fastapi.responses import JSONResponse

        # Patches stdlib json so the client's numpy-encoded frames DECODE. It does not cover
        # the response: FastAPI serialises return values through jsonable_encoder, which
        # knows nothing about numpy. /act therefore builds its body with json_numpy.dumps and
        # returns a raw Response.
        json_numpy.patch()
        api = FastAPI()

        @api.get("/health")
        def health():
            cfg = self.policy.config
            return {
                "status": "ok",
                # The reason this endpoint exists. Poll it after every deploy until it
                # reports the checkpoint you meant to test.
                "checkpoint": CHECKPOINT,
                "gpu": GPU,
                "dtype": str(self.dtype),
                "action_horizon": ACTION_HORIZON,
                "chunk_size": cfg.chunk_size,
                "uses_state": self.wants_state,
                "input_features": sorted(cfg.input_features),
                "wire_keys": sorted(WIRE_TO_FEATURE),
            }

        @api.post("/reset")
        def reset():
            self.policy.reset()
            return {"status": "reset"}

        @api.post("/act")
        async def act(request: Request):
            """Wire format, matching libero_closed_loop.query_server:
                in  {image: HWC uint8, wrist_image: HWC uint8, instruction: str, state: (8,)}
                out {actions: (10, 7), dt_ms: float}
            `instruction` is accepted and ignored -- ACT has no language input.
            """
            t0 = time.time()
            payload = json_numpy.loads(await request.body())

            required = list(WIRE_TO_FEATURE) + (["state"] if self.wants_state else [])
            missing = [k for k in required if k not in payload]
            if missing:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"payload missing {missing}; got {sorted(payload)}. "
                                      f"Run the sim client with --payload-keys libero."},
                )

            # UNBATCHED: AddBatchDimensionProcessorStep adds the leading dimension itself.
            obs = {}
            for wire_key, feature in WIRE_TO_FEATURE.items():
                img = np.asarray(payload[wire_key])
                if img.ndim != 3 or img.shape[2] != 3:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"{wire_key} must be HWC RGB, got {img.shape}"},
                    )
                # HWC uint8 [0,255] -> CHW float [0,1]. That is the range the training
                # dataset's frames were decoded into, and ACT normalises VISUAL with MEAN_STD
                # against statistics measured in it -- so this scaling is not cosmetic, it is
                # the input convention. copy=True because json_numpy returns a read-only
                # buffer that torch warns loudly about wrapping.
                t = self.torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1)
                obs[feature] = t.to(dtype=self.dtype) / 255.0

            if self.wants_state:
                state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
                expected = self.policy.config.input_features["observation.state"].shape[0]
                if state.shape[0] != expected:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"state must be {expected}-D [eef_pos(3), "
                                          f"axisangle(3), gripper_qpos(2)], got {state.shape}"},
                    )
                # copy=True for the same reason as the images above: json_numpy hands back a
                # read-only buffer and torch warns that writing through the wrapped tensor is
                # undefined. np.asarray alone does NOT copy when the dtype already matches.
                obs["observation.state"] = self.torch.from_numpy(
                    np.array(state, copy=True)).to(dtype=self.dtype)

            actions = self._predict(obs)
            return Response(
                content=json_numpy.dumps(
                    {"actions": actions, "dt_ms": 1000 * (time.time() - t0)}
                ),
                media_type="application/json",
            )

        return api


@app.local_entrypoint()
def smoke_test():
    """`modal run act/act_modal.py` -- bring the app up, hit /health, POST one synthetic
    observation, and check the returned chunk's shape and range.

    A synthetic frame proves the plumbing, NOT the policy: white noise is far out of
    distribution and the actions it produces mean nothing. Shape, dtype and a finite range
    are the whole assertion.
    """
    import json

    import json_numpy
    import numpy as np
    import requests

    json_numpy.patch()
    url = ACTServer().serve.get_web_url()
    print("url:", url)

    print("health:", json.dumps(requests.get(f"{url}/health", timeout=600).json(), indent=2))

    payload = {
        "image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "instruction": "pick up the green ball and put it in the green container",
        "state": np.zeros(8, dtype=np.float32),
    }
    resp = requests.post(f"{url}/act", data=json_numpy.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"}, timeout=600)
    resp.raise_for_status()
    out = json_numpy.loads(resp.text)
    actions = np.asarray(out["actions"])
    print(f"actions {actions.shape} dtype {actions.dtype} "
          f"range [{actions.min():.3f}, {actions.max():.3f}] in {out['dt_ms']:.0f} ms")
    assert actions.shape == (ACTION_HORIZON, 7), actions.shape
    assert np.isfinite(actions).all()
    print("OK")
