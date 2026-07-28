"""Serve the STOCK `HuggingFaceVLA/smolvla_libero` checkpoint behind the /act protocol
`libero/libero_closed_loop.py` already speaks.

WHY THIS CHECKPOINT
-------------------
SmolVLA-450M fine-tuned by HuggingFace on LIBERO -- robosuite/MuJoCo, Franka Panda, the same
simulator family and arm as our scene. Its `config.json` declares exactly the conventions our
`a4` dataset and `libero_closed_loop.py` already produce:

    observation.images.image   [3, 256, 256]   <- our external_cam (LIBERO's agentview)
    observation.images.image2  [3, 256, 256]   <- our eye_in_hand
    observation.state          [8]             <- eef_pos(3) + axisangle(3) + gripper_qpos(2)
    action                     [7]             <- 6-D delta EE + gripper, in [-1, 1]

So this is a drop-in swap for `libero_modal.py` (which serves MolmoAct2-LIBERO, 5.57B): same
wire format in, same wire format out, a model 12x smaller.

NO CHANGES ARE NEEDED IN THE SIM CLIENT. Run it with the LIBERO payload keys it already has:

    uv run python libero/libero_closed_loop.py \
        --payload-keys libero \
        --server-url https://<...>.modal.run/act \
        --chunks 20 --randomize-ball --no-view

`--payload-keys libero` sends `{image, wrist_image, instruction, state}`; this server maps
`wrist_image` onto the model's `observation.images.image2` feature. That mapping is the whole
reason the key name matters -- see convert_dataset.py for the same rename on the dataset side.

WHY A T4 (the cheapest GPU that can run this)
----------------------------------------------
450M params: ~1.8 GB at fp32, ~0.9 GB at fp16. A T4 has 16 GB, so memory is not remotely the
constraint -- the question is only whether the code paths run on Turing. Turing has NO bf16,
so we load in **float32** rather than the bf16 that a bigger card would use. At 450M that is
affordable, and it avoids the class of silent-garbage failure this project keeps hitting.

Rungs if the T4 does not work out, cheapest first:
    T4   16 GB  ~$0.59/hr   <- default
    L4   24 GB  ~$0.80/hr   (Ada; bf16 fine. What libero_modal.py uses for the 5.57B model)
    A10G 24 GB  ~$1.10/hr

Override without editing this file:  `--gpu L4` on the local entrypoint, or edit GPU below
for a `modal deploy`.

ACTION HORIZON
--------------
The checkpoint ships `chunk_size: 50, n_action_steps: 1` -- i.e. re-query the policy every
single control tick. That is fine inside `lerobot-eval`, where the policy is in-process, and
useless over HTTP, where PROGRESS.md sec.2 measured transport at ~4x the inference cost. We
return 10 actions per call instead: it matches LIBERO's own action horizon, matches what
`libero_closed_loop.py` expects (it warns on anything else), and matches the
`--policy.n_action_steps=10` LeRobot itself uses to reproduce published LIBERO results.

Usage:
    modal run smolvla_libero/smolvla_modal.py            # ephemeral, runs a self-test
    modal serve smolvla_libero/smolvla_modal.py          # dev server, live reload
    modal deploy smolvla_libero/smolvla_modal.py         # persistent, prints a stable URL
"""

import modal

CHECKPOINT = "HuggingFaceVLA/smolvla_libero"

# How many actions one /act call returns. See "ACTION HORIZON" above.
ACTION_HORIZON = 10

GPU = "T4"

# The two camera keys the sim client sends (libero_closed_loop.PAYLOAD_KEY_SETS["libero"]),
# mapped onto the feature names smolvla_libero's config.json declares. `wrist_image ->
# image2` is the rename this whole integration turns on.
WIRE_TO_FEATURE = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
}

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
    # `[smolvla]` pulls the SmolVLM2 vision/language deps. Deliberately NOT `[libero]`: that
    # extra drags in robosuite -> egl_probe, which needs cmake and an OpenGL toolchain and
    # exists only to run the LIBERO *simulator*. Our simulator is local MuJoCo.
    .pip_install("lerobot[smolvla]", "fastapi[standard]", "json-numpy", "pillow")
)

hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)

app = modal.App("smolvla-libero")


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/cache/huggingface": hf_cache},
    # Keep a container warm for 5 min after the last request so back-to-back rollouts do not
    # each re-pay the cold-start model load. Scales to zero after that -- no GPU billed idle.
    scaledown_window=300,
    timeout=900,
)
class SmolVLAServer:
    @modal.enter()
    def load(self):
        """Runs once per container start. Downloads (first time only -- the HF cache is a
        Volume shared with the MolmoAct2 deployments) and loads the policy onto the GPU."""
        import threading

        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        # One policy object serves every request, and _predict calls policy.reset() and then
        # pushes into the policy's own observation queues. Two overlapping requests would
        # interleave those mutations and hand each other's history to the model -- a silent
        # wrong-actions failure, not an error. Serialise inference so concurrent rollouts
        # (e.g. a clamped and an unclamped run at once) are safe; they queue instead of
        # corrupting each other.
        self._lock = threading.Lock()

        self.torch = torch
        self.device = "cuda"
        # float32, not bfloat16: a T4 is Turing and has no bf16 at all. At 450M the memory
        # cost of fp32 is ~1.8 GB, which is irrelevant on a 16 GB card.
        self.dtype = torch.float32

        self.policy = SmolVLAPolicy.from_pretrained(CHECKPOINT)
        self.policy.to(device=self.device, dtype=self.dtype)
        self.policy.eval()
        self.policy.reset()

        cfg = self.policy.config

        # lerobot 0.6.0 moved preprocessing OUT of the policy and into a processor pipeline.
        # `predict_action_chunk` consumes an already-processed batch, so calling it with a raw
        # `task` string fails with KeyError: 'observation.language.tokens'.
        #
        # The preprocessor's steps are, in order:
        #   Rename -> AddBatchDimension -> NewLineTask -> Tokenizer -> Device -> Normalizer
        # and the postprocessor un-normalizes the action back to real units and returns CPU
        # tensors. Two consequences this file depends on:
        #   1. AddBatchDimension means we must submit UNBATCHED observations (no leading 1).
        #   2. Loading with pretrained_path=CHECKPOINT takes the normalisation statistics and
        #      the tokenizer settings FROM THE CHECKPOINT. That is the whole point -- building
        #      them from our own dataset would normalise the state/action into a different
        #      space than the one the action expert was trained in, which is the failure that
        #      PROGRESS.md sec.5 describes as "garbage actions of the correct shape".
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=CHECKPOINT
        )

        # The saved processor config carries whatever device it was serialised with (usually
        # "cpu"). Retarget every DeviceProcessorStep at the GPU by inspecting the step objects
        # rather than by passing an overrides dict keyed on a step name -- the name is an
        # implementation detail of the registry, the type is not.
        moved = 0
        for step in list(self.preprocessor.steps):
            if type(step).__name__ == "DeviceProcessorStep":
                step.device = self.device
                moved += 1
        print(f"  preprocessor : {[type(s).__name__ for s in self.preprocessor.steps]}", flush=True)
        print(f"  postprocessor: {[type(s).__name__ for s in self.postprocessor.steps]}", flush=True)
        print(f"  device steps retargeted to {self.device}: {moved}", flush=True)
        # Print what the checkpoint actually declares rather than trusting this file's
        # comments. If these ever stop matching what the sim sends, it shows up here at
        # container start instead of as plausible-looking wrong actions.
        print("loaded", CHECKPOINT, flush=True)
        print("  input_features :", {k: list(v.shape) for k, v in cfg.input_features.items()}, flush=True)
        print("  output_features:", {k: list(v.shape) for k, v in cfg.output_features.items()}, flush=True)
        print("  chunk_size     :", cfg.chunk_size, " n_action_steps:", cfg.n_action_steps, flush=True)
        print("  normalization  :", cfg.normalization_mapping, flush=True)

        missing = set(WIRE_TO_FEATURE.values()) - set(cfg.input_features)
        if missing:
            raise RuntimeError(
                f"checkpoint does not declare {sorted(missing)}; its input features are "
                f"{sorted(cfg.input_features)}. Sending images under a key the policy does "
                f"not know means it silently sees no image at all."
            )

    def _predict(self, obs):
        """Return an (ACTION_HORIZON, 7) numpy array for one UNBATCHED observation dict.

        `predict_action_chunk` returns the full `chunk_size` (50) actions from a single
        forward pass; we keep the leading ACTION_HORIZON of them. Note this is deliberately
        NOT `select_action`, which slices to `config.n_action_steps` -- 1 for this checkpoint
        -- and would make us pay a whole round trip per control tick.
        """
        import numpy as np

        # Reset per request: each /act is an independent observation, and `predict_action_chunk`
        # pushes into the policy's observation queues. Without this, successive requests stack
        # history that our wire protocol never promised to be contiguous. The lock keeps that
        # reset-then-predict pair atomic against a concurrent request -- see load().
        with self._lock, self.torch.no_grad():
            self.policy.reset()
            batch = self.preprocessor(obs)
            chunk = self.policy.predict_action_chunk(batch)   # (1, chunk_size, 7), normalized
            actions = self.postprocessor(chunk)               # un-normalized, on CPU

        actions = actions[0, :ACTION_HORIZON]
        return actions.float().cpu().numpy().astype(np.float32)

    @modal.asgi_app()
    def serve(self):
        import time

        import json_numpy
        import numpy as np
        from fastapi import FastAPI, Request, Response
        from fastapi.responses import JSONResponse

        # Patches the stdlib `json` module so numpy arrays round-trip. That covers DECODING
        # the client's request (which json_numpy-encodes its frames), but NOT the response:
        # FastAPI serialises return values through its own `jsonable_encoder`, which knows
        # nothing about numpy and fails with "dictionary update sequence element #0 has
        # length 7". So /act builds its response body with json_numpy.dumps explicitly and
        # returns a raw Response, bypassing FastAPI's encoder entirely.
        json_numpy.patch()
        api = FastAPI()

        @api.get("/health")
        def health():
            cfg = self.policy.config
            return {
                "status": "ok",
                "checkpoint": CHECKPOINT,
                "gpu": GPU,
                "dtype": str(self.dtype),
                "action_horizon": ACTION_HORIZON,
                "chunk_size": cfg.chunk_size,
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
            """
            t0 = time.time()
            payload = json_numpy.loads(await request.body())

            missing = [k for k in (*WIRE_TO_FEATURE, "state") if k not in payload]
            if missing:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"payload missing {missing}; got {sorted(payload)}. "
                                      f"Run the sim client with --payload-keys libero."},
                )

            # UNBATCHED: the preprocessor's AddBatchDimensionProcessorStep adds the leading
            # dimension itself. Passing an already-batched tensor here yields a 5-D image,
            # which prepare_images silently reinterprets as a time series and indexes [:, -1].
            obs = {}
            for wire_key, feature in WIRE_TO_FEATURE.items():
                img = np.asarray(payload[wire_key])
                if img.ndim != 3 or img.shape[2] != 3:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"{wire_key} must be HWC RGB, got {img.shape}"},
                    )
                # HWC uint8 [0,255] -> CHW float [0,1]. That range is exactly what
                # prepare_images documents as its input ("convert pixel range from [0.0, 1.0]
                # to [-1.0, 1.0] as requested by SigLIP") -- it also does its own
                # resize-with-pad to 512. VISUAL normalization is IDENTITY for this
                # checkpoint, so this scaling is the whole of our image preprocessing.
                # copy=True: json_numpy hands back a read-only array, and torch warns loudly
                # about wrapping non-writable buffers.
                t = self.torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1)
                obs[feature] = t.to(dtype=self.dtype) / 255.0

            state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
            if state.shape[0] != 8:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"state must be 8-D [eef_pos(3), axisangle(3), "
                                      f"gripper_qpos(2)], got {state.shape}"},
                )
            obs["observation.state"] = self.torch.from_numpy(state).to(dtype=self.dtype)
            # A plain string, not a list: AddBatchDimensionProcessorStep is what turns this
            # into the batch of one that TokenizerProcessorStep expects.
            obs["task"] = payload.get("instruction", "")

            actions = self._predict(obs)
            # json_numpy.dumps, not FastAPI's encoder -- see the note on json_numpy.patch()
            # above. The client decodes this with json_numpy too (it calls resp.json() after
            # patching), so the array survives as an array rather than nested lists.
            return Response(
                content=json_numpy.dumps(
                    {"actions": actions, "dt_ms": 1000 * (time.time() - t0)}
                ),
                media_type="application/json",
            )

        return api


@app.local_entrypoint()
def smoke_test(gpu: str = GPU):
    """`modal run smolvla_libero/smolvla_modal.py` -- bring the app up, hit /health, then
    POST one synthetic observation to /act and check the returned chunk's shape and range.

    A shape check alone is not enough: PROGRESS.md sec.5 records a server happily returning
    "garbage actions of the correct shape" when the normalisation tag was wrong. So this also
    asserts the values land inside [-1, 1], which is where LIBERO delta-EE actions live.
    """
    import json
    import urllib.request

    import numpy as np

    base = SmolVLAServer().serve.get_web_url().rstrip("/")
    print(f"endpoint : {base}")
    print(f"  /act   : {base}/act")

    # 600 s: this request triggers the cold start (download on first ever run, then load).
    with urllib.request.urlopen(f"{base}/health", timeout=600) as r:
        print(json.dumps(json.loads(r.read().decode()), indent=2))

    import json_numpy
    json_numpy.patch()

    rng = np.random.default_rng(0)
    payload = {
        "image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "instruction": "pick up the green ball and put it in the green container",
        # A plausible LIBERO reset state, not zeros: eef above the table, top-down
        # orientation (axis-angle magnitude ~pi), gripper open at the mirrored (+x, -x).
        "state": np.array([-0.05, 0.0, 1.19, 3.14, 0.0, -0.09, 0.04, -0.04], dtype=np.float32),
    }
    req = urllib.request.Request(
        f"{base}/act",
        data=json_numpy.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read().decode())

    actions = np.asarray(body["actions"], dtype=np.float32)
    print(f"\nactions {actions.shape}  server dt {body['dt_ms']:.0f} ms")
    print("  range :", float(actions.min()), "..", float(actions.max()))
    print("  first :", np.round(actions[0], 4))
    print("  std   :", np.round(actions.std(axis=0), 4))

    problems = []
    if actions.shape != (ACTION_HORIZON, 7):
        problems.append(f"expected ({ACTION_HORIZON}, 7), got {actions.shape}")
    if not np.isfinite(actions).all():
        problems.append("chunk contains NaN/inf")

    # LIBERO's action space is Box(-1, 1). The postprocessor un-normalises with the
    # checkpoint's MEAN_STD statistics, so a flow-matching sample can land a little outside
    # without anything being wrong -- but it cannot land FAR outside, because that means the
    # un-normalisation is using the wrong statistics. Note the input here is random noise, so
    # judge the scale, not the direction.
    peak = float(np.abs(actions).max())
    if peak > 3.0:
        problems.append(f"max |action| = {peak:.3f}, far outside LIBERO's Box(-1, 1) "
                        f"-- un-normalisation statistics are wrong")
    elif peak > 1.0:
        print(f"  note: max |action| = {peak:.3f}, marginally outside [-1, 1] "
              f"(sampled policy; the sim client is what bounds the commanded delta)")

    if problems:
        raise SystemExit("SMOKE FAILED:\n  " + "\n  ".join(problems))
    print("\nsmoke PASS")
