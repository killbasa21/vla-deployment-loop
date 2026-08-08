"""Modal image, volume and policy server.

One image definition, used by both serving and (later) training, so the two can
never drift onto different lerobot versions.

    modal run infra/modal_app.py::smoke          # load + one forward pass
    modal deploy infra/modal_app.py              # bring the server up
    curl https://<app>.modal.run/health

The server exposes:
    GET  /health  -> which checkpoint is actually loaded
    POST /act     -> {instruction, state[9], images{key: b64 png}} -> {actions[T][7]}
    POST /reset   -> clear any per-episode policy state
"""

import json
import os

import modal

APP_NAME = "greenbox"
LEROBOT_VERSION = "0.6.1"

# Cheap GPUs, in the order worth trying. L4 (24 GB) is the cheapest card that
# comfortably holds SmolVLA-450M at inference and at frozen-VLM fine-tuning.
SERVE_GPU = os.environ.get("GREENBOX_SERVE_GPU", "L4")

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(f"{APP_NAME}-vol", create_if_missing=True)
VOL = "/vol"

hf_cache = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)
HF_CACHE = "/hf-cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        f"lerobot[smolvla]=={LEROBOT_VERSION}",
        "fastapi[standard]",
        "pillow",
        "num2words",
        "accelerate",
    )
    .env({"HF_HOME": HF_CACHE, "TRANSFORMERS_VERBOSITY": "error"})
)

with image.imports():
    import base64
    import io

    import numpy as np
    import torch
    from PIL import Image

# Kept in sync with greenbox/task_spec.py. Duplicated rather than imported
# because the Modal container does not have the local package on its path.
STATE_DIM = 9
ACTION_DIM = 7
IMAGE_KEYS = ["observation.images.agentview", "observation.images.wrist"]
STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_SIZE = 256

DEFAULT_CHECKPOINT = "lerobot/smolvla_base"


# --------------------------------------------------------------------- helpers


def build_policy(checkpoint: str, stats: dict, device: str = "cuda"):
    """Instantiate SmolVLA with *our* feature shapes and load `checkpoint` into it.

    The stock checkpoint declares a 6-D state / 6-D action robot with three
    cameras. None of the transformer weights actually depend on that: SmolVLA
    pads state and action to `max_state_dim`/`max_action_dim` (32), so
    `state_proj` is Linear(32 -> H) and `action_out_proj` is Linear(E -> 32)
    regardless of the robot. The only shape-dependent tensors are the
    normalization buffers, which we rebuild from `stats`. So we construct a
    policy with our features and load the checkpoint non-strictly, then report
    exactly which keys did not transfer instead of silently ignoring them.
    """
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    for key in IMAGE_KEYS:
        input_features[key] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)
        )
    output_features = {
        ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        load_vlm_weights=False,
        device=device,
    )

    stats_t = {
        k: {kk: torch.tensor(vv, dtype=torch.float32) for kk, vv in v.items()}
        for k, v in stats.items()
    }
    policy = SmolVLAPolicy(cfg, dataset_stats=stats_t)

    src = SmolVLAPolicy.from_pretrained(checkpoint)
    missing, unexpected = policy.load_state_dict(src.state_dict(), strict=False)
    transferable = [k for k in missing if "normalize" not in k and "unnormalize" not in k]
    report = {
        "checkpoint": checkpoint,
        "missing_non_norm": transferable,
        "unexpected": [k for k in unexpected
                       if "normalize" not in k and "unnormalize" not in k],
    }
    del src

    policy.to(device)
    policy.eval()

    # lerobot 0.6 moved tokenization, normalization and device placement out of
    # the policy and into processor pipelines: `predict_action_chunk` reads
    # `observation.language.tokens`, which only the preprocessor produces.
    from lerobot.policies import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=stats_t
    )
    return policy, cfg, report, preprocessor, postprocessor


def load_stats(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------- server


@app.cls(
    image=image,
    gpu=SERVE_GPU,
    volumes={VOL: volume, HF_CACHE: hf_cache},
    scaledown_window=300,
    timeout=1800,
    max_containers=1,
)
class PolicyServer:
    checkpoint: str = modal.parameter(default=DEFAULT_CHECKPOINT)
    stats_path: str = modal.parameter(default=f"{VOL}/stats.json")

    def _resolve(self):
        """Which checkpoint to serve. `/vol/serve.json` wins over the default so a
        new fine-tune can be swapped in without redeploying the app."""
        cfg_path = f"{VOL}/serve.json"
        checkpoint, stats_path = self.checkpoint, self.stats_path
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                sel = json.load(fh)
            checkpoint = sel.get("checkpoint", checkpoint)
            stats_path = sel.get("stats_path", stats_path)
        local_stats = os.path.join(checkpoint, "greenbox_stats.json")
        if os.path.exists(local_stats):
            stats_path = local_stats  # a fine-tune carries the stats it trained with
        return checkpoint, stats_path

    @modal.enter()
    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_selected()

    def _load_selected(self):
        volume.reload()
        self.checkpoint, self.stats_path = self._resolve()
        stats = load_stats(self.stats_path)
        (self.policy, self.cfg, self.report,
         self.preprocessor, self.postprocessor) = build_policy(
            self.checkpoint, stats, self.device
        )
        n_total = sum(p.numel() for p in self.policy.parameters())
        n_train = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        self.report["params_total"] = n_total
        self.report["params_trainable"] = n_train
        self.report["stats_path"] = self.stats_path
        print(json.dumps(self.report, indent=2)[:4000])

    def _decode(self, b64):
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    @modal.method()
    def act(self, payload: dict) -> dict:
        state = torch.tensor(payload["state"], dtype=torch.float32,
                             device=self.device)[None]
        batch = {STATE_KEY: state, "task": [payload["instruction"]]}
        for key in IMAGE_KEYS:
            arr = self._decode(payload["images"][key])
            t = torch.from_numpy(arr).to(self.device).permute(2, 0, 1).float() / 255.0
            batch[key] = t[None]

        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.preprocessor(batch))
            chunk = self.postprocessor(chunk)
        chunk = chunk[0, :, :ACTION_DIM].float().cpu().numpy()
        # The env clips anyway, but clipping here keeps the logged actions in the
        # same space the expert's were, so the two are comparable.
        chunk = np.clip(chunk, -1.0, 1.0)
        return {"actions": chunk.tolist(), "checkpoint": self.checkpoint}

    @modal.method()
    def health(self) -> dict:
        return dict(self.report, device=self.device)

    @modal.method()
    def reset(self) -> dict:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        return {"ok": True}

    @modal.method()
    def reload(self) -> dict:
        self._load_selected()
        return dict(self.report, device=self.device)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request

        api = FastAPI()
        server = self

        @api.get("/health")
        async def _health():
            return server.health.local()

        @api.post("/reset")
        async def _reset():
            return server.reset.local()

        @api.post("/reload")
        async def _reload():
            return server.reload.local()

        @api.post("/act")
        async def _act(request: Request):
            return server.act.local(await request.json())

        return api


# ----------------------------------------------------------------- smoke tests


@app.function(image=image, gpu=SERVE_GPU, volumes={VOL: volume, HF_CACHE: hf_cache},
              timeout=1800)
def smoke(checkpoint: str = DEFAULT_CHECKPOINT):
    """Load the stock checkpoint into our feature shapes and run one forward pass."""
    stats_path = f"{VOL}/stats.json"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"{stats_path} missing -- run:\n"
            "  uv run python tools/dump_stats.py\n"
            f"  modal volume put {APP_NAME}-vol assets/stats.json /stats.json"
        )
    stats = load_stats(stats_path)
    policy, cfg, report, preprocessor, postprocessor = build_policy(checkpoint, stats)

    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"total params     {n_total / 1e6:.1f} M")
    print(f"trainable params {n_train / 1e6:.1f} M  ({n_train / n_total:.1%})")
    print(f"expert hidden    {policy.model.vlm_with_expert.expert_hidden_size}")
    print(f"vlm hidden       "
          f"{policy.model.vlm_with_expert.config.text_config.hidden_size}")
    print(f"chunk_size {cfg.chunk_size}  n_action_steps {cfg.n_action_steps}")
    print(f"missing non-norm keys: {len(report['missing_non_norm'])}")
    for k in report["missing_non_norm"][:20]:
        print("   ", k)
    print(f"unexpected keys: {len(report['unexpected'])}")
    for k in report["unexpected"][:20]:
        print("   ", k)

    batch = {
        STATE_KEY: torch.zeros(1, STATE_DIM, device="cuda"),
        "task": ["put the green box in the green container"],
    }
    for key in IMAGE_KEYS:
        batch[key] = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
    with torch.inference_mode():
        raw = policy.predict_action_chunk(preprocessor(batch))
        chunk = postprocessor(raw)
    print(f"action chunk shape {tuple(chunk.shape)}")
    print(f"first action (normalized)   {raw[0, 0, :ACTION_DIM].float().cpu().numpy()}")
    print(f"first action (unnormalized) {chunk[0, 0, :ACTION_DIM].float().cpu().numpy()}")
    return {"ok": True, "chunk_shape": list(chunk.shape),
            "params_total": n_total, "params_trainable": n_train}


@app.local_entrypoint()
def main(checkpoint: str = DEFAULT_CHECKPOINT):
    print(smoke.remote(checkpoint))


# --------------------------------------------------------------------- training


TRAIN_GPU = os.environ.get("GREENBOX_TRAIN_GPU", "L4")


@app.function(
    image=image,
    gpu=TRAIN_GPU,
    volumes={VOL: volume, HF_CACHE: hf_cache},
    timeout=24 * 3600,
)
def train(
    run_name: str = "ft1",
    steps: int = 10000,
    batch_size: int = 16,
    lr: float = 1e-4,
    warmup: int = 500,
    save_every: int = 2000,
    log_every: int = 50,
    data_dir: str = f"{VOL}/demos",
    base_checkpoint: str = DEFAULT_CHECKPOINT,
    num_workers: int = 8,
    seed: int = 0,
):
    """Fine-tune the action expert (and the projections) on the collected demos.

    The VLM and the vision encoder stay frozen -- that is `train_expert_only` and
    `freeze_vision_encoder`, the same recipe the base checkpoint was trained with,
    and it is what keeps this on a cheap GPU.
    """
    import glob
    import math
    import random
    import time

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    stats = load_stats(f"{VOL}/stats.json")
    policy, cfg, report, preprocessor, postprocessor = build_policy(
        base_checkpoint, stats, "cuda"
    )
    policy.train()

    chunk = cfg.chunk_size
    files = sorted(glob.glob(os.path.join(data_dir, "**", "ep_*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no episodes under {data_dir}")

    class Demos(torch.utils.data.Dataset):
        """One sample = one timestep, plus the `chunk`-long action window after it."""

        def __init__(self, files):
            self.files = files
            self.index = []
            self.lengths = {}
            for fi, f in enumerate(files):
                with np.load(f, allow_pickle=True) as d:
                    n = len(d["state"])
                self.lengths[fi] = n
                self.index.extend((fi, t) for t in range(n))

        def __len__(self):
            return len(self.index)

        def __getitem__(self, i):
            fi, t = self.index[i]
            with np.load(self.files[fi], allow_pickle=True) as d:
                state = d["state"][t]
                actions = d["action"][t : t + chunk]
                imgs = {
                    "observation.images.agentview": d["agentview"][t],
                    "observation.images.wrist": d["wrist"][t],
                }
                task = json.loads(str(d["meta"]))["instruction"]

            pad = chunk - len(actions)
            is_pad = np.zeros(chunk, dtype=bool)
            if pad > 0:
                # Repeat the last action rather than zero-padding: a zero action is
                # a real command (hold still), so it would teach the wrong thing if
                # the mask were ever ignored.
                actions = np.concatenate([actions, np.repeat(actions[-1:], pad, 0)])
                is_pad[len(actions) - pad :] = True

            out = {
                STATE_KEY: torch.from_numpy(np.asarray(state, np.float32)),
                ACTION_KEY: torch.from_numpy(np.asarray(actions, np.float32)),
                "action_is_pad": torch.from_numpy(is_pad),
                "task": task,
            }
            for key, buf in imgs.items():
                arr = np.asarray(Image.open(io.BytesIO(bytes(buf))).convert("RGB"))
                out[key] = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return out

    def collate(samples):
        out = {"task": [s["task"] for s in samples]}
        for key in samples[0]:
            if key == "task":
                continue
            out[key] = torch.stack([s[key] for s in samples])
        return out

    ds = Demos(files)
    print(f"{len(files)} episodes, {len(ds)} frames, chunk {chunk}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate, drop_last=True, persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"trainable {n_train / 1e6:.1f} M / {n_total / 1e6:.1f} M "
          f"({n_train / n_total:.1%})")

    opt = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=1e-10)

    def lr_at(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(steps - warmup, 1)
        return 0.025 + 0.975 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ckpt_root = f"{VOL}/checkpoints/{run_name}"
    os.makedirs(ckpt_root, exist_ok=True)

    def save(step):
        path = f"{ckpt_root}/step_{step:06d}"
        policy.save_pretrained(path)
        with open(f"{path}/greenbox_stats.json", "w") as fh:
            json.dump(stats, fh)
        volume.commit()
        print(f"saved {path}", flush=True)
        return path

    step, t0, losses = 0, time.time(), []
    log_path = f"{ckpt_root}/train_log.jsonl"
    with open(log_path, "a") as logf:
        while step < steps:
            for batch in loader:
                if step >= steps:
                    break
                batch = {k: (v.to("cuda", non_blocking=True)
                             if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = policy.forward(preprocessor(batch))
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

                losses.append(loss.detach().item())
                step += 1
                if step % log_every == 0:
                    rec = {
                        "step": step,
                        "loss": float(np.mean(losses[-log_every:])),
                        "lr": sched.get_last_lr()[0],
                        "grad_norm": float(gnorm),
                        "steps_per_s": step / (time.time() - t0),
                    }
                    print(json.dumps(rec), flush=True)
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                if step % save_every == 0:
                    save(step)

    final = save(step)
    return {"run": run_name, "steps": steps, "final": final,
            "loss": float(np.mean(losses[-200:]))}


@app.local_entrypoint()
def launch_train(run_name: str = "ft1", steps: int = 10000, batch_size: int = 16):
    print(train.remote(run_name=run_name, steps=steps, batch_size=batch_size))


@app.function(image=image, gpu=SERVE_GPU, volumes={VOL: volume, HF_CACHE: hf_cache},
              timeout=1800)
def inspect_params(checkpoint: str = DEFAULT_CHECKPOINT):
    """Enumerate exactly which tensors carry gradients, grouped by module."""
    from collections import defaultdict

    stats = load_stats(f"{VOL}/stats.json")
    policy, cfg, _report, _pre, _post = build_policy(checkpoint, stats)

    groups = defaultdict(lambda: [0, 0, 0])  # [trainable, frozen, n_tensors]
    for name, p in policy.named_parameters():
        if name.startswith("model.vlm_with_expert.lm_expert"):
            key = "action expert (lm_expert)"
        elif "vision_model" in name or "vision_tower" in name:
            key = "vision encoder (SigLIP)"
        elif "connector" in name or "modality_projection" in name:
            key = "VLM vision->text connector"
        elif name.startswith("model.vlm_with_expert"):
            key = "VLM decoder (16 layers)"
        elif "_proj" in name or "time_mlp" in name:
            key = f"projection: {name.split('.')[1]}"
        else:
            key = "other"
        g = groups[key]
        g[0 if p.requires_grad else 1] += p.numel()
        g[2] += 1

    print(f"{'module':<34}{'trainable':>14}{'frozen':>14}{'tensors':>9}")
    tot_t = tot_f = 0
    for key in sorted(groups, key=lambda k: -groups[k][0]):
        t, f, n = groups[key]
        tot_t += t
        tot_f += f
        print(f"{key:<34}{t:>14,}{f:>14,}{n:>9}")
    print(f"{'TOTAL':<34}{tot_t:>14,}{tot_f:>14,}")
    print(f"\ntrainable fraction {tot_t / (tot_t + tot_f):.1%}")

    print("\nevery trainable tensor outside the action expert:")
    for name, p in policy.named_parameters():
        if p.requires_grad and not name.startswith("model.vlm_with_expert.lm_expert"):
            print(f"  {name:<48} {tuple(p.shape)}")
    return {"trainable": tot_t, "frozen": tot_f}
