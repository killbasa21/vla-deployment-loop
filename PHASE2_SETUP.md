# Phase 2 — MolmoAct2-DROID server setup (as actually done)

This documents the exact steps taken to get the MolmoAct2-DROID inference server running
on a rented vast.ai GPU instance, including the real problems hit and how they were
resolved. Written so the same setup can be replicated quickly on a new instance.

## 1. Verify the GPU/environment

```bash
nvidia-smi
python3 --version
```

Confirmed: RTX 5090, 32607MiB VRAM, driver reporting CUDA 13.0, Python 3.12.

## 2. Validate PyTorch Blackwell (sm_120) support *before* installing anything

The RTX 5090 uses the Blackwell architecture (compute capability `sm_120`), which needs
PyTorch 2.7.0+ built against CUDA 12.8 (`+cu128`) for proper support. This vast.ai
template happened to ship several pre-built venvs (`/venv/torch-2.7.1`, `/venv/torch-2.9.1`,
`/venv/main` with torch 2.10.0) that already had this — checked each with:

```bash
python3 -c "
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))
print('sm_120' in torch.cuda.get_arch_list())
"
```

All three confirmed `sm_120` support. Used `/venv/main` (newest torch) going forward.

**If a fresh instance doesn't have this pre-built:** install explicitly with
`pip install torch --index-url https://download.pytorch.org/whl/cu128` and re-run the
check above before doing anything else — renting time is wasted if this doesn't check out.

## 3. Clone MolmoAct2

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/allenai/molmoact2.git
```

**Problem hit:** a plain `git clone` failed — one LFS-tracked test-fixture image
(`experiments/lerobot/tests/artifacts/...png`) 404'd on the LFS server. Skipping LFS
smudge entirely fixed it (we don't need that test fixture).

## 4. Do NOT blindly `uv sync` — the repo's pinned torch is wrong for Blackwell

`molmoact2/pyproject.toml` pins:
```toml
dependencies = ["torch==2.5.1", "torchvision==0.20.1", ...]
[tool.uv.sources]
torch = { index = "pytorch-cu121" }   # CUDA 12.1 -- predates Blackwell support
```

Running `uv sync` here would silently replace the working `sm_120`-capable torch with a
`cu121` build that doesn't support the 5090 at all. Instead, install everything **except**
torch/torchvision into the already-verified venv explicitly:

```bash
uv pip install --python /venv/main/bin/python \
  "transformers>=4.57,<4.58" "accelerate>=1.0" "safetensors>=0.4" \
  "huggingface-hub[cli]>=0.36" "hf-transfer>=0.1.8" "pillow>=10" \
  "numpy>=1.26,<3" "einops>=0.7" "sentencepiece>=0.2" "protobuf>=4.25" \
  "fastapi>=0.116" "uvicorn[standard]>=0.35" "json-numpy>=2.1.0" \
  "tyro>=1.0.13" "imageio>=2.37.3" "transforms3d>=0.4.2"
```

Then re-ran the torch/sm_120 check to confirm it wasn't touched.

## 5. Read the actual server code before running it

Read `examples/droid/host_server_droid.py` to get the exact wire protocol:
- `POST /act` body: `{external_cam, wrist_cam, instruction, state}` — `state` is `(8,)
  float32 = [q1..q7, gripper]`
- Response: `{actions: (N,8) float32, dt_ms: float}`

Read `sim_eval/inference/common.py` and `sim_eval/inference/client.py` (MolmoAct2's own
sim evaluation harness) to resolve what the gripper axis's units actually are:
`DroidClient.action_adapter = None` and `droid_state_adapter` passes the gripper value
straight through with **no normalization** — it's the raw radian position of a Robotiq
2F-85 knuckle joint on the real DROID rig, not a normalized 0-1 value.

## 6. Download the checkpoint and start the server

```bash
tmux new-session -d -s molmoact_server \
  "HF_HUB_ENABLE_HF_TRANSFER=0 /venv/main/bin/python examples/droid/host_server_droid.py \
   --host 0.0.0.0 --port 18000 --dtype bfloat16 2>&1 | tee server.log"
```

Checkpoint is `allenai/MolmoAct2-DROID`, ~21.8GB (5 safetensors shards + small config/
tokenizer files), downloaded automatically via `snapshot_download` on first run.

**Problem hit — `hf_transfer` stalled twice**, each time on one of the last two shards,
silently hanging indefinitely with no error (confirmed by watching the `.incomplete`
blob's byte count freeze for 15+ minutes while the process stayed alive). Fixed by
running with `HF_HUB_ENABLE_HF_TRANSFER=0` (disabling the accelerated Rust downloader)
and manually resuming the stuck files with the plain downloader:
```bash
HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download allenai/MolmoAct2-DROID <shard-name>
```
This is why the server is started with `HF_HUB_ENABLE_HF_TRANSFER=0` above — once
everything's cached, it doesn't matter, but avoids re-triggering the same stall behavior
if the cache ever needs a partial re-fetch.

**Problem hit — warmup failed with `TypeError: predict_action() got an unexpected keyword
argument 'action_mode'`.** The repo's own `CLAUDE.md` claims the DROID checkpoint uses
`action_mode="continuous"` (defaulted) while only the YAM checkpoint needs
`inference_action_mode`. This is stale: the actual `modeling_molmoact2.py` downloaded
from the Hub now requires `inference_action_mode` unconditionally (raises `ValueError` if
omitted) for the DROID checkpoint too — the Hub-hosted model code had been updated after
that doc was written. Fixed with a one-line edit in `host_server_droid.py`:
```python
# was: action_mode="continuous",
inference_action_mode="continuous",
```
Confirmed correct by grepping the actual downloaded `modeling_molmoact2.py`'s
`predict_action` signature rather than trusting the repo's documentation.

After this fix: `Warmup OK (566.0ms)`, server listening.

## 7. Confirm the server actually works end-to-end

```bash
python3 -c "
import numpy as np, requests, json_numpy
json_numpy.patch()
payload = {
    'external_cam': np.zeros((360,640,3), dtype=np.uint8),
    'wrist_cam': np.zeros((360,640,3), dtype=np.uint8),
    'instruction': 'pick up the red box and put it in the green container',
    'state': np.zeros(8, dtype=np.float32),
}
resp = requests.post('http://localhost:18000/act', data=json_numpy.dumps(payload),
                      headers={'Content-Type':'application/json'}, timeout=30)
print(resp.status_code, resp.json()['actions'])
"
```
Got back `200` and a real `(N, 8)` action chunk — server confirmed working.

## 8. Networking: reaching the server from your local machine

**Don't rely on an SSH `-L` tunnel if avoidable.** We initially used
`ssh -L 8101:localhost:8101 ...`, which worked for small GET requests but consistently
hung on POST bodies over a few hundred KB — traced (via junk-payload size sweeps that
bypassed the model entirely) to this instance's network path itself, not our code, the
tunnel, or vast.ai's infra specifically. The same hang happened even hitting vast.ai's
**direct external port** (no SSH tunnel at all), which ruled out SSH-tunnel multiplexing
as the cause. Root cause: the GPU instance was hosted in China, and this pattern (small
requests fine, larger uploads silently stalling) is a known signature of the Great
Firewall's handling of cross-border encrypted traffic.

**To find the real external port**: vast.ai's Docker Options `-p X:X` in the template do
**not** guarantee that literal port number is what's externally reachable. Check the
actual mapping in the vast.ai console: instance card → the small IP-and-port info popup
→ "Open Ports" list, e.g. `1.193.138.57:34011 -> 18000/tcp`. That external port
(`34011`), not the internal one (`18000`), is what a local client should connect to.

**Mitigation applied (client-side) for the network instability**: shrink the request
payload. MolmoAct2's own image processor resizes every input to 378x378 internally
regardless of what's sent (see `processor_config.json`), so there's no quality reason to
send anything larger — we render camera images at 128x128 instead of 640x360, cutting
the payload from ~1.8MB to ~131KB, which made requests complete reliably again (though
still slow — network overhead dominates: ~420ms of actual GPU inference vs. several
seconds of network transit per request on this China-hosted instance).

## Key files/paths on the instance
- `/workspace/molmoact2/` — the cloned repo
- `/workspace/molmoact2/examples/droid/host_server_droid.py` — the server (patched, see step 6)
- `/workspace/molmoact2/server.log` — server stdout, via the `tmux` session `molmoact_server`
- `/root/.cache/huggingface/hub/models--allenai--MolmoAct2-DROID/` — the downloaded checkpoint
- `/venv/main` — the venv used (torch 2.10.0+cu128, confirmed sm_120)
