# Fine-tune run: ae_train / run_20260719_ae500

Action-expert-only fine-tune of MolmoAct2-DROID on the green-ball pick-and-place task.

- **Date:** 2026-07-19
- **Task / instruction:** "pick up the green ball and put it in the green container"
- **Base checkpoint:** `allenai/MolmoAct2-DROID`
- **Fine-tune mode:** action-expert-only (`ft_vlm=false`, `ft_action_expert=true`,
  `ft_embedding=none`, `lora_enable=false`) — VLM/vision/connector frozen, only the
  continuous flow-matching action expert is trained.
- **Compute:** Modal, 1× H100. App: https://modal.com/apps/hardikkapoor1021/main/ap-gHaye5SPHowp4tXkv5PnpK

## Dataset
- LeRobot repo id `greenbox/green_ball_pick`, 50 episodes / 6050 frames, 15 fps.
- Collected by `phase4_collect_demos.py` (IK-scripted expert), written by `lerobot_writer.py`.
- Converted v2.1 → v3.0 on the Modal volume (`molmoact2-lerobot-data`) before training.
- Cameras: `observation.images.external_cam`, `observation.images.wrist_cam`.
- State/action: 8-D `[q1..q7, gripper_rad]`, `norm_mode=min_max` (writer emits
  min/max/mean/std, not q01/q99, so the trainer's default q01_q99 is overridden).

## Hyperparameters
| | |
|---|---|
| max_steps (`--max_duration`) | 500 |
| global_batch_size | 8 |
| device_batch_size | 1 |
| save_interval | 250 (checkpoints at step 250, 500) |
| action_expert_learning_rate | 5e-5 |
| action_horizon / n_action_steps | 15 |
| packing | false |
| dynamic_seq_len | true |

## Launch command
```bash
modal run phase4_modal_train.py::main --mode ae_only --max-steps 500 \
    --save-interval 250 --global-batch-size 8 --device-batch-size 1 --exp-name ae_train
```

## Outputs
- Modal volume path: `molmoact2-checkpoints:/checkpoints/checkpoints/finetune/ae_train/`
- Pulled into: `./checkpoints/` (next to this file) via
  `modal volume get molmoact2-checkpoints checkpoints/finetune/ae_train fine_tunes/pick_up_tasks/ae_train/run_20260719_ae500/checkpoints`
- Correct recursive pull command used for the final local copy:
  ```bash
  uv run modal volume get --force molmoact2-checkpoints \
      checkpoints/finetune/ae_train/step500 \
      fine_tunes/pick_up_tasks/ae_train/run_20260719_ae500/checkpoints
  ```
- Earlier failed/ambiguous pull was moved aside at
  `checkpoints/step500.partial_before_recursive_pull/`.

## Results
Run completed cleanly (500/500 steps, single H100, `global_batch_size=1`, per-step
logging). The action-expert flow-matching loss fell from ~0.18 to ~0.01, confirming the
expert adapted to the green-ball data:

| step | action_flow_loss |
|---|---|
| 1   | 0.176 |
| 100 | 0.153 |
| 150 | 0.062 |
| 250 | 0.076 |
| 350 | 0.019 |
| 450 | 0.002 |
| 500 | 0.021 |

Per-step loss is noisy (batch size 1), but the trend is a clear decrease into the
~0.002–0.02 band by the back half. (20-step smoke reference reached ≈0.0506.) The loss
was flat rather than still descending at step 500, so 500 steps was sufficient.

**Config verified** (from `checkpoints/config.yaml`): `ft_action_expert=true`,
`ft_vlm=false`, `ft_embedding=none`, `lora_enable=false` (action-expert-only);
`norm_mode=min_max`; dataset `greenbox/green_ball_pick`; base
`allenai/MolmoAct2-DROID`. This is exactly the intended run.

**Note:** a first attempt (`global_batch_size=8`, `log_interval=20`) was aborted after
16 min — that config ran 8 sequential micro-batches per step with no logging until step
20, so it crawled with zero visibility on a 70%-idle H100. Relaunched with the config
above (fast + per-step logs).

### Checkpoint (this folder)
- `checkpoints/step500/` — final checkpoint, **PyTorch Distributed Checkpoint (DCP)**
  format: `model_and_optim/` = 16 `__0_N.distcp` shards + `.metadata` (model **and**
  optimizer state, ~25 GB), plus `train/rank0.pt` (RNG/step state) and `config.yaml`.
- `checkpoints/config.yaml` — top-level run config.
- `step250` also exists on the Modal volume (`molmoact2-checkpoints`) but was not pulled.

## Serving this checkpoint
The DCP format above is a **training** checkpoint. The base Phase 3 server
(`phase3_modal.py`) loads the released Hugging Face checkpoint through
`host_server_droid.py`, so it is still the right server for `allenai/MolmoAct2-DROID`.

For this fine-tuned checkpoint, use the new `phase3_modal_finetuned.py` wrapper. It is
intended to load the raw DCP checkpoint directly from the Modal checkpoint volume using
`molmoact2/experiments/scripts/serve_policy.py`, with:
- checkpoint: `/checkpoints/checkpoints/finetune/ae_train/step500`
- image preset: `droid` (`external_cam`, `wrist_cam`)
- norm tag: `green_ball_pick`
- action mode: `continuous`

Current status:
- `phase3_modal_finetuned.py` has been added.
- It has not yet been deployed.
- `/health` has not yet been checked.
- `phase3_closed_loop.py` has not yet been run against the fine-tuned endpoint.
- No training, download, or simulation process is currently running.

Next commands:
```bash
modal deploy phase3_modal_finetuned.py
uv run python phase3_closed_loop.py --model-path droid \
    --server-url <printed-finetuned-modal-url>/act \
    --request-timeout 600 --chunks 5
```

Fallback path if direct DCP serving fails: convert the checkpoint to Hugging Face format
with `olmo.hf_model.convert_molmoact2_to_hf`, then adjust a serving wrapper to load the
exported HF directory from a Modal volume.
