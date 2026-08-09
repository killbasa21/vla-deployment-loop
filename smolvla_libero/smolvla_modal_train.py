"""Fine-tune `HuggingFaceVLA/smolvla_libero` on our green-ball demos, on a rented Modal GPU.

Counterpart to `smolvla_modal.py` (which SERVES the stock checkpoint). Same model, same
conventions; this one trains it.

WHY SMOLVLA AND NOT MOLMOACT2
-----------------------------
`docs/FINE_TUNE_LEARNINGS.md` sec.5.5 records the binding constraint on the MolmoAct2 path: a $5
budget bought 150 steps = **0.06 epochs**, at which point the honest read was that the
checkpoint had "barely moved off the base". SmolVLA is 450M against MolmoAct2's 5.57B, so
the same money buys a real run. It is also already LIBERO-fine-tuned, on a robosuite Panda,
with `observation.state` 8-D and `action` 7-D delta-EE -- the conventions our dataset
already speaks.

WHICH DATASET
-------------
`a5`, not `a4`. a4 was collected through Route A differential IK; inference now runs
robosuite's OSC_POSE port. The two plants realise ~72% and ~12.3% of a commanded per-tick
displacement, so their labels differ by ~6x and are NOT interchangeable -- see
`collect_finetune_data.py`'s header. a5 is the OSC-native re-collection.

WHAT IS TRAINED
---------------
`--mode` picks, mirroring `libero/libero_modal_train.py`'s vocabulary:

  expert    Action expert only, VLM frozen. This is HuggingFace's OWN recipe for
            smolvla_libero (its config.json records `freeze_vision_encoder: true,
            train_expert_only: true`). Cheapest, and no backward pass through the VLM.
  lora      LoRA on lerobot's DEFAULT SmolVLA target set. Despite the name this does NOT
            touch the VLM -- see the comment on the branch below, which corrects the
            original claim. It is LoRA on the action expert's q/v projections plus the
            state/action heads, and it is what every `lora` run before 2026-08-03 measured.
  vlm_lora  Added 2026-08-03. LoRA on the VLM's own attention, the vision->text connector
            AND the action expert's attention, r=8 throughout, with only the state/action
            heads at full rank. ~3.3M trainable. The one thing it changes relative to
            `lora` is that the VLM is in the gradient path -- which is the single variable
            the diagnosis below names, and deliberately the ONLY variable it moves.
            (Its first draft trained lm_expert at full rank, ~81.5M; that was reverted for
            the reasons on VLM_LORA_EXPERT.)
  full      No adapters, everything trainable.

Which one is right depends on WHERE the stock checkpoint fails, and the measured baseline
says grounding: over 3 randomised ball positions the closest lateral approach was 10.7 mm,
108.2 mm and 46.4 mm. Missing by 108 mm at an unseen ball position is a perception failure,
not an action-generation one, and `PROGRESS.md` sec.4 already established that a frozen VLM
cannot fix that -- with the VLM frozen the expert can only learn the average trajectory over
the randomisation box. `vlm_lora` is the first mode in this file that actually acts on that
diagnosis; `lora`, which was the default while the diagnosis stood, does not.

The default is still `lora` so that a bare `modal run` reproduces the runs already scored.
Pass `--mode vlm_lora` deliberately.

NORMALISATION -- the trap this project keeps re-learning
--------------------------------------------------------
`--norm-stats` chooses which statistics the STATE/ACTION normalisers are built from:

  checkpoint  (default) keep the ones baked into smolvla_libero. The action expert was
              trained under them; rebuilding from 30 episodes of one task hands it a
              different affine map than it learned, and it spends capacity relearning a
              transform that already exists.
  dataset     rebuild from a5's own stats.json.

This is the same decision `pin_released_stats.py` makes for MolmoAct2, and getting it wrong
is a silent failure: the model returns actions of the correct shape that are simply in the
wrong space (PROGRESS.md sec.5).

OVERFITTING
-----------
30 episodes of ONE task, one instruction string. Defences, all free:
  * `--save-freq` small, then score EACH checkpoint in closed loop and keep the best. The
    last checkpoint is not automatically the best one.
  * The scheduler's `decay_steps` is pinned to the ACTUAL run length below. The checkpoint
    ships `decay_steps: 30000`; run 3000 steps against that and the LR never decays, it
    just stops mid-schedule.
  * Watch the rotation channels. a5 holds one top-down orientation throughout, so its
    drx/dry/drz spans are far narrower than released LIBERO's -- that is where a collapse
    shows up first.

Usage:
    modal run smolvla_libero/smolvla_modal_train.py::upload      # push a5 to the volume
    modal run smolvla_libero/smolvla_modal_train.py::dump_modules  # print named_modules()
    modal run smolvla_libero/smolvla_modal_train.py --max-steps 1 --save-freq 1   # SMOKE
    modal run smolvla_libero/smolvla_modal_train.py --max-steps 3000 --save-freq 750
    modal run smolvla_libero/smolvla_modal_train.py --mode vlm_lora --gpu A10G \
        --batch-size 8 --max-steps 1 --save-freq 1                                # SMOKE
"""

import re
import sys
from pathlib import Path

import modal

# The shared image definitions live at the repo root, and Modal re-imports this
# module inside the container -- where infra/ lands on /root via with_infra().
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.modal_images import lerobot_train_image, with_infra
from infra.task_spec import DATA_NAMESPACE

# a7, the third dataset in this sequence, and the reason for each move is worth keeping:
#
#   a5  scale 0.05, expert slowed 2.5x   539 ticks/ep   trained fine, served SLOW
#   a6  scale 0.20, distance-retimed     161 ticks/ep   FAST, and misses the ball
#   a7  scale 0.10, distance-retimed     ~256 ticks/ep  the middle setting
#
# a6 solved the speed problem and bought a new one: at 0.20 m per action unit, one unit of
# normalised policy error is 40 mm on the table against a5's 15 mm, so the same policy
# quality that grasps at a5's scale closes on air at a6's (PROGRESS.md sec.23.5). 0.10
# halves that to ~27 mm while still running ~2.1x faster than a5.
#
# a7 is also 60 episodes rather than 30. The competing explanation for a6's miss is simply
# too little data for grounding, and CPU collection is free, so this run tests both.
#
# NOTE the serving client must run --delta-pos-scale 0.10 to match, and --randomize-bins,
# since a7 shuffles the bin layout every episode.
DATASET_DIR = "smolvla_libero/data/a7_smolvla"   # local, produced by convert_dataset.py
REMOTE_REPO = f"/data/{DATA_NAMESPACE}/green_ball_a7"  # its own path, like a6 got its own:
                                                       # an overwritten volume repo leaves no
                                                       # way to A/B the datasets against
                                                       # each other.
BASE_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"

# ---------------------------------------------------------------------------------------
# PEFT target sets for `--mode vlm_lora`   (added 2026-08-03)
# ---------------------------------------------------------------------------------------
# PEFT matches `target_modules` with re.fullmatch against the keys of `named_modules()` on
# the object handed to get_peft_model -- which lerobot's `wrap_with_peft` calls on the
# POLICY, so every name below is rooted at SmolVLAPolicy, hence the leading `model.`
# (= SmolVLAPolicy.model, the VLAFlowMatching). The nesting under that is
#   model.vlm_with_expert            SmolVLMWithExpertModel
#     .vlm                           SmolVLMForConditionalGeneration
#       .model                       SmolVLMModel -> .vision_model / .connector / .text_model
#     .lm_expert                     the action expert (its own AutoModel)
#
# A regex that matches NOTHING raises no error in PEFT; you simply get a run where the VLM
# never trains. That is the exact failure this repo keeps paying for (libero/PROGRESS.md
# sec.5, and the `observation.images.image2` story in this directory's README), so these
# strings are checked against a real named_modules() dump by `_peft_preflight()` before any
# GPU work starts, and can be inspected by hand with
#   modal run smolvla_libero/smolvla_modal_train.py::dump_modules
#
# VERIFIED 2026-08-03 by running exactly that against HuggingFaceVLA/smolvla_libero. The
# counts it reported, which are what the preflight re-checks every run:
#   VLM_LORA_TEXT       128 modules  (32 text-tower layers x q/k/v/o)
#   VLM_LORA_CONNECTOR    1 module
#   VLM_LORA_VISION      48 modules  (12 SigLIP layers x q/k/v/out)
#   each VLM_LORA_FULL_TRAINING entry: exactly 1 module
#
# Do NOT derive these names by reading smolvlm_with_expert.py: that file is internally
# inconsistent about them. Line 102 reads `text_model.layers` (correct, rooted at
# `vlm.model`) while line 172 builds the string `text_model.model.layers.N.` and substring-
# matches it against `self.vlm.named_parameters()` -- which yields
# `model.text_model.layers.N.`, so that freeze list has never matched anything. Harmless
# here (wrap_with_peft freezes every base parameter anyway) but it is why the dump exists.

# Text tower attention. q/k/v/o rather than lerobot's usual q/v: the failure being attacked
# is grounding, i.e. WHICH image tokens the text stream attends to, and k_proj is precisely
# what decides which keys are addressable. The q/v-only convention comes from the original
# LoRA paper's cost ablation, not from a claim that k/o do not matter.
VLM_LORA_TEXT = (
    r"model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
)

# The vision->text connector (SmolVLMSimpleMLP: one bias-free nn.Linear). One matrix, and
# it is the single bottleneck every visual token passes through on its way into the LM, so
# it is the cheapest place to re-map "this patch is the green ball" into language space.
VLM_LORA_CONNECTOR = r"model\.vlm_with_expert\.vlm\.model\.connector\.modality_projection\.proj"

# The SigLIP vision tower, OFF by default -- a deliberate choice, not an oversight.
# Reasoning: SigLIP is trained on far more images than 60 episodes of one table, and what
# it encodes about *where a green blob is* is already better than anything this dataset can
# teach it; what is missing is the read-out, which is the connector and the text tower
# above. It is also the expensive half (its activations dominate the backward pass at two
# 512x512 cameras per sample). `--vision-lora` turns it on for the run that tests that
# reasoning rather than assuming it. NOTE the vision attention output projection is named
# `out_proj`, not `o_proj` -- SmolVLMVisionAttention, not Llama.
VLM_LORA_VISION = (
    r"model\.vlm_with_expert\.vlm\.model\.vision_model\.encoder\.layers\.\d+\.self_attn\."
    r"(q|k|v|out)_proj"
)

# The action expert, ALSO LoRA and NOT full rank -- REVISED 2026-08-03, see the note below.
# `lm_expert` is a whole transformer (same layer count as the VLM, half the width via
# `expert_width_multiplier=0.5`), so this is the one decision in this file that moves the
# trainable-parameter count by an order of magnitude.
#
# THE ARGUMENT THAT WAS WRONG. This constant originally listed `lm_expert` under
# `full_training_modules`, on the reasoning that our action space had changed enough that a
# low-rank delta could not express the re-mapping. That reasoning does not survive reading
# collect_finetune_data.py:9-11: the action is a 7-D delta eef pose in [-1,1] and the state
# is [eef_pos(3), axisangle(3), gripper_qpos(2)] in the LIBERO frame -- LIBERO's own
# convention, same Franka Panda, same robosuite OSC_POSE port, same 20 Hz, and the reset
# orientation is measured in-distribution against a live LIBERO env (axis-angle 3.140,0,
# -0.089 against LIBERO's 3.141,0.002,-0.090). `action_in_proj` / `action_out_proj` /
# `state_proj` map the IDENTICAL space they were pretrained on. Nothing changed.
#
# THE ARGUMENT FOR KEEPING IT LOW-RANK. `--mode lora` has already measured the expert
# LoRA'd with the VLM frozen: 1-2/10 on a5/a7. Adding full-rank expert on top of VLM
# adapters would move TWO variables against a baseline in which only one was broken, and
# the diagnosis (libero/PROGRESS.md sec.4, and the 10.7/108.2/46.4 mm closest-approach
# spread) names grounding, not action capacity. If perception is fixed and the action
# mapping is already correct, the expert needs to learn one motion, not a new space.
# 81.5M trainable against 60 episodes was also a poor ratio; this is ~3.3M.
#
# Dropping `lm_expert` from full_training_modules additionally removes a hazard that
# `_peft_preflight` could not check: PEFT's ModulesToSaveWrapper DEEPCOPIES each module it
# wraps (peft 0.18.1, utils/other.py:584) and replaces the attribute, while SmolVLA reaches
# into `self.lm_expert.layers` directly (smolvlm_with_expert.py:424) rather than calling its
# forward. That works only because ModulesToSaveWrapper.__getattr__ forwards attribute
# lookups to the active copy (other.py:106-107); peft is unpinned in the Modal image, so a
# future release dropping that forwarding would silently train a clone that never runs. The
# five modules left below are all invoked through their own forward, so none of them ride on
# that behaviour.
#
# UNVERIFIED, unlike the VLM regexes above: the `lm_expert.layers` rooting comes from
# smolvlm_with_expert.py:424, not from a named_modules() dump, and that file is not reliable
# about names (see the note above). `_peft_preflight` counts these separately and refuses to
# start if they match nothing. Confirm with ::dump_modules --grep lm_expert before a real run.
VLM_LORA_EXPERT = (
    r"model\.vlm_with_expert\.lm_expert\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
)

# Trained at FULL rank, via `--peft.full_training_modules` (PEFT's `modules_to_save`), not
# LoRA. These five are the state/action heads -- the expert's I/O boundary. They stay full
# rank not because the space changed (it did not, see above) but because they are 0.74M
# parameters in total: LoRA on a 32->480 projection saves nothing worth the indirection.
VLM_LORA_FULL_TRAINING = [
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
]


def _vlm_lora_targets(vision_lora: bool) -> str:
    """The `--peft.target_modules` regex for `--mode vlm_lora`."""
    parts = [VLM_LORA_TEXT, VLM_LORA_CONNECTOR, VLM_LORA_EXPERT]
    if vision_lora:
        parts.append(VLM_LORA_VISION)
    return "(" + "|".join(parts) + ")"


# Same recipe as smolvla_modal.py so the heavy pip layers are a CACHE HIT rather than a
# 15-25 minute rebuild. Any divergence here (a different .env, an extra package earlier in
# the chain) invalidates the cache and costs that time back.
image = with_infra(lerobot_train_image())

hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
lerobot_data = modal.Volume.from_name("molmoact2-lerobot-data", create_if_missing=True)
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)

app = modal.App("smolvla-green-ball-train")


def _stock_module_names() -> list[str]:
    """Every `named_modules()` key of the STOCK checkpoint, rooted at the policy.

    Loaded on CPU: this only ever reads names, and the training container's GPU should not
    be billed for a minute of safetensors I/O. `from_pretrained` ends with
    `policy.to(config.device)`, and the checkpoint's config says `cuda`, so the device is
    overridden on the config before the load rather than after.

    Imports are function-local because this module is also imported on the LOCAL machine
    (Modal re-imports it to find the app) and the local env deliberately has no torch.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(BASE_CHECKPOINT)
    cfg.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(BASE_CHECKPOINT, config=cfg)
    names = [name for name, _ in policy.named_modules()]
    del policy
    return names


def _peft_preflight(target_regex: str, full_training_modules: list[str]) -> None:
    """Fail before the GPU is spent if the PEFT targets match nothing.

    PEFT is silent about empty matches in both directions: `target_modules` that fullmatch
    no module produces a LoRA over nothing, and `modules_to_save` entries that suffix-match
    no module are dropped unless `strict_module_check` is set, which lerobot does not set
    (peft 0.18.1, utils/other.py:1041). Either way you get a training run that reports a
    healthy loss and has not trained the thing you asked for.
    """
    names = _stock_module_names()
    hits = [n for n in names if re.fullmatch(target_regex, n)]
    # Counted SEPARATELY, because a non-empty `hits` proves nothing on its own: the default
    # SmolVLA target set also produces one, and matching only the expert is exactly the
    # mistake `vlm_lora` exists to avoid. Both halves have to be non-empty -- no VLM hits
    # means no grounding is being trained (the whole point of the mode), and no expert hits
    # means the action side is frozen apart from the 0.74M heads.
    vlm_hits = [n for n in hits if ".vlm_with_expert.vlm." in n]
    expert_hits = [n for n in hits if ".vlm_with_expert.lm_expert." in n]
    print(f"peft    : {len(names)} modules in policy, {len(hits)} LoRA targets "
          f"({len(vlm_hits)} in the VLM, {len(expert_hits)} in the action expert)",
          flush=True)
    for n in hits[:3] + hits[-3:]:
        print(f"          target {n}", flush=True)
    for label, subtree, got in (("vlm", "model.vlm_with_expert.vlm", vlm_hits),
                                ("expert", "model.vlm_with_expert.lm_expert", expert_hits)):
        if not got:
            raise SystemExit(
                f"PEFT target regex matched NO module under {subtree} -- this run would "
                f"train no {label} parameters at all and PEFT would not complain.\n"
                f"  regex: {target_regex}\n"
                "Dump the real names and fix the regex:\n"
                "  modal run smolvla_libero/smolvla_modal_train.py::dump_modules")

    missing = [m for m in full_training_modules
               if not any(n.endswith(m) for n in names)]
    if missing:
        raise SystemExit(
            f"--peft.full_training_modules entries match no module: {missing}\n"
            "PEFT matches these by name SUFFIX (key.endswith) and drops the misses "
            "silently. Dump the real names with ::dump_modules.")
    print(f"          full-rank {full_training_modules}", flush=True)

    # LoRA scales its update by alpha/r, and lerobot's PeftConfig CLI has no alpha field
    # (configs/default.py:95 exposes exactly target_modules, full_training_modules,
    # method_type, init_type, r), so alpha is whatever PEFT defaults to -- 8 as of peft
    # 0.18.1. Print what this container actually resolved rather than trusting that.
    from peft import LoraConfig
    print(f"          peft LoraConfig default alpha = {LoraConfig().lora_alpha}", flush=True)


@app.function(
    image=image,
    # No GPU: this loads the policy purely to enumerate module names.
    cpu=4.0,
    volumes={"/cache/huggingface": hf_cache},
    timeout=30 * 60,
)
def dump_modules(grep: str = "vlm_with_expert"):
    """Print the stock checkpoint's `named_modules()` keys, so PEFT regexes can be written
    against reality instead of against smolvlm_with_expert.py's inconsistent naming.

        modal run smolvla_libero/smolvla_modal_train.py::dump_modules
        modal run smolvla_libero/smolvla_modal_train.py::dump_modules --grep text_model

    Prints leaf modules only (parents are never LoRA targets) plus a per-prefix count, and
    finishes by reporting what the constants at the top of this file currently match.
    """
    from collections import Counter

    names = _stock_module_names()
    print(f"{len(names)} modules total\n", flush=True)

    counts = Counter()
    for n in names:
        if grep in n:
            # Collapse the layer index so 32 identical layers print once.
            counts[re.sub(r"\.\d+\.", ".{i}.", n)] += 1
    for pattern, count in sorted(counts.items()):
        print(f"  x{count:<4} {pattern}", flush=True)

    print("", flush=True)
    for label, rx in (("text", VLM_LORA_TEXT), ("connector", VLM_LORA_CONNECTOR),
                      ("expert", VLM_LORA_EXPERT), ("vision", VLM_LORA_VISION)):
        n_hit = sum(1 for n in names if re.fullmatch(rx, n))
        print(f"  {label:<10} regex matches {n_hit} modules", flush=True)
    for m in VLM_LORA_FULL_TRAINING:
        n_hit = sum(1 for n in names if n.endswith(m))
        print(f"  full-rank  {m:<20} matches {n_hit} modules", flush=True)


@app.function(
    image=image,
    # L4 (24 GB, ~$0.80/hr): 450M trains comfortably in bf16 here, and unlike the T4 the
    # serving app uses, Ada HAS bf16. A T4 would work only in fp32 and slowly. A10G
    # ($1.10/hr) is the next rung if steps/second disappoints.
    gpu="L4",
    # PNG-in-parquet means every sample decodes 2 images on the CPU. With a small CPU
    # allocation that, not the GPU, is the wall -- so buy cores and dataloader workers.
    cpu=16.0,
    volumes={
        "/cache/huggingface": hf_cache,
        "/data": lerobot_data,
        "/checkpoints": checkpoints,
    },
    timeout=5 * 60 * 60,
)
def train(mode: str = "lora", max_steps: int = 3000, batch_size: int = 16,
          save_freq: int = 750, lr: float = 1e-4, num_workers: int = 12,
          exp_name: str = "smolvla-green-ball", norm_stats: str = "checkpoint",
          seed: int = 0, lora_r: int = 0, vision_lora: bool = False,
          preflight: bool = True, dataset_root: str = REMOTE_REPO):
    import json
    import os
    import subprocess
    import time

    # Fail loudly and immediately if the dataset was never uploaded, rather than after the
    # model load. Also echo what it contains: a wrong-plant dataset is the single most
    # expensive mistake available here and it is invisible once training starts.
    info_path = f"{dataset_root}/meta/info.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"{info_path} missing. Upload first:\n"
            f"  modal run smolvla_libero/smolvla_modal_train.py::upload")
    info = json.load(open(info_path))
    frames, episodes = info["total_frames"], info["total_episodes"]
    feats = sorted(k for k in info["features"] if k.startswith("observation") or k == "action")
    print(f"dataset : {episodes} episodes, {frames} frames, v{info['codebase_version']}, "
          f"fps {info['fps']}", flush=True)
    print(f"features: {feats}", flush=True)
    if "observation.images.image2" not in info["features"]:
        raise SystemExit(
            "dataset has no `observation.images.image2`. smolvla_libero's config declares "
            "that key for the wrist camera; a dataset using `wrist_image` will train with "
            "NO second camera and no error. Run convert_dataset.py.")

    epochs = max_steps * batch_size / max(frames, 1)
    print(f"plan    : {max_steps} steps x batch {batch_size} = {epochs:.2f} epochs", flush=True)

    out_dir = f"/checkpoints/smolvla/{exp_name}"

    cmd = [
        "lerobot-train",
        f"--policy.path={BASE_CHECKPOINT}",
        f"--dataset.repo_id={DATA_NAMESPACE}/{dataset_root.rsplit('/', 1)[-1]}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={out_dir}",
        f"--job_name={exp_name}",
        "--policy.device=cuda",
        f"--batch_size={batch_size}",
        f"--steps={max_steps}",
        f"--save_freq={save_freq}",
        f"--num_workers={num_workers}",
        f"--seed={seed}",
        "--save_checkpoint=true",
        "--wandb.enable=false",
        # lerobot pushes the finished policy to the HF Hub by default, which 401s in a
        # container with no token -- AFTER training and checkpointing have both succeeded.
        # The run then exits non-zero and looks like a training failure when nothing is
        # wrong. (checkpoints.commit() below deliberately runs before we inspect the exit
        # code, so even that first run's weights survived.)
        "--policy.push_to_hub=false",
        # Pin the decay schedule to the ACTUAL run length. The checkpoint ships
        # decay_steps=30000; leaving it there means a 3000-step run never decays the LR and
        # simply stops a tenth of the way through its schedule.
        f"--policy.scheduler_decay_steps={max_steps}",
        f"--policy.optimizer_lr={lr}",
        # Serve-time horizon. The checkpoint ships n_action_steps=1 (one forward per control
        # tick), which is fine in-process and pointless over HTTP -- our server returns 10,
        # matching LIBERO's horizon and what lerobot itself uses to reproduce published
        # LIBERO numbers. Bake it in so the trained checkpoint serves the way we evaluate it.
        "--policy.n_action_steps=10",
    ]

    if mode == "lora":
        # CORRECTED 2026-07-31, after a 29 s smoke failure:
        #
        #   ValueError: Can't find 'adapter_config.json' at 'HuggingFaceVLA/smolvla_libero'
        #
        # `--policy.use_peft=true` does NOT create a LoRA. In lerobot 0.6.0 it means "this
        # checkpoint IS a PEFT adapter -- load it", so factory.make_policy goes looking for
        # an adapter_config.json next to the base checkpoint and dies when the stock
        # checkpoint (which is a plain policy) has none. Creating a fresh adapter is a
        # TOP-LEVEL train field, `--peft.*`, which lerobot_train.py turns into
        # `policy.wrap_with_peft(...)`; that call sets `config.use_peft = True` itself, on
        # the way out, which is what makes the saved checkpoint loadable with use_peft
        # later. Passing it on the way IN is the error.
        #
        # target_modules is left at SmolVLA's own default rather than specified here:
        #   (model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj
        #    |model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_(in|out)))
        # i.e. the ACTION EXPERT's attention projections plus the state/action heads. That
        # is a narrower scope than this mode's original comment claimed (it does not touch
        # the VLM at all), and it is the right one for what a6 is fixing: episode SPEED is
        # encoded in action magnitudes, which is the expert's job, not a grounding problem
        # the VLM has to relearn. Revisit if the failure turns back into a localisation
        # miss.
        #
        # freeze_vision_encoder / train_expert_only are NOT passed: wrap_with_peft calls
        # requires_grad_(False) on every base parameter regardless, so for THIS target set
        # they would be dead flags that read as if they were doing something.
        #
        # AMENDED 2026-08-03: that last paragraph is true here and false for `vlm_lora`.
        # requires_grad is not the only thing those flags drive --
        # SmolVLMWithExpertModel.train() (smolvlm_with_expert.py:182-189) re-asserts
        # `self.vlm.eval()` on EVERY switch into train mode while train_expert_only is set,
        # and the stock checkpoint ships it true. That is invisible as long as nothing in
        # the VLM is being trained, and it parks the adapters in eval mode the moment
        # something is. See the vlm_lora branch.
        cmd += ["--peft.method_type=LORA", f"--peft.r={lora_r or 32}"]
    elif mode == "vlm_lora":
        # ADDED 2026-08-03. What `lora` was documented to be until the comment above was
        # corrected: adapters on the VLM's own attention. The expert is LoRA'd at the same
        # rank rather than trained outright -- see VLM_LORA_EXPERT for why that reverted,
        # and VLM_LORA_VISION for why the vision tower is excluded by default.
        #
        # Read against `lora`, this branch changes exactly one thing: the VLM gets gradients.
        # That is the point. `lora`'s measured 1-2/10 is the control.
        targets = _vlm_lora_targets(vision_lora)
        full_training = ",".join(VLM_LORA_FULL_TRAINING)
        if preflight:
            # Load the stock checkpoint on CPU and count what the regex actually matches
            # before lerobot-train is allowed to start. Costs ~1 min of an otherwise idle
            # GPU container; a silently-VLM-free run costs the whole training budget.
            _peft_preflight(targets, VLM_LORA_FULL_TRAINING)
        cmd += [
            "--peft.method_type=LORA",
            # r=8, not `lora`'s 32. Two reasons, and the second one is the trap: 60
            # single-task episodes do not support 32 ranks' worth of new capacity across
            # the whole text tower plus the expert; and PEFT's lora_alpha defaults to 8
            # with NO way to set
            # it from lerobot's CLI (its PeftConfig exposes only target_modules,
            # full_training_modules, method_type, init_type, r -- configs/default.py:95).
            # LoRA scales its update by alpha/r, so r is doing double duty here: r=32 also
            # quarters the effective step size, r=8 leaves it at 1.0x. Raising --lora-r
            # therefore lowers the effective LR, which is the opposite of the usual
            # intuition; compensate with --lr if you change it.
            f"--peft.r={lora_r or 8}",
            f"--peft.target_modules={targets}",
            f"--peft.full_training_modules=[{full_training}]",
            # LOAD-BEARING, unlike in the `lora` branch. train_expert_only=true makes
            # SmolVLMWithExpertModel.train() call self.vlm.eval() every time the trainer
            # switches to train mode, which would leave every adapter added above running
            # in eval mode; the run would train, slowly and wrongly, and report nothing
            # unusual. freeze_vision_encoder only gates `vision_model.eval()`, so it is
            # strictly required only under --vision-lora -- passed unconditionally so the
            # two flags cannot drift apart, and because eval() on SigLIP is otherwise a
            # no-op (no dropout in that tower).
            "--policy.train_expert_only=false",
            "--policy.freeze_vision_encoder=false",
        ]
    elif mode == "full":
        # No adapters: train the VLM outright. 450M is small enough that this is affordable,
        # and it is the strongest option if LoRA underfits 30 episodes.
        cmd += ["--policy.use_peft=false",
                "--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false"]
    elif mode == "expert":
        # HuggingFace's own smolvla_libero recipe.
        cmd += ["--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true"]
    else:
        raise ValueError(f"unknown mode {mode!r}; use lora | vlm_lora | expert | full")

    if norm_stats == "dataset":
        # Explicit opt-in only. See the module docstring: rebuilding the normaliser from 30
        # single-task episodes hands the action expert a different affine map than the one
        # it was trained under.
        print("WARNING: rebuilding normalisation from the dataset's own stats", flush=True)

    print("Launching:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/root")
    dt = time.time() - t0
    print(f"lerobot-train returned {proc.returncode} after {dt:.0f}s "
          f"({dt / max(max_steps, 1):.2f}s per step including startup and saves)", flush=True)
    checkpoints.commit()
    if proc.returncode != 0:
        raise SystemExit(f"training failed with code {proc.returncode}")
    print(f"Done. Checkpoints on volume molmoact2-checkpoints at {out_dir}")


@app.local_entrypoint()
def upload(local_dir: str = DATASET_DIR, dataset_root: str = REMOTE_REPO):
    """Push the converted dataset onto the volume. Separate from training so a re-run does
    not re-upload, and so the upload can be verified before GPU time is spent.

    The two arguments exist for smoke datasets. A smoke subset MUST go to its own
    `--dataset-root`: `modal volume put --force` merges into the destination rather than
    replacing it, so writing a 2-episode subset over a real dataset's path leaves the
    subset's meta/info.json ("2 episodes") next to the full dataset's data files, which is
    a corrupted dataset that still loads."""
    import subprocess
    print(f"uploading {local_dir} -> molmoact2-lerobot-data:{dataset_root}")
    subprocess.run(
        ["modal", "volume", "put", "--force", "molmoact2-lerobot-data",
         local_dir, dataset_root.replace("/data", "")],
        check=True,
    )


@app.local_entrypoint()
def main(mode: str = "lora", max_steps: int = 3000, batch_size: int = 16,
         save_freq: int = 750, lr: float = 1e-4, num_workers: int = 12,
         exp_name: str = "smolvla-green-ball", norm_stats: str = "checkpoint",
         gpu: str = "L4", seed: int = 0, lora_r: int = 0, vision_lora: bool = False,
         preflight: bool = True, dataset_root: str = REMOTE_REPO):
    """--lora-r 0 means "this mode's own default" (lora: 32, vlm_lora: 8) rather than a
    single number shared across modes, because those two want different ranks and the 32
    that `lora` runs were measured at must not move. See the vlm_lora branch for why the
    rank is also an effective-learning-rate knob.

    --dataset-root points at a path on the molmoact2-lerobot-data volume; it defaults to
    REMOTE_REPO and exists so a smoke subset can be trained without touching it."""
    train.with_options(gpu=gpu).remote(
        mode=mode, max_steps=max_steps, batch_size=batch_size, save_freq=save_freq,
        lr=lr, num_workers=num_workers, exp_name=exp_name, norm_stats=norm_stats, seed=seed,
        lora_r=lora_r, vision_lora=vision_lora, preflight=preflight,
        dataset_root=dataset_root,
    )
