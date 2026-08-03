"""Print the state/action normalisation buffers a SmolVLA checkpoint actually carries.

`convert_dataset.py` (sec. "STATS") notes the checkpoint carries its own normalisation
buffers and the dataset's stats.json is not consulted at all. That means b1's state was
mapped through stock LIBERO's statistics, and a mismatch there is silent: the model
returns actions of the right shape in the wrong space.

    uv run modal run smolvla_libero/dump_norm_stats.py
    uv run modal run smolvla_libero/dump_norm_stats.py --checkpoint /checkpoints/...
"""
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.modal_images import lerobot_serve_image, with_infra

app = modal.App("smolvla-dump-norm")
hf_cache = modal.Volume.from_name("molmoact2-hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("molmoact2-checkpoints", create_if_missing=True)


@app.function(
    image=with_infra(lerobot_serve_image()),
    gpu="T4",
    volumes={"/cache/huggingface": hf_cache, "/checkpoints": checkpoints},
    timeout=20 * 60,
)
def dump(checkpoint: str = "HuggingFaceVLA/smolvla_libero"):
    import numpy as np
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # from_pretrained directly rather than make_policy: the latter wants dataset metadata
    # we do not have here, and the buffers are already on the module.
    policy = SmolVLAPolicy.from_pretrained(checkpoint)

    print(f"checkpoint: {checkpoint}")
    print(f"normalization_mapping: {getattr(policy.config, 'normalization_mapping', None)}")
    for attr in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
        mod = getattr(policy, attr, None)
        print(f"\n--- {attr}: {type(mod).__name__ if mod is not None else 'ABSENT'}")
        if mod is None:
            continue
        sd = mod.state_dict()
        if not sd:
            print("    EMPTY state_dict -- no statistics stored on this module")
        for k, v in sd.items():
            a = v.detach().float().cpu().numpy().ravel()
            print(f"    {k:<58} {np.round(a, 4) if a.size <= 32 else f'<{a.size} values>'}")


@app.local_entrypoint()
def main(checkpoint: str = "HuggingFaceVLA/smolvla_libero"):
    dump.remote(checkpoint=checkpoint)
