"""Pin the released LIBERO normalisation statistics into one of our datasets.

WHY THIS EXISTS
---------------
`train_lerobot.py` defaults to `--norm_mode q01_q99`, and
`lerobot_utils/stats.py:_collect_tagged_stats` builds a tag's normaliser by merging the
`meta/stats.json` of every repo registered under that tag. Fine-tune on our data alone and
the `libero` tag's normaliser is rebuilt from our ~19k frames, throwing away the
calibration the action expert was pretrained under. Our action span is only 0.4-0.7 of the
released one per channel (README sec.6.5), so the rebuilt normaliser stretches our labels
to fill the same [-1, 1] the pretrained head already knows -- and the adapter spends its
small capacity relearning an affine map that already exists.

WHY NOT JUST MIX IN THE RELEASED DATASET
----------------------------------------
That is the documented alternative and it also works, but `olmo/data/data_loader.py:
_build_mixture` calls `get_dataset_by_name` on EVERY repo in a mixture group regardless of
its sampling rate, so a repo present only for its statistics is still fully materialised:
35 GB of inline-PNG parquet downloaded onto the Modal volume to obtain 9 kB of JSON.
(Sampling rate and stats collection ARE separate knobs -- per-repo `sampling_rate` scales
the group-relative rate, while stats merge count-weighted over all repos in the tag -- so
mixing in the released data at a low rate is the right move if we ever want replay against
forgetting. It is just not the cheap way to fix normalisation.)

WHAT IT CHANGES
---------------
Only the two keys `_collect_tagged_stats` actually reads: `action` and
`observation.state`. Image/index/timestamp stats stay ours. The measured file is kept
beside the replacement as `meta/stats_measured.json`, so the divergence is visible on disk
rather than hidden inside a wrapper script -- after this runs, `meta/stats.json` no longer
describes this dataset, and that is the whole point.

Usage:
    uv run python libero/fine_tune/pin_released_stats.py libero/fine_tune/a4
    uv run python libero/fine_tune/pin_released_stats.py libero/fine_tune/a4 --restore
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

RELEASED_STATS_URL = (
    "https://huggingface.co/datasets/allenai/MolmoAct2-LIBERO-Dataset"
    "/resolve/main/meta/stats.json"
)
PINNED_KEYS = ("action", "observation.state")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="dataset root, e.g. libero/fine_tune/a4")
    p.add_argument("--restore", action="store_true",
                   help="put meta/stats_measured.json back as meta/stats.json")
    p.add_argument("--released-stats", default=None,
                   help="local copy of the released meta/stats.json (default: download)")
    args = p.parse_args()

    meta = Path(args.root) / "meta"
    live, measured = meta / "stats.json", meta / "stats_measured.json"

    if args.restore:
        if not measured.exists():
            raise SystemExit(f"{measured} does not exist; nothing to restore")
        shutil.copyfile(measured, live)
        print(f"restored {live} from {measured}")
        return

    if args.released_stats:
        released = json.loads(Path(args.released_stats).read_text())
    else:
        print(f"fetching {RELEASED_STATS_URL}")
        with urllib.request.urlopen(RELEASED_STATS_URL) as r:
            released = json.loads(r.read())

    ours = json.loads(live.read_text())

    # Keep the first backup: re-running must not overwrite the measured file with an
    # already-pinned one, which would silently destroy the record of our own distribution.
    if not measured.exists():
        shutil.copyfile(live, measured)
        print(f"backed up measured stats -> {measured}")
    else:
        print(f"{measured} already exists, leaving it alone")

    for key in PINNED_KEYS:
        if key not in released:
            raise SystemExit(f"released stats has no key {key!r}")
        if key not in ours:
            raise SystemExit(f"{live} has no key {key!r}")
        n_ours = len(ours[key]["mean"])
        n_rel = len(released[key]["mean"])
        if n_ours != n_rel:
            raise SystemExit(
                f"dimension mismatch on {key}: ours {n_ours}-D, released {n_rel}-D")
        before = (ours[key]["q01"], ours[key]["q99"])
        ours[key] = released[key]
        print(f"{key}: {n_ours}-D")
        print(f"   q01 {[round(v, 3) for v in before[0]]}")
        print(f"    -> {[round(v, 3) for v in released[key]['q01']]}")
        print(f"   q99 {[round(v, 3) for v in before[1]]}")
        print(f"    -> {[round(v, 3) for v in released[key]['q99']]}")

    live.write_text(json.dumps(ours))
    print(f"\nwrote {live} with released {'/'.join(PINNED_KEYS)} statistics")


if __name__ == "__main__":
    main()
