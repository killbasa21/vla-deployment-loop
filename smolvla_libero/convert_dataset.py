"""Copy `libero/fine_tune/a4` into this folder, renamed to SmolVLA-LIBERO's conventions.

WHY THIS EXISTS
---------------
`HuggingFaceVLA/smolvla_libero` is the official SmolVLA-450M checkpoint fine-tuned on
LIBERO. Its `config.json` declares:

    observation.images.image   [3, 256, 256]      <- LIBERO's agentview
    observation.images.image2  [3, 256, 256]      <- LIBERO's robot0_eye_in_hand
    observation.state          [8]                <- eef_pos(3) + axisangle(3) + grip(2)
    action                     [7]                <- 6-D delta EE + gripper, in [-1, 1]
    normalization  VISUAL=IDENTITY, STATE=MEAN_STD, ACTION=MEAN_STD

`a4` already matches THREE of those four exactly -- it was built against LIBERO's own
convention (see libero/fine_tune/README.md sec.2). The one divergence is the wrist camera's
key name: a4 calls it `observation.images.wrist_image`, SmolVLA calls it
`observation.images.image2`.

That rename is not cosmetic. LeRobot's LIBERO docs state it directly:

    "naming keys are encoded inside the normalization statistics layer"

A policy built from `smolvla_libero`'s config looks up `observation.images.image2`. Handed a
dataset that only has `wrist_image`, it does not error -- it finds no second camera, and the
wrist view (the cue for the last centimetre of the grasp) silently vanishes. This project has
already lost weeks to exactly this class of silent key/convention mismatch: PROGRESS.md sec.5
records `host_server_droid.py`'s hardcoded `NORM_TAG` yielding "garbage actions of the correct
shape". Rename it here, once, in every place it appears.

THE FOUR PLACES THE KEY APPEARS
-------------------------------
Renaming only the obvious one leaves a dataset that fails deep inside the loader:

  1. `data/chunk-*/file-*.parquet`  -- the Arrow column itself, AND the `huggingface`
     schema metadata blob, which is what `datasets` reads to type the column as an Image
     feature. Renaming the column without the metadata gives a struct<bytes,path> that is
     never decoded into pixels.
  2. `meta/info.json`              -- the `features` dict.
  3. `meta/stats.json`             -- the per-feature normalisation stats.
  4. `meta/episodes/chunk-*/file-*.parquet` -- five flattened per-episode stats columns
     (`stats/observation.images.wrist_image/{min,max,mean,std,count}`).

THE STATS SWAP (the second, less obvious fix)
---------------------------------------------
`a4/meta/stats.json` does NOT currently hold a4's own statistics. `pin_released_stats.py`
overwrote it with MolmoAct2's released LIBERO stats (`count = 273465` against a4's 19440) so
that the MolmoAct2 fine-tune would inherit the pretrained q01/q99 calibration it normalises
with. That was correct for MolmoAct2 and is WRONG for SmolVLA, which normalises STATE and
ACTION with MEAN_STD, not q01/q99, and whose own pretraining distribution is a different
dataset entirely.

`a4/meta/stats_measured.json` is the untouched backup of a4's real measured statistics. This
script installs that as `stats.json` and keeps the MolmoAct2 one alongside as
`stats_molmoact2_released.json` so the provenance is not lost.

(Open question deliberately NOT decided here: for a fine-tune you may instead want
`HuggingFaceVLA/libero`'s stats, so the action expert keeps the normalisation it was trained
under. That matters only when training starts; for running the STOCK checkpoint the policy
carries its own normalisation buffers and this file's stats are not consulted at all.)

Usage:
    uv run python smolvla_libero/convert_dataset.py
    uv run python smolvla_libero/convert_dataset.py --src libero/fine_tune/a4 --out smolvla_libero/data/a4_smolvla
    uv run python smolvla_libero/convert_dataset.py --verify-only
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OLD_KEY = "observation.images.wrist_image"
NEW_KEY = "observation.images.image2"

REPO_ROOT = Path(__file__).resolve().parents[1]
# a7 by default since 2026-08-01: a4 is ik-plant; a5 is the 27 s/episode collection whose
# slowness the first fine-tune reproduced; a6 fixed the speed and lost the grasp to a
# 4x-coarser action unit (PROGRESS.md sec.23.5). a7 is scale 0.10, distance-retimed,
# 60 episodes.
DEFAULT_SRC = REPO_ROOT / "libero" / "fine_tune" / "a7"
DEFAULT_OUT = REPO_ROOT / "smolvla_libero" / "data" / "a7_smolvla"

# What smolvla_libero's config.json declares. Checked against the converted dataset at the
# end so a future change to either side fails loudly here rather than at training time.
EXPECTED = {
    "observation.images.image": None,   # image feature, shape lives in info.json
    NEW_KEY: None,
    "observation.state": 8,
    "action": 7,
}


def rename_in_schema_metadata(schema: pa.Schema) -> pa.Schema:
    """Rewrite the `huggingface` metadata blob so `datasets` still types the renamed column
    as an Image feature. Without this the column decodes as a raw struct and the policy gets
    no pixels."""
    meta = dict(schema.metadata or {})
    blob = meta.get(b"huggingface")
    if blob is None:
        raise SystemExit(
            "parquet has no `huggingface` schema metadata -- this is not a v3.0 LeRobot "
            "dataset written by lerobot_v30_writer.py"
        )
    text = blob.decode()
    if OLD_KEY not in text:
        return schema  # already converted
    meta[b"huggingface"] = text.replace(OLD_KEY, NEW_KEY).encode()
    return schema.with_metadata(meta)


def convert_data_parquet(path: Path) -> None:
    """Rename the wrist column in one data shard, in place, preserving Arrow types and the
    huggingface metadata. Read-modify-write of the whole file: shards are capped at 100 MB
    by the writer, so this stays well inside memory."""
    table = pq.read_table(path)
    if OLD_KEY not in table.column_names:
        return
    # Capture the metadata BEFORE renaming: pyarrow's rename_columns() drops schema
    # metadata, so reading it off the renamed table finds nothing.
    metadata = dict(rename_in_schema_metadata(table.schema).metadata or {})
    names = [NEW_KEY if n == OLD_KEY else n for n in table.column_names]
    table = table.rename_columns(names).replace_schema_metadata(metadata)
    pq.write_table(table, path)


def convert_episodes_parquet(path: Path) -> None:
    """Rename the five flattened per-episode stats columns
    (`stats/<key>/{min,max,mean,std,count}`)."""
    table = pq.read_table(path)
    names = [n.replace(OLD_KEY, NEW_KEY) if OLD_KEY in n else n for n in table.column_names]
    if names == table.column_names:
        return
    pq.write_table(table.rename_columns(names), path)


def rename_json_key(path: Path, container: str | None = None) -> None:
    """Rename OLD_KEY -> NEW_KEY in a JSON file, keeping insertion order so the converted
    file diffs cleanly against the original."""
    if not path.exists():
        return
    obj = json.loads(path.read_text())
    target = obj[container] if container else obj
    if OLD_KEY not in target:
        return
    # Rebuild rather than pop+insert, so the renamed key keeps its original position.
    renamed = {(NEW_KEY if k == OLD_KEY else k): v for k, v in target.items()}
    if container:
        obj[container] = renamed
    else:
        obj = renamed
    path.write_text(json.dumps(obj, indent=4) + "\n")


def install_measured_stats(meta: Path) -> str:
    """Make a4's OWN measured statistics the active `stats.json`, preserving the MolmoAct2
    released stats under a name that says what they are. Returns a one-line report."""
    stats, measured = meta / "stats.json", meta / "stats_measured.json"
    if not measured.exists():
        return "stats_measured.json absent -- left stats.json untouched"

    active_count = json.loads(stats.read_text())["action"]["count"][0]
    measured_count = json.loads(measured.read_text())["action"]["count"][0]
    if active_count == measured_count:
        return f"stats.json already the measured stats (count {active_count})"

    shutil.move(stats, meta / "stats_molmoact2_released.json")
    shutil.copy(measured, stats)
    return (f"stats.json: MolmoAct2 released (count {active_count}) -> source measured "
            f"(count {measured_count}); old kept as stats_molmoact2_released.json")


def verify(out: Path) -> None:
    """Fail loudly on anything a policy built from smolvla_libero's config would need."""
    info = json.loads((out / "meta" / "info.json").read_text())
    feats = info["features"]
    problems = []

    for key, dim in EXPECTED.items():
        if key not in feats:
            problems.append(f"info.json missing feature {key}")
        elif dim is not None and feats[key]["shape"][0] != dim:
            problems.append(f"{key} shape {feats[key]['shape']} != [{dim}]")
    if OLD_KEY in feats:
        problems.append(f"info.json still carries {OLD_KEY}")

    stats = json.loads((out / "meta" / "stats.json").read_text())
    if NEW_KEY not in stats:
        problems.append(f"stats.json missing {NEW_KEY}")
    if stats["action"]["count"][0] != info["total_frames"]:
        problems.append(
            f"stats.json count {stats['action']['count'][0]} != info total_frames "
            f"{info['total_frames']} -- foreign stats are still installed"
        )

    for shard in sorted((out / "data").rglob("*.parquet")):
        schema = pq.ParquetFile(shard).schema_arrow
        if NEW_KEY not in schema.names:
            problems.append(f"{shard.name}: no {NEW_KEY} column")
        if OLD_KEY in schema.names:
            problems.append(f"{shard.name}: still has {OLD_KEY}")
        hf = (schema.metadata or {}).get(b"huggingface", b"").decode()
        if f'"{NEW_KEY}": {{"_type": "Image"}}' not in hf:
            problems.append(f"{shard.name}: {NEW_KEY} not typed as Image in hf metadata")

    for shard in sorted((out / "meta" / "episodes").rglob("*.parquet")):
        names = pq.ParquetFile(shard).schema_arrow.names
        if any(OLD_KEY in n for n in names):
            problems.append(f"{shard.name}: still has stats/{OLD_KEY}/* columns")

    if problems:
        raise SystemExit("VERIFY FAILED:\n  " + "\n  ".join(problems))

    print("\nverify PASS")
    print(f"  episodes / frames : {info['total_episodes']} / {info['total_frames']}")
    print(f"  codebase_version  : {info['codebase_version']}  fps {info['fps']}")
    print(f"  features          : {', '.join(k for k in feats if k.startswith('observation') or k == 'action')}")
    # The source dataset's OWN count. a4 is the only source where this is not automatic:
    # pin_released_stats.py overwrote its stats.json with MolmoAct2's released LIBERO stats
    # (count 273465), so restore_measured_stats swaps the measured file back in. a5/a6 were
    # never pinned, so their stats.json is already their own and the swap is a no-op.
    print(f"  action count      : {stats['action']['count'][0]} (source's own)")
    print("  action mean       :", [round(v, 3) for v in stats["action"]["mean"]])
    print("  action std        :", [round(v, 3) for v in stats["action"]["std"]])
    print("  state mean        :", [round(v, 3) for v in stats["observation.state"]["mean"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-run the checks on an already-converted --out and exit")
    args = ap.parse_args()

    if args.verify_only:
        verify(args.out)
        return

    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists; pass --force to overwrite")
        shutil.rmtree(args.out)
    if not args.src.exists():
        raise SystemExit(f"source dataset {args.src} not found")

    print(f"copying {args.src} -> {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.src, args.out)

    meta = args.out / "meta"

    print(install_measured_stats(meta))

    rename_json_key(meta / "info.json", container="features")
    for name in ("stats.json", "stats_measured.json", "stats_molmoact2_released.json"):
        rename_json_key(meta / name)

    shards = sorted((args.out / "data").rglob("*.parquet"))
    for i, shard in enumerate(shards, 1):
        convert_data_parquet(shard)
        print(f"  data shard {i}/{len(shards)}: {shard.name}", flush=True)

    for shard in sorted((meta / "episodes").rglob("*.parquet")):
        convert_episodes_parquet(shard)
        print(f"  episodes shard: {shard.name}")

    print(f"\nrenamed {OLD_KEY} -> {NEW_KEY}")
    verify(args.out)


if __name__ == "__main__":
    main()
