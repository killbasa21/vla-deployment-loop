"""Cut the first N episodes out of a converted LeRobot v3.0 dataset, for smoke runs.

    uv run python smolvla_libero/make_smoke_subset.py \
        --src smolvla_libero/data/a5_smolvla --out smolvla_libero/data/smoke_a5_2ep -n 2

WHY A SUBSET AND NOT JUST `--max-steps 1` ON THE REAL SET
---------------------------------------------------------
A 1-step run against `a5_smolvla` still uploads 397 MB and still has the loader index the
whole thing. A smoke run exists to prove build -> load -> step -> checkpoint-save, and none
of those need 30 episodes. Two episodes is 26 MB and uploads in seconds.

WHAT HAS TO BE REWRITTEN, AND WHAT MUST NOT BE
----------------------------------------------
v3.0 keeps global row indices in three places that have to stay consistent, which is why
this is a script and not a `cp` plus a hand-edited info.json:

  data/chunk-000/file-*.parquet    row-sliced. The `huggingface` SCHEMA METADATA blob is
                                   carried over verbatim -- it is what types the two image
                                   columns as Image features, and a slice that drops it
                                   yields struct<bytes,path> columns that are never decoded
                                   into pixels (convert_dataset.py documents this exact
                                   failure).
  meta/episodes/...parquet         row-sliced to the same N. Its `dataset_from_index` /
                                   `dataset_to_index` are global row offsets, and because
                                   we always take a PREFIX they stay correct untouched.
  meta/info.json                   total_episodes, total_frames, splits.

  meta/stats.json                  COPIED UNCHANGED, deliberately. These are the full
                                   dataset's statistics, not the subset's. Under
                                   `--norm-stats checkpoint` (the default) they are not
                                   consulted for normalisation at all, and recomputing them
                                   from 2 episodes would produce a normaliser that is wrong
                                   in a way a smoke run cannot detect. A subset is for
                                   exercising the code path, never for a number.

Prefix-only by construction: an arbitrary episode selection would require renumbering
`episode_index`, `index`, and both dataset_*_index columns, and every extra rewrite is
another way for a smoke fixture to differ from the real thing.
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("-n", "--episodes", type=int, default=2)
    args = p.parse_args()

    src, out = args.src, args.out
    if out.exists():
        shutil.rmtree(out)

    ep_path = next((src / "meta" / "episodes").rglob("*.parquet"))
    ep = pq.read_table(ep_path)
    if args.episodes > ep.num_rows:
        raise SystemExit(f"{src} has {ep.num_rows} episodes, asked for {args.episodes}")
    ep_keep = ep.slice(0, args.episodes)
    df = ep_keep.to_pandas()

    # Global row range of the kept prefix, read off the episode table rather than assuming
    # every episode is the same length (a5's are, a re-collection's need not be).
    n_frames = int(df["dataset_to_index"].iloc[-1])
    used_files = sorted(set(zip(df["data/chunk_index"], df["data/file_index"])))
    print(f"{src.name}: keeping episodes 0..{args.episodes - 1} = {n_frames} frames "
          f"from {len(used_files)} data file(s)")

    (out / "meta" / "episodes" / ep_path.parent.name).mkdir(parents=True, exist_ok=True)
    pq.write_table(ep_keep.replace_schema_metadata(ep.schema.metadata),
                   out / "meta" / "episodes" / ep_path.parent.name / ep_path.name)

    # Row-slice each data file the prefix touches. `frame_index` is per-episode and `index`
    # is global; both are already correct for a prefix, so only the row count changes.
    kept = 0
    for chunk, fidx in used_files:
        rel = Path(f"data/chunk-{chunk:03d}/file-{fidx:03d}.parquet")
        t = pq.read_table(src / rel)
        take = min(t.num_rows, n_frames - kept)
        sliced = t.slice(0, take).replace_schema_metadata(t.schema.metadata)
        (out / rel.parent).mkdir(parents=True, exist_ok=True)
        pq.write_table(sliced, out / rel)
        kept += take
        print(f"  {rel}: {t.num_rows} -> {take} rows")
    assert kept == n_frames, f"sliced {kept} rows, episode table says {n_frames}"

    for name in ("stats.json", "tasks.parquet", "cohorts.json"):
        if (src / "meta" / name).exists():
            shutil.copy2(src / "meta" / name, out / "meta" / name)

    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_episodes"] = args.episodes
    info["total_frames"] = n_frames
    info["splits"] = {"train": f"0:{args.episodes}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
