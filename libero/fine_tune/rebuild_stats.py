"""Rebuild `meta/stats.json` for a v3.0 dataset whose writer died after the parquet files.

WHY THIS EXISTS
---------------
`lerobot_v30_writer.finalize()` writes, in order: the data parquets, `meta/episodes/`,
`meta/tasks.parquet`, `meta/info.json`, then `meta/stats.json`, then `meta/cohorts.json`.
The image-stats block just before `stats.json` builds

    sample = [f for e in self._episodes for f in e["frames"][cam][::5]]
    arr = np.asarray(sample, dtype=np.float32) / 255.0

which materialises every fifth frame of the ENTIRE dataset as one float32 array -- for `a7`
(60 episodes, 20034 frames, 256x256x3, two cameras) about 4007 frames x 786 KB = 3.1 GB per
camera, on top of a collector process already holding every raw frame in RAM. `a7` was
killed there: the tree it left has data, episodes, tasks and info.json, and no stats.

Re-collecting is ~25 minutes of CPU and would hit the same wall. The parquets are complete
and authoritative, so this script recomputes the same statistics FROM THEM, streaming, with
bounded memory.

WHAT IT DOES NOT RECOVER
------------------------
`meta/cohorts.json`, the sidecar recording which cohort (reach / noise / recover) each
episode came from. That lives only in the collector's in-memory `extra` dict and is not
written into the parquets. It is not part of the LeRobot spec and no training or serving
path reads it -- it exists for after-the-fact ablations. A dataset repaired here can be
trained on; it just cannot be split by cohort.

EXACTNESS
---------
The low-dim stats are computed from the same values over the same axis as the writer, so
they match. The image stats differ in the last bits only: the writer reduces a float32
array, this accumulates in float64 over the same subsampled frames (every 5th frame of each
episode, `count` set to the dataset's total frames -- both conventions copied deliberately).
Normalisation cannot tell the difference.

Usage:
    uv run python libero/fine_tune/rebuild_stats.py libero/fine_tune/a7
    uv run python libero/fine_tune/rebuild_stats.py libero/fine_tune/a7 --check
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# Imported rather than re-implemented: if the writer's convention changes (quantile set,
# integral columns, the (3,1,1) nesting), this repair follows it instead of drifting.
from lerobot_v30_writer import INTEGRAL_COLUMNS, _lowdim_stats

# The writer's own subsample stride for image statistics.
IMAGE_STRIDE = 5


def _data_files(root):
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no data parquets under {root / 'data'}")
    return files


class _ImageAccumulator:
    """Streaming per-channel min/max/mean/std over [0,1] pixels, shaped (3,1,1).

    Holds three float64 triples and a count, so memory is O(1) in the dataset size -- which
    is the entire point of this file.
    """

    def __init__(self):
        self.n = 0                                  # pixels seen, per channel
        self.lo = np.full(3, np.inf)
        self.hi = np.full(3, -np.inf)
        self.total = np.zeros(3)
        self.total_sq = np.zeros(3)
        self.frames = 0

    def add(self, frame_uint8):
        a = np.asarray(frame_uint8, dtype=np.float64).reshape(-1, 3) / 255.0
        self.lo = np.minimum(self.lo, a.min(0))
        self.hi = np.maximum(self.hi, a.max(0))
        self.total += a.sum(0)
        self.total_sq += (a * a).sum(0)
        self.n += a.shape[0]
        self.frames += 1

    def result(self, count):
        if self.frames == 0:
            raise RuntimeError("no frames accumulated")
        mean = self.total / self.n
        # Population variance, matching np.std's default ddof=0 in the writer. Clipped at 0
        # because the sum-of-squares form can go a few ulps negative on a constant channel.
        var = np.maximum(self.total_sq / self.n - mean * mean, 0.0)

        def c31(x):
            return np.asarray(x, dtype=np.float64).reshape(3, 1, 1).tolist()

        return {
            "min": c31(self.lo),
            "max": c31(self.hi),
            "mean": c31(mean),
            "std": c31(np.sqrt(var)),
            "count": [int(count)],
        }


def rebuild(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    total_frames = info["total_frames"]
    image_keys = [k for k, v in info["features"].items() if v["dtype"] == "image"]
    lowdim_keys = ["observation.state", "action"]
    index_cols = ["timestamp", "frame_index", "episode_index", "index", "task_index"]

    print(f"{root}: {info['total_episodes']} episodes, {total_frames} frames, "
          f"v{info['codebase_version']}")
    print(f"  image features : {image_keys}")

    accum = {k: _ImageAccumulator() for k in image_keys}
    lowdim = {k: [] for k in lowdim_keys}
    idx = {c: [] for c in index_cols}
    seen = 0

    for path in _data_files(root):
        pf = pq.ParquetFile(path)
        # Row-group at a time: one group is a few hundred MB of PNG bytes at most, against
        # the whole file (~100 MB compressed, several GB decoded) if read at once.
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            for k in lowdim_keys:
                lowdim[k].append(np.asarray(table[k].to_pylist(), dtype=np.float64))
            for c in index_cols:
                idx[c].append(np.asarray(table[c].to_pylist(), dtype=np.float64))

            # Subsample by the writer's rule -- every 5th frame OF EACH EPISODE, which is
            # `frame_index % 5 == 0`, not every 5th row of the file. The two differ whenever
            # an episode length is not a multiple of the stride, i.e. always.
            frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
            wanted = np.nonzero(frame_index % IMAGE_STRIDE == 0)[0]
            for k in image_keys:
                col = table[k].to_pylist()
                for i in wanted:
                    accum[k].add(Image.open(io.BytesIO(col[i]["bytes"])).convert("RGB"))
            seen += table.num_rows
        print(f"  {path.name}: {seen}/{total_frames} frames", flush=True)

    if seen != total_frames:
        raise SystemExit(
            f"read {seen} rows but info.json declares total_frames={total_frames}. The data "
            f"files are incomplete; stats built from them would be wrong. Re-collect.")

    stats = {}
    for k in lowdim_keys:
        stats[k] = _lowdim_stats(np.concatenate(lowdim[k]), with_quantiles=True)
    for k in image_keys:
        # count is the dataset's frame total, NOT the number of frames sampled -- the
        # writer's own convention, and what the released stats.json carries.
        stats[k] = accum[k].result(total_frames)
        print(f"  {k}: sampled {accum[k].frames} frames, "
              f"mean {np.ravel(stats[k]['mean']).round(4).tolist()}")
    for c in index_cols:
        stats[c] = _lowdim_stats(np.concatenate(idx[c]),
                                 integral=c in INTEGRAL_COLUMNS)

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="dataset root, e.g. libero/fine_tune/a7")
    ap.add_argument("--check", action="store_true",
                    help="compute and print, but do not write stats.json")
    args = ap.parse_args()

    out = args.root / "meta" / "stats.json"
    if out.exists() and not args.check:
        raise SystemExit(
            f"{out} already exists. This script is for repairing a dataset that has none; "
            f"refusing to overwrite. Delete it first if that is really what you want.")

    stats = rebuild(args.root)

    action = stats["action"]
    print("\naction, the channel that decides whether this dataset is usable:")
    print(f"  q01  {np.round(action['q01'], 3).tolist()}")
    print(f"  q99  {np.round(action['q99'], 3).tolist()}")
    print(f"  std  {np.round(action['std'], 3).tolist()}")
    print(f"  count {action['count'][0]}")

    if args.check:
        print("\n--check: nothing written")
        return
    out.write_text(json.dumps(stats, indent=1))
    print(f"\nwrote {out}")
    print("NOTE: meta/cohorts.json is NOT recoverable from the parquets -- see the module "
          "docstring. Cohort-split ablations on this dataset are not available.")


if __name__ == "__main__":
    main()
