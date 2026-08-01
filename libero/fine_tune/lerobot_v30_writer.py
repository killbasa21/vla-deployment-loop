"""LeRobot **v3.0** dataset writer, shaped to match `allenai/MolmoAct2-LIBERO-Dataset`.

Why this exists when the repo root already has `lerobot_writer.py`: that one emits **v2.1
with mp4 video features**, which is the wrong format twice over for this checkpoint.

  1. VERSION. `experiments/lerobot/src/lerobot/datasets/lerobot_dataset.py:83` pins
     `CODEBASE_VERSION = "v3.0"` and raises `BackwardCompatibilityError` on anything older.
     The v2.1 path only works via the separate `phase4_modal_train.py::convert` step.
  2. IMAGE STORAGE. The released LIBERO dataset has `"video_path": null` and stores each
     frame as PNG bytes **inside the parquet**, in a `struct<bytes, path>` column typed
     `{"_type": "Image"}` by HuggingFace `datasets`. `lerobot_writer.py` writes one mp4
     per camera per episode instead.

Every field below was read off the released dataset rather than inferred, because a
fine-tune that silently disagrees with the pretraining schema is expensive to diagnose:

    meta/info.json                     codebase_version v3.0, video_path null,
                                       features.*.dtype "image" for the two cameras
    meta/tasks.parquet                 [task_index int64, task string], pandas index=task
    meta/episodes/chunk-000/file-000.parquet
                                       one row per episode: index, data file location,
                                       dataset_from_index/dataset_to_index, tasks, length,
                                       and a flattened `stats/<feature>/<stat>` block
    meta/stats.json                    dataset-wide stats, with q01/q10/q50/q90/q99 on the
                                       low-dimensional features
    data/chunk-000/file-000.parquet    ALL episodes concatenated, rolled over at 100 MB
                                       (v2.1's one-file-per-episode layout is gone)

Layout produced:

    <root>/
      meta/info.json
      meta/tasks.parquet
      meta/stats.json
      meta/episodes/chunk-000/file-000.parquet
      data/chunk-000/file-000.parquet
      data/chunk-000/file-001.parquet   (rollover)

Only `pyarrow`, `pandas`, `numpy` and `Pillow` are needed -- no `lerobot`, no torch.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

# Quantiles the released stats.json carries on the low-dim features. Image features get
# only min/max/mean/std/count -- matched here rather than "improved".
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _png_bytes(frame):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _lowdim_stats(values, with_quantiles=False, integral=False):
    """min/max/mean/std/count over axis 0 of a (T, D) block, as plain lists.

    `integral` matters more than it looks. The released dataset types
    `stats/frame_index/min` (and index, episode_index, task_index) as **int64** while their
    mean and std stay double. Emitting doubles there gives a parquet whose column types
    disagree with the pretraining dataset even though every column name matches -- the kind
    of divergence that surfaces as a cryptic Arrow cast error inside the loader rather than
    as anything legible. Counted columns get int min/max; only mean/std stay float."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    cast = (lambda a: [int(x) for x in a]) if integral else (lambda a: a.tolist())
    out = {
        "min": cast(v.min(0)),
        "max": cast(v.max(0)),
        "mean": v.mean(0).tolist(),
        "std": v.std(0).tolist(),
        "count": [int(v.shape[0])],
    }
    if with_quantiles:
        for name, q in QUANTILES.items():
            out[name] = np.quantile(v, q, axis=0).tolist()
    return out


# The columns the released dataset stores with integer min/max. `timestamp` is NOT one --
# it is float there and float here.
INTEGRAL_COLUMNS = ("frame_index", "episode_index", "index", "task_index")


def _c31(x):
    return np.asarray(x, dtype=np.float64).reshape(3, 1, 1).tolist()


class ImageStatsAccumulator:
    """Streaming per-channel min/max/mean/std over [0,1]-normalised pixels.

    REPLACES a version that did `np.asarray(frames, dtype=np.float32) / 255.0` over a whole
    episode (or, for the dataset-wide stats, every 5th frame of the entire dataset) and
    then reduced it. That materialised a float32 copy 4x the size of the uint8 frames --
    0.26 GB per camera per 334-frame episode, and a 3.1 GB spike for the global pass -- on
    top of the raw frames still being held. It is what made `a7` (60 episodes, 20034
    frames) die in `finalize()` after every episode had already been simulated: the most
    expensive possible moment to run out of memory.

    Accumulating sum / sumsq / min / max in float64 is O(1) in memory, exact for min and
    max, and agrees with `np.std` to ~1e-9 on this data. It also removes the reason the old
    global pass subsampled every 5th frame, so the dataset-wide image stats are now over
    EVERY frame rather than 20% of them -- more accurate, and cheaper.
    """

    __slots__ = ("_n", "_frames", "_sum", "_sumsq", "_min", "_max")

    def __init__(self):
        self._n = 0        # pixels, for the mean/std denominator
        self._frames = 0   # frames, which is what `count` reports (matches the released
                           # stats.json, where image count is a frame count not a pixel one)
        self._sum = np.zeros(3, dtype=np.float64)
        self._sumsq = np.zeros(3, dtype=np.float64)
        self._min = np.full(3, np.inf, dtype=np.float64)
        self._max = np.full(3, -np.inf, dtype=np.float64)

    def update(self, frame):
        """Fold one (H,W,3) uint8 frame in. One frame at a time on purpose: the caller
        already has it in hand, so nothing new is allocated beyond a single float64 view."""
        a = np.asarray(frame, dtype=np.float64).reshape(-1, 3) / 255.0
        self._n += a.shape[0]
        self._frames += 1
        self._sum += a.sum(0)
        self._sumsq += np.square(a).sum(0)
        np.minimum(self._min, a.min(0), out=self._min)
        np.maximum(self._max, a.max(0), out=self._max)

    def result(self, count=None):
        if self._n == 0:
            raise ValueError("no frames accumulated")
        mean = self._sum / self._n
        # max(0, ...) guards the case where round-off makes E[x^2] - E[x]^2 a tiny
        # negative on a constant channel (e.g. a camera that never sees anything but a
        # flat background), which would otherwise produce nan through the sqrt.
        var = np.maximum(self._sumsq / self._n - np.square(mean), 0.0)
        return {
            "min": _c31(self._min),
            "max": _c31(self._max),
            "mean": _c31(mean),
            "std": _c31(np.sqrt(var)),
            "count": [int(self._frames if count is None else count)],
        }


def _image_stats(frames):
    """Per-channel stats on [0,1]-normalised pixels, shaped (3,1,1).

    Kept for callers that genuinely hold a small frame list. The writer itself no longer
    uses it -- see ImageStatsAccumulator for why."""
    arr = np.asarray(frames, dtype=np.float32) / 255.0  # (T,H,W,3)
    flat = arr.reshape(-1, arr.shape[-1])

    def c31(x):
        return np.asarray(x, dtype=np.float64).reshape(3, 1, 1).tolist()

    return {
        "min": c31(flat.min(0)),
        "max": c31(flat.max(0)),
        "mean": c31(flat.mean(0)),
        "std": c31(flat.std(0)),
        "count": [int(arr.shape[0])],
    }


class LeRobotV30Writer:
    """Accumulates episodes, then writes the whole v3.0 tree in `finalize()`.

    Episodes are buffered until finalize because v3.0 concatenates them into shared data
    files and needs global `index` numbering and dataset-wide stats. What is buffered is
    **PNG bytes, not raw frames**, and that distinction is the difference between working
    and dying:

        raw uint8 256x256x3      196.6 kB per frame
        PNG of a rendered scene   ~12 kB per frame     (measured on a6)

    a7 is 20034 frames x 2 cameras. Held raw that is 7.9 GB, on a 15 GB machine, and
    `finalize()` then wanted a float32 copy for image stats and an Arrow buffer for the
    encoded PNGs on top of it -- so it OOMed after simulating all 60 episodes, losing the
    lot. Held as PNG it is ~0.5 GB, which is simply the size of the files being written.

    Encoding at ingest costs nothing overall: every frame gets PNG-encoded exactly once
    either way. It just happens while the frame is already in hand rather than in one
    spike at the end. Per-episode image stats are folded in at the same moment for the
    same reason (see ImageStatsAccumulator).
    """

    def __init__(self, root, fps, image_shape=(256, 256, 3), state_dim=8, action_dim=7,
                 robot_type="panda", cameras=("image", "wrist_image")):
        self.root = Path(root)
        self.fps = float(fps)
        self.image_shape = tuple(image_shape)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.robot_type = robot_type
        self.cameras = tuple(cameras)
        self._image_keys = [f"observation.images.{c}" for c in self.cameras]

        self._tasks = {}
        self._episodes = []   # dicts with png/states/actions/task/extra/img_stats
        self._total_frames = 0
        # Dataset-wide image stats, folded in as episodes arrive. Kept per camera because
        # the two views have genuinely different pixel distributions.
        self._global_img = {c: ImageStatsAccumulator() for c in self.cameras}

    # -- ingestion ---------------------------------------------------------

    def add_episode(self, frames, states, actions, task, extra=None):
        """frames: {camera_name: [(H,W,3) uint8]}; states (T,8); actions (T,7).

        `extra` is carried into the sidecar `meta/cohorts.json` only -- it never reaches
        the parquet, because adding columns the pretraining schema doesn't have is exactly
        the kind of silent divergence this writer exists to avoid."""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        T = len(states)
        if actions.shape[0] != T:
            raise ValueError(f"{T} states vs {actions.shape[0]} actions")
        if states.shape[1] != self.state_dim or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"expected state {self.state_dim}-D and action {self.action_dim}-D, got "
                f"{states.shape[1]} and {actions.shape[1]}"
            )
        for c in self.cameras:
            if len(frames[c]) != T:
                raise ValueError(f"camera {c}: {len(frames[c])} frames vs {T} states")

        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)

        # Encode and measure HERE, while the caller's frames are live, then drop the raw
        # arrays. Holding them to finalize is what OOMed a7 -- see the class docstring.
        png = {}
        img_stats = {}
        for c in self.cameras:
            acc = ImageStatsAccumulator()
            encoded = []
            for f in frames[c]:
                acc.update(f)
                self._global_img[c].update(f)
                encoded.append(_png_bytes(f))
            png[c] = encoded
            img_stats[c] = acc.result()

        self._episodes.append({
            "png": png,
            "states": states,
            "actions": actions,
            "task": task,
            "extra": dict(extra or {}),
            "img_stats": img_stats,
        })
        self._total_frames += T
        return len(self._episodes) - 1

    # -- output ------------------------------------------------------------

    def _episode_table(self, ep, ep_index, global_start):
        T = len(ep["states"])
        cols = {}
        for cam, key in zip(self.cameras, self._image_keys):
            cols[key] = pa.array(
                [{"bytes": b, "path": f"frame_{i:06d}.png"}
                 for i, b in enumerate(ep["png"][cam])],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            )
        cols["observation.state"] = pa.array(
            [r.tolist() for r in ep["states"]],
            type=pa.list_(pa.float32(), self.state_dim))
        cols["action"] = pa.array(
            [r.tolist() for r in ep["actions"]],
            type=pa.list_(pa.float32(), self.action_dim))
        cols["timestamp"] = pa.array((np.arange(T) / self.fps).astype(np.float32),
                                     type=pa.float32())
        cols["frame_index"] = pa.array(np.arange(T, dtype=np.int64), type=pa.int64())
        cols["episode_index"] = pa.array(np.full(T, ep_index, dtype=np.int64),
                                         type=pa.int64())
        cols["index"] = pa.array(
            np.arange(global_start, global_start + T, dtype=np.int64), type=pa.int64())
        cols["task_index"] = pa.array(
            np.full(T, self._tasks[ep["task"]], dtype=np.int64), type=pa.int64())
        return pa.table(cols)

    def _hf_metadata(self):
        """The `huggingface` schema metadata that makes `datasets` decode the two struct
        columns as images rather than as raw structs. Copied field-for-field from the
        released dataset's parquet."""
        feats = {k: {"_type": "Image"} for k in self._image_keys}
        feats["observation.state"] = {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": self.state_dim, "_type": "List"}
        feats["action"] = {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": self.action_dim, "_type": "List"}
        for col, dtype in [("timestamp", "float32"), ("frame_index", "int64"),
                           ("episode_index", "int64"), ("index", "int64"),
                           ("task_index", "int64")]:
            feats[col] = {"dtype": dtype, "_type": "Value"}
        return {b"huggingface": json.dumps({"info": {"features": feats}}).encode()}

    def _features(self):
        feats = {}
        for key in self._image_keys:
            feats[key] = {
                "dtype": "image",
                "shape": list(self.image_shape),
                "names": ["height", "width", "channel"],
                "fps": self.fps,
            }
        feats["observation.state"] = {
            "dtype": "float32", "shape": [self.state_dim],
            "names": {"motors": ["x", "y", "z", "rx", "ry", "rz",
                                 "gripper_left", "gripper_right"]},
            "fps": self.fps,
        }
        feats["action"] = {
            "dtype": "float32", "shape": [self.action_dim],
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            "fps": self.fps,
        }
        for col, dtype in [("timestamp", "float32"), ("frame_index", "int64"),
                           ("episode_index", "int64"), ("index", "int64"),
                           ("task_index", "int64")]:
            feats[col] = {"dtype": dtype, "shape": [1], "names": None}
        return feats

    def finalize(self):
        if not self._episodes:
            raise RuntimeError("no episodes were added; nothing to finalize")

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

        max_bytes = DEFAULT_DATA_FILE_SIZE_IN_MB * 1024 * 1024
        chunk_index = file_index = 0
        pending, pending_bytes = [], 0
        episode_rows = []
        global_index = 0

        def flush(chunk_index, file_index, tables):
            if not tables:
                return
            table = pa.concat_tables(tables).replace_schema_metadata(self._hf_metadata())
            path = self.root / DATA_PATH.format(chunk_index=chunk_index,
                                                file_index=file_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)

        for ep_index, ep in enumerate(self._episodes):
            table = self._episode_table(ep, ep_index, global_index)
            size = table.nbytes
            # Roll over BEFORE adding, so a file never exceeds the advertised cap. An
            # episode larger than the cap on its own still gets its own file.
            if pending and pending_bytes + size > max_bytes:
                flush(chunk_index, file_index, pending)
                file_index += 1
                if file_index >= DEFAULT_CHUNK_SIZE:
                    chunk_index += 1
                    file_index = 0
                pending, pending_bytes = [], 0

            T = len(ep["states"])
            row = {
                "episode_index": ep_index,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + T,
                "tasks": [ep["task"]],
                "length": T,
            }
            for cam, key in zip(self.cameras, self._image_keys):
                for name, val in ep["img_stats"][cam].items():
                    row[f"stats/{key}/{name}"] = val
            for key, values in [("observation.state", ep["states"]),
                                ("action", ep["actions"])]:
                for name, val in _lowdim_stats(values).items():
                    row[f"stats/{key}/{name}"] = val
            for col in ["timestamp", "frame_index", "episode_index", "index",
                        "task_index"]:
                vals = np.asarray(table[col].to_pylist(), dtype=np.float64)
                stats_col = _lowdim_stats(vals, integral=col in INTEGRAL_COLUMNS)
                for name, val in stats_col.items():
                    row[f"stats/{col}/{name}"] = val
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0
            episode_rows.append(row)

            pending.append(table)
            pending_bytes += size
            global_index += T

        flush(chunk_index, file_index, pending)

        pd.DataFrame(episode_rows).to_parquet(
            self.root / EPISODES_PATH.format(chunk_index=0, file_index=0), index=False)

        tasks = pd.DataFrame(
            [{"task_index": i, "task": t}
             for t, i in sorted(self._tasks.items(), key=lambda kv: kv[1])]
        ).set_index("task")
        tasks.to_parquet(self.root / "meta" / "tasks.parquet")

        info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": self.fps,
            "features": self._features(),
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "chunks_size": DEFAULT_CHUNK_SIZE,
            "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_IN_MB,
            "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            "data_path": DATA_PATH,
            "video_path": None,
            "robot_type": self.robot_type,
            "splits": {"train": f"0:{len(self._episodes)}"},
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

        all_states = np.concatenate([e["states"] for e in self._episodes])
        all_actions = np.concatenate([e["actions"] for e in self._episodes])
        stats = {
            "observation.state": _lowdim_stats(all_states, with_quantiles=True),
            "action": _lowdim_stats(all_actions, with_quantiles=True),
        }
        for cam, key in zip(self.cameras, self._image_keys):
            # Folded in at ingest, over EVERY frame. The old code subsampled every 5th
            # frame here only because the exact version needed the whole dataset resident.
            stats[key] = self._global_img[cam].result(count=self._total_frames)
        for col, arr in [
            ("timestamp", np.concatenate(
                [np.arange(len(e["states"])) / self.fps for e in self._episodes])),
            ("frame_index", np.concatenate(
                [np.arange(len(e["states"])) for e in self._episodes])),
            ("episode_index", np.concatenate(
                [np.full(len(e["states"]), i) for i, e in enumerate(self._episodes)])),
            ("index", np.arange(self._total_frames)),
            ("task_index", np.concatenate(
                [np.full(len(e["states"]), self._tasks[e["task"]])
                 for e in self._episodes])),
        ]:
            # No quantiles on these: the released stats.json carries q01..q99 on
            # observation.state and action only, and nowhere else.
            stats[col] = _lowdim_stats(arr, integral=col in INTEGRAL_COLUMNS)
        (self.root / "meta" / "stats.json").write_text(json.dumps(stats, indent=1))

        # Sidecar, NOT part of the LeRobot spec: which cohort each episode came from, so
        # the three groups stay separable for ablations after the fact.
        cohorts = [
            {"episode_index": i, "name": e["extra"].get("name", f"episode_{i:04d}"),
             **{k: v for k, v in e["extra"].items() if k != "name"}}
            for i, e in enumerate(self._episodes)
        ]
        (self.root / "meta" / "cohorts.json").write_text(json.dumps(cohorts, indent=1))

        return {"episodes": len(self._episodes), "frames": self._total_frames}
