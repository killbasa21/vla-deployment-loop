"""
Minimal LeRobot v2.1 dataset writer (no `lerobot`/torch dependency).

Emits the on-disk layout `LeRobotDataset` expects, so the demos collected by
phase4_collect_demos.py can be loaded by the LeRobot ecosystem (and, once MolmoAct2's
fine-tuning code ships, fed to it) without this lightweight client machine having to
install the full `lerobot` package + its torch stack. We only depend on `pyarrow`
(parquet) and `imageio[ffmpeg]` (mp4 video encode).

Layout produced (v2.1):

    <root>/
      meta/
        info.json            # feature schema, fps, counts, path templates
        tasks.jsonl          # {task_index, task}
        episodes.jsonl       # {episode_index, tasks, length}
        episodes_stats.jsonl # per-episode min/max/mean/std/count for every feature
      data/chunk-000/
        episode_000000.parquet   # per-frame state/action/index columns (no pixels)
      videos/chunk-000/
        observation.images.<cam>/episode_000000.mp4

Images are stored as `video` features (mp4), not inline in the parquet -- the parquet
holds only the low-dimensional columns, and LeRobotDataset decodes the matching mp4 by
timestamp (frame_index / fps) at load time. Camera frames are therefore assumed to be
recorded at the same fps as the control loop, one frame per parquet row.
"""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000  # episodes per data/videos chunk dir; our runs fit in chunk-000

# Video encoding settings, mirrored into info.json's per-feature "info" block so a
# LeRobot loader decodes with matching parameters.
_VIDEO_CODEC = "h264"
_VIDEO_PIX_FMT = "yuv420p"


def _feature_stats(values):
    """min/max/mean/std/count for a stack of low-dim vectors, shape (T, D).
    Returned as plain Python lists (length D), the shape LeRobot stores per feature."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [int(values.shape[0])],
    }


def _image_stats(frames):
    """Per-channel min/max/mean/std for a list of (H,W,3) uint8 frames, normalized to
    [0,1], shaped (3,1,1) -- LeRobot's channel-first image-stat convention."""
    arr = np.asarray(frames, dtype=np.float32) / 255.0  # (T,H,W,3)
    per_ch = arr.reshape(-1, arr.shape[-1])  # (T*H*W, 3)

    def c31(x):
        return x.reshape(3, 1, 1).tolist()

    return {
        "min": c31(per_ch.min(0)),
        "max": c31(per_ch.max(0)),
        "mean": c31(per_ch.mean(0)),
        "std": c31(per_ch.std(0)),
        "count": [int(arr.shape[0])],
    }


class LeRobotDatasetWriter:
    def __init__(self, root, fps, cameras, image_shape, state_dim, action_dim,
                 robot_type="unknown"):
        self.root = Path(root)
        self.fps = float(fps)
        self.cameras = tuple(cameras)
        self.image_shape = tuple(image_shape)  # (H, W, C)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.robot_type = robot_type

        self.meta_dir = self.root / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self._image_keys = [f"observation.images.{c}" for c in self.cameras]
        self._tasks = {}          # task string -> task_index
        self._episodes = []       # {episode_index, tasks, length}
        self._episodes_stats = [] # {episode_index, stats}
        self._total_frames = 0
        self._episode_index = 0

    def _task_index(self, task):
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    def _chunk(self, episode_index):
        return episode_index // CHUNK_SIZE

    def add_episode(self, frames, states, actions, task):
        """frames: {cam_name: [ (H,W,3) uint8, ... ]}; states/actions: (T, D) arrays."""
        ep = self._episode_index
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        T = len(states)
        assert actions.shape[0] == T
        for c in self.cameras:
            assert len(frames[c]) == T, f"{c}: {len(frames[c])} frames vs {T} states"

        task_index = self._task_index(task)
        chunk = self._chunk(ep)

        # --- per-frame parquet columns (pixels live in the mp4s, not here) ---
        timestamps = (np.arange(T) / self.fps).astype(np.float32)
        frame_index = np.arange(T, dtype=np.int64)
        global_index = np.arange(self._total_frames, self._total_frames + T, dtype=np.int64)
        table = pa.table({
            "observation.state": pa.array([r.tolist() for r in states],
                                          type=pa.list_(pa.float32())),
            "action": pa.array([r.tolist() for r in actions],
                               type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(T, ep, dtype=np.int64), type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(np.full(T, task_index, dtype=np.int64), type=pa.int64()),
        })
        data_path = self.root / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)

        # --- one mp4 per camera ---
        for cam, key in zip(self.cameras, self._image_keys):
            vid_path = self.root / f"videos/chunk-{chunk:03d}/{key}/episode_{ep:06d}.mp4"
            vid_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                vid_path, frames[cam], fps=self.fps, codec="libx264",
                pixelformat=_VIDEO_PIX_FMT, macro_block_size=None,
            )

        # --- per-episode stats (v2.1 episodes_stats.jsonl) ---
        stats = {
            "observation.state": _feature_stats(states),
            "action": _feature_stats(actions),
        }
        for cam, key in zip(self.cameras, self._image_keys):
            stats[key] = _image_stats(frames[cam])
        for col, dtype in [("timestamp", np.float32), ("frame_index", np.int64),
                           ("episode_index", np.int64), ("index", np.int64),
                           ("task_index", np.int64)]:
            col_vals = table[col].to_numpy().astype(dtype).reshape(-1, 1)
            stats[col] = _feature_stats(col_vals)
        self._episodes_stats.append({"episode_index": ep, "stats": stats})

        self._episodes.append({"episode_index": ep, "tasks": [task], "length": T})
        self._total_frames += T
        self._episode_index += 1

    def _features(self):
        motor_names = lambda p, n: [f"{p}_{i}" for i in range(n)]
        feats = {}
        for key in self._image_keys:
            feats[key] = {
                "dtype": "video",
                "shape": list(self.image_shape),
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": self.fps,
                    "video.height": self.image_shape[0],
                    "video.width": self.image_shape[1],
                    "video.channels": self.image_shape[2],
                    "video.codec": _VIDEO_CODEC,
                    "video.pix_fmt": _VIDEO_PIX_FMT,
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        feats["observation.state"] = {
            "dtype": "float32", "shape": [self.state_dim],
            "names": motor_names("state", self.state_dim),
        }
        feats["action"] = {
            "dtype": "float32", "shape": [self.action_dim],
            "names": motor_names("action", self.action_dim),
        }
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            feats[col] = {
                "dtype": "float32" if col == "timestamp" else "int64",
                "shape": [1], "names": None,
            }
        return feats

    def finalize(self):
        n_ep = self._episode_index
        if n_ep == 0:
            raise RuntimeError("no episodes were added; nothing to finalize")
        n_chunks = self._chunk(n_ep - 1) + 1

        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": n_ep,
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": n_ep * len(self.cameras),
            "total_chunks": n_chunks,
            "chunks_size": CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self._features(),
        }
        (self.meta_dir / "info.json").write_text(json.dumps(info, indent=4))

        with open(self.meta_dir / "tasks.jsonl", "w") as f:
            for task, idx in sorted(self._tasks.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

        with open(self.meta_dir / "episodes.jsonl", "w") as f:
            for ep in self._episodes:
                f.write(json.dumps(ep) + "\n")

        with open(self.meta_dir / "episodes_stats.jsonl", "w") as f:
            for es in self._episodes_stats:
                f.write(json.dumps(es) + "\n")
