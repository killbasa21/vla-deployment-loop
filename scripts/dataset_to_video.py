#!/usr/bin/env python3
"""Render LeRobot image-dtype episodes to side-by-side mp4.

The a5/a6/a7 fine-tune datasets store both cameras as PNG bytes inside the
parquet files (`dtype: "image"` in `meta/info.json`), so there is no `videos/`
directory to point a player at. This decodes them, hstacks external + wrist,
and writes one mp4 per episode.

Local-env only: pyarrow / pillow / imageio. No torch, no lerobot.

    uv run python scripts/dataset_to_video.py libero/fine_tune/a7 --episodes 0,1,2
    uv run python scripts/dataset_to_video.py libero/fine_tune/a7 --out /tmp/a7_vids
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def parse_episodes(spec: str | None) -> set[int] | None:
    """"0,3,5" or "0-9" or "0-4,7" -> set of indices. None means every episode."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def image_keys(info: dict) -> list[str]:
    keys = [k for k, v in info["features"].items() if v.get("dtype") == "image"]
    if not keys:
        video = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        raise SystemExit(
            f"No image-dtype features. Found video-dtype {video} — that dataset "
            "already has mp4s under videos/, play those instead."
        )
    # external camera first, wrist second
    return sorted(keys, key=lambda k: ("wrist" in k, k))


def decode(cell) -> np.ndarray:
    """Parquet image cell -> RGB array. LeRobot stores {'bytes':..., 'path':...}."""
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = 4 + 12 * i
        draw.text((5, y + 1), line, fill=(0, 0, 0))  # shadow, for light scenes
        draw.text((4, y), line, fill=(255, 255, 0))
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="dataset root, e.g. libero/fine_tune/a7")
    ap.add_argument("--episodes", help='"0,1,2" or "0-9"; default all')
    ap.add_argument("--out", type=Path, help="output dir; default <dataset>/videos_rendered")
    ap.add_argument("--fps", type=float, help="override fps from meta/info.json")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip the frame/gripper text overlay")
    args = ap.parse_args()

    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    keys = image_keys(info)
    fps = args.fps or float(info.get("fps", 20))
    wanted = parse_episodes(args.episodes)
    out_dir = args.out or (args.dataset / "videos_rendered")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.dataset.glob("data/chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.dataset}/data/chunk-*/")

    # v3.0 packs several episodes per file, v2.1 is one file per episode; group by
    # episode_index either way. Read one file at a time so memory stays bounded.
    writers: dict[int, imageio.Writer] = {}
    counts: dict[int, int] = {}
    try:
        for f in files:
            cols = keys + ["episode_index", "frame_index", "action"]
            table = pq.read_table(f, columns=cols)
            episodes = table.column("episode_index").to_pylist()
            if wanted is not None and not (wanted & set(episodes)):
                continue
            rows = table.to_pylist()
            for row in rows:
                ep = row["episode_index"]
                if wanted is not None and ep not in wanted:
                    continue
                frame = np.hstack([decode(row[k]) for k in keys])
                if not args.no_overlay:
                    action = row.get("action") or []
                    grip = f" grip={action[-1]:+.2f}" if action else ""
                    frame = annotate(frame, [f"ep{ep:03d} f{row['frame_index']:04d}{grip}"])
                if ep not in writers:
                    path = out_dir / f"episode_{ep:06d}.mp4"
                    writers[ep] = imageio.get_writer(
                        path, fps=fps, codec="libx264",
                        macro_block_size=1, ffmpeg_log_level="error",
                    )
                writers[ep].append_data(frame)
                counts[ep] = counts.get(ep, 0) + 1
    finally:
        for w in writers.values():
            w.close()

    if not counts:
        raise SystemExit("no frames matched — check --episodes")
    for ep in sorted(counts):
        print(f"episode_{ep:06d}.mp4  {counts[ep]:5d} frames  "
              f"{counts[ep] / fps:6.1f}s  ({' | '.join(keys)})")
    print(f"\n{len(counts)} videos in {out_dir} @ {fps:g} fps")


if __name__ == "__main__":
    main()
