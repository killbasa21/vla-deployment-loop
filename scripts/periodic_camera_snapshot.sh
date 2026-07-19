#!/usr/bin/env bash
# Periodically copies the latest external ("standing") and wrist camera frames
# from a running phase3_closed_loop.py session into trial/, timestamped, so
# there's an easy-to-browse sample of the feed without digging through the
# full sequentially-numbered frame dump.
#
# Usage: scripts/periodic_camera_snapshot.sh <frames-dir> [interval-seconds]

set -euo pipefail

FRAMES_DIR="${1:?usage: periodic_camera_snapshot.sh <frames-dir> [interval-seconds]}"
INTERVAL="${2:-10}"
OUT_DIR="trial"

mkdir -p "$OUT_DIR"

while true; do
    latest_ext=$(ls "$FRAMES_DIR"/frame_*.png 2>/dev/null | sort | tail -1 || true)
    latest_wrist=$(ls "$FRAMES_DIR"/wrist/frame_*.png 2>/dev/null | sort | tail -1 || true)

    if [ -n "$latest_ext" ] && [ -n "$latest_wrist" ]; then
        ts=$(date +%Y%m%d_%H%M%S)
        cp "$latest_ext" "$OUT_DIR/standing_cam_${ts}.png"
        cp "$latest_wrist" "$OUT_DIR/wrist_cam_${ts}.png"
        echo "$(date '+%H:%M:%S') snapshot -> standing_cam_${ts}.png / wrist_cam_${ts}.png (source: $latest_ext)"
    fi

    sleep "$INTERVAL"
done
