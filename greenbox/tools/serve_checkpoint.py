"""Point the deployed server at a checkpoint and confirm it actually cut over.

A `modal deploy` that returns quickly has not necessarily swapped the weights, so
this writes the selection, calls /reload, and prints what /health reports back --
never trust the request, trust the reply.

    uv run python tools/serve_checkpoint.py --checkpoint lerobot/smolvla_base
    uv run python tools/serve_checkpoint.py \
        --checkpoint /vol/checkpoints/ft1/step_010000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import requests

# Your own deployment's URL. Modal derives it from your workspace name, so there is no
# shareable default -- `modal deploy infra/modal_app.py` prints it, and GREENBOX_SERVER_URL
# saves passing --url every time.
DEFAULT_URL = os.environ.get("GREENBOX_SERVER_URL")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--url",
        default=DEFAULT_URL,
        required=DEFAULT_URL is None,
        help="policy server URL; defaults to $GREENBOX_SERVER_URL",
    )
    p.add_argument("--volume", default="greenbox-vol")
    p.add_argument("--stats-path", default=None)
    args = p.parse_args()

    sel = {"checkpoint": args.checkpoint}
    if args.stats_path:
        sel["stats_path"] = args.stats_path

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(sel, fh)
        tmp = fh.name
    subprocess.run(
        ["modal", "volume", "put", args.volume, tmp, "/serve.json", "--force"],
        check=True,
    )

    r = requests.post(f"{args.url}/reload", timeout=900)
    r.raise_for_status()
    health = r.json()
    print(json.dumps(health, indent=2))

    if health.get("checkpoint") != args.checkpoint:
        print(f"\nSERVER DID NOT CUT OVER: asked for {args.checkpoint!r}, "
              f"serving {health.get('checkpoint')!r}")
        sys.exit(1)
    print(f"\nserving {args.checkpoint}")


if __name__ == "__main__":
    main()
