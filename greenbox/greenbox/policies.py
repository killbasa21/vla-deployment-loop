"""Action sources. One interface, so the expert and a served checkpoint are
driven, watched and scored through exactly the same code path.
"""

from __future__ import annotations

import base64
import io

import numpy as np

from greenbox import task_spec as spec
from greenbox.expert import ExpertConfig, ScriptedExpert


class ExpertSource:
    name = "expert"

    def __init__(self, seed: int = 0, action_noise: float = 0.0,
                 waypoint_noise: float = 0.0):
        self.expert = ScriptedExpert(
            ExpertConfig(
                action_noise=action_noise,
                waypoint_noise=waypoint_noise,
                rng=np.random.default_rng(seed),
            )
        )

    def reset(self, env):
        self.expert.reset(env)

    def act(self, env, obs):
        return self.expert.act(env)

    def waypoint(self, env):
        return self.expert.waypoint(env)[0]

    @property
    def status(self) -> str:
        return f"phase={self.expert.phase}"

    @property
    def finished(self) -> bool:
        return self.expert.finished


class ServerSource:
    """Talks to the policy server; consumes one action chunk at a time.

    `chunk_reuse` is how many actions of a returned chunk are executed before
    asking again. 0 means the whole chunk. Smaller values re-observe more often
    (better closed-loop reactivity, more requests).
    """

    name = "server"

    def __init__(self, url: str, chunk_reuse: int = 0, timeout: float = 120.0):
        import requests

        self.session = requests.Session()
        self.url = url.rstrip("/")
        self.chunk_reuse = chunk_reuse
        self.timeout = timeout
        self.queue: list[np.ndarray] = []
        self.last_chunk_len = 0
        self.n_calls = 0

    def reset(self, env):
        self.queue.clear()
        self.n_calls = 0
        try:
            self.session.post(f"{self.url}/reset", timeout=self.timeout)
        except Exception:
            pass  # server need not implement /reset

    def act(self, env, obs):
        if not self.queue:
            payload = {
                "instruction": spec.INSTRUCTION,
                "state": env.policy_state().tolist(),
                "images": {
                    key: encode_png(obs[f"{cam}_image"][::-1])
                    for cam, key in spec.CAMERAS.items()
                },
            }
            r = self.session.post(f"{self.url}/act", json=payload, timeout=self.timeout)
            r.raise_for_status()
            chunk = np.asarray(r.json()["actions"], dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk[None]
            self.last_chunk_len = len(chunk)
            self.n_calls += 1
            keep = self.chunk_reuse or len(chunk)
            self.queue = list(chunk[:keep])
        return self.queue.pop(0)

    def waypoint(self, env):
        return None

    @property
    def status(self) -> str:
        return f"queued={len(self.queue)}/{self.last_chunk_len} calls={self.n_calls}"

    @property
    def finished(self) -> bool:
        return False


class RandomSource:
    """Uniform random actions -- the floor any real policy must clear."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, env, obs):
        return self.rng.uniform(-1.0, 1.0, size=spec.ACTION_DIM).astype(np.float32)

    def waypoint(self, env):
        return None

    @property
    def status(self) -> str:
        return ""

    @property
    def finished(self) -> bool:
        return False


def encode_png(img: np.ndarray) -> str:
    import imageio.v3 as iio

    buf = io.BytesIO()
    iio.imwrite(buf, img, extension=".png")
    return base64.b64encode(buf.getvalue()).decode()


def make_source(args):
    if args.policy == "expert":
        return ExpertSource(args.seed, args.action_noise, args.waypoint_noise)
    if args.policy == "random":
        return RandomSource(args.seed)
    return ServerSource(args.server_url, args.chunk_reuse)
