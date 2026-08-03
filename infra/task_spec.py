"""The task, defined once.

Every track -- `libero/`, `act/`, `smolvla_libero/`, and the retired `droid/` -- sends a
language instruction with each observation, and the string has to be byte-identical at
collection time and at serving time. It was previously copied into five files. That is a
silent failure: a policy fine-tuned on one wording and served another does not error, it
just conditions on a prompt it never saw, which is the same shape of bug as the
`observation.images.image2` rename (`smolvla_libero/README.md`) and the hardcoded
`NORM_TAG` (`libero/PROGRESS.md` sec.5).

Kept import-free on purpose. This module is shipped into every Modal container by
`with_infra()`, including the serving images that deliberately have no mujoco and no numpy
at import time, so it must stay pure data.

2026-08-03: the pick target changed from a 40 mm sphere to a 40 mm cube and the instruction
changed with it. See `libero/PROGRESS.md` sec.26. Datasets a5/a6/a7 and every score derived
from them predate the change and describe a different task.
"""

# The one string. Present tense, no article before "green box", matching the phrasing of
# LIBERO's own task descriptions ("pick up the alphabet soup and place it in the basket").
INSTRUCTION = "pick up the green box and put it in the green container"

# The pick target's body and geom names in the scene XMLs. Here rather than in
# `libero_closed_loop.py` because `droid/phase4_collect_demos.py` drives a different scene
# file and would otherwise keep its own copy.
TARGET_BODY = "green_box"
TARGET_GEOM = "green_box_geom"

# Half-extent of the cube in metres, i.e. a 40 mm box. This is the value in the scene XMLs;
# the XML stays the source of truth and callers read the COMPILED geom, per CLAUDE.md. It is
# duplicated here only as the fallback for tooling that has no model to compile (score_runs.py
# reading an old log, for one).
TARGET_HALF_EXTENT = 0.02
