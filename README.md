# vla-deployment-loop

**One robot task. Twelve policies. The one that works is the smallest.**

The task: *pick up the green box and put it in the green container.* The box and the
container move to random spots every episode. A run only counts if the box ends up in the
container — getting close doesn't count.

> **The target used to be a ball.** For most of this project it was a 40 mm green sphere and
> the instruction said *"green ball"*. It was changed to a 40 mm cube, and the instruction
> changed with it. A sphere has no orientation, so a policy could grab it any way it liked; a
> cube punishes a bad wrist angle. Everything measured before the switch describes the ball,
> and this page says which is which. Ball and box numbers are not comparable.

---

## 1. The pieces

| piece | what it does here |
|---|---|
| **MuJoCo** | the physics simulator, running locally on a laptop |
| **MuJoCo Menagerie** | where the Franka Panda arm model comes from |
| **robosuite** | used in the second, rebuilt version of the task — it ships the LIBERO-style controller and scene conventions the pretrained models expect |
| **LeRobot** | the dataset format the demos are written in, and the trainer for two of the four policies |
| **Modal** | rents the GPU. Both training and serving run there |
| **FastAPI + HTTP** | the wire between the two halves. One `POST /act` endpoint, JSON in, actions out |

Nothing about the model runs locally. The laptop only simulates and draws.

## 2. The loop

```
 ┌──────────── local (this repo) ─────────────┐        ┌──── Modal GPU ────┐
 │ MuJoCo: Franka Panda + green box + bins    │  HTTP  │ policy server      │
 │  - renders external_cam + wrist_cam        │ ─────► │ POST /act          │
 │  - reads proprioception                    │  JSON  │ in:  images,       │
 │  - applies the returned action chunk       │ ◄───── │      instruction,  │
 │  - steps physics, logs every chunk         │        │ out: (N, 8) chunk  │
 └────────────────────────────────────────────┘        └────────────────────┘
```

Each request sends two camera images, the arm's current state, and the instruction string.
The policy replies with a **chunk** — N future actions, not one. The client plays the chunk
out, then asks again. That's one network round trip per chunk instead of per step, which is
the only reason a remote GPU is usable at 20 Hz.

Every policy in this repo speaks that same contract. Same client, same scorer, same wire
format. Only the server changes.

## 3. What's in the repo

| path | what |
|---|---|
| `libero/` | the shared infrastructure — the scene, the controller, the closed-loop client, the demo collector and the scorer that every other track uses |
| `act/` | ACT trained from scratch. The best result |
| `smolvla_libero/` | SmolVLA-450M serving and LoRA training |
| `droid/` | the first attempt. Retired |
| `greenbox/` | the task rebuilt from scratch in robosuite, with cleaner metrics |
| `scenes/` | the MuJoCo scene XMLs. `mujoco_menagerie/` is a pinned submodule and stays pristine |
| `infra/` | Modal images, and the task instruction, each defined exactly once |
| `docs/` | old plans and postmortems, each labelled current or superseded |

Two things live in exactly one file on purpose: every Modal image (`infra/modal_images.py`)
and the instruction string (`infra/task_spec.py`). The instruction used to be copy-pasted into
five files. Training on one wording and serving another doesn't throw an error — it just
quietly asks the policy a question it never studied.

```bash
git clone --recursive <this repo>    # the scenes need the mujoco_menagerie submodule
uv sync
```

Install, run, collect and train commands are in [`docs/SETUP.md`](docs/SETUP.md). Everything
runs from the repo root.

## 4. How the training data was made

There is no human teleoperation here. A **scripted expert** does the task: move above the box,
descend, close, lift, carry, release. It's driven by the same controller the policy will later
drive, which matters more than it sounds.

Each episode randomises the box position and the bin layout. Episodes are **rejection
sampled** — only successful ones are kept. Everything is written out as a LeRobot dataset:
one parquet of all episodes, images inlined as PNGs.

Two bugs in this stage cost more time than any model did.

**The labels were measuring error, not intent.** The arm used position actuators that were too
soft, so it never quite reached the pose it was told to hold — it sagged under gravity and
stayed there. The collector wrote every label as `(target − current) / scale`, which is the
*tracking error*. The sag got baked into every label as a constant offset. Over 8700 frames,
the 1st percentile of the sideways `dx` channel was **−0.08** — the data had almost no
examples of pulling back, which was exactly the move that kept failing. Fixed by replacing the
position actuators with an operational-space controller on torque actuators. Sag went from
4.84 mm to **0.000 mm**. Datasets `a1`–`a4` were thrown out.

> **Lesson:** labels must come from the controller that consumes them. Any label written as
> `target − current` inherits the robot's tracking error as training signal.

**A rotation took the long way round.** In the rebuilt task the scripted expert — the thing
that generates *all* the data — scored 0/10. Cause: `q` and `−q` describe the same rotation,
but only one unwraps the short way. A wrist error of −0.77 rad came out as **5.51 rad**, which
maxed out the rotation channels; because the controller mixes rotation into position, the arm
was pulled *up* instead of down onto the box. Flipping the sign when `w < 0` took the expert
from **0/10 to 10/10**.

The datasets that survived:

| dataset | episodes | ticks/ep | action scale | target |
|---|---|---|---|---|
| `a5` | 30 | 539 | 0.05 | ball |
| `a6` | 30 | 161 | 0.20 | ball |
| `a7` | 60 | ~334 | 0.10 | ball |
| `b1` | 40 | ~410 | 0.05 | **box** |
| `greenbox` | 300 | — | 0.05 | box, rebuilt scene |

## 5. Picking a model, and what each one taught

### 5.1 MolmoAct2-DROID — the training data was the wrong world

The first choice, and the wrong one. **DROID is pretrained on video of real robots** — real
lighting, real cameras, an FR3 arm. Our input is a flat-shaded MuJoCo render of a Panda. That
is two gaps at once: simulated-vs-real, and one arm vs another. It never placed the box, and
neither did a 500-step fine-tune on top of it.

There was also a bug underneath, which is the more useful story. **The simulator was ignoring
97% of every command.** The client stepped physics *once* per action; the demos held each
action for **33** steps. Every command got 2 ms of simulated time instead of 66 ms.

It was found by feeding the expert's own perfect actions back through the client's code path.
The box never moved from where it spawned. A perfect policy scored zero on that harness — so
all three DROID fine-tunes had been graded against a ceiling of zero. Nine days went into
arguing about which model was worse.

> **Lesson:** run the expert through the inference path before blaming a policy. It is nearly
> free, and it is the only thing that tells "bad policy" apart from "the environment isn't
> listening."

### 5.2 MolmoAct2-LIBERO — right world, but the model is too big to afford

LIBERO is a simulated-Panda benchmark, so a checkpoint pretrained on it starts in roughly our
world. That was the right correction. The problem was size: **5.57 B parameters**, needing a
24 GB GPU just to serve.

The switch forced a line-by-line comparison against the reference environment, which turned up
four errors in a scene that had been rendering plausibly for weeks: the table was 100 mm too
low, the grip point was 9.5 mm off **and rotated 90°**, the reset pose was wrong, and the world
origin had the wrong x. The same exercise produced the most useful measurement in the project
— the stock checkpoint scored **3/3 on a real benchmark task** through the reference
simulator. That settled the argument: the download was fine and the wire format was fine, so
everything failing after that was our scene, our data, or our control.

> **Lesson:** check robot parameters by compiling the model and reading real values, not by
> reading the XML.

But the money ran out before the training did. On a $5 budget, a LoRA run bought **150 steps =
0.06 epochs** of the dataset. Saving a single checkpoint cost 150 s of GPU time, about 4% of
the whole budget. At 0.06 epochs the model has barely moved off its base weights — the risk
isn't overfitting, it's that no training happened at all. Later, properly funded fine-tunes on
`a5` and `a7` scored 2/10 and 1/10, which is indistinguishable from the untrained baseline.

> **Lesson:** a large model doesn't just cost more per step, it converts a fixed budget into
> fewer steps. Check what your budget buys in *epochs* before picking the model.

### 5.3 SmolVLA-450M — small enough to actually train, but it needs data

**12× smaller**, serves on the cheapest GPU available. The same $5 becomes a real training run
instead of a rounding error. That's the whole argument for it, and it held up.

The results split cleanly by how much data it got:

- **30–60 demos:** poor. Its first fine-tune did finish the task, but slowly — about 54 action
  chunks — because its demos were slow (539 ticks each). Copying a slow teacher gives you a
  slow student. Later fine-tunes on 60 demos scored 0/13.
- **300 demos, in a scene built to match its pretraining conventions:** **36%**, and the curve
  was still climbing when training stopped.

> **Lesson:** the small model wasn't weak — it was starved. What moved it was more demos, and
> demos that looked like what it was pretrained on.

One clear negative result: a variant that **unfroze the vision encoder** did worse, and
steadily worse. Across 5 checkpoints × 10 seeds, the best score came from the *earliest*
checkpoint (1/10 at step 488) and every later one scored zero. Unfreezing a pretrained vision
tower and LoRA-ing it on 40 episodes damages features that were already good.

### 5.4 ACT — no pretraining at all

ACT was added to settle a question: was SmolVLA missing the grasp because of *units*, or
because it couldn't *see* well enough? ACT answers it directly by having no frozen vision at
all — a ResNet18 in the gradient path from step 0, **51.6 M** trainable parameters, no
pretraining to preserve.

Trained on the same 60 demos the SmolVLA LoRA failed on, it placed the ball 5 times out of 6
and picked it up **12 out of 12**. So it was seeing, not units. A model that can adapt its
vision learns this task from 60 episodes; a frozen-tower LoRA on the same data does not.

## 6. How it was fine-tuned

| policy | trainable | hardware | steps | notes |
|---|---|---|---|---|
| MolmoAct2-LIBERO | LoRA r32 | L4 24 GB | 150 → later runs longer | $5 bought 0.06 epochs |
| SmolVLA (`a5`/`a7`) | action expert only | T4 16 GB | 5000 | vision + language frozen |
| SmolVLA (`b1`) | action expert **+ vision** | L4 | 2438 | the unfreeze experiment |
| SmolVLA (rebuilt) | 99.9 M of 450 M (22.2%) | L4 | 12 000 | batch 16, lr 1e-4 cosine, bf16, ~96 min |
| ACT | 51.6 M, all of it | L4 | paused at 30 k of 60 k | batch 16, saves every 10 k |

Two practical notes. ACT's training was **dataloader-bound, not GPU-bound** — the GPU spent
more time waiting for PNG decodes (0.110 s) than computing (0.099 s), and doubling the worker
count only moved the bottleneck onto CPU cores, which are billed separately. And every
training run starts with a **1-step smoke run** that proves build → load → step → save. More
than one step proves nothing extra and costs real money.

## 7. Results

`placed` means the task succeeded. `lift` means the box left the table in the gripper — worth
tracking separately, because for most of these it's as far as they ever got.

| # | policy | data | target | placed | lift |
|---|---|---|---|---|---|
| 1 | ~~MolmoAct2-DROID, stock~~ | — | ball | 0 | 0 |
| 2 | ~~MolmoAct2-DROID, `ae_train`~~ | `a1`–`a4` | ball | 0 | 0 |
| 3 | ~~MolmoAct2-DROID, `lora_train`~~ | `a1`–`a4` | ball | 0 | 0 |
| 4 | MolmoAct2-LIBERO, stock (baseline) | — | ball | 0/3 | 1/3 |
| 5 | MolmoAct2-LIBERO, fine-tuned | `a5` | ball | 2/10 | 3/10 |
| 6 | MolmoAct2-LIBERO, fine-tuned | `a7` | ball | 1/10 | 2/10 |
| 7 | SmolVLA-450M, LoRA | `a5` | ball | completes † | — |
| 8 | SmolVLA-450M, LoRA | `a7` | ball | 0/13 | 2/13 |
| 9 | SmolVLA-450M, LoRA + vision, 5 ck | `b1` | **box** | 1/52 | 2/52 |
| 10 | **ACT, from scratch** | `a7`, 60 eps | ball | **5/6** | **6/6** |
| 11 | ACT, trained 3× longer | `a7` | ball | 6/8 | 8/8 |
| 12 | **SmolVLA-450M, rebuilt task** | 300 demos | box | **36%** | 84% grasped |

Rows 1–3 measured the decimation bug, not a policy. Row 7 † finished the task but has no
scored rollouts — only the observation that it worked, slowly. Row 10's 5/6 is the six
evaluation seeds; across all twelve logged rollouts, including six deliberately awkward box
positions, it's 7/12 placed and 12/12 lifted. Row 12 is a different scene with harder
language — three trays whose colours shuffle every episode, so "the green one" can't be
memorised as a position — scored over 25 episodes against a scripted expert at 100% and random
actions at 0%.

Three findings came out of comparing them.

**The gripper was a channel nobody taught.** Across 20 MolmoAct2 rollouts, every single lift
and every single placement came from a run where the gripper closed *at all* — and it closed
in exactly half of them. Worse, closing was nearly uncorrelated with the hand being near the
box: one run closed 0.7 mm away but 58 chunks too late, another did a full close-carry-release
on an empty hand 39 mm away. The cause is in the data: the expert only ever closes on a
perfectly centred, stationary box, and rejection sampling deletes every episode where contact
went wrong. The states the policy is actually in when it has to decide are absent from
training **by construction**.

**In the rebuilt task, the bottleneck was the wrist, not the language.** Colour grounding was
never the problem — across every checkpoint, the policy put the box in a wrong-coloured tray
**0 times out of 25**. But it grasped 84% of the time and only lifted 36%. The measurements
say why: at the moment it closes, the gripper is 0.045 m from the box (close enough) but
**0.394 rad off in wrist angle**, against 0.048 rad it had already achieved earlier in the
same episode. It reaches the right place with the right angle, then rotates away before
closing. This is where the ball-to-box change bites — a sphere would have forgiven it.

**The last checkpoint is not the best one.** Shown three times, on three different models. ACT
at 10 k had a clean rule for when it let go — every `dx ≤ −0.048` held on, everything above
released, no exceptions. At 30 k it placed slightly more often but that rule went ragged, even
as its motion got smoother and its grasps got tighter. SmolVLA's 3 k checkpoint beat its 5 k
one. The unfreeze sweep peaked at its very first checkpoint. **Score intermediate
checkpoints.**

One more, found by accident: **the scorer itself was wrong.** A run reported `placed 32%` and
`released 8%`, which is impossible — you have to let go before it counts as placed. `released`
was being recorded the *first* time the gripper lost contact, and contact flickers during
transport. Changed to the last let-go, the numbers lined up.

> **Lesson:** build the metric as a chain where each stage requires the one before it, then
> look for totals that can't happen.

## 8. What's next

**Fine-tune the VLM properly.** The unfreeze experiment failed, but it failed on 40 episodes.
The question of whether the language model can be adapted rather than damaged is still open,
and 300+ demos is the setting to ask it in.

**A small patch-based policy for lightweight tasks.** Most of what this task needs is local:
where the box is relative to the gripper. A policy operating on image patches instead of a
full vision-language stack should be far cheaper for that class of problem.

**ACT inside a mixture of experts, for many tasks.** ACT's output is a densely encoded action
chunk — a lot of behaviour packed into a small model. That makes it a good candidate as one
expert among several, routed per task, instead of retraining a whole model per skill.

**An RL environment on top of SmolVLA.** Behaviour cloning can only copy the demonstrations,
which is why the slow expert produced a slow policy and why the untrained states around a
failed grasp were never learned. Fine-tuning with reinforcement learning in the same simulator
would let it improve past what the expert showed it.

## Reading further

The detail behind every number on this page:

1. [`docs/SETUP.md`](docs/SETUP.md) — install, run, collect, train
2. [`PROGRESS.md`](PROGRESS.md) — the cross-track story, one section per track
3. The track you care about: [`act/`](act/README.md), [`libero/`](libero/README.md),
   [`smolvla_libero/`](smolvla_libero/README.md), [`greenbox/`](greenbox/README.md),
   [`droid/`](droid/README.md)
4. That track's `PROGRESS.md` — the chronological attempt log, **wrong turns kept in**
5. [`docs/README.md`](docs/README.md) — old plans and postmortems, each labelled current or
   superseded

**When two documents disagree, newest wins**, in this order: `libero/PROGRESS.md` and
`act/PROGRESS.md` (the measurements) → the subproject `README.md` → this file → `docs/`.
Numbers written before the decimation fix and the controller port are evidence of what was
believed at the time, not current fact. Re-measure anything load-bearing.

A note on the progress logs: they keep the wrong turns in on purpose. Corrections are appended
as new sections rather than edited into old ones, because knowing *which* conclusions were
reversed is most of the value.

## License

MIT — see [LICENSE](LICENSE).
