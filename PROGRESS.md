# Project progress — the cross-track story

How this project got from "run a pretrained VLA in MuJoCo" to "an ACT trained from scratch
places the ball 5 times in 6". One section per track, in the order they were attempted,
each ending with what it **proved** and what it **cost**.

This file is deliberately shallow. Every claim here is a summary of a numbered section in a
track's own log, and that log is where the measurement lives:

| track | attempt log |
|---|---|
| MolmoAct2-LIBERO, SmolVLA, the scene, the plant | [`libero/PROGRESS.md`](libero/PROGRESS.md) |
| ACT | [`act/PROGRESS.md`](act/PROGRESS.md) |
| MolmoAct2-DROID (retired) | [`docs/`](docs/README.md) |

Newest at the bottom. Corrections are appended, never edited into earlier sections.

---

## 1. Phases 0-2 — sim, scene, and a remote GPU brain (2026-07-18)

Built the local half first and verified it by looking at it rather than by trusting the
maths: Panda from MuJoCo Menagerie, a pick-and-place scene, two cameras, and renders
inspected by eye ([`docs/PLAN.md`](docs/PLAN.md)). Then stood MolmoAct2-DROID's FastAPI
`/act` server up on a rented vast.ai RTX 5090
([`docs/PHASE2_SETUP.md`](docs/PHASE2_SETUP.md)).

**Proved.** The deployment shape works: local sim, remote GPU, action chunks over HTTP, one
round trip per chunk rather than per step. That architecture survived every later track
unchanged — four different policies have since been served through the same `/act`
contract.

**Cost.** Manual GPU rental was the friction, and it is why everything after phase 3 runs
on Modal instead.

## 2. Phase 3 — the closed loop, and a bug that made every result meaningless (2026-07-18 → 07-27)

`droid/phase3_closed_loop.py` closed the loop; `droid/phase3_modal.py` moved the server to
Modal. The stock DROID checkpoint did not do the task. Neither did a 500-step
action-expert fine-tune (`ae_train`, phase 4).

Then, on 2026-07-27, [`libero/PROGRESS.md` §1](libero/PROGRESS.md): the client called
`mj_step` **once** per action, while demos held each action for `decimation = 33` steps.
Every command got 2 ms of physics instead of 66 ms. Proven by oracle replay — the ground
truth expert setpoints, pushed through the client's own apply-and-step path, left the ball
**untouched from spawn**.

**Proved.** Every evaluation before that date — stock DROID, `lora_train`, `ae_train` — had
a **ceiling of zero**. They measured the bug, not the policies. That is why the entire
`droid/` track is retired rather than compared against.

**Cost.** Roughly nine days of model-blaming. The lesson that stuck: **replay the expert
through the inference path before concluding anything about a policy.** It is cheap and it
is the only thing that separates "the policy is bad" from "the environment discards 97% of
what the policy says".

## 3. Phase 4 — expert demos, and where the labels came from (2026-07-19)

An IK-scripted waypoint expert, recorded in the 8-D DROID convention, written out as a
LeRobot dataset ([`docs/PHASE4_PLAN.md`](docs/PHASE4_PLAN.md)). Fine-tuning ran on Modal.

**Proved.** The pipeline — collect → LeRobot → Modal fine-tune → serve — is real and
reproducible. Every later track reuses it with a different writer and a different trainer.

**Cost.** The dataset had a bug that took another nine days to find. See §5.

## 4. Switching checkpoints: DROID → LIBERO (2026-07-26 onward)

MolmoAct2-**DROID** is pretrained on real-robot footage of an FR3 — flat-shaded MuJoCo
Panda renders are a distribution gap on two axes at once. MolmoAct2-**LIBERO** is pretrained
on simulated Panda scenes, so `libero/` is a deliberate re-port to close that gap
([`libero/README.md`](libero/README.md)).

Building it turned up **four measured errors in our own scene** ([§21](libero/PROGRESS.md)):
the table was 100 mm too low relative to the robot base, `grip_site` was 9.5 mm out and
yawed 90°, the reset pose was not LIBERO's, and the origin offset had the wrong x. A
benchmark diagnostic ([§20](libero/PROGRESS.md)) then scored the checkpoint **3/3 on a real
LIBERO task through robosuite's own OSC**.

**Proved.** The checkpoint and our serving path are both fine. Whatever fails after this
point is our scene, our data or our control — not the model download and not the wire
format. That single measurement is what made the rest of the project debuggable.

## 5. Servo droop — the labels were an error term (2026-07-28)

Full postmortem: [`docs/SERVO_DROOP.md`](docs/SERVO_DROOP.md). The Panda ran on
overdamped position actuators, so it never reached the pose it was told to hold. The
collector labelled every tick `(target − current) / 0.05` — an **error** term — so the
standing gravity sag was written into every training label as a near-constant offset. `dx`
became one-sided: its 1st percentile over 8700 frames was −0.08, meaning the data contained
essentially no signal for retreating in −x, which was exactly the failing behaviour.

The fix was not the stiffer gains that document proposes. It was
[§22](libero/PROGRESS.md): **a native port of robosuite's `OSC_POSE`** onto torque
actuators (`libero/osc_controller.py`, `scene_libero_osc.xml`). Measured sag **0.000 mm**
against 4.84 mm. There is no joint setpoint to lag behind, so the bug cannot exist.

**Proved.** Any label defined as `target − current` inherits the plant's tracking error.
The collector's rule since: **labels come from the controller that consumes them.**

**Cost.** Datasets `a1`-`a4` were all collected through the old plant and had to be thrown
away. `docs/SERVO_DROOP.md` is superseded *in its premise* one day after being written —
it is kept for its method, not its conclusion.

## 6. SmolVLA — right diagnosis, wrong bottleneck (2026-07-31)

A LoRA fine-tune of SmolVLA-450M on `a5` **completed the task** — approach, grasp,
transport, place — in ~54 action chunks, which is slow. Diagnosed correctly
([§23](libero/PROGRESS.md)): `a5`'s episodes are 539 ticks each because
`OSC_SPEED_SCALE = 2.5` had been used to stop action labels clipping, which is the wrong
knob (the ceiling is set by `DELTA_POS_SCALE`) and did not even work — `a5` still clips
`dx` on 3.07% of frames. `a6` re-collected at scale 0.20 with distance-retimed segments:
161 ticks/episode, 0.00% saturation, `dx q01 = -0.681` against released LIBERO's -0.679.

**Proved.** The action-space geometry can be diagnosed from the dataset alone and fixed
without touching the policy. Also, the smaller model was the first thing to complete the
task at all.

**Cost.** See §7 — the fix worked and bought nothing.

## 7. Both MolmoAct2 fine-tunes measured properly — it is the gripper (2026-08-02)

10 rollouts each, identical seeds, randomised ball and bins, each served at its own
collection scale ([§25](libero/PROGRESS.md)):

| | placed | grasp-and-lift | gripper ever closed |
|---|---|---|---|
| `a5` (scale 0.05, 539-tick expert) | 2/10 | 3/10 | 5/10 |
| `a7` (scale 0.10, 334-tick expert) | 1/10 | 2/10 | 5/10 |
| stock checkpoint (n=3) | 0/3 | 1/3 | — |

**Every lift and every placement across all 20 rollouts came from a run where the gripper
fired at all**, and closure is close to uncorrelated with whether the hand is on the ball —
one run closed 0.7 mm from the ball but 58 chunks too late, another did a full
close-transport-release on an empty hand 39 mm away.

The speed fix from §6 is real and delivered exactly the ~3× it predicted, in chunks-to-first-
close. It converted into no additional placements.

**Proved.** Three datasets and two fine-tunes went into the action space's geometry while
the binding constraint was a **binary channel neither dataset teaches**: the expert only
ever closes on a stationary, perfectly centred ball, and rejection sampling on
`lifted and placed` deletes every episode where contact went wrong. The states the policy
occupies at decision time are absent from the training data *by construction*.

**Also unresolved:** five of the 20 rollouts truncated early with stderr discarded, because
the client exits rather than retries on a failed request. Every rate above carries that
noise.

## 8. ACT — the smallest model wins (2026-08-01)

`act/` exists to separate two explanations for SmolVLA missing the grasp — units versus
grounding — by removing the frozen vision tower entirely: ACT trained **from scratch** on
`a7`, ResNet18 in the gradient path from step 0, 51.6 M learnable params, on an L4
([`act/PROGRESS.md` §1](act/PROGRESS.md)).

**ck10000: 5/6 placements, 6/6 grasp-and-lift**, against a baseline of 0/3 and 1/3. The
gripper closure that gated every MolmoAct2 run is simply not the failure mode here.

The remaining failure is its mirror image ([§7.4](act/PROGRESS.md)): on 3 of 9 rollouts the
gripper **never opens** — it carries the ball to the bin and holds it — and at ck10000 that
was a clean function of `dx`: every `dx ≤ −0.048` held, everything above released, no
exceptions. An inward carry does not release.

At ck30000 the release rate improved 4/7 → 5/7 but the **structure vanished**
([§7.5](act/PROGRESS.md)) — the `dx` boundary went jagged, while motion quality and grasp
precision both *improved*. That is fitting episodes rather than the rule.

**Proved.**
- Grounding, not units. A model with an adaptable vision path learns this task from 60
  episodes; frozen-tower LoRAs on the same data do not.
- **The last checkpoint is not the best one** — now shown twice on two architectures (here,
  and SmolVLA's ck3000 beating ck5000 at [§23.5](libero/PROGRESS.md)). Score intermediate
  checkpoints.
- The decision boundary getting *noisier* while the motion gets *tighter* is a real,
  separable signal. Measure them separately, and compare only within the same outcome class
  — a held run hovers rather than retreating, so path length across outcomes is confounded.

**State.** Training paused at 30 k of 60 k, resumable. **ck20000 is unscored and is the
interesting one**: if 4/7 → ? → 5/7 passes through a peak, the right stopping point is
earlier than either endpoint.

---

## What to do next

In the order the evidence argues for:

1. **Score ACT ck20000** on the same seven `dx`-chosen seeds. Cheapest informative
   measurement available, and it decides whether resuming to 60 k is worth anything.
2. **Fix the release failure**, not by training longer — §7.5 shows longer made it less
   predictable. A third bin slot at low x would separate "inward carry" from "far ball"
   ([`act/PROGRESS.md`](act/PROGRESS.md), open items).
3. **Teach closure explicitly** if the MolmoAct2 path is revived: stop rejection-sampling
   away the failed-contact episodes, since that is what deletes the decision states.
4. **Make the client retry.** Five truncated rollouts is noise on every rate in §7, and one
   transient server error currently ends a run silently.
