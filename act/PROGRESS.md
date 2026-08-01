# ACT on the green-ball task — attempt log

Companion to `libero/PROGRESS.md`. One numbered section per thing learned, each carrying the
measurement it came from. Corrections to `act/README.md` go here rather than being edited
silently into the plan.

Sections 1-4 are the state of the world this path starts from. Section 5 onward is measured.
Run 1 on `a7` **paused at 30 k of 60 k** (2026-08-01), resumable. ck10000 releases on 4/7
matched seeds, ck30000 on 5/7 — but ck30000's failures no longer follow the `dx` structure
that ck10000's did (§7.4, §7.5). ck20000 unscored.

---

## 1. Why this directory exists (2026-08-01)

`libero/PROGRESS.md` §23.5 ended with two unseparated explanations for the SmolVLA `a6`
fine-tune closing the gripper on air:

| explanation | prediction if true |
|---|---|
| **units** — 0.20 m per action unit, so 0.9 normalised units of error = 36 mm | `a7` at scale 0.10 halves the miss to ~14-18 mm, inside the ball's 20 mm radius |
| **grounding** — 30 episodes too few, and the LoRA touches only the action expert | a bigger/adapted vision path grasps where the LoRA misses, at the same scale |

ACT tests the second one directly: no pretraining to preserve, no frozen tower, the ResNet18
is in the gradient path from step 0. Run both policies on `a7` at `--delta-pos-scale 0.10`
and the split is informative either way.

The measured baseline to beat, all from §23.5, SmolVLA `a6` LoRA at scale 0.20:

| run | ball | first close | lateral err | vertical |
|---|---|---|---|---|
| ck5000, randomised | (0.569, 0.104) | chunk 10 | 88 mm | on the table |
| ck5000, nominal | (0.560, 0.000) | chunk 3 | 36 mm | on the table |
| ck3000, nominal | (0.560, 0.000) | chunk 4 | 11 mm | **40 mm high** |

And the earlier MolmoAct2-LIBERO stock number (`README.md` §9): **0/3 placements, 1/3
grasp-and-lift**.

## 2. What ACT inherits for free, and the one contract that changes

Inherited unchanged: the demos (`libero/fine_tune/a7`), the sim client
(`libero_closed_loop.py --payload-keys libero`), the scorer (`libero/score_runs.py`), the
`/act` wire format, and the OSC control path (`libero/PROGRESS.md` §22).

**Changed: the normalisation decision inverts.** For SmolVLA the rule is *keep the
checkpoint's statistics* — the action expert was pretrained under them, and rebuilding from
30 single-task episodes hands it an affine map it never learned (`PROGRESS.md` §5 is the
failure: "garbage actions of the correct shape"). ACT has no pretrained statistics at all,
so the dataset's own `stats.json` is the only correct source.

This makes `pin_released_stats.py` an active hazard on this path rather than a fix. It
overwrites `a*/meta/stats.json` with MolmoAct2's released LIBERO numbers (`count = 273465`
against the dataset's own). `act_modal_train.py` therefore asserts
`stats.json` count == `info.json` total_frames before spending GPU time, and says to restore
`stats_measured.json` if it does not.

**Also changed: no dataset conversion.** `smolvla_libero/convert_dataset.py` exists only
because `smolvla_libero`'s config hardcodes `observation.images.image2` for the wrist camera
and LeRobot resolves feature keys by name inside the normaliser — the wrong name means no
wrist camera and no error. ACT is built from the dataset's metadata and takes any
`observation.image*` key, so `a7` uploads verbatim and the wrist camera keeps
`observation.images.wrist_image`. The identity `WIRE_TO_FEATURE` in `act_modal.py` is still
checked against `cfg.input_features` at container start; the guard is cheap and the failure
it catches is silent.

## 3. Open questions this path is expected to answer

- **Does ACT grasp at `--delta-pos-scale 0.10`?** If yes and SmolVLA does not, the SmolVLA
  gap is adaptation, not data.
- **Does ACT take the proprioception shortcut?** Signature: accurate on the nominal ball,
  wide on a randomised one, with a near-identical trajectory across seeds. `--no-state` is
  the ablation; it is only worth running if that signature appears (`README.md` §3.2).
- **Which checkpoint is best?** Not assumed to be the last — §23.5 measured ck3000 at 3x the
  lateral accuracy of ck5000 on the same run.

## 4. Traps carried over, worth not re-learning

- **A `modal deploy` is not a cutover.** The previous container answers from inside its 300 s
  `scaledown_window`; §23.5 discarded a run that looked like its *best* result because
  `/health` was still reporting the old checkpoint. Poll `/health` for the step you meant.
- **Smokes are 1 step.** `--max-steps 1 --save-freq 1` proves build → load → step → save.
  Anything beyond that is money for no extra signal (CLAUDE.md).
- **Serving flags must match collection flags.** `a7` is `--delta-pos-scale 0.10
  --bin-layout shuffle`; the client must run `--delta-pos-scale 0.10 --randomize-bins` or the
  rollout is not measuring `a7`.

---

## 5. `a7` was OOM-killed inside `finalize()` — and was repairable (2026-08-01)

The collection "failed", but not where that word suggests. All 60 episodes were simulated and
all six data parquets were written. What is missing is the tail of `finalize()`.

`lerobot_v30_writer.finalize()` writes in this order: data parquets → `meta/episodes/` →
`meta/tasks.parquet` → `meta/info.json` → **`meta/stats.json`** → `meta/cohorts.json`. The
tree left behind has everything up to and including `info.json`. So the process died in the
image-statistics block, and that block has an obvious reason to die:

```python
sample = [f for e in self._episodes for f in e["frames"][cam][::5]]
arr = np.asarray(sample, dtype=np.float32) / 255.0
```

Every fifth frame of the entire dataset, materialised at once, per camera: 4031 frames ×
256×256×3 × 4 bytes ≈ **3.1 GB each**, on top of a collector already holding every raw frame
in RAM (measured 8.2 GB RSS, 51% of this machine). The comment above it says frames are
subsampled "because the exact mean of every pixel of every frame is not worth the memory" —
the subsample was not nearly enough.

**The parquets are complete and authoritative, so the fix is to recompute rather than
re-collect** (~25 min of CPU that would hit the same wall).
`libero/fine_tune/rebuild_stats.py` streams the parquet row-groups with O(1) memory
accumulators and writes an equivalent `stats.json`. Two details in it are load-bearing:

- The subsample rule is **`frame_index % 5 == 0`**, not every fifth row of the file. The
  writer subsampled per episode; those differ whenever an episode length is not a multiple
  of 5, i.e. always.
- Image `count` is the dataset's **total** frames, not the number sampled — the writer's own
  convention, and what released LIBERO's `stats.json` carries.

**`meta/cohorts.json` is not recoverable.** Which cohort (reach / noise / recover) each
episode came from lives only in the collector's in-memory `extra` dict and never reaches the
parquets. It is a sidecar, not part of the LeRobot spec, and nothing in training or serving
reads it — but cohort-split ablations on `a7` are gone.

### 5.1 What `a7` actually is

Recomputed from the repaired dataset, beside its predecessors:

| dataset | scale | ep | frames | ticks/ep | dx std | mm per 1.0 normalised error | dx clip % |
|---|---|---|---|---|---|---|---|
| `a5` (slow, **grasps**) | 0.05 | 30 | 16170 | 539 | 0.309 | 15.5 | 3.07 |
| `a6` (fast, **misses**) | 0.20 | 30 | 4825 | 161 | 0.200 | 40.0 | 0.00 |
| **`a7`** | 0.10 | **60** | **20034** | **334** | **0.235** | **23.5** | **1.35** |

`libero/PROGRESS.md` §23.5 predicted ~256 ticks/ep and ~27 mm for this setting. Both are
close: 334 and 23.5. So `a7` is the middle rung it was meant to be, and it doubles the
episode count, which is the other half of §23.5's open question.

One number worth watching: **dx clips on 1.35% of frames**, versus `a6`'s 0.00%. Halving
`--delta-pos-scale` doubles every action in normalised units, so saturation comes back —
`a7` sits between `a5`'s 3.07% and `a6`'s zero. Not obviously disqualifying (`a5` grasped at
3.07%), but if ACT's failures cluster on fast lateral motion, this is the first place to
look.

## 5.2 First training run — measured throughput, and the dataloader is the wall (2026-08-01)

`--max-steps 60000 --save-freq 10000`, L4, `cpu=16.0`, `num_workers=12`, batch 16.

```
step:1K smpl:16K epch:0.80 loss:1.822 grdn:44.849 lr:1.0e-05 updt_s:0.099 data_s:0.110 smp/s:76
```

`num_learnable_params = 51,574,663` — the README's 51 M arithmetic estimate was right, so
the ~80 M figure from the ACT paper does not describe this configuration.

**GPU compute is 0.099 s/step. `data_s` is 0.110 — the GPU spends more than half its time
waiting for the dataloader.** Two PNG decodes per sample at batch 16 is 32 decodes per step
and twelve workers do not keep up. Measured wall rate 0.216 s/step (800 steps in 173 s), so
60 k steps is ~3.6 h rather than the ~2.5 h predicted from GPU time alone.

A 400-step probe under a separate `--exp-name` (never against the live run's `output_dir`)
at `--num-workers 24`, same `cpu=16.0`:

| workers | updt_s | data_s | s/step (steps 200-400) | 60 k projection |
|---|---|---|---|---|
| 12 | 0.099 | 0.110 | 0.216 | 3.6 h |
| 24 | 0.119 | **0.057** | **0.18** | 3.0 h |

Doubling workers halves the wait and *raises* GPU step time, because the workers and the
training process share one `cpu=16.0` allocation. So worker count is not the binding
constraint — cores are, and cores are billed separately from the GPU on Modal, which is why
this is not obviously worth buying. **`cpu=16.0` is a material share of this run's cost, not
a free knob**; check the actual split on the Modal dashboard before raising it.

Left running at 12 workers: recovering 35 minutes did not justify discarding 8 minutes of
progress plus another container start.

## 5.3 `lerobot-train` refuses a pre-existing `output_dir` (2026-08-01)

```
FileExistsError: Output directory /checkpoints/act/act-green-ball already exists and
resume is False.
```

The 1-step smoke wrote to the same default `exp_name` as the real run, so the real run died
on its own smoke — and died in `cfg.validate()`, i.e. **after** the container was up and
billing (27 s of L4). Two fixes, both in `act_modal_train.py`:

- a preflight that checks `out_dir` before anything else and prints the fix, ~2 s in;
- an explicit `--overwrite` that deletes it, deliberately opt-in because a scored checkpoint
  that has not been copied off the volume is gone.

Give smokes and probes their own `--exp-name`. Also note `modal run act/act_modal_train.py`
with no `::main` fails outright — the module has two local entrypoints.

## 5.4 The success criterion, and a scoring bug found before using it (2026-08-01)

`libero/score_runs.py` is the arbiter, and it uses the same test the demo collector uses to
accept an episode, so "success" means the same thing in training data and in evaluation:

| outcome | test |
|---|---|
| **lifted** | ball centre ever exceeds `TABLE_TOP_Z + 0.02 (radius) + 0.05` — 50 mm of clearance under the ball |
| **placed** | FINAL ball xy within ±50 mm of the green bin (axis-aligned box, despite the name `BIN_RADIUS`) **and** final z below `TABLE_TOP_Z + 0.06` |

Reported as rates over N logs, plus four diagnostics: `best_lateral_mm` (closest the hand
ever came to the ball in the table plane), `ball_max_z_mm`, `gripper_close_pct`, and the
per-channel action std. **The rates are the score; `best_lateral_mm` is what makes a failure
readable** — it is the number that separated ck3000 from ck5000 in §23.5, and both of those
were `lifted=False, placed=False`.

Known limits, worth stating before quoting any number from it: `placed` looks only at the
last logged ball position, so a place-then-knock-out reads as failure and a lucky roll into
the bin reads as success; and nothing checks the ball got there *via* the gripper.

### The bug

`score_runs.green_bin_xy()` re-reads the bin position from the **scene XML**.
`--randomize-bins` shuffles the three bins across `BIN_SLOTS` at build time by mutating
`model.body_pos`, and `build_sim` discarded `randomize_bins`'s return value — so the true
green bin position was **never written to the log**.

Consequence: every randomised-bins run would have been scored against a bin up to 500 mm
from where it actually was, `placed` always False, and no way to notice from the output.
This is the same shape as `PROGRESS.md` §5's `NORM_TAG` and the `image2` rename — correct
types, correct-looking output, wrong reference.

Fixed on both sides before any ACT checkpoint is scored:
- `libero_closed_loop.py` reads `green_bin` off the **built model** (not the XML) and writes
  `green_bin_xy` into every log entry;
- `score_runs.py` prefers that value and prints a loud warning naming any log that lacks it.

All eight pre-existing logs lack the field and are now flagged `(XML)`. They happen to be
scoreable — they were not run with `--randomize-bins` — but nothing in the log proves that,
which is precisely the point.

## 6. `lerobot-train` has no feature-exclusion flag (2026-08-01)

Written down because the plan's `--no-state` ablation (§3, README §3.2) initially assumed one
existed. It does not: neither `DatasetConfig` nor `PreTrainedConfig` carries an
exclude/select field. `input_features` is normally left `None` and **inferred from the
dataset**, which would pick `observation.state` straight back up.

The only lever is to stop the inference by supplying `--policy.input_features` outright — and
because the same code path fills both, `--policy.output_features` with it.
`act_modal_train.py` builds both dicts from `info.json` when `--use-state=false`. **Untested
until that mode gets its own 1-step smoke.** If draccus rejects the nested dict on the
command line, the fallback is a copy of the dataset with the column dropped from
`meta/info.json` and `meta/stats.json` — more work, no ambiguity.

---

## 7. ck10000 works — 5/6 placements — and the one failure is the RELEASE (2026-08-01)

The first checkpoint of run 1: 10 k steps, 8 epochs, one sixth of the planned run. Served on
a T4 through `act_modal.py` (its first ever execution — loaded and answered correctly on the
first try), scored through `libero/score_runs.py`.

| run | ball start | green bin | lift | place | lateral | released |
|---|---|---|---|---|---|---|
| nominal | (0.560, 0.000) | (0.560, 0.250) | ✓ | ✓ | 5.0 mm | ✓ |
| random | (0.570, −0.113) | (0.546, 0.268) | ✓ | ✓ | 7.7 mm | ✓ |
| r2 | — | (0.792, 0.013) | ✓ | ✓ | 2.9 mm | ✓ |
| r3 | — | (0.789, 0.012) | ✓ | ✓ | 7.1 mm | ✓ |
| r4 | — | (0.560, 0.269) | ✓ | ✓ | 2.5 mm | ✓ |
| seed 7 | (0.619, −0.008) | (0.571, 0.239) | ✓ | **✗** | 1.7 mm | **✗** |

**5/6 placements, 6/6 grasp-and-lift**, against a baseline of 0/3 placed, 1/3 lifted.

### 7.1 What this settles

§1 asked whether `a6`'s miss was the units or the grounding. **It was neither — it was
adaptation.** ACT closes to 2.5-7.7 mm and completes the task on the same scene, same
collector, same OSC plant, same `--delta-pos-scale 0.10` where the SmolVLA LoRA missed by
11-88 mm and never moved the ball. The "60 episodes is too few to ground" explanation also
loses: ACT grounds fine on exactly that data.

**The §3.2 proprioception-shortcut worry is dead.** A ball displaced 113 mm from nominal
still gives 7.7 mm lateral, and the far bin slot (0.792, 0.013) works as well as the near
one. The policy is using the cameras.

### 7.2 The failure mode is the release, and the dataset is not at fault

Checked first, because a policy faithfully reproducing a broken label is the cheaper
explanation: **60/60 `a7` episodes end with the gripper commanded open**, and the overall
channel is 60.0% open / 39.7% closed. The demos release correctly.

Seed 7, chunk by chunk: grasps at 8, lifts to 178 mm, arrives **7.1 mm** over the bin at
chunk 12 — then descends and hovers, holding, for 27 more chunks. The gripper command drifts
down at the very end (+1.06 → +0.81 → +0.86) without ever crossing zero. So it is not a
grasp failure or a transport failure; the policy stalls on the release decision boundary.

That is a plausible thing for more training to sharpen, and an argument for letting the run
finish rather than banking ck10000.

### 7.3 `score_runs.py` scored the held ball as a successful placement

Found by **watching the viewer**, not by the scorer. `placed` tested only the ball's final xy
against the bin and `z < TABLE_TOP_Z + 0.06`. With `TABLE_TOP_Z = -0.012` that ceiling is
0.048, and a ball still gripped 42 mm above the table sits at 0.042 — inside it. Seed 7 was
reported `placed=True` with the ball in the hand.

Fixed: `placed` now also requires `released` (last commanded gripper action < 0), `rel` is a
column, and any held run is named in a warning. The test is the gripper command rather than a
tighter height because the heights would have to be fitted to this scene's bin floor (resting
ball 0.012 vs held 0.042, only 30 mm apart) whereas "did the policy let go" is what the word
means in any scene.

Third instance of the same shape in this project — `NORM_TAG`, the `image2` rename, the bin
position — a check that returns a plausible value computed against the wrong reference. Note
the `grip%` column showed it all along: 83.5% for seed 7 against 28-35% for the runs that
released. The signal was already in the output and nobody was reading it.

### 7.4 Hypothesis: the release fails on an INWARD carry — every scene variable checked

Prompted by the suggestion that the container's orientation matters, not just the ball. It
does not, but checking it forced the full enumeration below, and the two-variable version of
this section was too narrow.

**Bin orientation is constant and cannot be the cause.** All three bins in
`scene_libero_hand.xml` / `scene_libero_osc.xml` are identical axis-aligned boxes (a base
plus four walls, same `size`, no `quat` or `euler`), and `randomize_bins` writes **only**
`model.body_pos`. Orientation is therefore byte-identical in every run ever made. Ruled out
by construction, not by measurement.

**Everything the seed actually varies**, recovered by replaying the RNG, against the release
outcome. `dx`/`dy` are green-bin minus ball:

| run | ball (x, y) | green slot | green (x, y) | blue | yellow | dx | dy | released |
|---|---|---|---|---|---|---|---|---|
| r2 | (0.471, −0.054) | FAR | (0.792, 0.013) | (0.544, 0.254) | (0.569, −0.262) | +0.321 | +0.067 | ✓ |
| r3 | (0.492, +0.056) | FAR | (0.789, 0.012) | (0.563, −0.266) | (0.557, 0.249) | +0.297 | −0.044 | ✓ |
| hix49 | (0.659, +0.002) | FAR | (0.804, −0.004) | (0.565, −0.244) | (0.541, 0.261) | +0.145 | −0.006 | ✓ |
| r4 | (0.495, +0.089) | near+y | (0.560, 0.269) | (0.543, −0.246) | (0.795, 0.012) | +0.065 | +0.180 | ✓ |
| nominal | (0.560, 0.000) | near+y | (0.560, 0.250) | XML | XML | 0.000 | +0.250 | ✓ |
| random | (0.570, −0.113) | near+y | (0.546, 0.268) | (0.552, −0.253) | (0.813, −0.004) | −0.024 | +0.381 | ✓ |
| **view** | (0.619, −0.008) | near+y | (0.571, 0.239) | (0.792, 0.015) | (0.540, −0.237) | **−0.048** | +0.247 | **✗** |
| **hix20** | (0.657, +0.047) | near+y | (0.558, 0.235) | (0.561, −0.254) | (0.783, −0.016) | **−0.099** | +0.188 | **✗** |
| **hix53** | (0.660, +0.065) | near+y | (0.553, 0.267) | (0.784, 0.007) | (0.573, −0.242) | **−0.107** | +0.202 | **✗** |

Taking the candidates one at a time:

| variable | separates? | why not |
|---|---|---|
| bin orientation | **n/a** | constant by construction — never randomised |
| green bin slot | **no** | FAR always releases (3/3), but `near+y` both releases (3) and holds (3) |
| ball x | **no** | hix49 has the 2nd-highest ball x (0.659) and releases cleanly |
| ball y | **no** | held y ∈ {−0.008, +0.047, +0.065} sits inside released y ∈ {−0.113 … +0.089} |
| dy | **no** | r4 (+0.180) releases, hix20 (+0.188) holds |
| distractor arrangement | **no** | yellow-at-FAR appears in both r4 (✓) and hix20 (✗) |
| **dx = green_x − ball_x** | **YES** | perfect split at ≈ −0.035; every ✗ is negative, every ✓ is ≥ −0.024 |
| **grip%** (an outcome, not an input) | **YES** | 74–84 held vs 22–35 released, no overlap |

So `dx` survives and nothing else does. It also explains the slot result as a special case:
the FAR slot forces `dx` strongly positive whatever the ball does, while `near+y` gives
`dx ≈ 0.56 − ball_x`, which goes negative exactly when the ball is far out.

**Physical reading: the failures are the runs that carry the ball INWARD, toward the robot.**
Positive `dx` = carry outward or sideways; negative = retract while holding. All three
failures retract; no success does by more than 24 mm.

**What the failure looks like.** Not a positioning error: the failing runs arrive 1.7–2.8 mm
laterally from the bin centre, better than several successes. They grasp, lift to ~190 mm,
arrive over the bin, descend — then hover, holding, for the remaining ~27 chunks. The gripper
command drifts toward zero at the end (+1.06 → +0.81 → +0.86) without crossing. Stalled on a
decision boundary, not lost.

**Hypothesis: under-fitting in a thinly-covered corner.**

- The demos are correct — 60/60 `a7` episodes end with the gripper commanded open (§7.2).
- The corner is *inside* the training distribution: `collect_finetune_data.py:841` samples the
  ball from the same `L.BALL_SAMPLE_X = (0.46, 0.66)` the client evaluates over, and the same
  `BIN_SLOTS`. But it needs ball x ≳ 0.59 **and** green drawn to a near slot — jointly about
  (0.35 × 2/3) ≈ 23% of draws, so roughly 14 of 60 episodes, and only a few of those are
  strongly negative `dx`.
- ck10000 is **8 epochs**. The release is one channel crossing zero — a decision boundary,
  which is what gets learned last and worst where data is thinnest.
- Mechanism for the drift: L1's optimum is the conditional median, which on a bimodal ±1
  target snaps to a mode *once the conditioning is confident*. Under-determined context puts
  the median between modes — exactly the observed +0.8 plateau that never reaches −1.

**Discriminating test, free and already available**: rerun seeds 7, 20 and 53 against
ck20000 / ck30000 / ck60000. Release sharpening with training confirms under-fitting and the
fix is steps. Surviving to 48 epochs makes it data coverage, and the fix is targeted
collection at negative-`dx` configurations.

**Limit worth stating: `dx` and ball x cannot be fully separated in this scene.** `BIN_SLOTS`
offers only two x values (0.56 and 0.80, ±20 mm jitter), so `dx ≤ −0.048` *requires*
ball x ≳ 0.59. hix49 is the one run that pulls them apart — highest-but-one ball x, positive
`dx`, clean release — and it is a single run. A third bin slot at low x would settle it, and
is only worth building if the failure survives more training.

**Also unsampled: green never drew the `near−y` slot** (0.56, −0.25) in any of these eight
seeds. A third of the bin-permutation space is untested.

## 7.5 ck30000: +1 placement, but the failure became UNPREDICTABLE (2026-08-01)

Training paused at 30 k (`modal app stop`; `010000`/`020000`/`030000` all on the volume with
`training_state`, and `010000` pulled to `fine_tunes/act_green_ball_a7/`). ck30000 deployed,
`/health` cutover confirmed, then the same seven seeds run against both checkpoints.

| seed | dx | ck10000 | ck30000 | |
|---|---|---|---|---|
| 1 | −0.024 | rel @21 | **HELD** | regressed |
| 2 | +0.321 | rel @20 | rel @24 | |
| 3 | +0.298 | rel @20 | rel @22 | |
| 4 | +0.066 | rel @22 | rel @17 | |
| 7 | −0.048 | HELD | **rel @25** | fixed |
| 20 | −0.099 | HELD | HELD | |
| 53 | −0.106 | HELD | **rel @24** | fixed |

**Released 4/7 → 5/7.** Two fixed, one regressed.

### The headline is not the +1, it is the loss of structure

At ck10000 the release outcome was a **clean function of `dx`**: every `dx ≤ −0.048` held,
everything above released, no exceptions in nine runs (§7.4). At ck30000 the same axis
gives −0.024 ✗, −0.048 ✓, −0.099 ✗, −0.107 ✓ — scattered. The decision boundary did not
sharpen with training; it became jagged.

That is what fitting individual episodes rather than the rule looks like, and it is the same
shape as `libero/PROGRESS.md` §23.5, where ck3000 was three times more accurate than ck5000
from the same run. **The last checkpoint is not the best one** — now demonstrated twice, on
two different architectures.

### What did NOT get worse, checked before claiming it did

The first read of "the model looks confused at 30 k" was wandering, and the measurement does
not support it. Comparing **released runs only** (a held run hovers instead of retreating, so
path length and gripper-flip count are confounded and cannot be compared across outcomes):

| | eef path (m) | dpos std | gripper flips |
|---|---|---|---|
| ck10000, released runs | 1.98 | 0.147 | 2 |
| ck30000, released runs | **1.84** | **0.124** | 2 |

ck30000's motion is *tighter*, not messier. Grasp precision also improved — seeds 2 and 3
reach 0.0 mm lateral at ck30000 against 2.9 and 7.1 mm at ck10000.

So the regression is confined to the **release decision**, not to motion quality or
localisation. Delayed commitment is visible but weak and inconsistent (reopen chunk 20→24,
20→22, but 22→17), so "it hesitates longer" is not supported either; the honest statement is
that the release timing became noisy.

### Consequences

- **ck20000 is now the interesting checkpoint** and has not been scored. If 4/7 → ? → 5/7
  passes through a peak, the useful stopping point is earlier than either endpoint.
- **Resuming to 60 k is no longer obviously worth it.** The evidence between 10 k and 30 k is
  that placement rate is roughly flat while the failure mode gets less predictable. Resume is
  available (`--resume`, §Open items) if the 20 k score argues for it.
- **`dx` remains the right axis for *choosing* test seeds** even though it no longer predicts
  ck30000's outcome — it is what generates hard cases.
- Seed 20 has now failed at both 8 and 24 epochs, the only seed to do so. It is the single
  best candidate for a targeted look.

---

## Open items

- **Green bin position in scoring.** Decided 2026-08-01 not to spend more time here. State
  as it stands: `libero_closed_loop.py` now writes `green_bin_xy` per log entry and
  `score_runs.py` prefers it, so ACT's own runs will be scored against the real layout. Every
  log written before today lacks the field and falls back to the scene XML's
  `(0.560, 0.250)`, flagged `(XML)` in the output. That fallback is correct for those eight
  logs (none used `--randomize-bins`) but is NOT correct in general — a randomised run scored
  from the XML has a meaningless `placed`. Revisit if any historical randomised run needs
  re-scoring; there is no way to recover its layout, so it would have to be re-run.
- **`--use-state=false` ablation is unverified.** §6. Needs its own 1-step smoke before it is
  trusted, and its own `--exp-name`.
- **`meta/cohorts.json` for `a7` is gone.** §5. No cohort-split ablations without re-collecting.
- **`a7` clips `dx` on 1.35% of frames.** §5.1. First place to look if ACT fails on fast
  lateral motion.
- **Release reliability.** §7.2/§7.4 — 3 of 9 rollouts at ck10000 never opened the gripper,
  all three on an inward carry (`bin_x < ball_x`). Rerun those seeds against ck20000+ to
  decide under-fitting vs data coverage.
- **A third bin slot at low x** would separate "inward carry" from "far ball" (§7.4). Only
  worth building if the failure survives training.
- **`score_runs.py` will read a log that is still being written.** Fixed 2026-08-01: the
  client logs `chunks_requested` and the scorer prints an INCOMPLETE warning. Caught after
  nearly reporting a live run as a release failure — a partial log ends mid-carry and scores
  exactly like a policy that never let go.
- **Historical `placed` numbers predate the release check.** Anything quoted from before
  2026-08-01 may include held-ball false positives. Re-score before comparing.
