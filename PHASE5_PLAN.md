# Phase 5 — Diagnosis & Plan

Why the green-ball pick-and-place fine-tunes haven't worked, and what to do instead.

---

## 1. Aim & Scene

**Task:** Franka arm picks a green ball and places it in a green box. Both ball and box
positions randomize every episode.

**Scene** (`mujoco_menagerie/franka_emika_panda/scene_pick_place.xml`):
- Franka Panda + Robotiq 2F-85 gripper.
- `green_ball` — sphere, radius 0.02 m, freejoint. Sampled x∈[0.40,0.58], y∈[-0.16,0.14].
- `green_bin` / `blue_bin` / `yellow_bin` — 10×10 cm open boxes, welded to world,
  permuted across 3 anchor slots each episode (+/-2 cm jitter).
- `red_box` — distractor at (0.5, -0.28).
- Cameras: `external_cam` (free-standing, diagonal) + `wrist_cam` (hand-mounted).

**Model:** MolmoAct2-DROID. 8-D state `[q1..q7, gripper_rad]`, 8-D absolute joint-position
actions, 15 Hz, served over HTTP `/act`.

**Attempts so far:**

| # | Approach | Result |
|---|---|---|
| 1 | Base MolmoAct2-DROID, zero-shot | Never reaches the ball; hovers after a few steps |
| 2 | LoRA (VLM path) + action expert | Reaches near ball, closes gripper, no pick |
| 3 | Action-expert-only (`ae_train`, 500 steps) | Reaches near ball, closes gripper, no pick |

Training data for 2 and 3: 50 IK-scripted expert demos, LeRobot v2.1.

---

## 2. Current high-level issues

### 2.1 Inference runs the arm 33× too fast — **the primary bug**

Demo collection holds each action for `decimation` sim steps
(`phase4_collect_demos.py:286`):

```python
decimation = int(round((1.0 / model.opt.timestep) / CONTROL_HZ))   # = 33
for _ in range(decimation):
    mujoco.mj_step(model, data)
```

500 Hz sim / 15 Hz control → **66 ms of physics per action**.

Inference gives each action **one** step (`phase3_closed_loop.py:543`):

```python
for i, action in enumerate(actions):
    apply_action(data, action)
    mujoco.mj_step(model, data)      # 2 ms, not 66 ms
```

A 16-action chunk advances the sim 32 ms instead of 1.07 s. The position actuators never
converge on any commanded target; the arm creeps and stalls. This is exactly the "hovers
after a few steps" signature, and it capped every run including the fine-tuned ones.

Self-reinforcing: the arm barely moves → `state` stays constant → the model keeps emitting
the same absolute joint target.

### 2.2 Trained for less than one epoch

`ae_train` = 500 steps × `global_batch_size=8` = **4,000 samples**. Dataset is 6,050 frames.
The model saw ~66% of the data, once. Nothing converged. Both fine-tunes are essentially the
base prior plus a nudge — which is why they landed in the same place.

For 50 demos / 6k frames, expect **20k–30k steps** (≈30–65 epochs).

### 2.3 Demos contain zero recovery behavior

Every episode is a perfect open-loop IK trajectory. A BC policy trained only on perfect
trajectories has no idea what to do once it is 1.5 cm off — which is precisely the state it
lands in. Classic covariate shift.

### 2.4 No cheap evaluation signal

Every diagnosis so far cost a Modal deploy + a sim run, and produced one bit from one
rollout — while six independent failure modes were live simultaneously. One of those runs
was confounded by 2.1 the whole time.

---

## 3. Scene issues — divergence from the DROID pretraining distribution

Similarity matters in tiers. **Tier A is already correct** — that is why the base model does
something rather than nothing. **Tier B is where the gap is.**

### Tier A — must match exactly (✓ already compliant)

From `sim_eval/inference/common.py:44` and `client.py:144`:

```python
def droid_state_adapter(qpos):      # 13-D → 8-D
    return np.concatenate([qpos[:7], qpos[7:8]])

class DroidClient(_MolmoActHTTPClient):
    action_adapter = None            # raw model output, no rescaling
```

8-D state in raw radians, no normalization; absolute joint-position actions;
`external_cam` + `wrist_cam`; 15 Hz; gripper 0=open → 0.8=closed. All correct. Don't touch.

**One gap:** pretraining is on an **FR3** (`franka_droid.py` → `fr3_robotiq.urdf`); we train a
**Panda**. Same joint semantics, different link lengths — identical joint vectors put the EE
in different places. `mujoco_menagerie/franka_fr3/scene_droid.xml` is already vendored.

### Tier B — should match; cheap XML fixes; this is the real gap

| Issue | DROID / MolmoAct2 | Ours | Impact |
|---|---|---|---|
| **Gripper pad friction** | `static=2.0, dynamic=2.0`, `patch_radius=0.1` (`franka_droid.py:27`) | `friction="0.7"` / `"0.6"` (`2f85.xml:48,52`) | ~3× lower. A sphere has no form closure — the grasp is *purely* frictional. Scripted demos hit 98% only because alignment is perfect; margin is razor-thin. **Likely direct cause of "closed, didn't pick up."** |
| **Camera framing** | — | 1.40 m standoff, `fovy=71.5` | 5.3 mm/px → **ball = ~7 px**, smaller than one ViT patch (~14 px). It cannot be localized from the external cam. |
| **Aspect ratio** | `width=640, height=360` (16:9) | square 256²/378² | Every pretraining frame is wide; every one of ours is square. |
| **Wrist cam mount** | Rigid, fixed pose on `robotiq_arg2f_base_link` | `mode="targetbody" target="grasp_target"` | Ours **re-aims every frame**, pinning the target near center. A real wrist cam is bolted on and the target *drifts* — that drift **is** the servo error signal. We deleted the best cue for the last centimeter. |

**Camera fix:** ~0.8 m standoff, `fovy≈45` → 1.75 mm/px, ball ≈ **23 px** (~1.6 patches),
bin ≈ 57 px, still frames 0.66 m. Deviating from DROID's 71.5° spec is worth it — sub-patch
objects are a harder blocker than FOV shift. The diagonal *angle* is fine; the framing is not.

**Also consider:** ball radius 0.02 → 0.03. Easier to see *and* easier to grasp. Shrink back later.

### Tier C — doesn't need to match

Object identity, colors, table texture, lighting, layout, instruction wording. This is the
delta we're training. Don't spend effort here.

**Mental model:** Tier A → the action expert's motor prior transfers. Tier B → the vision
backbone's features transfer. Tier C → what we're fine-tuning.

Both failed runs were Tier-A-perfect and Tier-B-sloppy — consistent with what we saw:
sensible motion (motor prior fine) that never landed on target (visual features didn't).

---

## 4. Evals without training every time

An **eval ladder** — each rung ~10× cheaper than the next, and gating it.

### Rung 0 — Oracle replay (free, no GPU)

Replay recorded demo actions **through the inference code path**
(`phase3_closed_loop.apply_action` + its step loop), not the collector's. If ground-truth
expert actions don't pick up the ball, **no policy can**.

This is the upper bound, costs minutes, and would have caught the decimation bug on day one.
Re-run after every environment change (friction, cameras, FR3 port).

### Rung 1 — Perception probe (one forward pass, no training)

No need to train to learn whether the model can see a 7-px ball.
`examples/droid/host_server_droid.py:209` has `enable_depth_reasoning=False`, and the
response body (line 289) is only `{"actions", "dt_ms"}`. The spatial-reasoning output is
right there, switched off. Enable it, return it, ask the **base** checkpoint about our
frames. Decisive answer for one inference call.

### Rung 2 — Offline eval on held-out demos

Hold out 10 episodes; feed recorded frames, compare predicted vs recorded actions.
Two metrics: per-dim MSE (watch the **gripper channel** separately) and open-loop rollout
divergence over ~20 steps (catches compounding error that per-step MSE hides).

⚠️ In BC, validation MSE correlates **weakly** with task success. Use it as a *failure
detector* ("did this converge at all"), not a success predictor. It would have flagged the
0.66-epoch run instantly.

### Rung 3 — Batched headless sim eval

Pattern already vendored at `sim_eval/run_eval.py:210-255`:

```python
obs, _ = env.reset(seed=config.seed + ep)
...
summary = {"success_rate": float(np.mean(successes)), ...}
```

ManiSkill-based so not a drop-in, but copy the structure: **20–50 episodes, headless,
success rate + summary JSON**. Current eval writes 2 PNGs per sim step and syncs a live
viewer — enormously slow, and n=1.

### Rung 4 — The GIF

Keep it, but only to diagnose a failure already detected at Rung 3. Never as the metric.

### Process changes that cost nothing

- **Checkpoint every 2k steps**, eval several checkpoints from one run → a learning curve
  instead of a point; kill divergent runs early.
- **Keep one `modal serve` warm** and swap checkpoints. Deploy-per-checkpoint pays cold
  start every time.
- **Ablate by substitution:** script the transport/place, let the policy do only
  reach-and-grasp. Isolates the broken sub-skill instead of scoring the chain pass/fail.

There's no way around empirically evaluating a BC policy. But each evaluation can be 100×
cheaper, failures can be *diagnostic* rather than binary, and the free tests can run first.

---

## 5. Plan

**Cut task difficulty first.** Freeze the green bin at one position; randomize only the ball.
Nothing works end-to-end yet — halving what vision must learn buys a success signal to build
on. Re-enable bin randomization later.

### Tier 0 — free, no GPU

1. Fix the decimation bug in `phase3_closed_loop.py` (hold each action for `decimation` steps).
2. Raise gripper pad friction to ~2.0 in the 2F-85 XML.
3. Make `wrist_cam` a rigid mount — drop `mode="targetbody"`.
4. Re-frame `external_cam` (~0.8 m standoff, `fovy≈45`) and render **16:9**, not square.
5. Consider porting the scene to FR3.
6. Build **Rung 0** (oracle replay) and **Rung 2** (offline eval).
7. Re-evaluate the existing `ae_train` checkpoint — free, and the result will be materially
   different from what we've seen.

Items 2 and 3 are the ones most likely to move "closes on nothing," independent of retraining.

### Tier 1 — the one real run (~$20–40)

8. Recollect **~300 demos**: new cameras, 16:9, `--res` matched, noise-injected recovery
   labels (perturb the commanded target, then **re-solve IK from the perturbed state** so the
   recorded action is corrective), fixed bin.
9. Train `--mode lora`, **20–30k steps**, checkpoint every 2k.
10. Rung 2 → Rung 3. *Then* look at a GIF.

**Why LoRA over `ae_only`:** marginal cost on 1×H100 is ~1.5×, and it removes the suspected
perception bottleneck rather than leaving it to be re-litigated. One clear answer beats two
cheap ambiguous ones — the exact trap the last two runs fell into.

### Tier 2 — only if Tier 1 works

11. Re-enable bin randomization, collect more demos, retrain.
12. Shrink the ball back to 0.02 if it was enlarged.

---

## Key numbers

| Quantity | Value |
|---|---|
| Sim timestep / control rate | 0.002 s / 15 Hz → decimation = **33** |
| Current dataset | 50 episodes, 6,050 frames, 256×256 |
| `ae_train` steps | 500 × batch 8 = 4,000 samples (**0.66 epochs**) |
| Target training | 20–30k steps (30–65 epochs) |
| Ball @ current camera | ~7 px (< 1 ViT patch) |
| Ball @ proposed camera | ~23 px (~1.6 patches) |
| Pad friction: ours vs DROID | 0.7 / 0.6 vs **2.0 / 2.0** |
