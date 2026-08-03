# Learning to Fold — Toolbox, Prioritized by Transfer Value

Companion to `lehome-2026-learning-to-fold-rl-equations.md`.
Priority = **how much you'll need this to train your own policies on your own tasks**, not how much it won his competition.

🔑 load-bearing · ✅ he measured it · 🤷 asserted, no ablation · ⚠️ he flags it as a guess

---

# T0 — Foundational
*You cannot train a policy without understanding these. Not optional, not situational.*

- [ ] 🔑 **Behavior Cloning is the floor** — every technique here assumes *"a BC-pretrained policy with a non-zero success rate"* already exists (§2.2). BC is plain supervised learning, no RL needed. **This is your actual entry point** and where your MuJoCo pipeline already sits.
- [ ] 🔑 **Advantage `A = Q − V`** — the single quantity the entire paper manipulates. Every tool below is a way of computing, weighting, conditioning on, or degrading it.
- [ ] 🔑 **Generalized Advantage Estimation (GAE)** — how per-episode outcomes become per-frame training signal. You will reuse this on every project regardless of algorithm. `§6.4`
- [ ] 🔑 **Reward Derived From the Evaluator** — reuse the grader's own conditions as intermediate reward. **Buys:** dense signal with zero risk of optimizing something the grader doesn't measure. Applies to every task with a pass/fail spec — torque, seating tolerance, placement accuracy. `§6.1`
- [ ] 🔑 **Return-Preserving Shaping (failure withdrawal)** — dense reward inside the episode, all of it clawed back on failure, so `Σr = 1[success]` exactly. **Buys:** credit assignment with a provably zero reward-hacking surface. **Highest-transfer idea in the paper.** `§6.1`
- [ ] 🔑 **Potential-Based Reward Shaping** — `γΦ(s') − Φ(s)` provably cannot change the optimal policy. **Buys:** the one shaping form that is mathematically safe to apply anywhere, without thinking. `§6.4`
- [ ] 🔑 **Action Chunking** — predict H steps, execute a few, re-plan. **Buys:** the basic inference contract for every modern VLA. His bandit converged to executing only 3–5 of 30 — near-horizon predictions are far more reliable. ✅ `§7.1`
- [ ] 🔑 **Domain Randomization (episode-level vs. per-step)** — **Buys:** the primary lever against unseen-object generalization and sim-to-real gap. Unavoidable for anything trained in sim. `§3.3`
- [ ] 🔑 **Biased vs. Unbiased Data Accounting** — replay and mining episodes are deliberately easier/harder than fresh ones; exclude them from any statistic. **Buys:** numbers you can trust. Everything downstream breaks *silently* without this. Pure hygiene, universally applicable. `§3.2`

## Not in this paper, but T0 for you

He assumes these. You'll hit them first.

- [ ] 🔑 **Behavior cloning end to end** — dataset format, normalization, train/val split, loss curves.
- [ ] 🔑 **Evaluation protocol design** — what counts as success, how many trials, what variance. His whole system rests on a trustworthy binary grader you can call cheaply.
- [ ] 🔑 **Checkpoint scoring & model selection** — *your own repo already learned this the hard way: "The last checkpoint is not the best one. Demonstrated twice, on two architectures."* Score intermediate checkpoints.
- [ ] 🔑 **Train/serve config parity** — *also already in your `CLAUDE.md`: `--delta-pos-scale` at serving must equal the dataset's collection value.* A mismatch doesn't measure your model at all. He gets this free via a stateless server; you don't.

---

# T1 — First real project
*You'll reach for these the first time you post-train anything. Learn on demand, not upfront.*

- [ ] 🔑 🤷 **Policy As Its Own Value Function** — add value/progress heads to the policy instead of training a separate critic. **Buys:** one model to train, serve, version and certify. **Transfer note:** the architectural pattern generalizes fully; the *specific* heads he chose don't. `§5`
- [ ] 🔑 🤷 **Attention-Group Isolation (shortcut prevention)** — give the value head pixels only, no proprioception. **Buys:** heads that can't cheat. Structural, cheap, no loss hacks. Generalizes to any multi-head model. `§4.1`
- [ ] 🔑 ✅ **Censoring / Survivorship Bias in Terminal States** — successes terminate instantly while near-misses linger, so "almost done" frames come overwhelmingly from **failures**. **Buys:** explains a systematic value bias you'd otherwise chase for a week. **Universal in any early-terminating environment.** Zero math. `§6.2`
- [ ] 🔑 🤷 **Prioritized Sampling via the sampler, not the loss** — a downweighted sample still costs a full image decode and batch slot. **Buys:** 100% of batch capacity on data you care about. Applies anywhere you weight data. `§2.3`
- [ ] 🔑 🤷 **Importance Sampling / Inverse Propensity Weighting** — tilted sampling biases any head whose target is a true-distribution statistic. **Buys:** one batch serving two effective distributions. **You need this the moment you combine prioritized sampling with auxiliary heads** — and it's easy to not realize you needed it. `§2.3`
- [ ] 🔑 🤷 **Stop-Gradient Discipline** — `A = Q − V` only means something if V is estimated independently. **Buys:** heads that don't corrupt each other. General multi-head hygiene. `§5.2`
- [ ] 🔑 ✅ **Representation Overfitting Diagnostic** — resize 640→320→224 instead of 640→224 (invisible to the eye); sim success dropped significantly and the aux heads distinguished the two *perfectly*. **Buys:** a brutal, cheap test for latching onto rendering artifacts. **Make it a standing check.** `§9.1`
- [ ] 🔑 ✅ **Inference-Time Optimization as a Category** — the same checkpoint achieves very different success rates depending on how it's run. **Buys:** free performance at zero training cost. Most people tune training and ship whatever inference config they started with. `§7`
- [ ] 🤷 **Auxiliary-Head Regularization** — small heads overfit far faster than the backbone. Separate weight decay + label smoothing. `§5.3`
- [ ] 🤷 **Zero-Initialized Conditioning** — new conditioning vectors init to zero so old checkpoints resume identically. **Buys:** architecture changes become strict extensions mid-run. Small trick, disproportionately useful when iterating on a live training run. `§4.4`

---

# T2 — Situational
*Learn when you hit the trigger. Listed with the condition that should send you here.*

| Trigger | Tool | What it buys |
|---|---|---|
| Your critic is noisy and full baseline subtraction hurts | 🔑 **Control Variates / CUPED** `§6.2` | Principled dampening; correct at both limits |
| You're running **continuous** async collection, not discrete rounds | 🔑 **Predict-at-Collection-Time** `§6.5` | Value estimates that don't rot. Models overfit their own buffer — re-predicting gives garbage |
| ↳ same, and your buffer spans many policy versions | 🔑 **Graceful Estimator Degradation** `§6.5` | Advantage that gets less informative but never *wrong*. What makes the loop viable at all |
| Your task has natural intermediate checkpoints | **Segment Baselines** `§6.5` | Correct within-episode credit from outcomes alone, no critic |
| Your model ignores a conditioning input | 🔑 **AdaRMS / AdaLN / FiLM conditioning** `§4.4` | Conditioning that reaches every layer. He saw the model perform the *wrong garment's* motion despite correct input |
| An input available at train time is missing at eval | **Self-Prediction Bootstrap** `§4.2` | Model predicts it, votes, feeds its own answer back |
| Sim throughput is your bottleneck | **Success Replay** + **Physics Snapshotting** `§3.2` | Multiplies scarce positive data |
| ↳ and you know *where* it fails | 🔑 **Hard Negative Mining from Value Drops** `§3.2` | Targeted data at bottleneck states. **Uses the value head as a data-collection instrument** — the clever bit |
| Your policy is multimodal at decision points | ✅ **Best-of-N / Rejection Sampling** `§7.4` | Measurable gain at N=2–3. ⚠️ his scorer had ~zero outcome correlation and it *still* worked — mechanism is "avoid disasters," not "find optima" |
| You've built advantage conditioning | ✅ **Classifier-Free Guidance** `§7.3` | Amplified conditioning at 2× the *cheap* cost. Converged to α ≈ 7–9, far above diffusion norms |
| You have many inference knobs and no budget to grid-search | 🔑 ✅ **Thompson Sampling bandit** `§7.7` | Cheap tuning + hyperparameters that co-evolve with the policy + exploration doubling as training variance |
| Chunk boundaries cause visible jumps | **Soft Inpainting / Anchoring** `§7.2` | Smooth motion — but anchor only in the high-noise phase so the chunk can still self-correct |
| Your policy fails from OOD states and never recovers | 🔑 ⚠️ **DAgger** `§3.4` | In principle, recovery. **He reports it largely failed in sim** — *"the policy was simply better at folding than I was through teleop"* |
| Training has plateaued in a local optimum | ⚠️ **Checkpoint Rollback** `§2.5` | Revert a few days, retrain on everything since. π\*0.6 does this systematically |
| Your binary metric hides quality differences | **Margin-Ranked Quality Bonus** `§6.6` | Recovers the gradient the metric discards |

---

# T3 — Low transfer
*Read to understand his system; don't invest.*

- **Task-Relevant State Prediction (keypoint world-model substitute)** `§5.1` — the cheapest good idea in the paper, but it **needs privileged simulator state**. Doesn't transfer to real hardware without external measurement. Worth knowing *because* finding a real-world proxy would be genuinely differentiating.
- **On-Manifold vs. Off-Manifold argument** `§2.2` — ⚠️ intellectually the paper's foundation, but he states it as a subjective bet with no ablation. Understand the claim; don't treat it as established.
- **XSA** `§4.5` — *"I rely on 'recent fashion' here."*
- **TTC head, checkpoint head** — he calls both legacy; trained but unused downstream.
- **FAST tokens, cross-layer KV mixing, per-timestep normalization, garment-type token** — inherited architecture or task-specific, no evidence attached.
- **All specific constants** (α=0.5, 0.98 decay, loss weights) — tuned for his setup. Memorizing these is noise.

---

## Start here

1. **Return-Preserving Shaping** — ~100 lines of numpy, no GPU, no RL prerequisites, highest transfer.
2. **Censoring bias** — pure intuition, zero math, saves you a debugging week eventually.
3. **GAE** — the one piece of classical RL you'll use on literally every project.

## Prerequisite ladder (learn on demand, in order)

MDPs & returns → value functions → TD learning → advantage → policy gradient + baselines → **GAE** → KL-constrained improvement → **AWR** → (PPO/GRPO for contrast) → generative modelling → diffusion → **flow matching** → CFG

## The gap he never closed

**Autonomous exploration and recovery.** Reweighting methods are structurally incapable of discovering behaviour never seen — the direct cost of the on-manifold guarantee. He names it in §8.2 and §10. For industrial generalist robots this is the whole ballgame: 79.6% first-try is a demo, 99.5% eventual-success via autonomous retry is a product.

---

## How to dissect a tool

**Reproduce the problem before learning the solution.** Otherwise you memorize a name without the judgment of when to use it.

Per tool, five passes:

1. **Name what breaks without it** — one sentence, own words. Can't? Stop, you don't have it.
2. **Make it break** — smallest repro. Usually a toy problem, not the robot.
3. **Measure it** — get a number, or you can't tell if the fix worked.
4. **Apply the cheapest version** — not his; his has competition polish on it.
5. **Break the fix** — find where the tool stops working. That's where the intuition is.

**Depth by tier:** T0 → all five passes, build it. T1 → read now to recognize the symptom, dissect on contact. T2 → **index only**, it's a lookup table not a syllabus. T3 → read once.

**Toy problems, not the robot.** Gridworld for RL concepts; MuJoCo verifies ideas you already understand. Confounding "my concept is wrong" with "my sim is wrong" costs weeks.

**For anything marked 🤷:** the question isn't "how does this work" but **"would I notice if it were absent?"** He shipped ~50 things at once and won; a handful carried it.
