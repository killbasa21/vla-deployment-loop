# Learning to Fold — Prioritized Toolbox

Companion to `lehome-2026-learning-to-fold-rl-equations.md`.
Organized by **the problem you're facing** → **the tool** → **what it buys**. Algorithm internals are P1.

**Priority:** 🅿️0 learn the intuition · 🅿️1 learn the algorithm as a worked example · 🅿️2 skim or skip
**Weight:** 🔑 load-bearing (the system fails without it)
**Evidence:** ✅ measured · 🤷 asserted, no ablation · ⚠️ he explicitly flags it as a guess

> Caveat worth internalizing before you trust any of this: he ran **almost no ablations**. *"This report is an engineering case study, not a controlled experiment."* The 79.63% is a system-level result. Almost nothing below is individually attributed.

---

# 🅿️0 — The toolbox

## Problem: "My reward is binary and sparse"

- [ ] 🔑 🤷 **Reward Derived From the Evaluator** — *Don't invent a reward function; reuse the success checker's own conditions as intermediate checkpoints.* **Buys:** dense signal with zero risk of optimizing something the grader doesn't measure, and nothing new to justify later. `§6.1`
- [ ] 🔑 🤷 **Return-Preserving Shaping (failure withdrawal)** — *Pay dense reward inside the episode, claw all of it back on failure, so `Σr = 1[success]` exactly.* **Buys:** temporal credit assignment with a **provably zero reward-hacking surface** — no way to farm partial credit without actually succeeding. **The most transferable idea in the paper.** `§6.1`
- [ ] 🤷 **Monotone Proportional Shaping (running-minimum)** — *Spike rewards give bad credit assignment; allocate in proportion to the fraction of the gap closed, using a running minimum.* **Buys:** smooth gradient toward the goal, and progress is never un-credited by later drift. `§6.1`
- [ ] 🤷 **Margin-Ranked Quality Bonus** — *A binary metric scores a barely-passing result identically to a clean one.* **Buys:** recovers the quality gradient the metric discards; biases toward robust rather than marginal solutions. `§6.6`
- [ ] 🔑 🤷 **Potential-Based Reward Shaping** — *Shaping of the form `γΦ(s') − Φ(s)` telescopes to a constant.* **Buys:** free credit assignment that **provably cannot change the optimal policy**. The one shaping tool that's mathematically safe to apply anywhere. `§6.4`

## Problem: "I need a critic but don't want to train one"

- [ ] 🔑 🤷 **Policy As Its Own Value Function** — *Predicting success needs the same visual primitives as choosing actions.* **Buys:** one model to train, serve, version and certify instead of two; the value signal also regularizes the shared representation. **The paper's central architectural bet.** `§5`
- [ ] 🔑 🤷 **Value Function from a Binary Return** — *Once the return is exactly `1[success]`, V collapses to `P(success) − R_cum`.* **Buys:** your critic is now a sigmoid classifier trained with BCE. Follows directly from failure withdrawal above. `§6.2`
- [ ] 🔑 🤷 **Attention-Group Isolation (shortcut prevention)** — *Given access to joint state, a value head will overfit to proprioception instead of looking at the scene.* **Buys:** forces every prediction head to work from pixels alone. Cheap, structural, no loss-function hacks. `§4.1`
- [ ] 🔑 🤷 **Action-Conditional Value Head (Q) with Stop-Gradient** — *V tells you if the state is good; you need to know if these specific actions are good.* **Buys:** a Q-function for candidate ranking. Stop-grad prevents Q from corrupting V. `§5.2`
- [ ] 🔑 🤷 **Task-Relevant State Prediction (world-model substitute)** — *A full world model is expensive and most of the state is worthless to predict.* **Buys:** ~90% of the benefit of a world model at ~1% of the cost, by predicting only reward-relevant numbers. ⚠️ **needs privileged sim state — doesn't transfer to real hardware as-is.** `§5.1`
- [ ] 🤷 **Policy-Stable vs. Policy-Dependent Signals** — *Value drifts as the policy changes; task progress barely does.* **Buys:** the diagnostic for which of your signals survive off-policy staleness. Explains the whole design of §F below. `§6.3`

## Problem: "My advantage estimate is noisy or biased"

- [ ] 🔑 🤷 **Control Variates / CUPED** — *Full baseline subtraction is optimal only for a perfect critic; with a noisy one you inject its noise into the advantage.* **Buys:** a principled dampening coefficient. Sanity-checks correctly at both limits (perfect → full subtraction, noise → none). `§6.2`
- [ ] 🔑 ✅ **Censoring / Survivorship Bias in Terminal States** — *A success terminates instantly while a near-miss lingers, so "almost done" frames come overwhelmingly from **failures**.* **Buys:** explains a systematic value under-prediction you would otherwise chase for weeks. **The most non-obvious observation in the paper**, and it generalizes to any early-terminating environment. `§6.2`
- [ ] 🤷 **Offline Tail Correction + Terminal Loss Reweighting** — *Fix the above from both ends: interpolate stored values toward the known outcome, and upweight those frames in training.* **Buys:** unbiased value near episode end. `§6.2, §5.3`
- [ ] 🤷 **EMA Smoothing** — *Raw per-chunk head predictions are too noisy to consume directly.* **Buys:** usable signal for downstream consumers (advantage, live failure detection, snapshot triggers). `§6.2`
- [ ] 🤷 **Auxiliary-Head Regularization** — *Small linear heads overfit far faster than the backbone.* **Buys:** aux predictions that stay honest. Separate weight decay + label smoothing. `§5.3`

## Problem: "My data goes stale in a continuous training loop"

- [ ] 🔑 🤷 **Predict-at-Collection-Time (frozen values)** — *A model overfits data it has already trained on, so re-running your value head over your own replay buffer produces garbage.* **Buys:** value estimates that stay honest because they were made on-policy against unseen states. Non-obvious and easy to get wrong. `§6.5`
- [ ] 🔑 🤷 **Graceful Estimator Degradation** — *As data ages, migrate the advantage from learned-value GAE toward an outcome-only group-relative signal.* **Buys:** an estimator that gets **less informative but never wrong**. This is what makes continuous asynchronous collection viable at all. `§6.5`
- [ ] 🤷 **Segment Baselines with Conditional Probabilities** — *An episode that reached the checkpoint then failed was genuinely good, then genuinely bad.* **Buys:** correct within-episode credit assignment from **outcomes alone**, no value function needed. `§6.5`
- [ ] 🤷 **Scale Matching Between Estimators** — *You can't linearly blend two advantage estimators until they're on the same numeric scale.* **Buys:** the blend above actually meaning something. Easy to overlook. `§6.5`
- [ ] 🤷 **Exponential Dataset Decay + Runtime Multi-Dataset Sampling** — *Never merge datasets; hold all sources and sample by share at runtime.* **Buys:** the data mix becomes a config knob you change per iteration, not a preprocessing step you rerun. `§2.4`

## Problem: "I can't get enough good data out of a slow simulator"

- [ ] 🔑 ⚙️ 🤷 **Asynchronous Flywheel (no synchronization barriers)** — *Trainer, N rollout workers and a human station sharing state only through an artifact store.* **Buys:** scaling data collection = starting another machine. Nothing ever blocks on anything. `§2.1`
- [ ] 🤷 **Stateless Policy Server + Client-Side State** — *Everything that looks stateful at inference lives on the client.* **Buys:** trivially horizontal scaling and reproducible serving. `§3.1`
- [ ] 🔑 🤷 **Success Replay (rare-positive multiplication)** — *Successes on hard garments are precious; snapshot and re-run them under heavier augmentation.* **Buys:** multiplies scarce positive data — the binding constraint early in training. `§3.2`
- [ ] 🔑 🤷 **Hard Negative Mining from Value Drops** — *Snapshot exactly where the predicted success falls off its running max — the moment the policy visibly ruined a promising episode.* **Buys:** targeted data at the actual bottleneck states, instead of uniformly re-rolling. **Uses the value head as a data-collection instrument**, which is the clever bit. `§3.2`
- [ ] 🤷 **Physics State Snapshotting** — *Save particle + joint state so any moment is restorable.* **Buys:** the substrate all replay strategies run on. `§3.2`
- [ ] 🤷 **Difficulty-Targeted Curriculum** — *Prefer tasks near a target success rate.* **Buys:** training at the frontier of ability rather than on what's already solved. `§3.2`
- [ ] 🔑 🤷 **Biased vs. Unbiased Episode Accounting** — *Replay and mining episodes are deliberately easier/harder than fresh ones.* **Buys:** statistics you can trust. Everything downstream (baselines, bandits, success rates) breaks silently without this. `§3.2`
- [ ] 🔑 ⚠️ **DAgger (Dataset Aggregation)** — *Load a failure state, let a human fix it by teleop, hand back to the policy.* **Buys:** in principle, recovery behaviour. **He reports it largely failed in sim** — *"the policy was simply better at folding than I was through teleop"* — but was one of his most useful real-robot tools. `§3.4`

## Problem: "My policy ignores its conditioning inputs"

- [ ] 🔑 🤷 **Advantage Conditioning (RECAP)** — *Feed the advantage in as an **input**, not just a training weight, telling the model "predict good actions only."* **Buys:** a quality dial you can turn at inference — and it's what makes CFG possible below. `§4.3`
- [ ] 🔑 🤷 **Adaptive Normalization Conditioning (AdaRMS / AdaLN / FiLM)** — *A single prefix token is too weak; he observed the model performing the **wrong garment's** motion despite correct input.* **Buys:** conditioning that reaches every layer directly. Weak evidence: wrong-garment behaviour "visually reduced." `§4.4`
- [ ] 🤷 **Conditioning Dropout / Stochastic Masking** — *Show negative-advantage frames only in the unconditional branch.* **Buys:** the conditional/unconditional asymmetry that makes guidance meaningful. `§4.3`
- [ ] 🤷 **Zero-Initialized Conditioning** — *New conditioning vectors init to zero.* **Buys:** an older checkpoint resumes **identically** at step 0, so any conditioning change is a strict extension. Small trick, disproportionately useful when iterating on a live run. `§4.4`
- [ ] 🤷 **Self-Prediction Bootstrap ("System 2")** — *An input isn't available at eval time, so have the model predict it, vote over a few chunks, then feed its own answer back and freeze it.* **Buys:** a general pattern for closing the train/eval input gap. `§4.2, §7.6`

## Problem: "Same weights, worse results than I could be getting"

- [ ] 🔑 ✅ **Inference-Time Hyperparameter Optimization** — *The same checkpoint achieves very different success rates depending on how it's run.* **Buys:** free performance with zero training cost. **Underrated as a category** — most people tune training and ship whatever inference config they started with. `§7`
- [ ] 🔑 ✅ **Classifier-Free Guidance (CFG)** — *At inference you want the policy's best behaviour, not its average.* **Buys:** amplified conditioning at only 2× the *cheap* action-expert cost, since the prefix is shared. Bandit converged to **α ≈ 7–9**, far above diffusion-image norms — the most informative empirical number in the paper. `§7.3`
- [ ] 🔑 ✅ **Best-of-N / Rejection Sampling** — *At rare multimodal bottleneck states, sampling several chunks and rejecting bad ones helps.* **Buys:** measurably better rollouts at N=2–3, no gain beyond. ⚠️ **His scorer's correlation with actual outcomes was effectively zero and it still worked** — so the mechanism is "avoid disasters," not "find the optimum." `§7.4`
- [ ] 🔑 ✅ **Thompson Sampling / Multi-Armed Bandits** — *Grid search is expensive **and** the optimum moves as the policy trains.* **Buys:** three things at once — cheap tuning, hyperparameters that co-evolve with the policy, and **exploration that doubles as useful training variance**. `§7.7`
- [ ] ✅ **Short Execution Length / Frequent Re-planning** — *Bandit converged to executing only 3–5 of 30 predicted steps.* **Buys:** a real finding — near-horizon predictions are far more reliable than far ones, so re-planning against a fresh observation pays. `§7.7`
- [ ] 🤷 **Soft Inpainting / Action Anchoring** — *Re-planning from scratch makes trajectories jump at chunk boundaries.* **Buys:** mode stickiness and smooth motion — but anchor **only during the high-noise phase** so the chunk can still self-correct. Bandit converged to *light* anchoring. `§7.2`
- [ ] 🤷 **Noise Temperature + Correlated Noise** — *Seed noise should respect the action covariance, and τ<1 concentrates candidates near the mode.* **Buys:** better candidates for best-of-N. `§7.5`

## Problem: "It works in sim and dies on hardware"

- [ ] 🔑 ✅ **Representation Overfitting Diagnostic** — *Resize 640→320→224 instead of 640→224 — a change invisible to the eye. Sim success dropped significantly, and the aux heads could tell the two apart **perfectly**.* **Buys:** a cheap, brutal test for whether your policy has latched onto rendering artifacts. **Steal this as a standing check.** `§9.1`
- [ ] 🔑 🤷 **Domain Randomization (episode-level vs. per-step)** — *Two scopes with different purposes; episode-level is saved with the physics state so replays reproduce it.* **Buys:** the main lever against unseen-object generalization. `§3.3`
- [ ] 🤷 **Augmentation Intensity Scheduling** — *How hard you augment depends on who has to act on the frame: light when a live policy must perform, extreme for replays where performance doesn't matter.* **Buys:** aggressive augmentation without degrading collection. Genuinely non-obvious. `§3.3`
- [ ] 🤷 **Physics-Affecting vs. Visual-Only Augmentation** — *Scale, roughness and base jitter must be skipped when replaying a saved state.* **Buys:** replays that don't silently diverge. `§3.3`

## Problem: "Which RL family do I even reach for?"

- [ ] 🔑 ⚠️ **On-Manifold vs. Off-Manifold Updates** — *Valid actions occupy a thin manifold; "discourage bad actions" pushes predictions off it, while reweighting only redistributes mass among samples the policy already produces.* **Buys:** the decision rule for the whole algorithm choice. ⚠️ **He states this as a subjective bet with no ablation** — the single most load-bearing unverified claim in the paper. `§2.2`
- [ ] 🔑 🤷 **Exploration Cost of Reweighting Methods** — *The flip side: you can only sharpen behaviours already in the data.* **Buys:** knowing what you're giving up. Directly causes the recovery gap he never closed. `§2.2`
- [ ] 🔑 🤷 **Prioritized Sampling (via sampler, not loss)** — *A downweighted sample still costs a full image decode and batch slot.* **Buys:** 100% of batch capacity spent on data you actually care about. Pure efficiency, no algorithmic change. `§2.3`
- [ ] 🔑 🤷 **Importance Sampling / Inverse Propensity Weighting** — *Tilted sampling biases any head whose target is a statistic of the true distribution.* **Buys:** one batch serving two different effective distributions — tilted for the actor, unbiased for the critic. `§2.3`
- [ ] 🤷 **Difficulty-Proportional Priority** — *Data without advantages still needs prioritizing.* **Buys:** oversampling by failure rate. `§2.3`
- [ ] ⚠️ **Checkpoint Rollback** — *A policy that co-evolved with its own data settles into local optima; revert a few days and retrain on everything collected since.* **Buys:** "reliably kicked the policy out of local optima." π\*0.6 does this systematically. He did it 4 times total. `§2.5`

---

# 🅿️1 — Algorithms, as worked examples

Learn these when a 🅿️0 tool above stops making sense without them. **Dependency order:**

- [ ] **MDPs, returns, value functions** — the vocabulary. Everything else assumes it.
- [ ] **TD learning & bootstrapping** — how `V(s) = r + γV(s')` becomes a training target.
- [ ] **Advantage: `A = Q − V`** — the quantity the entire paper is built on.
- [ ] **Policy gradient / REINFORCE + baselines** — why variance reduction exists at all.
- [ ] 🔑 **Generalized Advantage Estimation (GAE)** — the γ/λ backward pass. **Hard prerequisite** for §6 of the equations file.
- [ ] 🔑 **KL-Constrained Policy Improvement → AWR** — why exponentiated advantage is the *exact* optimum, not a heuristic. The cleanest derivation in the field and directly explains the on-manifold property.
- [ ] **PPO** — the rejected baseline; learn what the clipped ratio buys.
- [ ] **GRPO** — rejected as the main algorithm, reused as the stale-data fallback.
- [ ] **Generative modelling → diffusion → flow matching** — prerequisite chain for everything inference-side.
- [ ] **Action chunking** — predict H, execute a few, re-plan.
- [ ] **Beta-Bernoulli conjugacy** — the one piece of math behind Thompson sampling. Genuinely small.

---

# 🅿️2 — Skim or skip

- [ ] ⚠️ **Exclusive Self-Attention (XSA)** — he is candid: *"I rely on 'recent fashion' here."* No ablation, unclear benefit.
- [ ] **AWAC** — sibling of AWR; read the abstract, skip the rest.
- [ ] **Time-to-completion head** — he calls it legacy; trained but unused downstream.
- [ ] **Checkpoint head** — explicitly legacy, predictions unused.
- [ ] **FAST tokens, cross-layer KV mixing, per-timestep normalization** — inherited architecture, not contributions, no evidence attached.
- [ ] **Specific constants** (0.98 decay, 0.05 loss weights, α=0.5) — tuned for his setup, not transferable numbers.

---

## Where to start

**Three tools, in this order:**

1. **Return-Preserving Shaping** — self-contained, ~100 lines of numpy, no GPU, no RL prerequisites. Transfers furthest beyond folding.
2. **Censoring bias in terminal states** — pure intuition, zero math, and it will save you a debugging week at some point.
3. **Policy-as-its-own-value-function** — the architectural bet everything else rests on.

Only after those does the 🅿️1 ladder pay off.

## The gap he never closed

**Autonomous exploration and recovery.** Reweighting methods are *structurally incapable* of discovering behaviour never seen — the direct cost of the on-manifold guarantee. He names it in §8.2 and §10. A gap in the method, not the write-up, and the place where the real differentiation likely sits.
