# Learning to Fold (LeHome 2026) — Core RL Equations

**Source:** Larchenko, arXiv:2606.27163v1 [cs.RO], Jun 2026. 1st/62 online (79.63%), 2nd real-world.
**Video:** youtube.com/watch?v=mTohakalKz8 · **Code:** hf.co/IliaLarchenko/lehome_sim
**Notes:** 2026-08-02

Flow-matching VLA (π0.5 lineage) post-trained with RL on bimanual garment folding. Central idea: **the policy is its own value function**. Advantage is consumed twice — AWR (sampling) + RECAP (conditioning). No critic, no PPO.

---

## 1. Why not PPO

Flow matching gives no tractable `log π` to differentiate. Deeper objection:

> "Valid actions occupy a tiny manifold... any algorithm that 'discourages bad actions' mostly pushes predictions off that manifold."

Reweighting/conditioning only redistribute mass among samples the policy already produces → always on-manifold. Cost: **no exploration**. He accepts this explicitly, and it's the system's main weakness (§7).

⚠️ Stated as a subjective bet. No ablation.

---

## 2. AWR — through the sampler

Standard AWR weight $w_i = e^{A_i/\beta}$, the closed-form solution to KL-constrained improvement: $\pi^* \propto \pi_{\text{data}}\,e^{A/\beta}$. β is the Lagrange multiplier. *That* identity is why AWR can't leave the manifold — the optimum is defined as a reweighting of the data.

His version applies weights via **sampling, not loss**:

$$\boxed{P(\text{sample frame } i) \propto e^{\,\mathrm{clip}(A_i,\,-2,\,2)}}$$

Batch loss is plain unweighted MSE. Equivalent in expectation, but a low-weight frame is *never loaded* rather than loaded-then-downweighted — no wasted image decode or batch slot. Clip bounds the best:worst ratio at $e^4 \approx 55\times$.

**Required correction.** The action head wants the tilted distribution; the aux heads don't (their targets are statistics of the *true* distribution). So every frame carries an inverse-propensity weight

$$\boxed{w_i = \frac{1}{N\,p_i\,T_{\text{ep}(i)}}}$$

applied to all aux losses, ignored by the action loss. One batch, two effective distributions.

BC/DAgger data has no advantage, so it gets difficulty priority instead: $P \propto e^{3(1-SR_{\text{garment}})}$ (a 50%-success garment sampled ~4.5× more).

---

## 3. Reward — densifying binary success

Success = conjunction of keypoint conditions on **distance ratios** $d^{(i)} = \text{dist}_i / \text{threshold}_i$ (proximity needs $d \le 1$, spread needs $d \ge 1$). He reuses *exactly these* for shaping — derive shaping from the evaluator, not intuition.

**Gradual first checkpoint** (tops). With $m_t = \min_{\tau\le t} d_\tau$ the running minimum, $t_1$ the frame the checkpoint is reached:

$$\boxed{R^{cp1}_t = 0.5\,\mathrm{clip}\!\left(\frac{d_0 - m_t}{d_0 - d_{t_1}},0,1\right), \quad r_t \mathrel{+}= R^{cp1}_t - R^{cp1}_{t-1}}$$

= *fraction of the gap closed* × 0.5. Running minimum makes it monotone — progress is never un-credited.

**Failure withdrawal** — the key move:

$$\boxed{\textstyle\sum_t r_t = \mathbb{1}[\text{success}]}$$

On failure all accumulated reward is withdrawn, spread uniformly from $t_p$ (last cumulative-reward peak) to avoid one terminal spike: $r_t \mathrel{-}= \frac{\sum_\tau r_\tau}{T - t_p - 1}$.

> Dense credit assignment *within* the episode; episode return stays exactly the true objective. No way to farm checkpoint reward without succeeding. **Most transferable idea in the paper** — works for any task with a crisp pass/fail spec.

**Precision boost.** Top 20% of successes per garment (by tightest final margin: $1-d^{(i)}$ proximity, $d^{(i)}-1$ spread) get $\Delta A = 0.3$ on every frame. Recovers the gradient binary metrics discard.

---

## 4. Policy as its own value function

Since the return *is* the success indicator:

$$\boxed{V(s_t) = P(\text{success}\mid s_t) - R^{\text{cum}}_t}$$

*(Total expected return is $P(\text{success}\mid s_t)$; $R^{\text{cum}}_t$ already paid out; remainder is the difference.)*

So V needs only a sigmoid head + BCE. No separate critic to train, serve, version-match. The head reads a **single query token in the image attention group** — sees pixels only, no proprioception, so it can't take a state shortcut.

**CUPED dampening.** Full baseline subtraction fails twice here: (1) checkpoint rewards cancel exactly — they enter both $P(\text{success})$ and $R^{\text{cum}}_t$ with opposite signs, giving *zero* advantage at a checkpoint; (2) variance-minimisation assumes a perfect predictor. Optimal control-variate coefficient (CUPED [22], = OLS coeff of V on $\hat V$):

$$\theta^* = \rho(V,\hat V)\frac{\sigma(V)}{\sigma(\hat V)}$$

Limits check: perfect → 1 (full subtraction); pure noise → 0 (don't subtract). Not computable, so a fixed constant is used:

$$\boxed{\hat V_t = \alpha_s\big(P(\text{success}\mid s_t) - R^{\text{cum}}_t\big), \quad \alpha_s = 0.5}$$

The same $\alpha_s$ damps *both* terms — which is what fixes problem (1): checkpoints are now only partially cancelled, so they retain positive advantage.

**Two corrections:** EMA $\bar S_t = \mathrm{EMA}(\hat P_t)$, $\alpha_{\text{EMA}}=0.2$. And a **value tail correction** — the head under-predicts near success because *a success terminates immediately while a near-miss sits in that identical-looking state for many frames*, so "almost done" frames come mostly from failures. Offline, last K=30 frames are interpolated toward the known outcome:

$$\bar S_{T-K+i} = \bar S_{T-K-1} + (y - \bar S_{T-K-1})\tfrac{i}{K}$$

Same bias attacked at training time via 20× BCE weight on the last 20 frames of successes.

**Completion head** — MSE on $t/T$, successful episodes only. Fixes three success-head weaknesses: noise, saturation above 90% success, and drift (same net predicts actions). Key property: **task progress is near policy-invariant** while value is definitionally policy-dependent. That's why completion survives staleness (§6) and success doesn't.

**Q-function analogue** on the FM query at the action-expert tail:

$$\boxed{\Delta_{\text{success}} = y - \mathrm{sg}\big[\hat P_{\text{success}}\big]}$$

"Given *these* actions, how much better/worse than the image-only baseline?" Stop-gradient is essential — $A = Q - V$ only means something if V is estimated independently. Loss scaled by $(1-t)$, samples with $t>0.5$ dropped (a mostly-noise chunk says nothing about the future).

This is the **world-model substitute**: predict only the reward-relevant numbers (keypoint distances, 30 frames ahead), not pixels or latents. Cheap — but needs privileged sim state, so it's sim-only.

---

## 5. Advantage — GAE over two heads

γ = 0.999, λ = 0.99, computed offline before each training iteration. Effective horizon $1/(1-\gamma\lambda) \approx 91$ frames ≈ **3 s at 30 Hz**.

**Success channel**, terminal pinned to truth ($\bar S_T = y$):

$$\boxed{\delta^s_t = r_t + \gamma\hat V_{t+1} - \hat V_t, \quad A^s_t = \delta^s_t + \gamma\lambda A^s_{t+1}}$$

Substituting $\hat V$ and $R^{\text{cum}}_{t+1} = R^{\text{cum}}_t + r_t$ (verified symbolically):

$$\delta^s_t = \underbrace{(1-\alpha_s\gamma)r_t}_{\text{damped reward}} + \underbrace{\alpha_s(\gamma\bar S_{t+1} - \bar S_t)}_{\Delta P(\text{success})} + \underbrace{\alpha_s(1-\gamma)R^{\text{cum}}_t}_{\to 0}$$

Third term scaled by 0.0005 → vanishes. With $1-\alpha_s\gamma = 0.5005$: **advantage ≈ half the reward earned + half the rise in predicted success.**

**Completion channel** — potential-based shaping, $\Phi_t = \alpha_c \bar C_t$, $\alpha_c = 0.5$:

$$\boxed{\delta^c_t = \gamma\Phi_{t+1} - \Phi_t, \quad \Phi^c_t = \delta^c_t + \gamma\lambda\Phi^c_{t+1}}$$

"Potential-based" is load-bearing (Ng et al. 1999, uncited): $\gamma\Phi(s')-\Phi(s)$ telescopes to a constant and is **provably policy-invariant** — free credit assignment, zero hacking risk. That's why it's applied at full strength to all data and never needs the $R^{\text{cum}}$ correction.

---

## 6. Stale rollouts — graceful degradation

`P(success)` is policy-dependent, and the model **overfits to data it already trained on** — you can't re-run the value head over your buffer. Fix: (1) record predictions *at collection time*, never re-predict; (2) decay dataset share ×0.98/iter; (3) blend:

$$\boxed{\tilde A_t = w A^s_t + (1-w)A^{\text{seg}}_t + \Phi^c_t, \quad w = \min(\text{sampling share},1)}$$

As data ages, $w\to 0$ and the advantage migrates to a GRPO-style outcome-only signal that uses no model prediction. **Less informative, never wrong.** $\Phi^c$ sits outside the blend (policy-invariant).

**Segment baseline** — episodes split at the first checkpoint, $p_{cp}$ = checkpoint rate, $SR$ = success rate:

$$A^{\text{seg}}_t = \begin{cases}(R_1 - \tfrac12 p_{cp})\,G(n_1)/n_1 & t \le t_1\\ \big(R_2 - (SR/p_{cp} - \tfrac12)\big)\,G(n_2)/n_2 & t > t_1\end{cases} \qquad G(n) = \frac{1-(\gamma\lambda)^n}{1-\gamma\lambda}$$

$(R - SR)\,G(T)/T$ if no checkpoint reached. $G(n)/n$ is the *average per-step GAE magnitude* — needed to put segment and GAE advantages on the same scale before blending. Second baseline $SR/p_{cp} = P(\text{success}\mid\text{checkpoint})$, the correct conditional.

> **Key property:** reach the checkpoint then fail → **positive advantage before, negative after**. Correct credit assignment from outcomes alone, no value function.

Normalised by one global σ over *unbiased rollouts only*, clipped to [−2,2].

**His own summary:** *"over-engineered and could probably be simplified."* Advantage is high when the action makes objective keypoint progress, raises predicted success, and raises predicted completion.

---

## 7. Conditioning & inference

**Advantage token gate.** $A<0$ → always masked (so negative frames are *only* seen unconditionally). $A\ge0$ → stochastic, $P(\text{neutral})$ ramping 0.5 at A=0 → 0.1 at A≥2. That asymmetry is what makes CFG meaningful.

**Classifier-free guidance:**

$$\boxed{\hat v = v_{\text{uncond}} + \alpha\,(v_{\text{cond}} - v_{\text{uncond}})}$$

Both passes share the prefix KV cache — guidance doubles only the cheap action expert. **α converged to 7–9**; he started with a 0–2 range and kept shifting up as the bandit pinned to the top arm. That's the most informative empirical result in the paper about advantage conditioning.

**Best-of-N:** $\text{score} = \tfrac12(\Delta^{\text{cond}}_{\text{success}} + \Delta^{\text{uncond}}_{\text{success}})$. Conditional head is optimistically biased (trained on positive-advantage frames only); unconditional is unbiased but scores guidance-shifted actions. Averaging is an admitted compromise. Retry with a larger batch if all N score negative.

⚠️ **His candid caveat:** *"The correlation between the FM head's prediction and the actual outcome was effectively zero in all my experiments — yet 2-3-candidate rollouts consistently beat single-candidate ones."* Hypothesis: best-of-N mostly **avoids disasters** at rare multimodal bottleneck states. Supported by N converging to 2–3 with no gain beyond.

**Noise:** $\Sigma_{\text{reg}} = \beta\Sigma + (1-\beta)I$, β=0.5; temperature scales by $\sqrt\tau$, converged τ ∈ 0.7–0.9.

**Thompson bandit** — factorised (independent posterior per parameter), tuned *online during collection*. Beta(α,β) per arm, baseline-subtracted reward:

$$\boxed{r = \text{success} - SR(\text{type}), \quad \alpha \mathrel{+}= \max(r,0), \quad \beta \mathrel{+}= \max(-r,0)}$$

Posteriors decay toward uniform each iteration → tracks the moving policy. Best benefit: exploration doubles as useful training variance (varying execution speed teaches the policy to go faster).

**Converged values (sim), H=30, S=10 Euler steps:**

| | top_long | top_short | pant_long | pant_short |
|---|---|---|---|---|
| $n_e$ executed | 5 | 5 | 3 | 3 |
| $n_a$ anchor | 6 | 3 | 3 | 3 |
| $t_{ip}$ inpaint onset | 0.4 | 0.4 | 0.5 | 0.5 |
| α guidance | 7 | 7 | 9 | 7 |
| τ noise temp | 0.9 | 0.7 | 0.7 | 0.7 |
| N candidates | 2 | 3 | 3 | 3 |

**What the directions say:** guidance → very high (advantage conditioning is a strong real signal). N > 1 helps, > 3 doesn't (avoiding bad ≠ finding optimal). **Execution length → only 3–5 of 30 steps** — near-horizon predictions are far more reliable, frequent re-planning pays. Inpainting → light (strong anchoring blocks self-correction).

---

## 8. Secondary equations

**AdaRMS multi-signal conditioning** — a single prefix token was too weak (model performed the wrong garment's motion), so the signal also modulates every RMSNorm:

$$c = \underbrace{\mathrm{MLP}(\mathrm{posemb}(t))}_{\text{flow time}} + \underbrace{g[\text{garment\_type}]}_{\text{4 learned}} + \underbrace{\mathbb{1}[\text{adv active}]\cdot a}_{\text{1 learned}}$$

New vectors zero-initialised → old checkpoints resume identically. Strict extension.

**XSA [19]** — removes the trivial self-routing path: $z_t = y_t - \frac{y_t\cdot v_t}{\lVert v_t\rVert^2+\epsilon}v_t$. He is candid: *"I rely on 'recent fashion' here."*

**Per-timestep action normalisation:** $\sigma_d(t) = a_d + s_d\sqrt{t + e_d}$. Actions are deltas, so spread grows like a random walk ($\sigma \propto \sqrt t$). One global scale would over-shrink the near steps — the ones actually executed — and let the high-variance tail dominate the loss.

**Aux loss weights** (action loss = 1.0): success 0.05 · ttc 0.05 · completion 0.02 · garment_type 0.02 · checkpoint 0.001 · keypoint_distance 0.02 · FAST 0.01 · wm_fast 0.01 · wm_flow {success 0.1, completion 0.05, keypoint 0.02}. Plus label smoothing $y' = y(1-\alpha)+\bar p_g\alpha$, α=0.05; aux-kernel weight decay 0.001 (they overfit massively).

---

## 9. Constants

| | | | |
|---|---|---|---|
| γ | 0.999 | $\alpha_s$ | 0.5 |
| λ | 0.99 | $\alpha_c$ | 0.5 |
| γλ | 0.989 → ~91 frames ≈ 3 s | $\alpha_{\text{EMA}}$ | 0.2 |
| adv clip | [−2,2] → $e^4\approx55\times$ | K tail | 30 |
| $1-\alpha_s\gamma$ | 0.5005 | ΔA precision | 0.3 |
| dataset decay | ×0.98/iter, floor 0.1 | BC priority | $e^{3(1-SR)}$ |
| H / S | 30 / 10 | CFG α | 7–9 |
| hardware | 1×H200, batch 192, ~300k steps | data | ~12.5k eps / 4.3M frames |

---

## 10. Table stakes vs. moat *(my read, not his)*

**Table stakes:** π0.5-class flow-matching VLA with action chunking · domain randomisation · async distributed rollouts (he used HF Hub as the message bus) · best-of-N.

**Plausibly differentiating:**
- **Evaluator-derived dense reward with return-preserving withdrawal (§3).** Transfers directly to any industrial task with a crisp spec — torque, seating tolerance, placement accuracy. Dense RL signal without inventing a reward you'll later have to defend.
- **Policy-as-value-function.** One artifact to train, serve, version *and certify*. On a factory floor that's a compliance advantage, not just engineering convenience.
- **Graceful advantage degradation (§6).** What makes a *continuous* collection loop viable. Unavoidable problem if you deploy fleets streaming data back forever.
- **Bandit-tuned inference params that co-evolve with the policy.** Per-SKU/per-station tuning at ~zero cost.

**The gap — and where the moat actually is.** He names it himself: **no autonomous recovery.** Trained on clean scripted demos plus its own successes; never learned to fix a ruined state. Sim DAgger failed — *"by the time I had it working the policy was simply better at folding than I was through teleop."*

> 79.6% first-try is a demo. 99.5% eventual-success via autonomous retry is a product. The conditioning/reweighting family is **structurally incapable** of discovering recovery behaviour it has never seen — the direct cost of the on-manifold guarantee in §1. Whoever solves autonomous recovery for deformable/contact-rich manipulation has the moat. Nothing here solves it, and he says so.

Second gap: the keypoint world-model substitute **needs privileged sim state**. Cheapest good idea in the paper, and it doesn't transfer to real-robot training without external measurement. A real-world proxy for that signal buys you a large chunk of this recipe on hardware.

---

## 11. Open questions

1. **Why does CFG want α ≈ 7–9?** Diffusion image models use 1.5–7.5; that's aggressive for a *policy*. Weak signal needing amplification, or is high-α extrapolation qualitatively different in action space?
2. **Best-of-N works at zero predictive correlation.** Testable: measure outcome variance across candidates per state, check whether the benefit concentrates in high-variance states.
3. **Which pieces carry the weight?** Of {gradual checkpoint, CUPED dampening, tail correction, completion shaping, segment blending, precision boost} — he says it's over-engineered. Ablate on a smaller task.
4. **Checkpoint rollbacks (§2.5).** Reverting to an older checkpoint and retraining on everything since "reliably kicked the policy out of local optima." π*0.6 does this systematically. Smells like it's really about the off-policy mixture ratio.
5. **Sim-overfitting diagnostic (§9.1).** Resizing 640→320→224 instead of 640→224 significantly dropped sim success, and the aux heads could *perfectly* distinguish the two. Striking, cheap to replicate, worth stealing as a standing test.

---

## 12. Next steps

**Read:** AWR [5] (short, the KL derivation) → RECAP/π*0.6 [6] → GAE [21] → Ng/Harada/Russell 1999 (potential-based shaping, uncited but it's the justification for §5) → HIL-SERL [11] (where the recovery gap points).

**Experiments** *(on the MuJoCo/MolmoAct2 pipeline in `docs/PLAN.md` — ask and I'll spec any properly):*

1. **Reward-withdrawal harness, no training.** Implement §3 on the existing pick-and-place: proximity condition from box→container distance, gradual 0.5 via running-minimum, withdraw on failure. Plot cumulative reward, success vs. near-miss. ~100 lines numpy, no GPU. Tests the best idea in isolation.
2. **Value-head-as-aux-head.** Sigmoid success head off image tokens only, BCE on outcomes — check whether the tail under-prediction bias (§4) appears. It should; the data asymmetry is universal.
3. **Sampler vs. loss weighting.** Measure the throughput claim directly. Reusable infra regardless of algorithm.
4. **Reproduce the resize diagnostic.** Cheapest sim-overfitting test; make it standing.

---

**Citations:** [5] AWR · [6] RECAP/π*0.6 · [7] BEHAVIOR-1K (his team) · [8] π0.5 · [9] AWAC · [10] DAgger · [11] HIL-SERL · [12] PPO · [13] GRPO · [14] difficulty curriculum · [15] SigLIP · [16] Gemma · [17] flow matching · [18] FAST tokens · [19] XSA · [20] chunk re-planning · [21] GAE · [22] CUPED · [23] RTC · [24] CFG · [25][26] Thompson sampling
