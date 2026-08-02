# Handoff: fine-tuning MolmoAct2-LIBERO for the green-ball pick-and-place

Written 2026-07-28. This file exists to start a **fresh agent session** on the fine-tune.
Everything below the horizontal rule is the prompt to paste. The notes after it are for
you, not for the agent.

---

## The prompt

```
Work in /home/hardik/Desktop/greenbox. Goal: fine-tune allenai/MolmoAct2-LIBERO so a
simulated Franka Panda picks up a green ball and places it in the green container.

STEP 1 — READ FIRST, IN THIS ORDER
- `docs/SERVO_DROOP.md` (was the top-level README until the 2026-08-02 reorg). At the time of writing this was the current state of the world. It documents a servo
  droop bug that poisoned the first two training datasets, the fix, and a measured
  closed-loop evaluation showing the stock checkpoint still fails the task.
- libero/README.md (conventions/spec), libero/fine_tune/README.md (dataset format),
  libero/PROGRESS.md (attempt log).
IMPORTANT: PROGRESS.md deliberately keeps wrong turns in, and several of its conclusions
were later reversed. docs/SERVO_DROOP.md §1.1, §5.1, §6.5 and §9 list numbers in the other docs that
are known stale or wrong. Where docs disagree, docs/SERVO_DROOP.md wins. Re-measure anything load
bearing rather than trusting a quoted figure.

STEP 2 — FIX THE DATA
Current dataset is libero/fine_tune/a3 (30 episodes: 10 reach / 10 noise / 10 recover,
5220 frames). a1/ and a2/ are SUPERSEDED — collected on the old plant, do not train on
them. a3's format is validated correct against the released dataset; the format is not the
problem. Two substantive problems remain, both described in docs/SERVO_DROOP.md §6, §6.5, §6.6:

(a) The dx action channel is one-sided: q01 = -0.072 against the released dataset's
    -0.679. Over the whole dataset there is essentially no signal for retreating in -x,
    which is the recovery behaviour the closed-loop runs fail at. Noise sigma CANNOT fix
    this — README §4.3 shows raising it stops episodes from succeeding at all. The
    recommended fix is to add a retreat-and-re-approach segment to the reference
    trajectory in libero/fine_tune/collect_finetune_data.py:waypoints(), which produces
    negative dx directly in the expert labels. Then regenerate as a4/ and validate.
(b) Normalisation. molmoact2/experiments/launch_scripts/train_lerobot.py defaults to
    --norm_mode q01_q99, and lerobot_utils/stats.py:_collect_tagged_stats builds the
    normaliser from whatever repos are in the mixture. Training on our data alone
    discards the pretrained calibration. Mix allenai/MolmoAct2-LIBERO-Dataset into the
    same 'libero' tag (273k frames vs our ~5k, so it dominates the quantiles and doubles
    as replay against forgetting), or inject the released meta/stats.json directly.

Validate any regenerated dataset before training: decode the parquet, check Arrow types
and the huggingface schema metadata against the released dataset, check action ranges and
per-channel quantiles against released, and LOOK AT the decoded PNG frames — the project
has been burned twice by concluding orientation from numbers alone.

STEP 3 — FINE-TUNE
No LIBERO training wrapper exists yet. droid/phase4_modal_train.py is the DROID/phase-4
counterpart and targets the 'green_ball_pick' mixture — use it as the structural template
but you will need a new mixture entry in data_mixtures.py (copy build_molmoact2_libero,
point it at our repo id), a volume upload, and a serving wrapper for the result.

TIME BUDGET: the fine-tune must finish within 1-2 hours of wall clock. This changes the
GPU choice — with a FIXED time window, "cheapest per hour" and "most training" conflict,
because a slow cheap card simply fits fewer steps into the same window. Optimise for the
most training completed inside the budget, not the lowest hourly rate, and say which you
picked and why. Budget backwards: measure seconds/step in the 1-step smoke run, then set
--max-steps so the run lands inside the window with margin for checkpoint save and
container startup (the base checkpoint is ~10GB to load).
Tune global_batch_size to hit that step count — the default is 128, which may be far more
per step than this dataset needs.
Size the GPU from the real memory requirement (5B params, bf16, LoRA). Do not guess.
Serving fits on an L4 (24GB, $0.80/hr); training needs more. Consider L4 -> A10 -> L40S ->
A100-40GB -> H100. Modal also bills warm idle containers at the GPU rate, so container
lifetime matters as much as the card.

NOTE, normalisation vs time budget: mixing allenai/MolmoAct2-LIBERO-Dataset into the
'libero' tag fixes normalisation (273k frames dominate the quantiles) but it also enters
the SAMPLING mixture, and at a high rate most of your limited steps would train on LIBERO's
tasks rather than the green-ball task. Stats collection and sampling rate are separate
knobs — set the released repo's rate low (or zero if the loader allows stats-only) so it
anchors the normaliser without eating the time budget. Verify which it actually does in
lerobot_utils/stats.py and data_mixtures.py rather than assuming.

SMOKE RUN FIRST, and keep it to --max-steps 1 --save-interval 1. One optimizer step plus
one checkpoint save proves the whole pipeline. Every extra step burns money for no signal.
This rule is in CLAUDE.md.

LoRA is required, not optional: PROGRESS.md §4 shows action-expert-only training drove
flow loss 0.176 -> 0.01 in 500 steps while task success stayed at zero, because with the
VLM frozen it can only learn the average trajectory.

TRAP THAT HAS ALREADY COST TIME: host_server_droid.py hardcodes NORM_TAG =
"franka_droid" at module level. Serving the LIBERO checkpoint with the DROID tag yields
garbage actions OF THE CORRECT SHAPE — a silent failure. libero/libero_modal.py retags it
after import; any new serving wrapper must do the same. Always confirm /act reports
norm_tag "libero" before trusting a rollout.

STEP 4 — EVALUATE
- libero/libero_closed_loop.py drives our green-ball scene. A LIBERO server is already
  deployed at
  https://hardikkapoor1021--molmoact2-libero-molmoactliberoserver-serve.modal.run/act
- libero/libero_benchmark_eval.py drives a real LIBERO task through robosuite's own OSC
  and scored 3/3 on the stock checkpoint — use it as a known-good control.
- The action expert is flow-matching, so it SAMPLES. One run is one draw; two runs from an
  identical start state have diverged completely. Score success rate over several runs and
  never read a single rollout as the model's behaviour.
- Baseline to beat, measured on the current corrected scene (README §9): 0/3 placements,
  1/3 grasp-and-lift.

CONSTRAINTS
- Modal costs real money. Confirm with me before any run that is not a 1-step smoke.
- Demo collection is local CPU MuJoCo and costs nothing — do not economise on demos.
- Don't commit anything under assets/ (bulky per-run debug artifacts, gitignored).
```

---

## Notes for you, not for the agent

### Two things to know when handing this off

**The plant change is not reflected in the older docs.** Arm gains in
`mujoco_menagerie/franka_emika_panda/panda_libero_hand.xml` were changed to `kp x2`,
`kd x0.7` on 2026-07-28. Documented in README §4.1 and in a long comment in the XML itself,
but `libero/README.md` and `PROGRESS.md` predate it. The new agent will be reading a scene
whose gains do not match some quoted numbers.

**`a3` may not be worth keeping.** If the agent adds the retreat segment it regenerates
anyway, and `a4` supersedes `a3` exactly as `a3` superseded `a1`. It may be cleaner to tell
it to go straight to `a4` rather than reason about which of four datasets is live.

### Two caveats on the 1-2 hour budget

**1-2 hours is plausible, and that is not a coincidence — it is a small LoRA on a tiny
dataset.** ~5000 frames of a single task. `PROGRESS.md` §4 records a 500-step
action-expert run completing on a single H100. At this scale the risk is not
undertraining, it is **overfitting**: README §6.1 notes the rotation channels are 3-6x
narrower than the released distribution and will be the first thing to collapse toward
zero if trained too long. A short budget is arguably the right call rather than a
compromise.

**The budget is on the training run, not on the work.** Dataset regeneration is local CPU
— `a3` took 918 s, and the retreat-segment change means regenerating. Add the smoke run,
deployment, and multi-run evaluation (the policy samples, so one rollout proves nothing).
The GPU meter reads 1-2 hours; the session will be longer.
