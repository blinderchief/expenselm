# ExpenseLM Learning Guide

Every concept the project touches, in the order the pipeline uses it.
Read a section *before* running that stage, then explain it back (to
Claude, per PRD §10 rule 2) without looking. The viva-style questions at
the end of each section are your self-check.

---

## 1. Why this project shape at all: distillation economics

A frontier model (Claude) can do expense extraction zero-shot, but every
message costs API money and leaves your infrastructure. The 2026 production
pattern the PRD mirrors: use the frontier model **once**, as a *teacher* —
to generate training data — and compress that narrow capability into a
small, cheap, private *student* model. That's **task-specific distillation**
(here: distillation through synthetic data + SFT, not logit distillation).

The economics: Claude at ~$7–10 per 1k messages vs. a 4B model at $0
marginal on hardware you own. If E3 ≥ E4 − ε on YOUR metrics, the small
model wins for this task. The eval harness exists to measure ε honestly.

**Self-check:** why does this only work for *narrow* tasks? (Because the
student has ~4B params of capacity; it can match the teacher on a thin
slice of behavior, never in general.)

## 2. The schema is the product (schema.py)

Structured extraction lives or dies on the output contract. Key ideas:

- **`extra="forbid"`** turns hallucinated fields into hard failures —
  otherwise your schema-validity metric silently overcounts.
- **Validate shape at parse time, judge correctness at grade time.** The
  schema answers "is this a legal record?"; the grader answers "is it the
  *right* record?". Mixing the layers makes both unmaintainable.
- **Closed enums everywhere possible** (categories, flags, confidence).
  Every open-ended string is a field you can't grade automatically.

**Self-check:** why is `date` a `str` with a regex, not `datetime.date`?

## 3. Evals before training (eval/, PRD's deliberate ordering)

Training without an eval is flying blind — you'll "feel" the model got
better. The harness is built Week 3, *before* any GPU spins, so that:

1. E0/E1/E4 baselines exist → you know the floor and the ceiling first.
2. Metric definitions are frozen → you can't (even unconsciously) tune the
   grader to flatter your model later.

The metric layers, and why each exists:

| Metric | Question it answers | Failure it catches |
|---|---|---|
| schema validity | can a program consume the output at all? | rambling, broken JSON, invented fields |
| expense F1 | did it find the right SET of expenses? | missed second expense, hallucinated extras |
| field accuracy | are individual values right? | wrong currency, date off-by-one |
| policy accuracy | did it *reason* about the rules? | limit not flagged, personal expense reimbursed |
| cost/latency | is it worth deploying? | a model that's right but too slow/expensive |

**The alignment problem** (grader.py): before grading fields you must
decide which predicted expense corresponds to which gold expense —
greedy match on (amount±0.01, category), relaxed to amount-only. Metrics
that skip this step (e.g. naive field-by-index comparison) silently break
on any reordered output.

**Self-check:** why is field accuracy denominated over *gold* expenses
rather than matched pairs? What would the alternative inflate?

## 4. Synthetic data that doesn't collapse (data/generate.py)

The failure mode: "generate 20 diverse examples" × 100 calls = 2,000
paraphrases of the same three cabs. Defenses used here:

1. **Explicit diversity axes** — every batch is pinned to a sampled corner
   of (source × language × ambiguity × multiplicity × policy case).
2. **Hand-written seed anchoring** — 3 of your seeds are shown per batch as
   style references, tying the synthetic distribution to reality.
3. **Schema gate** — generated gold that doesn't validate is discarded, so
   labeling errors of the *structural* kind never enter training.
4. **Human verification** — structure can be auto-checked; *judgment*
   (is this actually reimbursable?) cannot. Hence 100% review of test/dev.

**Self-check:** why generate input+gold *together* rather than generating
inputs and labeling them in a second pass? (One pass keeps the scenario's
intent attached; two-pass labeling re-introduces the ambiguity you designed
in, and doubles cost.)

## 5. Perturbation & contamination (perturb.py, contamination.py)

- Perturbation's iron rule: **transform X, never Y.** A typo doesn't change
  what was spent; deleting a currency symbol would. Every perturbation must
  be label-preserving or it's corrupting training data.
- Contamination is the report-killer: near-duplicate of a test input in
  train inflates E2/E3 and one reviewer question collapses the claim.
  Token-set fuzzy matching with digits normalized to `#` catches the
  "same message, different amount" duplicate class that generators produce.

**Self-check:** why normalize digits before fuzzy matching?

## 6. Chat templates — the silent killer (data/format.py, systems.py)

Models are trained on text rendered by a *chat template* (ChatML-style for
Qwen: `<|im_start|>role ... <|im_end|>`). Three rules:

1. Always render with `tokenizer.apply_chat_template`, never by hand.
2. Training and inference must use the **same** template (we guarantee it
   by routing both through `prompts.build_messages`).
3. At inference add `add_generation_prompt=True` (appends the assistant
   header the model expects to continue from); at training time don't.

Template mismatch is the classic "loss went down but the model outputs
garbage" bug.

## 7. LoRA / QLoRA (train/sft_unsloth.py)

- **Full fine-tuning** updates all ~4B weights → needs optimizer states for
  all of them (Adam: +8 bytes/param) → ~50GB+. Not a T4 story.
- **LoRA**: freeze W, learn ΔW = B·A with rank r≪d. Trainable params drop
  to <1%. Intuition: task adaptation lives in a low-dimensional subspace —
  you're steering the model, not re-teaching it.
- **QLoRA** = LoRA + storing the *frozen* base in 4-bit NF4. Gradients
  never flow through quantized weights (they're dequantized on the fly for
  forward passes); the trainable adapter stays bf16/fp16. Result: 7B-class
  training in ~5-6GB VRAM.
- The memory stack on a T4, and which trick attacks which piece:
  - weights → 4-bit NF4 (+ double quantization)
  - optimizer states → 8-bit Adam (`adamw_8bit`)
  - activations → gradient checkpointing (recompute in backward)
  - batch memory → gradient accumulation (small real batch, big effective)

**Self-check:** why is the LoRA learning rate (2e-4) ~100× higher than a
full-fine-tune LR? Why does `alpha=r` mean "scale 1.0"?

## 8. What Unsloth actually is (train/sft_trl_reference.py)

Not a new algorithm — an *implementation*: hand-written Triton kernels that
fuse operations (RoPE, RMSNorm, MLP, cross-entropy) HuggingFace executes as
separate GPU launches, plus smarter recomputation. Same math, ~2× speed,
~50-70% less VRAM. The raw-TRL reference run exists to make you *see* that:
identical loss curve shape, different tokens/sec and peak VRAM. That
distinction — algorithm vs. engineering — is the lesson.

## 9. DPO (train/dpo_unsloth.py, harvest_failures.py)

SFT teaches "here is the right answer". DPO teaches "prefer THIS over
THAT" — useful precisely when the model already produces *plausible but
wrong* outputs, which is what your harvested failures are.

- The loss raises the chosen-vs-rejected log-prob margin **relative to a
  frozen reference** (your SFT model). β controls the pull; the reference
  keeps the policy from wandering.
- Rejected samples must be *plausible* — the model's own dev-set mistakes.
  Corrupting gold labels randomly gives pairs the model already separates,
  i.e. zero gradient signal.
- The known failure mode (PRD §9): over-optimizing the margin degrades
  general behavior. Mitigation: β=0.1, one epoch, judge E3 vs E2 on dev
  before believing anything.
- The PEFT trick that makes it fit on a T4: reference log-probs come from
  the same model with the adapter *disabled* — no second model in memory.

**Self-check:** derive what happens to the DPO gradient when the model
already ranks chosen ≫ rejected. (σ saturates → gradient ≈ 0 → easy pairs
teach nothing. This is why harvesting real failures matters.)

## 10. Quantization for deployment (train/export_gguf.md)

Two different 4-bits in this project — don't conflate them in the viva:

| | NF4 (training) | Q4_K_M GGUF (deployment) |
|---|---|---|
| purpose | hold *frozen* base during QLoRA | shrink the *final merged* model |
| gradients | flow around it (through adapter) | none — inference only |
| ecosystem | bitsandbytes/CUDA | llama.cpp/LM Studio, runs on Mac |

E5 measures what Q4 costs on *your* task — JSON emission can be more
quantization-brittle than chat (one corrupted token breaks the parse).

## 11. Failure analysis (Week 7 — the most valuable section)

For every E3 test error, assign one taxonomy label (start from:
ambiguous-currency, date-resolution, category-confusion, missed-expense,
hallucinated-expense, policy-misapplication, schema-break) by reading
`results/e3_predictions.jsonl`. Count them. Show 2-3 verbatim examples per
bucket. This section is what separates eval work from homework — it tells
the reader *when* to trust the model, which is worth more than the F1.

## 12. The stretch ablation: your own mini scaling law

Train SFT at 250/500/1000/1500 examples (same hyperparams, same seed),
eval each on dev — plot accuracy vs. data size. If the curve is still
rising at 1500, more data is the cheapest win; if it plateaued at 500,
you've learned your task's data efficiency. Professors and interviewers
both love this because it's the *method* of scaling-law papers at toy cost.
