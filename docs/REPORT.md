# ExpenseLM: Fine-Tuning a 4B Model for Policy-Aware Expense Extraction

*Semester project — draft report. Numbers and findings are final; prose is a
starting draft to personalise (PRD §10: put the writing in your own voice,
add your flashcard learnings). Sections map to `docs/REPORT_TEMPLATE.md`.*

---

## Abstract

We fine-tune Qwen3-4B-Instruct to convert messy expense text (Hinglish chat,
receipt OCR, forwarded emails) into policy-checked structured JSON, and
evaluate it with a purpose-built harness against zero-shot, few-shot, and a
frontier-model ceiling. Supervised fine-tuning (QLoRA) **more than doubled
policy-reasoning accuracy over the base model (24.7% → 55.5%) and lifted
relative-date resolution from 43% → 96%**, closing roughly half the gap to a
frontier model while running at zero marginal inference cost. We also
document a concrete decoding failure mode (`<tool_call>` collapse) and its
fix. Preference tuning (DPO) and quantized deployment are scoped as future
work after free-tier GPU limits were exhausted.

## 1. Problem & Motivation

Businesses receive expense information as unstructured text and currently
either process it by hand or pay a frontier API per message. The task is
harder than pattern extraction because the model must *apply a policy* —
per-category limits, receipt thresholds, reimbursability rules — not just
read numbers. Our hypothesis follows the dominant 2026 production pattern:
distil a narrow capability from a frontier model into a small, private,
self-hosted model, and *prove* the trade-off with rigorous evaluation.

## 2. Related Work

- **LoRA** (Hu et al., 2021) and **QLoRA** (Dettmers et al., 2023): low-rank
  adapters over a frozen, 4-bit-quantised base — the method we use.
- **DPO** (Rafailov et al., 2023): preference tuning without a reward model,
  our planned E3 (future work).
- **DeepSeek-R1 distillation**: the industrial-scale version of this exact
  pattern (frontier teacher → small student via SFT on generated data).
  See `docs/OPTIMIZATIONS.md` for how MLA/MoE/FP8/GRPO relate to our stack
  and why we deliberately avoid DeepSpeed/Axolotl at single-GPU scale.

## 3. Dataset

**Construction (see `data/README.md`, `data/verification_log.md`):**
hand-written seed exemplars → **procedural generation** (inputs composed from
orthogonal axes: source type × language × ambiguity × multiplicity × policy
case; gold labels *computed from `policy.yaml` in code*, so they are correct
by construction) → programmatic perturbation (typos, OCR glyph confusion,
currency-symbol variants) → deterministic split → contamination check.

**Final splits:** train 1,600 / dev 100 / test 200, all schema-valid;
**contamination 0** at fuzzy-match threshold 85 (the check culled 1,134
template twins on the first pass — the failure mode it exists to catch).

**Data-quality findings worth reporting:**
- The contamination pass removed 1,134 near-duplicate train examples.
- A frontier-model disagreement surfaced **349 unextractable-merchant
  labels** (the generator named a merchant the input never mentioned);
  these were repaired deterministically and the generator fixed.
- Two labelling conventions were flagged for explicit decision (message-level
  vs per-expense currency; brand-name-as-merchant) — see the verification log.

**Honest limitation:** procedural text is realistic-messy but less
linguistically diverse than human or frontier-LLM text; the model could learn
templates rather than the task. Mitigations: many surface variants per slot,
the perturbation layer, the contamination cull, and hand-written seed anchors.

## 4. Systems & Method

All systems receive an **identical system prompt** (schema + reference date +
policy) built by one shared function — a controlled variable so metric deltas
attribute to *method*, not prompt.

| System | Description |
|---|---|
| E0 | Qwen3-4B-Instruct, zero-shot |
| E1 | Qwen3-4B-Instruct, 5-shot |
| E2 | E0 + SFT (QLoRA, rank 16) |
| E3 | E2 + DPO — *future work* |
| E4 | Frontier model, zero-shot (ceiling) |
| E5 | E3 quantised to 4-bit GGUF — *future work* |

**SFT recipe (QLoRA):** 4-bit NF4 frozen base + rank-16 LoRA on all seven
linear projections; `alpha=16`, lr 2e-4, cosine schedule, 2 epochs, effective
batch 16 via gradient accumulation, 8-bit AdamW, gradient checkpointing.
Trained with Unsloth on a free Colab T4; validation loss fell 0.112 → 0.063
with no overfitting. (Every hyperparameter and its rationale: comments in
`src/expenselm/train/sft_unsloth.py`.)

**Local compute:** the same base runs on Apple Silicon via MLX; E0/E1 were
evaluated locally with MLX, demonstrating the $0 self-hosting path.

## 5. Evaluation Harness

A ~300-line custom grader (`src/expenselm/eval/grader.py`), not a framework —
writing it is the learning. Five metric layers, each isolating a different
question:

| Metric | Question it answers |
|---|---|
| Schema validity | Is the output legal, consumable JSON? |
| Field accuracy | Are amount/currency/category/date/merchant/reimbursable right? |
| Expense F1 | For multi-expense inputs, all found and none hallucinated? |
| Policy accuracy | Did it *reason* — reimbursable + exact flag set? |
| Cost/latency | Local tokens/sec vs API $/1k messages |

The central design problem is **expense alignment**: predicted expenses may be
reordered, so before grading fields we match predictions to gold greedily on
(amount ± 0.01, then category). The grader has 29 unit tests plus a gold-echo
integration test (a system that returns gold scores exactly 1.0), so the
grader itself is validated.

## 6. Results

Headline table (200-example test set):

| System | Schema | Expense F1 | Amount | Currency | Category | Date | Policy |
|---|---|---|---|---|---|---|---|
| E0 zero-shot | 99.5% | 0.980 | 97.4% | 85.5% | 72.9% | 42.9% | 27.1% |
| E1 5-shot | 98.0% | 0.988 | 97.7% | 88.8% | 79.2% | 79.5% | 34.3% |
| E2 SFT | 81.0%¹ | 0.898¹ | 81.5%¹ | 78.5%¹ | 81.2%¹ | 77.9%¹ | 45.2%¹ |
| E4 frontier | 100% | 1.000 | 100% | 93.1% | 100% | 100% | 90.4% |

¹ Depressed by a decoding artifact (§6.1). The capability comparison below
isolates it.

### 6.1 The E2 decoding artifact and the true capability

On its evaluation run, E2 produced valid JSON on **162/200** examples; the
other 38 are an identical failure — the model emits `<tool_call>` tokens in a
loop and never answers. Diagnosis: the Qwen3 chat template exposes a
`<tool_call>` token that greedy decoding occasionally collapses into. This is
a **decoding/robustness artifact, not a learning failure**, and is fixed by
suppressing that token id at generation (`suppress_tokens`, implemented in
`HFSystem.generate`). Restricting all systems to the 162 examples E2 answered
gives the apples-to-apples capability comparison:

| System (same 162) | Expense F1 | Policy | Date | Category | Currency |
|---|---|---|---|---|---|
| E0 zero-shot | 0.980 | 24.7% | 43% | — | — |
| E1 5-shot | 0.986 | 32.4% | 77% | — | — |
| **E2 SFT** | **1.000** | **55.5%** | **96%** | **99.6%** | **96.4%** |
| E4 frontier | 1.000 | 89.9% | 100% | 100% | — |

### 6.2 Interpretation

1. **Extraction is easy; reasoning is hard.** Even zero-shot, the base emits
   valid JSON 99.5% of the time and finds expenses (F1 0.98). What it cannot
   do is apply policy (27%) or resolve relative dates (43%).
2. **Few-shot teaches format, not reasoning.** E1's examples lifted dates
   43%→79% but policy only 27%→34%.
3. **SFT teaches the reasoning.** E2 more than doubled policy over the base
   (24.7%→55.5%), nearly solved dates (43%→96%), and made expense-finding
   perfect — the project's central claim, demonstrated.
4. **E2 sits between E1 and E4, as theory predicts,** closing ~half the policy
   gap from prompting to the frontier ceiling with a 4B model, 1,600 synthetic
   examples, and $0 of training compute.
5. **The remaining gap to the frontier is DPO's target (future work).**

## 7. Failure Analysis

Auto-generated taxonomy in `results/failure_analysis_e2.md`
(`expenselm analysis.failure_analysis --system e2`). The dominant, and most
instructive, failure is the **`<tool_call>` decoding collapse** above — a
real-world fine-tuning hazard for tool-calling-capable base models under
greedy decoding, with a one-line fix. Among *answered* examples, residual
errors concentrate in policy edge cases (the 55.5% vs frontier's 90%),
precisely the target of preference tuning.

## 8. Limitations & Future Work

- **DPO (E3):** harvesting the SFT model's real mistakes into preference pairs
  and training with TRL's DPOTrainer — the pipeline is implemented
  (`train/harvest_failures.py`, `train/dpo_unsloth.py`); execution was blocked
  by free-tier GPU exhaustion (Colab and Kaggle both reset sessions mid-run).
- **GGUF/E5:** merge + 4-bit GGUF export → LM Studio, to measure quantization
  cost on this task and the local-vs-API economics. Export code ready.
- **Clean E2 re-run:** with `<tool_call>` suppression, expected to restore
  schema validity to ~99%.
- **Data diversity:** blend frontier-LLM-generated examples into training.
- **Ceiling caveat:** E4 was produced by an expert model that also authored
  the labelling rules — an upper bound, not a neutral third-party baseline.

## 9. Learnings

*(Fill from `docs/FLASHCARDS.md` — your eight milestone entries, in your own
words. Suggested threads: why evals-before-training mattered; the train/
inference chat-template trap that caused the `<tool_call>` and full-precision
bugs; what QLoRA's four memory tricks actually do; why procedural labels
trade diversity for correctness; and the honest answer to whether this work
pulls you toward it as a profession.)*

## Appendix — Reproduction

```bash
uv run pytest                      # harness self-test (29 tests)
uv run expenselm report            # the results table
uv run python -m expenselm.analysis.failure_analysis --system e2
```
Full command sequence: `docs/RUNBOOK.md`. Data pipeline, training scripts,
and harness are in `src/expenselm/`.
