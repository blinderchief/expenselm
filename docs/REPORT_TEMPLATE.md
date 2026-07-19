# ExpenseLM — Semester Report Skeleton (Week 8)

Target: 8–12 pages. Every section below names its data source in this repo
so writing is assembly, not archaeology. Write the *prose* yourself (PRD
§10) — this skeleton just makes sure nothing is forgotten.

## 1. Problem & Motivation (~1 page)
- The extraction task + why policy application makes it non-trivial (PRD §1, §4)
- The distillation economics argument (docs/PROJECT_EXPLAINED.md §2)

## 2. Related Work (~1 page)
- LoRA (Hu et al. 2021), QLoRA (Dettmers et al. 2023), DPO (Rafailov et al. 2023)
- DeepSeek-R1 distillation as the industrial version of this pattern
  (docs/OPTIMIZATIONS.md has the framing + citations)
- One line each: GRPO vs DPO choice; why not DeepSpeed/Axolotl (scale mismatch)

## 3. Dataset (~1.5 pages)
- Construction pipeline: seeds → procedural generation (labels computed from
  policy, correct by construction) → perturbation → decontamination
  → repair passes. Numbers: data/README.md + verification_log.md
- **Data quality findings** (this subsection impresses reviewers):
  - the 1,134 template twins culled by the contamination check
  - the 349 unextractable-merchant labels found via frontier-disagreement
    and repaired deterministically
  - the currency-context ambiguity (message-level vs per-expense) — state
    the rule you chose and why
- Honest limitations: procedural text diversity vs LLM-generated; mitigation

## 4. Systems & Method (~2 pages)
- The fixed-prompt discipline (prompts.py — one builder for all systems)
- E0–E5 definitions; exact model/backend per row (results/*.json `model` field)
- SFT: QLoRA recipe + every hyperparameter WITH its reason
  (src/expenselm/train/sft_unsloth.py comments; MLX-local equivalent)
- Unsloth vs raw-TRL comparison numbers (tokens/sec, peak VRAM, loss shape)
- DPO: pair harvesting from real failures; β choice; the E3-vs-E2 dev gate

## 5. Evaluation Harness (~1.5 pages)
- The five metrics and what each isolates (docs/LEARNING_GUIDE.md §3 table)
- The expense-alignment problem and the greedy matching design
- Grader-of-the-grader: the 29 unit tests + gold-echo integration test
- E4 protocol note: frontier baseline run by Claude Fable 5 in-session;
  same model designed the labeling rules → read E4 as an expert ceiling

## 6. Results (~1 page)
- THE TABLE (`expenselm report`) + 3–4 sentences per comparison:
  E1−E0 (in-context value), E2−E0 (SFT value), E3−E2 (DPO value),
  E3 vs E4 (the headline), E5−E3 (quantization cost)
- Cost/latency table (API $/1k messages vs local tokens/sec)

## 7. Failure Analysis (~1.5 pages)
- Taxonomy table from `python -m expenselm.analysis.failure_analysis --system e3`
- 2–3 verbatim examples per major bucket + YOUR one-sentence "why"
- When should a user trust this model / route to a human?

## 8. (Stretch) Data-size Ablation (~0.5 page)
- Curve from `python -m expenselm.analysis.ablation table`; where it bends

## 9. Learnings (~1 page)
- Distill from docs/FLASHCARDS.md — the eight "what I now understand" entries
- The honest answer to the real question (PRD deliverable #5)

## Appendix
- Reproduction commands (docs/RUNBOOK.md), hyperparameter tables, policy.yaml
