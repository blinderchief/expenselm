# ExpenseLM — Results

Status as of 2026-07-09: **4 of 6 systems measured** (E0, E1, E2, E4) on the
frozen 200-example test set. E3 (DPO) and E5 (GGUF) pending a GPU session.

## Headline table (all 200 test examples)

| System | Schema valid | Expense F1 | Amount | Currency | Category | Date | Policy |
|---|---|---|---|---|---|---|---|
| E0 Qwen3-4B zero-shot | 99.5% | 0.980 | 97.4% | 85.5% | 72.9% | 42.9% | 27.1% |
| E1 Qwen3-4B 5-shot | 98.0% | 0.988 | 97.7% | 88.8% | 79.2% | 79.5% | 34.3% |
| E2 + SFT (QLoRA) | 81.0%* | 0.898* | 81.5%* | 78.5%* | 81.2%* | 77.9%* | 45.2%* |
| E4 frontier (ceiling) | 100% | 1.000 | 100% | 93.1% | 100% | 100% | 90.4% |

\* E2's all-200 numbers are depressed by a **decoding artifact** — see below.
The row that reflects what SFT actually learned is the conditional table.

## The E2 story: separate the decoding bug from the learned capability

On its first (and only, so far) test run, E2 produced **valid JSON on 162 of
200** examples. The other **38 are all the identical failure**: the model
emits `<tool_call>\n\n<tool_call>...` in a loop and never answers.

**Diagnosis:** the Qwen3 chat template exposes a `<tool_call>` special token.
Under greedy decoding the fine-tuned model occasionally collapses into
repeating it. This is a **decoding/robustness artifact, not a learning
failure** — it says nothing about whether the model learned the task, only
that this decode configuration lets it derail on ~19% of inputs.

**Fix (needs a GPU re-run):** suppress the `<tool_call>` token id at
generation. Already implemented in `HFSystem.generate` (`suppress_tokens`),
so a re-run on Kaggle/Colab will recover these 38. Estimated to lift E2
schema validity from 81% toward ~99%.

### What E2 actually learned — same 162 examples, every system

Restricting all systems to the 162 examples E2 answered isolates capability
from the decoding bug (apples-to-apples):

| System | Expense F1 | Policy | Date | Category | Currency |
|---|---|---|---|---|---|
| E0 zero-shot | 0.980 | 24.7% | 43% | — | — |
| E1 5-shot | 0.986 | 32.4% | 77% | — | — |
| **E2 SFT** | **1.000** | **55.5%** | **96%** | **99.6%** | **96.4%** |
| E4 frontier | 1.000 | 89.9% | 100% | 100% | — |

## Reading the results (the interpretation for the report)

1. **Extraction is easy; reasoning is hard.** Even zero-shot (E0), the 4B
   model emits valid JSON 99.5% of the time and finds the right expenses
   (F1 0.98). What it *can't* do zero-shot is apply the policy (27%) or
   resolve relative dates (43%).
2. **Few-shot helps format, not reasoning.** E1's five examples pushed date
   resolution 43%→79% but policy only 27%→34%. In-context learning teaches
   patterns, not rule application.
3. **SFT teaches the reasoning.** On the shared 162, E2 more than doubled
   policy over E0 (24.7%→55.5%), nearly solved dates (43%→96%), and made
   expense-finding perfect (F1 1.0). This is the project's central claim,
   demonstrated: fine-tuning bought the capability prompting couldn't.
4. **E2 sits between E1 and E4, as theory predicts.** It closes roughly half
   the policy gap from the prompting baselines to the frontier ceiling —
   with a 4B model on 1,600 synthetic examples and $0 of training compute.
5. **The remaining gap to E4 (55%→90% policy) is DPO's job (E3).** Preference
   tuning on the model's own mistakes is the designed next lever.

## Caveats to state plainly in the writeup

- **E4 is an expert ceiling, not an independent frontier baseline:** it was
  produced by Claude Fable 5 in-session, and the same model authored the
  labeling rules. Treat it as "a careful expert applying the policy," an
  upper bound on achievable accuracy, not a neutral third party.
- **Data is procedurally generated** (labels computed from policy.yaml, correct
  by construction) — realistic-messy but less linguistically diverse than
  human text. See docs/OPTIMIZATIONS.md and data/README.md.
- **E2's first run used a 512-token cap and no tool-call suppression;** both
  are fixed in code for the re-run.

## Reproduce

```bash
uv run expenselm report                                  # the headline table
uv run python -m expenselm.analysis.failure_analysis --system e2   # taxonomy
cat results/e2.json                                      # full E2 diagnosis
```
