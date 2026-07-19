# ExpenseLM

**A ~4B open model, fine-tuned to turn messy expense text into policy-checked
JSON — with a rigorous eval harness to prove whether it beats frontier-model
pricing.** Semester project + post-training learning vehicle + portfolio
artifact. Full spec: [expenselm-prd.md](expenselm-prd.md).

> *Success looks like: one table, six rows, five metrics, no vibes.*

## Results (4 of 6 systems done — full analysis in [RESULTS.md](RESULTS.md))

| System | Schema valid | Expense F1 | Amt/Cur/Cat/Date | Policy |
|---|---|---|---|---|
| E0 Qwen3-4B zero-shot | 99.5% | 0.980 | 97.4/85.5/72.9/42.9 | 27.1% |
| E1 Qwen3-4B 5-shot | 98.0% | 0.988 | 97.7/88.8/79.2/79.5 | 34.3% |
| E2 + SFT (QLoRA) | 81.0%* | 0.898* | 81.5/78.5/81.2/77.9* | 45.2%* |
| E3 + DPO | _pending GPU_ | | | |
| E4 frontier (ceiling) | 100% | 1.000 | 100/93.1/100/100 | 90.4% |
| E5 quantized GGUF | _pending GPU_ | | | |

\* E2's raw numbers are held down by a `<tool_call>` decoding artifact on 19%
of outputs (fixed in code). **On the 162 examples it answered, E2 scored F1
1.000, policy 55.5%, date 96%** — more than doubling the base model's policy
accuracy and nearly matching the frontier on extraction. The headline finding:
**SFT taught the policy-reasoning that prompting alone couldn't.** See
[RESULTS.md](RESULTS.md) for the apples-to-apples comparison and interpretation.

Regenerate the table anytime with `uv run expenselm report`.

## Repo map

```
expenselm-prd.md            the spec — read first
configs/policy.yaml         the expense policy (single source of truth)
src/expenselm/
  schema.py                 pydantic output contract        ⚠ rewrite yourself
  prompts.py                one prompt builder for all systems
  data/                     generate → perturb → decontaminate → format
  eval/grader.py            the metrics                     ⚠ rewrite yourself
  eval/harness.py           one command → metrics table
  systems.py                E0..E5 behind one interface
  train/                    Unsloth SFT · raw-TRL reference · DPO · GGUF export
data/seed/                  12 worked exemplars             ⚠ write your own 50
tests/                      the harness's own eval — keep green
docs/LEARNING_GUIDE.md      every concept, in pipeline order
docs/OPTIMIZATIONS.md       DeepSeek / DeepSpeed / Unsloth — what applies here
docs/FLASHCARDS.md          the milestone "what I now understand" log
docs/RUNBOOK.md             ▶ START HERE — where everything runs, phase by phase
notebooks/expenselm_colab.ipynb   the Colab GPU notebook (E0/E1, SFT, DPO, GGUF)
```

⚠ = PRD §10: the three things you must write yourself to own the project.

## Quickstart (Mac)

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest               # harness self-test — must be green before anything else
```

(All `expenselm ...` commands below run as `uv run expenselm ...`.)

## The pipeline, in order

```bash
# Week 1-2 — data
#   (write your 50 seeds into data/seed/ first)
export GEMINI_API_KEY=...   # aistudio.google.com/apikey — free tier is enough
expenselm gen --n 1900 --out data/raw_generations/v1.jsonl
expenselm perturb --in data/raw_generations/v1.jsonl --out data/master.jsonl --rate 0.5
python -c "from expenselm.data.format import split; split('data/master.jsonl')"
expenselm decontaminate --train data/splits/train.jsonl --test data/splits/test.jsonl
#   → then HUMAN-VERIFY: 100% of test+dev, 20% of train. Log corrections.

# Week 3 — baselines BEFORE any training
expenselm eval --system e4 --split data/splits/test.jsonl          # Claude ceiling
expenselm eval --system e0 --split data/splits/test.jsonl          # on Colab GPU
expenselm eval --system e1 --split data/splits/test.jsonl

# Week 4 — SFT (on Colab: src/expenselm/train/sft_unsloth.py) → E2
expenselm format --in data/splits/train.jsonl --out data/chatml/train.jsonl
# Week 5 — DPO: harvest_failures.py → dpo_unsloth.py → E3
# Week 6 — GGUF export → LM Studio → E5 (src/expenselm/train/export_gguf.md)

expenselm report   # the table
```

## Non-goals (scope fence — PRD §3)

No web app. No models >8B. No multi-task. One task, six systems, one table.
