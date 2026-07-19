# RUNBOOK — where everything runs, in what order

Your execution map. Work top to bottom; each phase says **where** it runs,
**what** you do, and its **definition of done**. Tick the boxes.

## The map: two machines (updated 2026-07-08 — the Mac does more than planned)

| Place | What runs there | Why |
|---|---|---|
| **Your Mac (M5 Pro, MLX)** | data pipeline, verification, E4 baseline, **E0/E1 baselines, MLX-LoRA SFT → E2, DPO-pair harvesting** (`EXPENSELM_BACKEND=mlx`, script: `src/expenselm/train/sft_mlx.sh`), E5 (LM Studio), analysis, report | mlx-lm makes Apple Silicon a real training box; the PRD's "local alternative" became the primary path when free-tier quotas stalled |
| **Colab free T4** ([notebook](../notebooks/expenselm_colab.ipynb)) | **unsloth SFT + raw-TRL reference run** (the engineering-comparison learning goal), **DPO training** (needs TRL/CUDA), **GGUF export** | the pieces MLX can't do |

Backup GPU if Colab throttles you: Kaggle notebooks (free ~30 h/week of
T4/P100 — same scripts work). Note for the report: if both MLX-local and
unsloth-Colab SFT runs exist, they are two implementations of the same
recipe — comparing their E2 rows is a free robustness check.

## Status

Already done (verified on this machine):
- ✅ Repo scaffold, all pipeline code, 29 tests green (`uv run pytest`)
- ✅ Harness integration test: gold-echo system scores exactly 1.0 on every metric
- ✅ **Phase 1 (2026-07-07), via the procedural generator** (`data/procedural.py`
  — Gemini free-tier quota was too slow, so gold labels are *computed from
  policy.yaml in code*; automated text↔label audit: 0 problems in 2,400):
  - 50 seeds + 2,400 procedural → perturbed (~50%) → split
  - splits: **train 1,600 / dev 100 / test 200**, contamination = 0 hits
    (first decon pass culled 1,134 template twins; train topped up from a
    fresh-seed batch screened against frozen test+dev)
  - ChatML files ready: `data/chatml/train.jsonl`, `data/chatml/dev_chatml.jsonl`
- ⬜ OPTIONAL top-up: `uv run expenselm gen --engine gemini` when quota allows —
  mixing LLM-written examples into TRAIN improves linguistic diversity
  (never regenerate test/dev — they're frozen).

Everything below is yours to run.

---

## Phase 0 — Accounts (30 min, Mac)

- [ ] **Gemini API key** — aistudio.google.com/apikey. Add to `~/.zshrc`:
      `export GEMINI_API_KEY=...`. Free tier covers data generation
      (gemini-flash-latest); E4 uses gemini-2.5-pro — free-tier daily quota
      may split the 200-call run over 2 days, or ~$1–2 paid.
- [ ] **Weights & Biases** — wandb.ai free account (loss curves).
- [ ] **HuggingFace** — account + `hf auth login` on Colab later (model download
      + eventually uploading your adapters/GGUF, PRD deliverable 3).
- [ ] **LM Studio** — lmstudio.ai, install on the Mac (needed Week 6).
- [ ] **Google Drive** — space for ~5 GB of artifacts.

## Phase 1 — Data, Weeks 1–2 (Mac)

- [ ] **Write your 50 seed examples** into `data/seed/seed_examples.jsonl`,
      replacing my 12 exemplars (study them first — data/README.md explains
      each labeling decision). PRD §10: non-negotiable that this is you.
      Check: `uv run pytest tests/test_schema.py -q` validates all seeds.
- [ ] Generate: `uv run expenselm gen --n 1900 --out data/raw_generations/v1.jsonl`
      (~95 Gemini calls, free tier, ~30–60 min. Restartable; inspect the
      first batch before letting it run out.)
- [ ] Perturb: `uv run expenselm perturb --in data/raw_generations/v1.jsonl --out data/master.jsonl --rate 0.5`
- [ ] Split: `uv run python -c "from expenselm.data.format import split; split('data/master.jsonl')"`
- [ ] Decontaminate: `uv run expenselm decontaminate --train data/splits/train.jsonl --test data/splits/test.jsonl`
- [ ] **Human verification** — still yours, but the question changed:
      procedural gold labels are computed, so instead of "is the label
      right?" you check "is the TEXT realistic?" Read all 300 test+dev
      inputs; flag any that no human would ever write, log in
      `data/verification_log.md`, and replace stinkers with hand-written
      ones (rerun decon if you touch anything).
- [x] **FREEZE test.jsonl.** Done 2026-07-07 — read exactly once per system.

**Done when:** splits exist, verification log has entries, tests still green.

## Phase 2 — Eval baselines, Week 3 (Mac for E4; Colab for E0/E1)

- [ ] E4 on Mac: `uv run expenselm eval --system e4 --split data/splits/test.jsonl`
      Uses **gemini-flash-latest** (free tier has ZERO 2.5-Pro quota — verified
      2026-07-07, `limit: 0`). The run is resumable: quota walls / 429s / 5xxs
      only pause it; rerun the same command to continue. With paid billing you
      can redo the ceiling: `EXPENSELM_E4_MODEL=gemini-2.5-pro uv run expenselm
      eval --system e4 ...` — report whichever model filled the row.
- [ ] Zip/copy the repo folder (with `data/splits/`) to Drive as `MyDrive/expenselm`.
- [ ] Open [notebooks/expenselm_colab.ipynb](../notebooks/expenselm_colab.ipynb)
      in colab.research.google.com → GitHub/upload tab → T4 runtime.
      Run sections 0 and Week-3 (E0, E1). Copy `results/` back to Drive → Mac.
- [ ] `uv run expenselm report` — three rows filled. Stare at E4 vs E0:
      that gap is what fine-tuning must close.

## Phase 3 — Training, Weeks 4–5 (Colab notebook, sections Week-4/5)

- [ ] SFT run (~1–2 h on T4). Watch W&B: train loss down, eval loss not rising.
- [ ] E2 eval → copy adapter + results to Drive.
- [ ] Raw-TRL reference mini-run; record tokens/sec + peak VRAM vs Unsloth.
- [ ] Harvest failures (prints the failure-mode table — save it for the report),
      DPO run (~30–60 min), E3 eval.
- [ ] **Gate:** E3 vs E2 on *dev*, per failure mode, before trusting E3 (PRD §9).
- [ ] Flashcards for Weeks 4 & 5 (docs/FLASHCARDS.md) — from memory.

## Phase 4 — Local deployment, Week 6 (Colab export cell, then Mac)

- [ ] GGUF export cell (Q4_K_M + Q8_0 control) → download from Drive.
- [ ] LM Studio: load the GGUF, chat with your own model, then
      Developer tab → Start Server.
- [ ] `uv run expenselm eval --system e5 --split data/splits/test.jsonl`
- [ ] Record tokens/sec (LM Studio UI) for the cost table.

## Phase 5 — Analysis & writeup, Weeks 7–8 (Mac)

- [ ] Failure analysis: read every E3 error in `results/e3_predictions.jsonl`,
      assign taxonomy labels, count, pick verbatim examples (LEARNING_GUIDE §11).
- [ ] Stretch: data-size ablation (250/500/1000/1500) — rerun the SFT notebook
      section with `!head -N train.jsonl > train.jsonl` variants.
- [ ] `uv run expenselm report` → paste the final table into README + report.
- [ ] Report (8–12 pp), blog thread, upload adapters+GGUF to HF Hub.

---

## Time & money budget

| Item | Time | Cost |
|---|---|---|
| Seeds + verification (you, keyboard) | ~2–3 days spread out | $0 |
| Data generation (Gemini flash, free tier) | ~30–60 min | $0 |
| E4 baseline (Gemini 2.5 Pro) | ~30 min | $0–2 |
| E0/E1/SFT/DPO/export (Colab T4) | ~6–8 GPU-hours over 3 weeks | $0 |
| E5 + analysis (Mac) | ~2 days | $0 |

## When something breaks (PRD §10 rule 4)

Form your own hypothesis FIRST, write it down, then debug. Common ones:
- Colab OOM → batch 1 + accumulation 16; still OOM → Qwen3-1.7B (PRD §9).
- Colab disconnects → checkpoints are on Drive; resume with
  `trainer.train(resume_from_checkpoint=True)`.
- E2 outputs garbage but loss was fine → chat-template mismatch
  (LEARNING_GUIDE §6) — check `add_generation_prompt` and that train/infer
  both go through `prompts.build_messages`.
- DPO made dev metrics worse → lower beta to 0.05, never add epochs first.
