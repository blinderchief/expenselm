# ExpenseLM, explained from zero

You asked: *what are we exactly doing, why, what are the processes, how
will it work, what will the result be, and what exactly do we check in
evaluation?* This file answers those six questions in that order, in plain
language. (The deeper technical version of each topic is in
[LEARNING_GUIDE.md](LEARNING_GUIDE.md); the commands are in [RUNBOOK.md](RUNBOOK.md).)

---

## 1. WHAT are we doing?

We are teaching a small open-source AI model (Qwen3-4B — "4B" = 4 billion
parameters, small enough to run on a laptop) to do exactly ONE job:

> Read a messy piece of text about money someone spent
> ("bhai 450 ka cab liya airport tak") and produce a clean, structured
> JSON record: amount, currency, category, date, merchant, whether the
> company should reimburse it, and which policy rules it triggers.

Big models like Gemini 2.5 Pro can already do this. Our claim — the thing
the whole project proves or disproves — is:

> **A small model, fine-tuned on ~1,500 good examples of this one task,
> can match the big model's accuracy at ~zero cost per message.**

## 2. WHY are we doing it?

**The business reason.** A company processing 100k expense messages/month
through a frontier API pays real money per message, and every receipt
leaves their infrastructure. A fine-tuned 4B model runs on their own
hardware: $0 marginal cost, fully private. This "distill a narrow skill
from a big model into a small one" pattern is one of the most common
production plays in AI right now — learning to execute it end-to-end is
learning a genuinely employable skill.

**The learning reason (your primary goal).** Every stage of modern
post-training appears in this project at a size one person can actually
run and understand: dataset construction, supervised fine-tuning (SFT),
preference tuning (DPO), quantization, and — most importantly —
**evaluation**, which is the skill that separates people who *train*
models from people who know whether a model is *good*.

**Why the policy part matters.** If the task were only "pull out the
number", regex could do it. The model must also *apply rules* — is ₹7,800
over the travel limit? is bowling reimbursable if no client was there? —
which requires reading comprehension + reasoning, and that's what makes a
fine-tuned model interesting to measure.

## 3. WHAT are the processes? (the pipeline, step by step)

```
 seeds ──► synthetic generation ──► perturbation ──► split ──► decontaminate
 (50 by hand)  (Gemini writes ~1,900)  (add typos/noise)  (train/dev/test)  (remove leaks)
                                                              │
                            ┌─────────────────────────────────┘
                            ▼
                   HUMAN VERIFICATION  ◄── you, reading, fixing, logging
                            │
        ┌───────────────────┼──────────────────────┐
        ▼                   ▼                      ▼
   BASELINES first     TRAINING (Colab T4)     the frozen TEST set
   E0 zero-shot        SFT (QLoRA) ──► E2       (touched exactly once
   E1 five-shot        harvest failures          per system, at the end)
   E4 Gemini Pro       DPO ──► E3
                       quantize ──► E5 (GGUF, runs on your Mac)
                            │
                            ▼
                ONE TABLE: 6 systems × 5 metrics
```

Each step, and **why it exists**:

1. **50 hand-written seeds.** Writing them forces YOU to decide every
   ambiguous case (is Netflix "software"? what if the bill is lost?).
   Those decisions become the labeling rules in `data/README.md`. Your
   seed-013/014 experience is exactly why this step can't be skipped: your
   first labels had a wrong category, copied gold, and wrong flags — and
   now you know what "wrong" looks like *before* generating 1,900 more.
2. **Synthetic generation (Gemini).** Nobody hand-writes 1,900 examples.
   We ask a frontier model to write input+answer pairs, but pinned to a
   randomly-sampled scenario each batch (source type × language ×
   ambiguity × count × policy case) so we get genuine variety instead of
   1,900 paraphrases of one cab ride.
3. **Perturbation.** Synthetic text is too clean. We inject typos, OCR
   glyph confusion (0↔O, 1↔l), and currency-symbol variants — changing
   the *surface* while never changing what was actually spent.
4. **Split + decontamination.** 1,600 train / 100 dev / 200 test. Then we
   delete any train example that's a near-duplicate of a test example —
   otherwise the model "aces the exam because it saw the questions", and
   the whole result is fake.
5. **Human verification.** You read 100% of test+dev and 20% of train,
   fix wrong answers, and log every fix. The log becomes evidence of data
   quality in your report.
6. **Baselines BEFORE training.** We measure the untrained model (E0/E1)
   and the frontier model (E4) first, so when training lands we know
   exactly what it bought us. Training without a baseline is flying blind.
7. **SFT** — show the model 1,500 (input → correct JSON) pairs; it learns
   the task. **DPO** — collect the SFT model's *actual mistakes*, pair
   each with the correct answer, and train it to prefer right over its own
   characteristic wrong. **Quantize** — shrink the final model to ~2.5GB
   so it runs on your Mac.

## 4. HOW is it all going to work? (the mechanics, minus the math)

**Where things run.** Your Mac does everything except training (data
pipeline, API calls, final report). Google Colab's free T4 GPU does all
training — you open `notebooks/expenselm_colab.ipynb` there, and Google
Drive ferries files between the two. (You mentioned Colab in VS Code —
that works too, it's the same runtime; the browser version is simplest
because Drive mounting is one click.)

**How fine-tuning fits on a free GPU.** A 4B model normally needs ~50GB+
of GPU memory to train. We fit it in ~6GB with four tricks, each attacking
a different memory consumer:
- freeze the model and store it in 4-bit precision (weights),
- train only tiny "adapter" matrices bolted onto it — LoRA, <1% of
  parameters (gradients + optimizer),
- recompute intermediate values instead of storing them (activations),
- process 2 examples at a time but sum gradients over 8 rounds before
  updating, faking a batch of 16 (batch memory).

That combination is called **QLoRA**, and the library **Unsloth** makes it
~2× faster with hand-optimized GPU code. Same math, better engineering —
we run one training job in the plain library too, just to prove that to
ourselves.

**How DPO works, in one image.** After SFT, the model is like a student
who makes *specific, repeatable* mistakes (always calls auto rides
"other", forgets the second expense). We collect those real mistakes, show
the model pairs of (its wrong answer, the right answer), and nudge its
probabilities: right answer up, its own wrong answer down, with a frozen
copy of itself as an anchor so it doesn't drift into weirdness.

**How the model ends up on your Mac.** After training we merge the adapter
into the model, convert to the GGUF format, and compress to 4-bit. LM
Studio loads that file, and you literally chat with your own trained model
locally — then point the eval harness at it for the final row.

## 5. WHAT will the result be?

Concretely, four artifacts:

1. **One table** — 6 systems × 5 metrics on the same 200 hidden test
   examples. The headline is the E3-vs-E4 comparison: your fine-tuned 4B
   vs the frontier model. Any of three outcomes is a *good* project:
   - E3 ≈ E4 → thesis proven: small+tuned matches frontier at ~$0.
   - E3 well above E0 but below E4 → you quantify exactly what
     fine-tuning buys and what gap remains (honest negative results are
     rarer and more impressive than cherry-picked wins).
   - DPO or quantization *hurts* → you caught it with rigorous evals,
     which is the entire point of having them.
2. **A model** — LoRA adapters + a GGUF file anyone can run.
3. **A report** — methodology, the table, and a failure analysis.
4. **You, changed** — able to explain SFT/LoRA/DPO/quantization/evals
   from memory, with 8 weeks of evidence about whether this profession
   pulls you in.

## 6. WHAT exactly do we check in evaluation?

Five metrics, each answering a different question, all computed by our own
~300-line grader (`src/expenselm/eval/grader.py`) — never by eyeballing:

| # | Metric | The question | Example failure it catches |
|---|---|---|---|
| 1 | **Schema validity** | Did it output legal JSON a program can consume? | model writes "Here's the JSON:" + broken braces; invents a field `vendor_notes`; writes category "Lunch" instead of "meals" (your seed-013!) |
| 2 | **Field accuracy** (amount, currency, category, date, merchant, reimbursable — reported separately) | Are the individual values right? | date resolved to the wrong day; USD guessed as INR |
| 3 | **Expense-level F1** | For multi-expense messages, did it find ALL of them and ONLY them? | reads "lunch 800, cab 350" and returns just the lunch (recall ↓), or invents a third expense (precision ↓) |
| 4 | **Policy accuracy** | Did it *reason* about the rules — reimbursable + the exact flag set? | ₹7,800 flight recorded perfectly but `over_limit` flag missing |
| 5 | **Cost & latency** | Is it deployable — tokens/sec locally vs $ per 1k messages via API? | a model that's accurate but 10× too slow |

A subtlety worth understanding (it's the cleverest part of the grader):
before we can grade fields on a multi-expense message, we must decide
*which* predicted expense corresponds to *which* correct one — the model
may reorder them. The grader matches them by (amount within ±0.01, then
category). Take your seed-014: if the model answered with a hotel and a
dinner (like your original copy-pasted gold!), zero predictions would
match the real auto-ride expense → recall 0, precision 0, and every field
counted wrong — which is the truthful score for that answer.

And one rule above all: the **200-example test set is used exactly once
per system, at the end.** Every decision along the way (hyperparameters,
early stopping, "is DPO helping?") uses the dev set. The moment you peek
at test results and change something because of them, the test set stops
being an exam and becomes practice — and your table stops being evidence.

---

## Your immediate next steps (short version — full checklist in RUNBOOK)

1. `export GEMINI_API_KEY=...` in `~/.zshrc` (aistudio.google.com/apikey).
2. **Review the 38 seeds I wrote** (seed-015…050) the way I reviewed your
   two: check every flag against `configs/policy.yaml`, and log anything
   you'd change in `data/verification_log.md`. Disagreeing with a label
   and defending your version IS the calibration exercise.
3. `uv run expenselm gen --n 1900 --out data/raw_generations/v1.jsonl`
   — then read the first 20 generated examples before trusting the rest.
4. Perturb → split → decontaminate → verify (RUNBOOK Phase 1).
5. E4 baseline on the Mac, then the Colab notebook for E0/E1 and training.
