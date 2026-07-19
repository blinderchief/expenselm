"""Synthetic data generation via a frontier model (PRD §5, step 2 — Gemini).

STRATEGY — the anti-mode-collapse design:
LLMs asked for "20 diverse examples" 100 times produce 2,000 near-clones.
The fix (PRD §5): make diversity EXPLICIT by sampling a scenario from
orthogonal axes for every batch, so the generator is forced into a
different corner of the distribution each call:

    source type × language style × ambiguity × multi-expense × policy case

Your 50 hand-written seed examples anchor the distribution: 3 random seeds
are shown to the generator each batch as style references, so synthetic
data stays in the same "world" your seeds define.

Every generated example is validated against the pydantic schema before
being kept — malformed gold labels never enter the dataset.

COST: using the Gemini API (GEMINI_API_KEY env var). gemini-flash-latest's
free tier covers the ~95 batches for 1,900 examples at $0 — you may hit
requests-per-minute limits; the loop just runs slower, not worse. For the
E4 *baseline* we use the stronger gemini-2.5-pro (see systems.py) so the
"frontier ceiling" row is a real ceiling.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from expenselm.eval.grader import extract_json
from expenselm.prompts import build_system_prompt
from expenselm.schema import ExpenseRecord

# ---- the diversity axes (PRD §5). Add axes -> multiply coverage. ----------
SOURCE_TYPES = [
    "casual chat message to a manager",
    "forwarded email body with signature and quoted thread as noise",
    "receipt OCR output with broken lines, ALL CAPS and garbled characters",
    "voice-note transcription with filler words",
    "terse SMS-style note",
]
LANGUAGE_STYLES = [
    "plain English",
    "Hinglish (Hindi words in Latin script mixed with English, e.g. 'bhai 450 ka cab liya airport tak')",
    "formal corporate English",
    "abbreviated/slangy English with typos",
]
AMBIGUITY_TYPES = [
    "no ambiguity — everything explicit",
    "currency missing (bare numbers)",
    "relative date only ('yesterday', 'last Friday')",
    "unclear whether reimbursable (no purpose stated)",
    "amount written in words or Indian notation (e.g. '1.2k', '₹1,20,000')",
]
MULTI = ["exactly one expense", "two expenses", "three or more expenses"]
POLICY_CASES = [
    "all under limits, clearly reimbursable",
    "one expense over its category limit (must get over_limit flag)",
    "entertainment WITH client context (reimbursable)",
    "entertainment WITHOUT client context (personal_expense)",
    "large expense with no receipt mentioned (missing_receipt)",
    "alcohol-only expense (never reimbursable)",
]

BATCH_SIZE = 20
GENERATOR_MODEL = "gemini-flash-latest"  # free tier; bump to gemini-2.5-pro if
                                         # verification finds too many bad golds


def _batch_prompt(scenario: dict, seeds: list[dict], reference_date: str) -> str:
    seed_block = "\n".join(
        json.dumps({"input": s["input"], "gold": s["gold"]}, ensure_ascii=False)
        for s in seeds
    )
    return f"""You are generating training data for an expense-extraction model.

TARGET TASK — the extractor receives this system prompt (reproduce its logic
exactly when writing gold labels):
---
{build_system_prompt(reference_date)}
---

Generate {BATCH_SIZE} examples as a JSON object {{"examples": [...]}} where each
example is {{"input": "<messy text>", "reference_date": "{reference_date}", "gold": <object matching the schema above>}}.

THIS BATCH'S SCENARIO (every example must fit ALL of these):
- source type: {scenario['source']}
- language style: {scenario['language']}
- ambiguity: {scenario['ambiguity']}
- expenses per message: {scenario['multi']}
- policy situation: {scenario['policy']}

STYLE ANCHORS — hand-written examples from the real distribution (match their
realism and messiness, do NOT copy their content):
{seed_block}

Rules:
- gold labels must be EXACTLY what a careful human applying the policy would produce
- inputs must be genuinely messy for their source type (OCR = broken lines; email = signatures)
- vary merchants, amounts, cities, names across examples; use Indian context predominantly
- output ONLY the JSON object."""


def generate_dataset(n: int, out_path: str, seed_path: str, reference_date: str = "2026-03-15") -> None:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
    seeds = [json.loads(l) for l in open(seed_path) if l.strip()]
    rng = random.Random(42)  # reproducible scenario sampling

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # RESUME: a long free-tier run WILL be interrupted (rate limits, ctrl-C,
    # laptop sleep). Each batch is appended to disk the moment it's validated,
    # so re-running the same command just continues where it stopped.
    kept: list[dict] = [json.loads(l) for l in open(out)] if out.exists() else []
    if kept:
        print(f"resuming: {len(kept)} examples already in {out_path}")
    rejected = 0
    batch_num = 0

    while len(kept) < n:
        batch_num += 1
        scenario = {
            "source": rng.choice(SOURCE_TYPES),
            "language": rng.choice(LANGUAGE_STYLES),
            "ambiguity": rng.choice(AMBIGUITY_TYPES),
            "multi": rng.choice(MULTI),
            "policy": rng.choice(POLICY_CASES),
        }
        prompt = _batch_prompt(scenario, rng.sample(seeds, min(3, len(seeds))), reference_date)

        # response_mime_type forces syntactically-valid JSON output (Gemini's
        # constrained decoding) — the schema gate below still checks the
        # CONTENT of every gold label, which constrained decoding can't do.
        # temperature 1.0 on purpose: we WANT diverse generations here
        # (contrast with temperature 0 at eval time — measurement wants
        # determinism, data generation wants variety).
        # Retry with exponential backoff: free-tier 429s are normal traffic,
        # not failures. 5 attempts spans ~2.5 min — enough to ride out any
        # per-minute quota window.
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=GENERATOR_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=1.0,
                        max_output_tokens=16000,
                    ),
                )
                text = resp.text or ""
                break
            except Exception as e:
                wait = min(60, 5 * 2**attempt)
                print(f"  API error ({e.__class__.__name__}), retry {attempt+1}/5 in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(
                "5 consecutive API failures — likely daily quota exhausted. "
                "Progress is saved; rerun the same command tomorrow to resume."
            )

        blob = extract_json(text)
        if blob is None:
            rejected += BATCH_SIZE
            continue
        try:
            examples = json.loads(blob)["examples"]
        except (json.JSONDecodeError, KeyError):
            rejected += BATCH_SIZE
            continue

        batch_kept = []
        for ex in examples:
            # Gate: gold must validate against OUR schema, or it's poison.
            try:
                ExpenseRecord.model_validate(ex["gold"])
                assert isinstance(ex["input"], str) and ex["input"].strip()
            except Exception:
                rejected += 1
                continue
            ex["id"] = f"syn-{batch_num:03d}-{len(kept):04d}"
            ex["scenario"] = scenario  # kept for coverage analysis later
            kept.append(ex)
            batch_kept.append(ex)

        # Persist immediately — a crash can never cost more than one batch.
        with open(out, "a") as f:
            for ex in batch_kept:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"batch {batch_num}: kept {len(kept)}/{n} (rejected so far: {rejected})", flush=True)

    print(f"done: {len(kept)} examples in {out_path} "
          f"(schema-rejected this run: {rejected} — inspect a few, they reveal generator confusion)")
