"""The grader — computes PRD §7 metrics for ONE (prediction, gold) pair.

============================================================================
⚠️  PRD §10: the grading logic is one of the three things you must write
    yourself. Study this, then rewrite it. The design decisions below are
    exactly the kind of thing eval engineers get asked about.
============================================================================

THE CENTRAL DESIGN PROBLEM — expense alignment:
A message can contain several expenses, and the model may emit them in any
order, miss one, or hallucinate an extra. Before you can say "the currency
field is right", you must decide WHICH predicted expense corresponds to
WHICH gold expense. That's an assignment problem.

We use greedy matching on (amount within ±0.01, same category), then relax
to amount-only. Full Hungarian assignment is overkill for ≤5 expenses per
message and greedy is trivially explainable in the report. Matching on
(amount, category) follows PRD §7 metric 3 exactly.

METRIC LAYERS (each answers a different question):
  schema_valid    — did it emit parseable, contract-conforming JSON at all?
  expense P/R/F1  — did it find the right SET of expenses?
  field accuracy  — for matched expenses, are individual fields right?
  policy accuracy — reimbursable + policy_flags (the reasoning metric)

Field accuracy is computed ONLY over matched pairs: penalising the currency
field because the whole expense was missed would double-count the miss —
the expense-level recall already caught it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from expenselm.schema import ExpenseRecord

AMOUNT_TOL = 0.01  # PRD §7: amounts match within ±0.01

FIELD_NAMES = ("amount", "currency", "category", "merchant", "date", "reimbursable")


# ---------------------------------------------------------------------------
# Step 1 — parsing: raw model text -> ExpenseRecord or a labelled failure
# ---------------------------------------------------------------------------

def extract_json(text: str) -> str | None:
    """Pull the first JSON object out of possibly-noisy model output.

    Small models love to wrap JSON in ```json fences or prepend "Here is".
    We tolerate that (every system gets the same tolerance, so it's fair),
    but the *cleanliness* still shows up indirectly: malformed JSON that
    can't be recovered fails schema validity.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # First balanced {...} span.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_prediction(raw_text: str) -> tuple[ExpenseRecord | None, str | None]:
    """Returns (record, None) on success or (None, failure_reason)."""
    blob = extract_json(raw_text)
    if blob is None:
        return None, "no_json_found"
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None, "invalid_json"
    try:
        return ExpenseRecord.model_validate(data), None
    except ValidationError as e:
        # Keep the first error type — feeds the failure taxonomy (PRD §7).
        first = e.errors()[0]
        return None, f"schema_violation:{first['type']}:{'.'.join(map(str, first['loc']))}"


# ---------------------------------------------------------------------------
# Step 2 — alignment: which predicted expense is which gold expense?
# ---------------------------------------------------------------------------

def _amount_eq(a: float, b: float) -> bool:
    return abs(a - b) <= AMOUNT_TOL


def match_expenses(pred: ExpenseRecord, gold: ExpenseRecord) -> list[tuple[int, int]]:
    """Greedy 1:1 matching. Pass 1: amount+category. Pass 2: amount only.

    Pass 2 exists so a category confusion (travel vs other) still counts as
    a *found* expense with a *wrong category field* — which is the truthful
    description — instead of one miss plus one hallucination.
    """
    pairs: list[tuple[int, int]] = []
    used_p, used_g = set(), set()

    for strict in (True, False):
        for gi, g in enumerate(gold.expenses):
            if gi in used_g:
                continue
            for pi, p in enumerate(pred.expenses):
                if pi in used_p or not _amount_eq(p.amount, g.amount):
                    continue
                if strict and p.category != g.category:
                    continue
                pairs.append((pi, gi))
                used_p.add(pi)
                used_g.add(gi)
                break
    return pairs


# ---------------------------------------------------------------------------
# Step 3 — per-example grading result
# ---------------------------------------------------------------------------

@dataclass
class ExampleGrade:
    example_id: str
    schema_valid: bool
    parse_failure: str | None = None
    # expense-level counts (for corpus micro-F1)
    n_gold: int = 0
    n_pred: int = 0
    n_matched: int = 0
    # per-field: {field: (n_correct, n_matched)}
    field_correct: dict[str, int] = field(default_factory=dict)
    # policy: reimbursable + flags jointly correct, over matched pairs
    policy_correct: int = 0
    confidence_pred: str | None = None


def grade_example(example_id: str, raw_text: str, gold_data: dict) -> ExampleGrade:
    gold = ExpenseRecord.model_validate(gold_data)  # gold must always validate
    pred, failure = parse_prediction(raw_text)

    if pred is None:
        # Schema-invalid output scores zero recall on everything: the
        # business consumer of this model can't use unparseable output.
        return ExampleGrade(
            example_id=example_id,
            schema_valid=False,
            parse_failure=failure,
            n_gold=len(gold.expenses),
        )

    pairs = match_expenses(pred, gold)
    g = ExampleGrade(
        example_id=example_id,
        schema_valid=True,
        n_gold=len(gold.expenses),
        n_pred=len(pred.expenses),
        n_matched=len(pairs),
        confidence_pred=pred.confidence,
    )

    for f in FIELD_NAMES:
        g.field_correct[f] = 0

    for pi, gi in pairs:
        p, gl = pred.expenses[pi], gold.expenses[gi]
        g.field_correct["amount"] += int(_amount_eq(p.amount, gl.amount))
        g.field_correct["currency"] += int(p.currency == gl.currency)
        g.field_correct["category"] += int(p.category == gl.category)
        g.field_correct["date"] += int(p.date == gl.date)
        g.field_correct["reimbursable"] += int(p.reimbursable == gl.reimbursable)
        # merchant: case-insensitive, None==None counts as correct
        pm = p.merchant.lower().strip() if p.merchant else None
        gm = gl.merchant.lower().strip() if gl.merchant else None
        g.field_correct["merchant"] += int(pm == gm)
        # policy metric: reimbursable AND the flag SET must both be right
        g.policy_correct += int(
            p.reimbursable == gl.reimbursable
            and set(p.policy_flags) == set(gl.policy_flags)
        )
    return g


# ---------------------------------------------------------------------------
# Step 4 — corpus aggregation (micro-averaged)
# ---------------------------------------------------------------------------

def aggregate(grades: list[ExampleGrade]) -> dict:
    n = len(grades)
    n_valid = sum(g.schema_valid for g in grades)
    tp = sum(g.n_matched for g in grades)
    total_pred = sum(g.n_pred for g in grades)
    total_gold = sum(g.n_gold for g in grades)

    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    fields = {}
    for f in FIELD_NAMES:
        correct = sum(g.field_correct.get(f, 0) for g in grades)
        # Denominator = gold expenses, NOT matched pairs: an unmatched gold
        # expense means every one of its fields was effectively wrong.
        # This makes field accuracy an end-to-end number you can quote alone.
        fields[f] = correct / total_gold if total_gold else 0.0

    policy_correct = sum(g.policy_correct for g in grades)

    return {
        "n_examples": n,
        "schema_validity": n_valid / n if n else 0.0,
        "expense_precision": round(precision, 4),
        "expense_recall": round(recall, 4),
        "expense_f1": round(f1, 4),
        "field_accuracy": {k: round(v, 4) for k, v in fields.items()},
        "policy_accuracy": round(policy_correct / total_gold, 4) if total_gold else 0.0,
        "parse_failures": sorted(
            {g.parse_failure for g in grades if g.parse_failure is not None}
        ),
    }
