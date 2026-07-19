"""Schema contract tests. If any of these break, the whole eval is suspect."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from expenselm.schema import Expense, ExpenseRecord

VALID = {
    "expenses": [
        {
            "amount": 450.0,
            "currency": "INR",
            "category": "travel",
            "merchant": None,
            "date": "2026-03-14",
            "description": "cab to airport",
            "reimbursable": True,
            "policy_flags": [],
        }
    ],
    "confidence": "high",
}


def test_valid_record_passes():
    rec = ExpenseRecord.model_validate(VALID)
    assert rec.expenses[0].amount == 450.0


def test_extra_key_rejected():
    bad = json.loads(json.dumps(VALID))
    bad["expenses"][0]["vendor_notes"] = "hallucinated field"
    with pytest.raises(ValidationError):
        ExpenseRecord.model_validate(bad)


def test_unknown_currency_literal_allowed():
    ok = json.loads(json.dumps(VALID))
    ok["expenses"][0]["currency"] = "UNKNOWN"
    ExpenseRecord.model_validate(ok)


@pytest.mark.parametrize("cur", ["inr", "RUPEES", "Rs", "IN", ""])
def test_bad_currency_shape_rejected(cur):
    bad = json.loads(json.dumps(VALID))
    bad["expenses"][0]["currency"] = cur
    with pytest.raises(ValidationError):
        ExpenseRecord.model_validate(bad)


@pytest.mark.parametrize("d", ["2026-02-30", "14-03-2026", "2026/03/14", "yesterday"])
def test_bad_dates_rejected(d):
    bad = json.loads(json.dumps(VALID))
    bad["expenses"][0]["date"] = d
    with pytest.raises(ValidationError):
        ExpenseRecord.model_validate(bad)


def test_invented_policy_flag_rejected():
    bad = json.loads(json.dumps(VALID))
    bad["expenses"][0]["policy_flags"] = ["sounds_expensive"]
    with pytest.raises(ValidationError):
        ExpenseRecord.model_validate(bad)


def test_negative_amount_rejected():
    bad = json.loads(json.dumps(VALID))
    bad["expenses"][0]["amount"] = -100
    with pytest.raises(ValidationError):
        ExpenseRecord.model_validate(bad)


def test_empty_expenses_rejected():
    with pytest.raises(ValidationError):
        ExpenseRecord.model_validate({"expenses": [], "confidence": "low"})


def test_all_seed_examples_validate():
    """Week-1 definition of done (PRD §8): schema validates all seeds."""
    seed = Path(__file__).parents[1] / "data" / "seed" / "seed_examples.jsonl"
    lines = [json.loads(l) for l in seed.read_text().splitlines() if l.strip()]
    assert len(lines) >= 10
    for ex in lines:
        ExpenseRecord.model_validate(ex["gold"])
