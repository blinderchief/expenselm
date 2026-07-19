"""Grader tests — grade KNOWN outputs and assert exact metric values.

An eval harness is itself a program that can be wrong, and a wrong grader
is worse than no grader (confident nonsense). These tests are the
harness's own eval.
"""

import json

from expenselm.eval.grader import aggregate, extract_json, grade_example, parse_prediction

GOLD_TWO = {
    "expenses": [
        {"amount": 800.0, "currency": "INR", "category": "meals", "merchant": None,
         "date": None, "description": "lunch", "reimbursable": True,
         "policy_flags": ["missing_receipt"]},
        {"amount": 350.0, "currency": "INR", "category": "travel", "merchant": None,
         "date": None, "description": "cab back", "reimbursable": True,
         "policy_flags": []},
    ],
    "confidence": "high",
}


def as_output(record: dict) -> str:
    return json.dumps(record)


def test_extract_json_from_fenced_block():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
    assert json.loads(extract_json(text)) == {"a": 1}


def test_extract_json_balanced_braces():
    text = 'prefix {"a": {"b": 2}} suffix'
    assert json.loads(extract_json(text)) == {"a": {"b": 2}}


def test_perfect_prediction_scores_one():
    g = grade_example("x", as_output(GOLD_TWO), GOLD_TWO)
    assert g.schema_valid and g.n_matched == 2
    m = aggregate([g])
    assert m["expense_f1"] == 1.0
    assert m["policy_accuracy"] == 1.0
    assert all(v == 1.0 for v in m["field_accuracy"].values())


def test_missed_expense_hits_recall_not_precision():
    pred = {"expenses": [GOLD_TWO["expenses"][0]], "confidence": "high"}
    m = aggregate([grade_example("x", as_output(pred), GOLD_TWO)])
    assert m["expense_precision"] == 1.0
    assert m["expense_recall"] == 0.5


def test_hallucinated_expense_hits_precision():
    pred = json.loads(json.dumps(GOLD_TWO))
    pred["expenses"].append(
        {"amount": 999.0, "currency": "INR", "category": "other", "merchant": None,
         "date": None, "description": "ghost", "reimbursable": True, "policy_flags": []})
    m = aggregate([grade_example("x", as_output(pred), GOLD_TWO)])
    assert m["expense_recall"] == 1.0
    assert m["expense_precision"] < 1.0


def test_wrong_category_still_matches_but_field_drops():
    """Pass-2 (amount-only) matching: category confusion is a field error,
    not a missed expense."""
    pred = json.loads(json.dumps(GOLD_TWO))
    pred["expenses"][1]["category"] = "other"
    g = grade_example("x", as_output(pred), GOLD_TWO)
    assert g.n_matched == 2
    m = aggregate([g])
    assert m["expense_f1"] == 1.0
    assert m["field_accuracy"]["category"] == 0.5


def test_wrong_flags_break_policy_only():
    pred = json.loads(json.dumps(GOLD_TWO))
    pred["expenses"][0]["policy_flags"] = []  # dropped missing_receipt
    m = aggregate([grade_example("x", as_output(pred), GOLD_TWO)])
    assert m["policy_accuracy"] == 0.5
    assert m["field_accuracy"]["reimbursable"] == 1.0


def test_amount_tolerance():
    pred = json.loads(json.dumps(GOLD_TWO))
    pred["expenses"][0]["amount"] = 800.005  # within ±0.01
    g = grade_example("x", as_output(pred), GOLD_TWO)
    assert g.n_matched == 2


def test_garbage_output_fails_schema_and_zeroes_recall():
    g = grade_example("x", "I think the expense is about lunch?", GOLD_TWO)
    assert not g.schema_valid and g.parse_failure == "no_json_found"
    m = aggregate([g])
    assert m["schema_validity"] == 0.0
    assert m["expense_recall"] == 0.0


def test_schema_violation_reason_recorded():
    bad = json.loads(json.dumps(GOLD_TWO))
    bad["expenses"][0]["policy_flags"] = ["made_up_flag"]
    rec, failure = parse_prediction(as_output(bad))
    assert rec is None and failure.startswith("schema_violation:")
