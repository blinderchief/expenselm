"""Week 7 — failure analysis: "the most valuable section" (PRD §7).

Reads a system's saved predictions (results/{system}_predictions.jsonl) and
assigns every imperfect example to a taxonomy bucket, then prints counts
and verbatim examples ready to paste into the report.

The taxonomy is AUTO-ASSIGNED here as a first pass; the PRD's requirement
is that YOU then read the examples in each bucket and correct/refine the
assignment — automation finds the buckets, judgment explains them. The
"why did it do that" sentence per bucket is the part that can't be scripted.

Run:
    python -m expenselm.analysis.failure_analysis --system e3
    python -m expenselm.analysis.failure_analysis --system e2 --examples 3
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from expenselm.eval.grader import grade_example, parse_prediction

TAXONOMY_ORDER = [
    "schema_break",           # unparseable / contract-violating output
    "missed_expense",         # gold expense with no matching prediction
    "hallucinated_expense",   # predicted expense matching nothing in gold
    "category_confusion",     # matched, wrong category
    "currency_error",         # matched, wrong currency (incl. UNKNOWN-vs-INR)
    "date_resolution",        # matched, wrong/missing date
    "merchant_error",         # matched, wrong merchant
    "policy_misapplication",  # reimbursable or flag-set wrong
]


def categorize(record: dict) -> list[str]:
    """All applicable taxonomy labels for one example (may be several)."""
    labels = []
    raw, gold = record["raw_output"], record["gold"]
    pred, failure = parse_prediction(raw)
    if pred is None:
        return [f"schema_break ({failure})"]

    g = grade_example(record["id"], raw, gold)
    if g.n_matched < g.n_gold:
        labels.append("missed_expense")
    if g.n_pred > g.n_matched:
        labels.append("hallucinated_expense")
    fc = g.field_correct
    if fc["category"] < g.n_matched:
        labels.append("category_confusion")
    if fc["currency"] < g.n_matched:
        labels.append("currency_error")
    if fc["date"] < g.n_matched:
        labels.append("date_resolution")
    if fc["merchant"] < g.n_matched:
        labels.append("merchant_error")
    if g.policy_correct < g.n_matched:
        labels.append("policy_misapplication")
    return labels


def analyze(system: str, n_examples: int = 2) -> str:
    path = Path("results") / f"{system}_predictions.jsonl"
    records = [json.loads(l) for l in open(path)]

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    perfect = 0
    for r in records:
        labels = categorize(r)
        if not labels:
            perfect += 1
        for lb in labels:
            buckets[lb].append(r)

    lines = [f"# Failure analysis — {system} ({len(records)} examples, "
             f"{perfect} fully correct)\n"]
    lines.append("| bucket | count | share of examples |")
    lines.append("|---|---|---|")
    for key in TAXONOMY_ORDER:
        matching = [k for k in buckets if k.startswith(key)]
        n = sum(len(buckets[k]) for k in matching)
        if n:
            lines.append(f"| {key} | {n} | {n/len(records):.1%} |")
    lines.append("")

    for key in TAXONOMY_ORDER:
        for k in sorted(k for k in buckets if k.startswith(key)):
            lines.append(f"## {k} — {len(buckets[k])} examples\n")
            for r in buckets[k][:n_examples]:
                lines.append(f"**{r['id']}**")
                lines.append(f"- input: `{r['input'][:160]}`")
                lines.append(f"- predicted: `{r['raw_output'][:200]}`")
                lines.append(f"- gold: `{json.dumps(r['gold'], ensure_ascii=False)[:200]}`\n")
            lines.append("_Why (fill in by hand after reading the bucket):_\n")

    out = "\n".join(lines)
    report_path = Path("results") / f"failure_analysis_{system}.md"
    report_path.write_text(out)
    print(out[:3000])
    print(f"\n(full analysis -> {report_path})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--examples", type=int, default=2, help="verbatim examples per bucket")
    args = ap.parse_args()
    analyze(args.system, args.examples)
