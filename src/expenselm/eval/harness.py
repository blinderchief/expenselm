"""The harness — "one command produces the metrics table for any system".

That sentence is the Week-3 definition of done (PRD §8). Everything here is
plumbing around grader.py:

    dataset JSONL  ->  system.generate() per example  ->  grade  ->  table

It also records latency and token throughput per example, because
cost/latency is metric #5 (the business argument).

Run:
    expenselm eval --system e0 --split data/splits/test.jsonl
    expenselm report            # prints the combined E0..E5 table
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from expenselm.eval.grader import ExampleGrade, aggregate, grade_example

RESULTS_DIR = Path("results")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(system, examples: list[dict], system_name: str, limit: int | None = None) -> dict:
    """Evaluate `system` (anything with .generate(messages)->str) on examples.

    Predictions are also saved per-example so the failure analysis (PRD §7,
    "the most valuable section") can inspect raw outputs later without
    re-running inference.
    """
    from expenselm.prompts import build_messages

    if limit:
        examples = examples[:limit]

    # RESUME: predictions are appended to disk as they arrive, so an API
    # quota wall / ctrl-C / crash never loses work — rerunning the same
    # command re-grades the saved outputs (deterministic) and continues.
    RESULTS_DIR.mkdir(exist_ok=True)
    pred_path = RESULTS_DIR / f"{system_name}_predictions.jsonl"
    done: dict[str, dict] = {}
    if pred_path.exists():
        for line in open(pred_path):
            r = json.loads(line)
            done[r["id"]] = r
        if done:
            print(f"resuming: {len(done)} predictions already on disk")

    grades: list[ExampleGrade] = []
    latencies = []

    with open(pred_path, "a") as fout:
        for i, ex in enumerate(examples):
            if ex["id"] in done:
                prev = done[ex["id"]]
                grade = grade_example(ex["id"], prev["raw_output"], ex["gold"])
                grades.append(grade)
                latencies.append(prev.get("latency_s", 0.0))
                continue

            few_shot = getattr(system, "few_shot", None)
            messages = build_messages(ex["input"], ex["reference_date"], few_shot)

            t0 = time.perf_counter()
            raw = system.generate(messages)
            dt = time.perf_counter() - t0
            latencies.append(dt)

            grade = grade_example(ex["id"], raw, ex["gold"])
            grades.append(grade)
            fout.write(json.dumps(
                {"id": ex["id"], "input": ex["input"], "raw_output": raw,
                 "gold": ex["gold"], "grade": asdict(grade), "latency_s": round(dt, 3)},
                ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i+1}/{len(examples)}] {ex['id']}: "
                  f"{'ok' if grade.schema_valid else grade.parse_failure} ({dt:.1f}s)",
                  flush=True)

    metrics = aggregate(grades)
    metrics["latency_p50_s"] = round(statistics.median(latencies), 2) if latencies else None
    metrics["system"] = system_name
    metrics["model"] = getattr(system, "model", None)  # audit trail for the report

    with open(RESULTS_DIR / f"{system_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# --------------------------------------------------------------------------
# The final table — "one table, six rows, five metrics, no vibes."
# --------------------------------------------------------------------------

ROW_ORDER = ["e0", "e1", "e2", "e3", "e4", "e5"]
ROW_LABELS = {
    "e0": "E0 Qwen3-4B zero-shot",
    "e1": "E1 Qwen3-4B 5-shot",
    "e2": "E2 + SFT (QLoRA)",
    "e3": "E3 + DPO",
    "e4": "E4 frontier zero-shot (ceiling)",
    "e5": "E5 E3 quantized (GGUF Q4)",
}


def report() -> str:
    rows = []
    for name in ROW_ORDER:
        p = RESULTS_DIR / f"{name}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        fa = m["field_accuracy"]
        rows.append(
            f"| {ROW_LABELS[name]} | {m['schema_validity']:.1%} | "
            f"{m['expense_f1']:.3f} | {fa['amount']:.1%}/{fa['currency']:.1%}/"
            f"{fa['category']:.1%}/{fa['date']:.1%} | {m['policy_accuracy']:.1%} | "
            f"{str(m['latency_p50_s']) + 's' if m.get('latency_p50_s') is not None else '—'} |"
        )
    header = (
        "| System | Schema valid | Expense F1 | Amt/Cur/Cat/Date | Policy | p50 latency |\n"
        "|---|---|---|---|---|---|\n"
    )
    table = header + "\n".join(rows) if rows else "No results yet — run `expenselm eval` first."
    print(table)
    return table
