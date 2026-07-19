"""Build DPO preference pairs from the SFT model's REAL mistakes (PRD §5).

This is the step that makes your DPO section read like lab work instead of
a tutorial: rejected outputs are not synthetic corruptions — they are what
YOUR model actually got wrong on dev data, so DPO pushes down probability
mass exactly where your model puts it erroneously.

Pipeline:
  1. Run the SFT model (E2) over dev + a slice of train inputs.
  2. Keep outputs that are VALID JSON but WRONG — plausible failures.
     (Garbage outputs are useless for DPO: the model already assigns them
     low probability; there's no signal in pushing them lower.)
  3. Emit {prompt, chosen (gold), rejected (model's wrong answer)} JSONL.

Run AFTER E2 exists:
    python -m expenselm.train.harvest_failures \
        --splits data/splits/dev.jsonl data/splits/train.jsonl \
        --out data/dpo_pairs.jsonl --target 500
"""

from __future__ import annotations

import argparse
import json

from expenselm.eval.grader import grade_example, parse_prediction
from expenselm.prompts import build_messages


def is_useful_rejection(raw_output: str, gold: dict, example_id: str) -> tuple[bool, str]:
    """A useful rejected sample = schema-valid but substantively wrong.

    Returns (useful?, failure_mode_tag). The tag feeds your failure
    taxonomy AND lets you check DPO's before/after per failure mode
    (PRD §8 week 5: 'before/after diff on targeted failure modes').
    """
    pred, failure = parse_prediction(raw_output)
    if pred is None:
        return False, f"unusable:{failure}"  # garbage — skip

    g = grade_example(example_id, raw_output, gold)
    if g.n_matched < g.n_gold:
        return True, "missed_expense"
    if g.n_pred > g.n_gold:
        return True, "hallucinated_expense"
    fc = g.field_correct
    if fc["category"] < g.n_matched:
        return True, "wrong_category"
    if g.policy_correct < g.n_matched:
        return True, "policy_misapplication"
    if fc["date"] < g.n_matched:
        return True, "date_resolution"
    if fc["currency"] < g.n_matched:
        return True, "currency"
    if fc["merchant"] < g.n_matched:
        return True, "merchant"
    return False, "correct"  # model got it right — nothing to prefer against


def harvest(split_paths: list[str], out_path: str, target: int, adapter: str) -> None:
    import os

    if os.environ.get("EXPENSELM_BACKEND", "hf").lower() == "mlx":
        from expenselm.systems import MLXSystem

        system = MLXSystem(adapter_path=adapter)
    else:
        from expenselm.systems import HFSystem

        system = HFSystem(adapter_path=adapter)

    pairs, mode_counts = [], {}
    for path in split_paths:
        examples = [json.loads(l) for l in open(path) if l.strip()]
        for ex in examples:
            if len(pairs) >= target:
                break
            messages = build_messages(ex["input"], ex["reference_date"])
            raw = system.generate(messages)
            useful, mode = is_useful_rejection(raw, ex["gold"], ex["id"])
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if not useful:
                continue
            pairs.append({
                # TRL's DPOTrainer conversational format:
                "prompt": messages,  # system + user turns
                "chosen": [{"role": "assistant", "content": json.dumps(
                    ex["gold"], ensure_ascii=False, separators=(",", ":"))}],
                "rejected": [{"role": "assistant", "content": raw.strip()}],
                "failure_mode": mode,
                "source_id": ex["id"],
            })

    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"harvested {len(pairs)} pairs -> {out_path}")
    print("failure-mode distribution (this table goes in your report):")
    for mode, n in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode:24s} {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--out", default="data/dpo_pairs.jsonl")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--adapter", default="models/sft-adapter")
    args = ap.parse_args()
    harvest(args.splits, args.out, args.target, args.adapter)
