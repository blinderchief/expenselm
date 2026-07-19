"""Week 7 stretch — the data-size ablation (your own mini scaling law).

Creates nested training subsets (250 ⊂ 500 ⊂ 1000 ⊂ full) so each bigger
run strictly ADDS data — that's what makes the curve interpretable. Train
one adapter per subset with identical hyperparameters, evaluate each on
DEV (not test! the ablation is a design study, not a headline claim), and
plot accuracy vs. size.

Usage:
    python -m expenselm.analysis.ablation prepare        # writes the subsets
    # then per size (locally on MLX, or in the Colab notebook):
    #   mlx_lm lora ... --data data/ablation/n250 --adapter-path models/abl-250
    #   EXPENSELM_BACKEND=mlx expenselm eval --system e2 ... (adapter swapped)
    python -m expenselm.analysis.ablation table          # prints results table
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SIZES = [250, 500, 1000]  # plus the full set


def prepare(train_path: str = "data/chatml/train.jsonl",
            valid_path: str = "data/chatml/dev_chatml.jsonl") -> None:
    lines = [l for l in open(train_path) if l.strip()]
    rng = random.Random(99)
    rng.shuffle(lines)          # one shuffle; prefixes give nested subsets
    valid = Path(valid_path).read_text()

    for n in SIZES + [len(lines)]:
        d = Path(f"data/ablation/n{n}")
        d.mkdir(parents=True, exist_ok=True)
        (d / "train.jsonl").write_text("".join(lines[:n]))
        (d / "valid.jsonl").write_text(valid)
        print(f"wrote {min(n, len(lines))} -> {d}")


def table() -> None:
    """Collect results/abl-*.json (produced by eval runs) into a table."""
    rows = []
    for p in sorted(Path("results").glob("abl-*.json")):
        m = json.loads(p.read_text())
        rows.append((p.stem, m["expense_f1"], m["policy_accuracy"],
                     m["field_accuracy"]["category"]))
    print("| train size | expense F1 | policy acc | category acc |")
    print("|---|---|---|---|")
    for name, f1, pol, cat in rows:
        print(f"| {name.replace('abl-', '')} | {f1:.3f} | {pol:.1%} | {cat:.1%} |")
    if not rows:
        print("(no abl-*.json in results/ yet — run the per-size evals first)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    prepare() if cmd == "prepare" else table()
