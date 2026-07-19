"""Convert dataset examples to training-ready chat format (PRD §5, "Format").

One example -> one {"messages": [...]} line, using the SAME
prompts.build_messages() the eval harness uses. Train/inference prompt
identity is non-negotiable: if the model is trained with one system prompt
and evaluated with another, SFT gains partly measure prompt memorization,
not capability.

The assistant turn is the gold JSON, compact (no indentation) — fewer
tokens per example means more examples per VRAM-hour, and the model should
learn the content, not a pretty-printing style.

Also here: deterministic splitting (seeded shuffle) so anyone can
regenerate identical train/dev/test splits from the master file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from expenselm.prompts import build_messages


def to_chatml(example: dict) -> dict:
    messages = build_messages(example["input"], example["reference_date"])
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(example["gold"], ensure_ascii=False, separators=(",", ":")),
        }
    )
    return {"messages": messages}


def to_chatml_file(in_path: str, out_path: str) -> None:
    examples = [json.loads(l) for l in open(in_path) if l.strip()]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(to_chatml(ex), ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} ChatML examples -> {out_path}")


def split(
    master_path: str,
    out_dir: str = "data/splits",
    n_dev: int = 100,
    n_test: int = 200,
    seed: int = 13,
) -> None:
    """Shuffle once with a fixed seed, carve off dev and test, rest is train.

    Test is carved BEFORE any training experiments happen and then never
    edited — PRD §5: "touched exactly once, for the final table".
    """
    examples = [json.loads(l) for l in open(master_path) if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(examples)

    test, dev, train = (
        examples[:n_test],
        examples[n_test : n_test + n_dev],
        examples[n_test + n_dev :],
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("test", test), ("dev", dev), ("train", train)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for ex in part:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} -> {out / f'{name}.jsonl'}")
