"""Contamination check (PRD §5, step 5).

WHY THIS MATTERS MORE THAN ANYTHING ELSE IN THE DATA PIPELINE:
if near-duplicates of test inputs sit in the training set, E2/E3 scores are
inflated and your headline claim ("small model beats frontier pricing") is
built on leakage. One reviewer question — "how did you check for
contamination?" — and the whole result collapses. This is the difference
between eval work and homework.

METHOD: rapidfuzz token_set_ratio on normalized inputs. Token-set ignores
word order and duplication, so "cab 450 airport client meeting" matches
"took cab to airport for client meeting, 450" — exactly the near-duplicate
class synthetic generators produce. Embedding similarity would also work;
fuzzy matching is transparent, dependency-light, and explainable in the
report. (Say that sentence in the viva.)

Threshold 85 is a starting point — calibrate it: print the matches at
80-90 and eyeball where true duplicates end and mere topical similarity
begins. Record the calibrated value in the report.
"""

from __future__ import annotations

import json
import re

from rapidfuzz import fuzz


def normalize(text: str) -> str:
    # lowercase, strip digits punctuation -> compare STRUCTURE not amounts;
    # two messages differing only in "450" vs "480" are still duplicates.
    text = text.lower()
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"[^\w#\s]", " ", text)


def find_contaminated(
    train: list[dict], test: list[dict], threshold: float = 85.0
) -> list[tuple[str, str, float]]:
    """Returns (train_id, test_id, score) for every pair above threshold.

    O(train × test) comparisons — 1500×300 = 450k rapidfuzz calls, a few
    seconds. No need for anything cleverer at this scale (knowing when NOT
    to optimize is also the skill, PRD §6 note).
    """
    test_norm = [(t["id"], normalize(t["input"])) for t in test]
    hits = []
    for tr in train:
        tr_norm = normalize(tr["input"])
        for te_id, te_norm in test_norm:
            score = fuzz.token_set_ratio(tr_norm, te_norm)
            if score >= threshold:
                hits.append((tr["id"], te_id, score))
    return hits


def decontaminate(train_path: str, test_path: str, threshold: float = 85.0) -> None:
    """Remove offending TRAIN examples (never touch test — test is sacred,
    'touched exactly once' per PRD §5) and rewrite the train file."""
    train = [json.loads(l) for l in open(train_path) if l.strip()]
    test = [json.loads(l) for l in open(test_path) if l.strip()]

    hits = find_contaminated(train, test, threshold)
    bad_train_ids = {h[0] for h in hits}

    for tr_id, te_id, score in sorted(hits, key=lambda h: -h[2]):
        print(f"  CONTAMINATED train={tr_id} ~ test={te_id} (score {score:.0f})")

    kept = [t for t in train if t["id"] not in bad_train_ids]
    with open(train_path, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"removed {len(bad_train_ids)} train examples; {len(kept)} remain -> {train_path}")
