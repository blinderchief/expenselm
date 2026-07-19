"""Perturbation pass (PRD §5, step 3) — programmatic realism injection.

Synthetic text is suspiciously clean: perfect spelling, consistent currency
symbols, tidy line breaks. Real forwarded emails and OCR dumps are not.
These perturbations are applied to the INPUT ONLY — the gold label stays
untouched, because a typo in "recieved" doesn't change what was spent.

(That invariant — perturb X, never Y — is the whole trick. If a
perturbation could change the correct label, e.g. deleting the currency
symbol, it must NOT be applied blindly. That's why currency-symbol swaps
below substitute equivalent forms rather than removing information.)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ₹ / Rs / INR / rs. are the same information in different clothes —
# swapping among them changes surface form, never the gold label.
CURRENCY_VARIANTS = ["₹", "Rs ", "Rs. ", "INR ", "rs "]

KEYBOARD_NEIGHBORS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg",
    "g": "fh", "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k",
    "m": "n", "n": "bm", "o": "ip", "p": "o", "q": "wa", "r": "et",
    "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc",
    "y": "tu", "z": "x",
}

# What OCR engines actually confuse (visually similar glyphs).
OCR_CONFUSIONS = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "8": "B", "rn": "m"}


def typo(text: str, rng: random.Random) -> str:
    """One keyboard-neighbor substitution in a random word (not in numbers —
    corrupting an amount would change the gold label)."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(len(w))
    ch = w[j].lower()
    if ch in KEYBOARD_NEIGHBORS:
        words[i] = w[:j] + rng.choice(KEYBOARD_NEIGHBORS[ch]) + w[j + 1:]
    return " ".join(words)


def currency_variant(text: str, rng: random.Random) -> str:
    for v in CURRENCY_VARIANTS:
        if v in text:
            return text.replace(v, rng.choice([x for x in CURRENCY_VARIANTS if x != v]), 1)
    return text


def ocr_noise(text: str, rng: random.Random) -> str:
    """Glyph confusion + random line breaks — but never inside digits that
    are part of an amount context... simplest safe rule: only touch letters."""
    out = text
    for src, dst in rng.sample(list(OCR_CONFUSIONS.items()), k=2):
        if src.isalpha() and src in out:  # letters only: amounts stay intact
            out = out.replace(src, dst, 1)
    # inject a mid-text linebreak, as flatbed OCR does
    if len(out) > 40:
        pos = rng.randrange(20, len(out) - 10)
        space = out.find(" ", pos)
        if space != -1:
            out = out[:space] + "\n" + out[space + 1:]
    return out


def truncate_tail(text: str, rng: random.Random) -> str:
    """Chop trailing characters (cut-off screenshots / SMS). Max 10% so we
    never delete an amount that the gold label depends on — verify this
    when reviewing! Truncation is the riskiest perturbation."""
    cut = rng.randrange(1, max(2, len(text) // 10))
    return text[:-cut]


PERTURBATIONS = [typo, currency_variant, ocr_noise]  # truncate_tail opt-in only


def perturb_file(in_path: str, out_path: str, rate: float = 0.5, seed: int = 7) -> None:
    rng = random.Random(seed)
    examples = [json.loads(l) for l in open(in_path) if l.strip()]
    n_changed = 0
    for ex in examples:
        if rng.random() < rate:
            fn = rng.choice(PERTURBATIONS)
            new = fn(ex["input"], rng)
            if new != ex["input"]:
                ex["input"] = new
                ex["perturbation"] = fn.__name__
                n_changed += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"perturbed {n_changed}/{len(examples)} -> {out_path}")
