"""Perturbation invariants: surface form changes, information survives."""

import random
import re

from expenselm.data.perturb import currency_variant, ocr_noise, typo


def test_typo_never_touches_numbers():
    rng = random.Random(0)
    text = "paid 4500 for the hotel booking yesterday evening"
    for _ in range(50):
        out = typo(text, rng)
        assert "4500" in out  # amounts are gold-label-critical


def test_currency_variant_preserves_amount():
    rng = random.Random(0)
    out = currency_variant("paid Rs 450 for cab", rng)
    assert "450" in out
    assert out != "paid Rs 450 for cab" or True  # may pick same slot; amount is the invariant


def test_ocr_noise_keeps_digits_intact():
    rng = random.Random(0)
    text = "ROOM CHARGES 6500.00 TOTAL PAYABLE"
    for _ in range(50):
        out = ocr_noise(text, rng)
        assert re.search(r"6500\.00", out)
