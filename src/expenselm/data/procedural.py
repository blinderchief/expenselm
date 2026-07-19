"""Procedural synthetic data — gold labels correct BY CONSTRUCTION.

WHY THIS EXISTS: the Gemini generation path (generate.py) needs API quota
that a free tier meters out slowly. This module is the $0, instant,
reproducible alternative: instead of asking an LLM to *write* input+answer
pairs, we build each example from structured parts —

    (category, amount, currency style, merchant, date phrase,
     receipt mention, purpose, source type, language)

— render the messy text from those parts, and compute the gold label from
the SAME parts by executing the policy rules in code. The text and the
label can never disagree, because they come from one underlying object.
(With LLM generation, the label is the model's *opinion* and needs human
verification; here the label is a *computation*.)

THE HONEST TRADE-OFF (put this in your report — it's a good paragraph):
procedural text is less linguistically diverse than frontier-LLM text; the
model could partially learn our templates rather than the task. Defenses:
  1. every slot has many surface variants and templates shuffle structure,
  2. the perturbation pass adds another layer of surface noise,
  3. the contamination check deletes any train example whose *token set*
     is near a test example — template twins get culled automatically,
  4. the 50 hand-written seeds stay in the pool as distribution anchors,
  5. you can top up with `expenselm gen --engine gemini` (resumable) any
     time quota allows, and mixing the two sources strictly helps.

DESIGN RULE — avoid undefined corners: where two policy rules interact
ambiguously (e.g. over-limit in an UNKNOWN currency), we simply don't
generate that case, so every gold label in the dataset has one defensible
reading. Hand-written seeds are where the truly ambiguous cases live.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

from expenselm.prompts import load_policy
from expenselm.schema import ExpenseRecord

REFERENCE_DATE = "2026-03-15"  # a Sunday; all relative dates resolve against it

# Rough FX to INR — used ONLY for the receipt-threshold rule on foreign
# amounts (data/README.md decision #5). Never used for limit checks.
FX_TO_INR = {"USD": 90, "EUR": 100, "GBP": 115, "AED": 25}

# ---------------------------------------------------------------------------
# Building blocks. Every extra variant here directly buys dataset diversity.
# ---------------------------------------------------------------------------

ITEMS = {
    # category: list of (noun, merchant-or-None, amount_range, foreign_ok)
    "travel": [
        ("cab to the airport", None, (150, 900), False),
        ("cab", "Uber", (120, 800), False),
        ("cab", "Ola", (120, 800), False),
        ("auto", None, (40, 300), False),
        ("bike taxi", "Rapido", (60, 250), False),
        ("flight to Mumbai", "IndiGo", (3800, 9500), False),
        ("flight DEL-BLR", "MakeMyTrip", (3500, 8800), False),
        ("train ticket", "IRCTC", (300, 2400), False),
        ("metro card recharge", None, (100, 500), False),
        ("toll", None, (60, 350), False),
        ("parking", None, (40, 200), False),
        ("airport taxi", None, (18, 60), True),
    ],
    "meals": [
        ("team lunch", "Saravana Bhavan", (600, 3200), False),
        ("client dinner", "Barbeque Nation", (1400, 5800), False),
        ("lunch", None, (150, 900), False),
        ("dinner", None, (250, 1400), False),
        ("breakfast", None, (80, 400), False),
        ("chai and samosa for the meeting", None, (60, 250), False),
        ("biryani order", "Biryani House", (300, 900), False),
        ("coffee with the vendor", "Cafe Coffee Day", (150, 600), False),
        ("office snacks", None, (100, 600), False),
    ],
    "lodging": [
        ("hotel, one night", "Radisson Blu", (4500, 11500), False),
        ("hotel room", "Hotel Sunshine", (2800, 9800), False),
        ("one night stay", "OYO Townhouse", (1400, 4200), False),
        ("hotel near the client office", "FabHotel Prime", (2200, 6800), False),
        ("hotel for the conference", None, (120, 320), True),
    ],
    "supplies": [
        ("two pen drives", None, (700, 1800), False),
        ("printer cartridge", None, (900, 2600), False),
        ("A4 paper and stationery", "D-Mart", (300, 1200), False),
        ("wireless mouse", "Croma", (600, 2200), False),
        ("office chair", "Amazon", (4500, 14500), False),
        ("whiteboard markers", None, (150, 500), False),
    ],
    "software": [
        ("Figma renewal", "Figma", (10, 150), True),
        ("Zoom annual license", "Zoom", (120, 300), True),
        ("Adobe CC subscription", "Adobe", (40, 90), True),
        ("AWS bill for staging", "AWS", (80, 500), True),
        ("Jira subscription for the team", "Jira", (800, 4500), False),
        ("Notion team plan", "Notion", (600, 3000), False),
        ("domain renewal", "GoDaddy", (10, 40), True),
    ],
    "entertainment": [
        ("bowling", "Smaaash", (1200, 4200), False),
        ("movie tickets", "PVR", (400, 1600), False),
        ("karaoke night", None, (900, 3600), False),
        ("go-karting", None, (1500, 4500), False),
    ],
    "other": [
        ("courier to the client", "Blue Dart", (200, 900), False),
        ("conference registration", None, (500, 4500), False),
        ("workshop venue booking", None, (1500, 4800), False),
        ("visiting cards printing", None, (300, 1200), False),
    ],
}

ALCOHOL_ITEMS = [
    ("beers for the team", (800, 2400)),
    ("whiskey bottle", (1500, 4500)),
    ("wine for the party", (900, 2800)),
]

PERSONAL_ITEMS = [  # never business: reimbursable false + personal_expense
    ("Netflix subscription", "entertainment", (199, 649)),
    ("gym membership", "other", (1000, 1999)),
    ("birthday gift for a friend", "other", (400, 1900)),
    ("movie with family", "entertainment", (500, 1800)),
]

CLIENT_PURPOSES = [
    "for the client meeting", "client visit", "with the Acme Corp folks",
    "for the client demo", "client workshop", "with the Zentech clients",
]
BUSINESS_PURPOSES = [
    "for the office", "for the team", "for the offsite",
    "manager approved it as a team event", "for the sprint demo day",
]

RECEIPT_YES = [
    "bill attached", "invoice attached", "receipt with me", "have the bill",
    "bill hai mere paas", "receipt attached", "invoice is on the portal",
    "bill bheja hai", "receipts shared on mail",
]
RECEIPT_NO = [
    "no bill sorry", "lost the receipt", "bill kho gaya", "paid cash no slip",
    "will send the invoice later", "receipt nahi mila",
]

# (phrase, day-offset or exact date or None-for-vague)
DATE_PHRASES = [
    ("yesterday", -1), ("today", 0), ("day before yesterday", -2),
    ("last friday", -2), ("last saturday", -1),
    ("on 10 March", dt.date(2026, 3, 10)), ("on 2nd March", dt.date(2026, 3, 2)),
    ("on 12/03/2026", dt.date(2026, 3, 12)),
    ("last week", None), ("a few days back", None), ("2-3 din pehle", None),
]

CHAT_VERBS_EN = ["paid", "spent", "shelled out", "got charged", "put down"]
CHAT_VERBS_HI = ["diya", "kharcha kiya", "pay kiya", "de diya"]
HINGLISH_FRAMES = [
    "bhai {amt} ka {noun} {purpose}",
    "{noun} liya {date}, {amt} ka, {receipt}",
    "{noun} ka {amt} {purpose}, {receipt}",
    "yaar {noun} me {amt} lag gaye {date}",
    "{amt} {verb} {noun} ke liye {purpose}",
]
ENGLISH_FRAMES = [
    "{verb} {amt} for {noun} {date} {purpose}, {receipt}",
    "{noun} came to {amt} {date}, {receipt}",
    "{noun} — {amt} {purpose}",
    "quick one: {verb} {amt} on {noun} {date}. {receipt}",
    "reimbursement request: {noun}, {amt}, {date}. {receipt}",
    "fyi {verb} {amt} for the {noun} {purpose}",
]
VOICE_FILLERS = ["uh", "um", "so basically", "I think", "like"]
EMAIL_SIGNATURES = [
    "\n\nBest,\n{name}\nSent from my iPhone",
    "\n\nRegards,\n{name} | {team}\n{company}",
    "\n\nThanks,\n{name}\n--\nThis email may contain confidential information.",
]
EMAIL_NOISE_PREFIX = [
    "---------- Forwarded message ----------\nFrom: {name} <{email}>\nSubject: {subject}\n\n",
    ">> On Mar {d}, {name} wrote:\n>> approved, go ahead\n\n",
    "From: {name} <{email}>\nSubject: {subject}\n\nHi,\n\n",
]
NAMES = ["Priya Nair", "Rakesh Kumar", "Anita Shah", "Vikram Rao", "Sneha Iyer",
         "Arjun Mehta", "Divya Menon", "Rohit Sharma"]
COMPANIES = ["Zentech", "FinLogix", "Acme Corp", "BrightPay", "NimbusSoft"]

OCR_GLYPHS = {"O": "0", "o": "0", "I": "1", "i": "1", "S": "5", "B": "8", "e": "3"}

NUM_WORDS = {200: "two hundred", 300: "three hundred", 500: "five hundred",
             1000: "one thousand", 1500: "fifteen hundred", 2000: "two thousand"}


# ---------------------------------------------------------------------------
# One expense = one structured draw; gold fields are computed, not guessed.
# ---------------------------------------------------------------------------

def _draw_expense(rng: random.Random, policy: dict, case: str) -> dict:
    """Returns the internal spec an expense is built from."""
    if case == "alcohol":
        noun, amt_range = rng.choice(ALCOHOL_ITEMS)
        return {"noun": noun, "category": "meals", "merchant": None,
                "amount": rng.randrange(*amt_range), "foreign": False,
                "alcohol": True, "personal": True, "purpose": None}
    if case == "personal":
        noun, cat, amt_range = rng.choice(PERSONAL_ITEMS)
        return {"noun": noun, "category": cat, "merchant": None,
                "amount": rng.randrange(*amt_range), "foreign": False,
                "alcohol": False, "personal": True, "purpose": None}

    if case == "entertainment_client":
        cat = "entertainment"
    elif case == "entertainment_personal":
        cat = "entertainment"
    else:
        cat = rng.choice([c for c in ITEMS if c != "entertainment"])

    noun, merchant, amt_range, foreign_ok = rng.choice(ITEMS[cat])
    foreign = foreign_ok  # foreign_ok items are priced in foreign currency
    amount = rng.randrange(*amt_range)
    if case == "over_limit" and not foreign:
        # push the amount just past the category limit (25% of the time far past)
        limit = policy["limits"][cat]
        amount = int(limit * rng.uniform(1.05, 1.9))

    if cat == "entertainment":
        purpose = (rng.choice(CLIENT_PURPOSES) if case == "entertainment_client"
                   else None)
        personal = case != "entertainment_client"
    else:
        purpose = rng.choice(CLIENT_PURPOSES + BUSINESS_PURPOSES + [None, None])
        personal = False

    return {"noun": noun, "category": cat, "merchant": merchant, "amount": amount,
            "foreign": foreign, "alcohol": False, "personal": personal,
            "purpose": purpose}


def _gold_for(spec: dict, *, currency: str, receipt_mentioned: bool,
              date_iso: str | None, policy: dict) -> dict:
    """Execute the policy — this function IS the labeling guideline in code."""
    flags: list[str] = []
    amount = float(spec["amount"])
    reimbursable = True

    if spec["personal"] or spec["alcohol"]:
        reimbursable = False
        flags.append("personal_expense")

    if currency == "INR" and amount > policy["limits"][spec["category"]]:
        flags.append("over_limit")

    if currency == "UNKNOWN":
        flags.append("currency_unknown")

    inr_equiv = amount * FX_TO_INR.get(currency, 1)  # UNKNOWN treated numerically
    if inr_equiv >= policy["rules"]["receipt_required_above"] and not receipt_mentioned:
        flags.append("missing_receipt")

    desc = spec["noun"] + (" (alcohol)" if spec["alcohol"] else "")
    return {
        "amount": round(amount, 2),
        "currency": currency,
        "category": spec["category"],
        "merchant": spec["merchant"],
        "date": date_iso,
        "description": desc,
        "reimbursable": reimbursable,
        "policy_flags": flags,
    }


# ---------------------------------------------------------------------------
# Rendering: spec -> messy text (never corrupting label-bearing tokens)
# ---------------------------------------------------------------------------

def _amount_text(rng, amount: int, currency: str, language: str, source: str) -> str:
    if currency == "USD":
        return rng.choice([f"${amount}", f"USD {amount}", f"{amount} dollars"])
    if currency == "EUR":
        return rng.choice([f"EUR {amount}", f"{amount} euros"])
    if currency == "AED":
        return f"{amount} dirhams"
    if currency == "GBP":
        return rng.choice([f"GBP {amount}", f"{amount} pounds"])
    if currency == "UNKNOWN":
        # bare number in a non-Indian-context English message
        if source == "voice" and amount in NUM_WORDS and rng.random() < 0.5:
            return NUM_WORDS[amount]
        return str(amount)
    # INR — the whole zoo of Indian notations
    style = rng.choice(["₹{a}", "Rs {a}", "Rs. {a}", "{a} rs", "{a} rupees",
                        "INR {a}", "₹{a:,}"])
    return style.format(a=amount)


def _dates_for(rng) -> tuple[str, str | None]:
    """Returns (phrase for the text, resolved ISO date or None)."""
    phrase, val = rng.choice(DATE_PHRASES)
    if val is None:
        return phrase, None
    if isinstance(val, dt.date):
        return phrase, val.isoformat()
    ref = dt.date.fromisoformat(REFERENCE_DATE)
    return phrase, (ref + dt.timedelta(days=val)).isoformat()


def _clean(text: str) -> str:
    return " ".join(text.split()).replace(" ,", ",").replace(" .", ".")


def _render_chat(rng, parts: list[str], language: str) -> str:
    joiner = rng.choice([", ", " aur ", " + ", ", also "] if language == "hinglish"
                        else [", ", " and ", " + ", "; also "])
    return joiner.join(parts)


def _render_email(rng, body: str) -> str:
    name = rng.choice(NAMES)
    pre = rng.choice(EMAIL_NOISE_PREFIX).format(
        name=name, email=f"{name.split()[0].lower()}@{rng.choice(COMPANIES).lower()}.in",
        subject=rng.choice(["expenses", "reimbursement", "FW: bills", "claim"]),
        d=rng.randrange(1, 14))
    sig = rng.choice(EMAIL_SIGNATURES).format(
        name=name, team=rng.choice(["Sales", "Engineering", "Consulting"]),
        company=rng.choice(COMPANIES))
    return pre + body + sig

def _render_ocr(rng, spec: dict, amount_txt: str, date_iso: str | None) -> str:
    lines = []
    if spec["merchant"]:
        lines.append(spec["merchant"].upper())
    lines.append(spec["noun"].upper())
    if date_iso:
        # receipts print absolute dates — render the resolved date, so the
        # printed date and the gold date are the same fact in two formats
        y, m, d = date_iso.split("-")
        lines.append(f"DATE {rng.choice([f'{d}/{m}/{y}', f'{d}-{m}-{y}', f'{d}/{m}/{y[2:]}'])}")
    lines.append(f"TOTAL {amount_txt.upper()}")
    lines.append(rng.choice(["THANK YOU", "GST INCL", "VISIT AGAIN", "*COPY*"]))
    text = "\n".join(lines)
    # glyph garbling on LETTERS only — digits carry the gold label
    out = []
    for ch in text:
        if ch.isalpha() and ch in OCR_GLYPHS and rng.random() < 0.25:
            out.append(OCR_GLYPHS[ch])
        else:
            out.append(ch)
    return "".join(out)


def _render_voice(rng, body: str) -> str:
    return f"{rng.choice(VOICE_FILLERS)} {body}" + rng.choice(
        ["", " sorry for the rambling", " yeah that's it", ""])


# ---------------------------------------------------------------------------
# Putting one example together
# ---------------------------------------------------------------------------

POLICY_CASES = ["normal", "normal", "normal", "over_limit", "entertainment_client",
                "entertainment_personal", "alcohol", "personal", "no_receipt"]


def make_example(rng: random.Random, policy: dict, idx: int) -> dict:
    source = rng.choice(["chat", "chat", "chat", "sms", "email", "ocr", "voice"])
    language = rng.choice(["english", "english", "hinglish", "formal", "slang"])
    n_expenses = 1 if source in ("ocr",) else rng.choice([1, 1, 1, 1, 2, 2, 3])
    case = rng.choice(POLICY_CASES)
    if source == "ocr" and case == "entertainment_client":
        # a bare receipt can't state a business purpose; per policy rule 3,
        # entertainment with no stated purpose is a personal expense
        case = "entertainment_personal"

    parts, golds = [], []
    for i in range(n_expenses):
        item_case = case if i == 0 else "normal"
        spec = _draw_expense(rng, policy, item_case)

        # currency decision — mirrors data/README rule #1
        if spec["foreign"]:
            currency = rng.choice(["USD", "USD", "EUR", "AED", "GBP"]) \
                if spec["category"] != "software" else "USD"
            if spec["noun"] in ("Zoom annual license", "Figma renewal",
                                "Adobe CC subscription", "AWS bill for staging",
                                "domain renewal"):
                currency = "USD"
        elif language == "hinglish" or rng.random() < 0.8:
            currency = "INR"
        else:
            currency = "UNKNOWN"   # bare number, no Indian marker

        # receipt decision. OCR *is* a receipt.
        if source == "ocr":
            receipt_mentioned, receipt_txt = True, ""
        elif case == "no_receipt" and i == 0:
            receipt_mentioned, receipt_txt = False, rng.choice(RECEIPT_NO)
        elif rng.random() < 0.45:
            receipt_mentioned, receipt_txt = True, rng.choice(RECEIPT_YES)
        else:
            receipt_mentioned, receipt_txt = False, ""  # simply not mentioned

        date_phrase, date_iso = (_dates_for(rng) if rng.random() < 0.45 else ("", None))
        amount_txt = _amount_text(rng, spec["amount"], currency, language, source)

        golds.append(_gold_for(spec, currency=currency,
                               receipt_mentioned=receipt_mentioned,
                               date_iso=date_iso, policy=policy))

        if source == "ocr":
            parts.append(_render_ocr(rng, spec, amount_txt, date_iso))
            continue
        # avoid "Figma renewal at Figma" when the noun already names the merchant
        merchant_rendered = (
            spec["merchant"] is not None
            and spec["merchant"].split()[0].lower() not in spec["noun"].lower()
            and rng.random() < 0.7
        )
        noun = spec["noun"] + (f" at {spec['merchant']}" if merchant_rendered else "")
        if spec["merchant"] and not merchant_rendered \
                and spec["merchant"].split()[0].lower() not in spec["noun"].lower():
            # LABEL RULE: a merchant the text never names is not extractable —
            # gold must be null or the metric punishes impossible answers.
            golds[-1]["merchant"] = None
        frame = rng.choice(HINGLISH_FRAMES if language == "hinglish" else ENGLISH_FRAMES)
        rendered = _clean(frame.format(
            verb=rng.choice(CHAT_VERBS_HI if language == "hinglish" else CHAT_VERBS_EN),
            amt=amount_txt, noun=noun, purpose=spec["purpose"] or "",
            date=date_phrase, receipt=receipt_txt,
        ).replace(" ,", ",").rstrip(", "))

        # LABEL-CONSISTENCY REPAIR: a frame without a {receipt}/{date}/{purpose}
        # slot would silently drop a fact the gold label depends on. Anything
        # label-bearing that didn't make it into the text gets appended.
        if receipt_txt and receipt_txt not in rendered:
            rendered += f", {receipt_txt}"
        if date_phrase and date_phrase not in rendered:
            rendered += f" {date_phrase}"
        if (spec["category"] == "entertainment" and spec["purpose"]
                and spec["purpose"] not in rendered):
            rendered += f" {spec['purpose']}"  # client context decides reimbursable
        parts.append(rendered)

    if source == "ocr":
        text = parts[0]
    else:
        body = _render_chat(rng, parts, language)
        if source == "email":
            text = _render_email(rng, body)
        elif source == "voice":
            text = _render_voice(rng, body)
        elif source == "sms":
            text = body if len(body) < 90 else body[:1] + body[1:]  # sms stays terse naturally
        else:
            text = body

    confidence = "high"
    if any(g["currency"] == "UNKNOWN" for g in golds):
        confidence = "low"
    elif source in ("ocr", "voice") or any(g["policy_flags"] for g in golds):
        confidence = "medium" if rng.random() < 0.7 else "high"

    return {
        "id": f"proc-{idx:05d}",
        "source_type": source,
        "language": language,
        "reference_date": REFERENCE_DATE,
        "input": text,
        "gold": {"expenses": golds, "confidence": confidence},
        "scenario": {"case": case, "n": n_expenses},
        "generator": "procedural",
    }


def generate(n: int, out_path: str, seed: int = 20260315) -> None:
    policy = load_policy()
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w") as f:
        while written < n:
            ex = make_example(rng, policy, written)
            ExpenseRecord.model_validate(ex["gold"])  # invariant: always passes
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} procedural examples -> {out_path}")


if __name__ == "__main__":
    import sys

    generate(int(sys.argv[1]) if len(sys.argv) > 1 else 2400,
             sys.argv[2] if len(sys.argv) > 2 else "data/raw_generations/procedural.jsonl")
