# Verification log (PRD §5.4)

Every correction made while reviewing generated data goes here — one line
each. The stats from this log ARE the data-quality section of the report
(error rate of the generator, most common error patterns).

Format: `| date | example id | what was wrong | fix | error class |`

| Date | ID | What was wrong | Fix | Error class |
|---|---|---|---|---|
| 2026-07-07 | seed-013 | category "Lunch" not in enum (schema caught it); description copied from seed-012; flags [missing_receipt, unclear_purpose, currency_unknown] all wrong — currency IS inferable ("rupees"), amount 242.3 < 500 so no receipt needed; "just eaten" ⇒ date = reference date | category→meals, merchant→Chowman, date→2026-03-15, flags→[], confidence→medium | wrong-category, wrong-flag, wrong-date-resolution |
| 2026-07-07 | seed-014 | entire gold copy-pasted from seed-011 (two expenses: lodging+meals) — schema-VALID but semantically wrong; schema can't catch this, only review can | one travel expense, 357.0 (300 cash + 57 UPI summed = one transaction), date 2026-03-15 ("Today"), flags [] (357<500) | missed-expense-in-gold, wrong-category, wrong-flag |
| 2026-07-08 | 349 procedural examples (287 train / 22 dev / 40 test) | generator put a merchant in gold that the TEXT never mentions ("one night stay" → gold merchant "OYO Townhouse") — unextractable, discovered when the E4 frontier run disagreed with gold | deterministic repair: merchant→null wherever not present in input (OCR-glyph-aware fuzzy match ≥85); generator fixed in procedural.py | merchant-not-in-text |
| 2026-07-08 | ~21 test examples (OPEN — Suyash to decide) | currency ambiguity: bare number in a message that has ₹/Rs on ANOTHER expense ("Netflix INR 528 + paid 1552 for Notion"). Generator labeled per-expense (UNKNOWN); frontier reading says message-level context (INR). A human would likely say INR | NOT changed — decide the rule, then either relabel or accept; whichever way, document in data/README | currency-context-ambiguity |
| 2026-07-08 | ~7 test examples (OPEN — Suyash to decide) | "Netflix subscription" / brand-named nouns: gold merchant=null but the brand IS in the text — is Netflix the merchant? | NOT changed — pick a convention and relabel once | merchant-brand-in-noun |

Error classes to start with (extend as needed): wrong-flag, wrong-category,
missed-expense-in-gold, wrong-date-resolution, wrong-currency,
implausible-input, too-clean-input, policy-misapplied-in-gold.
