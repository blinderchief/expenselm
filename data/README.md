# Data directory

```
seed/seed_examples.jsonl   12 worked exemplars (see the warning below)
raw_generations/           synthetic batches from `expenselm gen` (gitignored)
master.jsonl               seed + synthetic + perturbed, before splitting
splits/{train,dev,test}.jsonl   the frozen splits (test = touched ONCE)
chatml/                    training-format files from `expenselm format`
dpo_pairs.jsonl            output of harvest_failures.py
verification_log.md        every correction you make during review (PRD §5.4)
```

## ⚠️ The seed examples are exemplars, not your seeds

PRD §10 rule 1: **you write the 50 seed examples yourself — this is
non-negotiable.** The 12 in `seed/` were written to demonstrate one worked
example per edge-case axis (Hinglish, multi-expense, OCR noise, over-limit,
entertainment ±client, alcohol, unknown currency, relative dates, forwarded
email, unclear purpose). Study *why* each gold label is what it is — e.g.
seed-002's lunch gets `missing_receipt` (800 ≥ 500, no bill mentioned) but
the 350 cab doesn't — then write your own 50 and **replace** these.
Writing them is how you calibrate your taste before synthetic generation.

## Example record format

```json
{
  "id": "seed-001",
  "source_type": "chat|email|ocr|sms|voice",
  "language": "english|hinglish|formal|slang",
  "reference_date": "2026-03-15",
  "input": "<the messy text>",
  "gold": { "expenses": [...], "confidence": "high|medium|low" }
}
```

## Labeling decisions (keep this list growing as you review)

1. Bare numbers WITH Indian context markers (₹/rs/Hinglish) → INR.
   Bare numbers with NO context → `UNKNOWN` + `currency_unknown` flag.
2. `missing_receipt` applies per expense-line, ≥500 INR, whenever no
   receipt/bill/invoice is mentioned — even on non-reimbursable lines.
3. Alcohol-only lines: category `meals`, `reimbursable: false`,
   flag `personal_expense` (closest allowed flag; policy rule 5).
4. Relative dates resolve against `reference_date`; vague ranges
   ("last week") stay `null` — don't guess a day.
5. Foreign-currency amounts are not converted for *limit* checks — avoid
   writing examples whose gold depends on a conversion rate. The *receipt*
   rule DOES apply to clearly non-trivial foreign amounts (≈500 INR+
   equivalent): e.g. $25 with no receipt mentioned → `missing_receipt`
   (seed-049), but $310 with "invoice on the portal" → no flag (seed-038).
6. **Line items of ONE transaction = ONE expense** (sum them): noodles +
   water on one restaurant bill (seed-013), cab fare + waiting charge paid
   together (seed-050), split payment methods for one ride (seed-014).
   Separate transactions = separate expenses, even same category/day
   (seed-027: two Ola rides; seed-042: three meals).
7. Entertainment with a stated *business* purpose (client OR e.g.
   manager-approved morale event, seed-040) → reimbursable; purely
   personal (seed-007, seed-028, seed-036 Netflix) → false + personal_expense.
8. `over_limit` is applied even when the expense is also non-reimbursable
   (seed-046: gym 12,000 vs other-limit 2,000).

## The verification log

Every time you correct a synthetic example during review, append a line to
`verification_log.md`: id, what was wrong, what you changed. The log's
*error-rate and error-pattern statistics* become the data-quality section
of your report (PRD §5.4).
