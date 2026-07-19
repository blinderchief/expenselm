"""The output schema — the contract every system E0..E5 is graded against.

============================================================================
⚠️  PRD §10 ("AI-Usage Protocol") says YOU must write the pydantic schema
    yourself. Treat this file as a *reference implementation to study, then
    rewrite from scratch without looking*. If you can't re-derive every
    validator here and explain WHY it exists, you don't own it yet.
============================================================================

Design decisions worth understanding (these are viva questions):

1. `extra="forbid"` — a model that hallucinates extra keys FAILS schema
   validation. Without this, pydantic silently ignores extra keys and your
   "schema validity rate" metric lies to you.

2. `amount` is float but graded within ±0.01 (PRD §7 metric 2). We do NOT
   round in the schema — rounding is a *grading* decision, not a *parsing*
   decision. Keep the layers separate.

3. `currency` is a 3-letter ISO 4217 code OR the literal "UNKNOWN".
   We validate the *shape* (3 uppercase letters), not membership in the full
   ISO list — a 200-entry enum would bloat the prompt for zero gain, and the
   model only ever sees INR/USD/EUR/GBP-ish currencies in this data.

4. `date` is a string, not `datetime.date` — because the JSON the model emits
   contains a string, and we want the schema to mirror the wire format
   exactly. A regex + real-date check catches "2026-02-30".

5. `policy_flags` is validated against the closed set in policy.yaml
   (mirrored here as ALLOWED_FLAGS). An open-ended list would let the model
   invent flags and make policy accuracy ungradeable.
"""

from __future__ import annotations

import re
from datetime import date as _date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Keep in sync with configs/policy.yaml `allowed_flags`.
ALLOWED_FLAGS = frozenset(
    {"over_limit", "missing_receipt", "personal_expense", "unclear_purpose", "currency_unknown"}
)

Category = Literal[
    "travel", "meals", "lodging", "supplies", "software", "entertainment", "other"
]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class Expense(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(gt=0, description="Positive numeric amount")
    currency: str = Field(description='ISO 4217 code, or "UNKNOWN" if not inferable')
    category: Category
    merchant: Optional[str] = None
    date: Optional[str] = Field(
        default=None, description="YYYY-MM-DD, resolved against the reference date"
    )
    description: str = Field(min_length=1)
    reimbursable: bool
    policy_flags: list[str] = Field(default_factory=list)

    @field_validator("currency")
    @classmethod
    def _currency_shape(cls, v: str) -> str:
        if v == "UNKNOWN":
            return v
        if not _CURRENCY_RE.match(v):
            raise ValueError(f"currency must be 3 uppercase letters or UNKNOWN, got {v!r}")
        return v

    @field_validator("date")
    @classmethod
    def _real_calendar_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not _DATE_RE.match(v):
            raise ValueError(f"date must be YYYY-MM-DD, got {v!r}")
        y, m, d = map(int, v.split("-"))
        _date(y, m, d)  # raises ValueError for 2026-02-30 etc.
        return v

    @field_validator("policy_flags")
    @classmethod
    def _known_flags_only(cls, v: list[str]) -> list[str]:
        unknown = set(v) - ALLOWED_FLAGS
        if unknown:
            raise ValueError(f"unknown policy_flags: {sorted(unknown)}")
        if len(v) != len(set(v)):
            raise ValueError("duplicate policy_flags")
        return v


class ExpenseRecord(BaseModel):
    """Top-level object the model must emit. One record per input message."""

    model_config = ConfigDict(extra="forbid")

    expenses: list[Expense] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


# The schema text that goes into the system prompt. Generated from the model
# itself so prompt and validator can never drift apart.
def schema_for_prompt() -> str:
    import json

    return json.dumps(ExpenseRecord.model_json_schema(), indent=2)
