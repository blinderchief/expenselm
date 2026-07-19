"""System-prompt construction — identical for every system E0..E5.

WHY ONE BUILDER: the experiment isolates *training method* differences
(zero-shot vs few-shot vs SFT vs DPO vs quantized). If each system got a
slightly different prompt, you could no longer attribute metric deltas to
the method. Fixed prompt = controlled variable. This is basic experimental
hygiene and a thing interviewers probe.

The prompt contains exactly what PRD §4 specifies:
  1. the JSON schema
  2. a reference date (so "yesterday" is resolvable)
  3. the expense policy (so reimbursable/policy_flags are decidable)
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "policy.yaml"


def load_policy(path: Path | None = None) -> dict:
    with open(path or POLICY_PATH) as f:
        return yaml.safe_load(f)


def policy_block(policy: dict) -> str:
    """Render the policy as compact prose the model can actually follow.

    We deliberately do NOT dump raw YAML: small models follow short
    numbered rules far better than nested config syntax.
    """
    limits = ", ".join(f"{cat} {amt} INR" for cat, amt in policy["limits"].items())
    rules = policy["rules"]
    return (
        "EXPENSE POLICY:\n"
        f"1. Per-expense limits: {limits}. Amounts above the limit keep "
        "reimbursable=true but MUST carry the 'over_limit' flag "
        "(unless another rule makes them non-reimbursable).\n"
        f"2. Reimbursable categories by default: "
        f"{', '.join(rules['reimbursable_categories'])}.\n"
        "3. 'entertainment' is reimbursable ONLY if a client/business purpose "
        "is stated; otherwise reimbursable=false and add 'personal_expense'.\n"
        f"4. Expenses of {rules['receipt_required_above']} INR or more need a "
        "receipt; if none is mentioned, add 'missing_receipt'.\n"
        "5. Alcohol-only expenses are never reimbursable.\n"
        "6. If an amount has no inferable currency, use currency='UNKNOWN' "
        "and add 'currency_unknown'.\n"
        f"7. Allowed policy_flags (use no others): "
        f"{', '.join(policy['allowed_flags'])}."
    )


def build_system_prompt(reference_date: str, policy: dict | None = None) -> str:
    from expenselm.schema import schema_for_prompt

    policy = policy or load_policy()
    return (
        "You extract structured expense records from messy text (chat messages, "
        "forwarded emails, receipt OCR). Respond with ONLY a single JSON object "
        "conforming exactly to this JSON schema — no markdown, no commentary:\n\n"
        f"{schema_for_prompt()}\n\n"
        f"REFERENCE DATE (today): {reference_date}. Resolve relative dates "
        "('yesterday', 'last Friday') against it; use null when no date is "
        "stated or inferable.\n\n"
        f"{policy_block(policy)}\n\n"
        "A message may contain multiple expenses — emit one entry per expense. "
        "Set confidence to 'low' when currency, amounts, or intent are ambiguous."
    )


def build_messages(
    input_text: str,
    reference_date: str,
    few_shot: list[dict] | None = None,
) -> list[dict]:
    """Build the chat `messages` array for one example.

    `few_shot`: list of dataset examples ({"input", "gold", ...}) to include
    as in-context demonstrations (E1). They are rendered as prior user/
    assistant turns — the standard few-shot chat format.
    """
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(reference_date)}
    ]
    for shot in few_shot or []:
        messages.append({"role": "user", "content": shot["input"]})
        messages.append(
            {"role": "assistant", "content": json.dumps(shot["gold"], ensure_ascii=False)}
        )
    messages.append({"role": "user", "content": input_text})
    return messages
