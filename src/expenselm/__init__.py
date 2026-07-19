"""ExpenseLM — fine-tuned expense intelligence.

Package map (mirrors the PRD's pipeline order):

    schema.py       -> the output contract (pydantic).      PRD §4
    prompts.py      -> system prompt builder (schema+date+policy)
    data/           -> seed, synthesize, perturb, decontaminate, format. PRD §5
    eval/           -> grader + harness. BUILT BEFORE TRAINING.  PRD §7
    systems/        -> the six systems E0..E5 behind one interface
    train/          -> Unsloth SFT, raw-TRL reference run, DPO.  PRD §6
"""

__version__ = "0.1.0"
