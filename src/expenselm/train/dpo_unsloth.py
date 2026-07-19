# ============================================================================
# DPO on top of the SFT adapter — the E3 run. Colab T4.
#
# WHAT DPO ACTUALLY DOES (own this for the viva):
# For each (prompt, chosen, rejected) pair, the loss raises the log-prob
# margin of chosen over rejected, RELATIVE to a frozen reference model
# (your SFT model). The reference model anchors the policy so it can't
# drift into degenerate outputs just to satisfy the margin.
#
#   L = -log σ( β·[ (logπ(chosen)-logπ_ref(chosen))
#                 - (logπ(rejected)-logπ_ref(rejected)) ] )
#
# β controls how hard you pull away from the reference. PRD §9 warns "DPO
# degrades the model (common!)" — the failure mode is β too high / too many
# epochs: the model wins the margin game by becoming weird everywhere else.
# Hence: small β, 1 epoch, and ALWAYS compare E3 vs E2 on dev before
# believing anything.
#
# Colab: !pip install unsloth trl peft bitsandbytes wandb
# ============================================================================

from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()  # unsloth's DPO speed/memory patch — call before DPOTrainer import

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

# Start from the SFT adapter — DPO refines the SFT model, it doesn't replace it.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="sft-adapter",       # loads base in 4-bit + your adapter on top
    max_seq_length=2048,
    load_in_4bit=True,
)

# Memory trick: with PEFT, TRL doesn't need a second copy of the model as
# the frozen reference — it just DISABLES the adapter to compute reference
# log-probs. Same base weights, adapter on = policy, adapter off = reference.
# This is why DPO fits on a T4 at all.

dataset = load_dataset("json", data_files={"train": "dpo_pairs.jsonl"})["train"]
dataset = dataset.remove_columns([c for c in dataset.column_names
                                  if c not in ("prompt", "chosen", "rejected")])

trainer = DPOTrainer(
    model=model,
    ref_model=None,                 # None + PEFT => adapter-disabled reference
    args=DPOConfig(
        beta=0.1,                   # the conservative default. If dev metrics
                                    # degrade, try 0.05 BEFORE trying more epochs.
        num_train_epochs=1,         # preference data over-trains fast; 1 epoch.
        per_device_train_batch_size=1,   # DPO holds 2 sequences (chosen+rejected)
        gradient_accumulation_steps=8,   # per sample -> halve the SFT batch size
        learning_rate=5e-6,         # 10-40× LOWER than SFT. You're nudging a
                                    # trained policy, not fitting fresh adapters.
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_8bit",
        max_length=2048,
        max_prompt_length=1536,     # our prompts are long (schema+policy)
        logging_steps=5,
        output_dir="outputs-dpo",
        report_to="wandb",
        run_name="expenselm-dpo-b0.1",
        seed=42,
        fp16=True,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
# Sanity metrics in W&B while training:
#   rewards/accuracies -> fraction of pairs where chosen margin > rejected.
#     Should climb toward 0.7-0.9. Stuck at 0.5 = model can't tell them apart
#     (pairs too subtle or LR too low). 1.0 instantly = pairs too easy.
#   rewards/margins    -> should grow smoothly. Exploding = β or LR too high.

model.save_pretrained("dpo-adapter")
tokenizer.save_pretrained("dpo-adapter")
# -> models/dpo-adapter locally; E3 = expenselm eval --system e3 ...
# THEN: run the dev set through BOTH e2 and e3 and diff per failure mode —
# that before/after table is the Week-5 deliverable.
