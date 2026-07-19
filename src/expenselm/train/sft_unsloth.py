# ============================================================================
# SFT with Unsloth + QLoRA — the E2 run. Designed for a free Colab T4 (16GB).
#
# PRD §8 Week 4 definition of done: "you can explain every hyperparameter
# you set". Every knob below carries its explanation. Before the viva,
# cover the comments and explain each one from memory.
#
# On Colab, first cell:
#   !pip install unsloth trl peft bitsandbytes wandb
#   (upload data/chatml/train.jsonl + dev.jsonl, or mount Drive)
# Then:  %run sft_unsloth.py
# ============================================================================

from unsloth import FastLanguageModel  # MUST be imported before transformers/trl —
                                       # unsloth monkey-patches them for the speedup

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# 1. Load the base model in 4-bit (the "Q" in QLoRA)
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 2048
# Why 2048: system prompt (~700 tok) + messy input (~200) + JSON output
# (~300) fits comfortably. Shorter sequences = quadratically less attention
# memory = bigger batches on the T4. Measure your actual p95 length first!

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507",  # unsloth's mirror, pre-patched
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,   # NF4 quantization of the FROZEN base weights.
                         # 4B params × ~0.5 bytes ≈ 2.2GB instead of 8GB bf16.
                         # Gradients never flow through these — only through
                         # the LoRA adapters, which stay in bf16.
)

# ---------------------------------------------------------------------------
# 2. Attach LoRA adapters (the "LoRA" in QLoRA)
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,               # adapter rank. The update ΔW is factored as B@A where
                        # A is (r × d), B is (d × r). r=16 ≈ 0.5-1% of params
                        # trainable. PRD default; the data-size ablation matters
                        # more than sweeping r — don't rabbit-hole here.
    lora_alpha=16,      # scaling: effective update = (alpha/r) · BA. alpha=r
                        # gives scale 1.0, the boring-and-good default.
    lora_dropout=0.0,   # unsloth's fast path requires 0; fine at this scale.
    target_modules=[    # WHICH weight matrices get adapters. Attention
        "q_proj", "k_proj", "v_proj", "o_proj",   # projections +
        "gate_proj", "up_proj", "down_proj",      # the MLP. "All linear
    ],                  # layers" is the post-2023 consensus (QLoRA paper).
    use_gradient_checkpointing="unsloth",  # recompute activations in the
                        # backward pass instead of storing them: ~70% less
                        # activation memory for ~20% more compute. THE
                        # classic memory/compute trade. "unsloth" variant is
                        # their optimized implementation.
    random_state=42,
)

# ---------------------------------------------------------------------------
# 3. Data — the ChatML files produced by `expenselm format`
# ---------------------------------------------------------------------------
dataset = load_dataset(
    "json",
    data_files={"train": "train.jsonl", "eval": "dev_chatml.jsonl"},
)

def render(example):
    # Render messages with the model's OWN chat template. Using the wrong
    # template (or a hand-rolled one) is the most common silent SFT bug:
    # loss goes down, model still outputs garbage at inference.
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    }

dataset = dataset.map(render)

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,

        per_device_train_batch_size=2,     # what fits in T4 VRAM
        gradient_accumulation_steps=8,     # effective batch = 2×8 = 16.
        # Gradient accumulation = run 8 small forward/backwards, sum the
        # gradients, THEN step. Mathematically ≈ a batch of 16 without the
        # memory of 16. This is how small GPUs fake big batches.

        num_train_epochs=2,                # 1500 examples is small; 2 epochs
        # is usually the sweet spot. 3+ risks memorizing synthetic phrasing.
        # Watch eval_loss: if it rises while train_loss falls -> overfitting.

        learning_rate=2e-4,                # standard for LoRA (10-100× higher
        # than full fine-tuning, because you're training a tiny adapter from
        # scratch, not nudging pretrained weights).
        lr_scheduler_type="cosine",        # smooth decay to ~0; boring, works.
        warmup_ratio=0.03,                 # ramp LR up over the first 3% of
        # steps so early noisy gradients don't blow up the fresh adapter.

        optim="adamw_8bit",                # 8-bit optimizer states: Adam keeps
        # 2 extra floats per trainable param; quantizing them saves memory
        # with negligible quality cost (bitsandbytes trick).

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,                     # dev-loss checkpoints for early-
        save_steps=50,                     # stopping decisions (PRD §5: dev
        save_total_limit=2,                # is FOR hyperparameter decisions)
        output_dir="outputs",              # -> point to Drive on Colab so a
        # session death mid-training doesn't cost you the run (PRD §9 risk 1)

        report_to="wandb",                 # loss curves for the report
        run_name="expenselm-sft-r16",
        seed=42,
        bf16=False, fp16=True,             # T4 is pre-Ampere: no bf16 support.
    ),
)

stats = trainer.train()
print(stats)

# ---------------------------------------------------------------------------
# 5. Save ONLY the adapter (a few hundred MB -> the base stays on HF Hub)
# ---------------------------------------------------------------------------
model.save_pretrained("sft-adapter")
tokenizer.save_pretrained("sft-adapter")
# Download this folder / copy to Drive. Locally it becomes models/sft-adapter
# and E2 is: expenselm eval --system e2 --split data/splits/test.jsonl
