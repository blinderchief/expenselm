# ============================================================================
# Raw TRL + PEFT mini-run — "so Unsloth isn't magic to you" (PRD §6).
#
# Same recipe as sft_unsloth.py but with the standard HuggingFace stack and
# ZERO unsloth. Run it on ~200 examples for a few hundred steps, then diff:
#
#   1. tokens/sec           (expect unsloth ~2x faster)
#   2. peak VRAM            (expect unsloth ~50-70% lower)
#   3. loss curve SHAPE     (should be near-identical — same math!)
#
# Point 3 is the lesson: unsloth is an ENGINEERING optimization (hand-fused
# Triton kernels, recomputed RoPE, chunked cross-entropy), not a different
# algorithm. If the loss curves diverged, something would be wrong.
#
# Colab: !pip install trl peft bitsandbytes datasets
# ============================================================================

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# --- 4-bit loading, spelled out (unsloth's load_in_4bit=True does this) ---
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4: quantile-based 4-bit type
                                        # designed for normally-distributed weights
                                        # (QLoRA paper, Dettmers 2023)
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants too
                                        # (~0.4 bits/param extra saving)
    bnb_4bit_compute_dtype=torch.float16,  # dequantize to fp16 for the matmuls
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)  # casts norms to fp32, enables
                                                # grad checkpointing hooks

# --- LoRA, spelled out (unsloth's get_peft_model does this) ---
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ),
)
model.print_trainable_parameters()  # expect ~0.5-1% trainable — say this
                                    # number out loud, it's the point of LoRA

dataset = load_dataset("json", data_files={"train": "train_small.jsonl"})

def render(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)}

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"].map(render),
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=1,      # raw HF eats more VRAM than
        gradient_accumulation_steps=16,     # unsloth — compensate here
        max_steps=200,                      # mini-run: comparison, not a model
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        logging_steps=10,
        output_dir="outputs-trl-ref",
        report_to="wandb",
        run_name="expenselm-sft-trl-reference",
        seed=42,
        fp16=True,
    ),
)

torch.cuda.reset_peak_memory_stats()
trainer.train()
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
# ^ record this and the same number from the unsloth run in your report.
