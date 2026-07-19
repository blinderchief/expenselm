# Week 6 — Export to GGUF & run locally (E5)

The loop-closer: your trained model, quantized, chatting on your own Mac,
measured by the same harness as everything else.

## Concepts first (viva material)

- **Merging.** LoRA kept the base frozen and trained a low-rank delta. For
  deployment you *merge*: `W_new = W + (alpha/r)·B@A`, producing a single
  standalone model. After merging there's no adapter anymore.
- **GGUF** is llama.cpp's file format: weights + tokenizer + chat template
  + quantization metadata in one file — the reason LM Studio can load it
  with zero config.
- **Q4_K_M** is a *post-training* 4-bit quantization (different beast from
  the NF4 4-bit you *trained* with — that one was only for holding frozen
  weights during training). K-quants use per-block scales with some layers
  kept at higher precision ("M" = medium mix). Q4_K_M is the community
  default: ~2.5GB for a 4B model, minimal quality loss — *measured
  generically*. **E5 exists to measure the loss on YOUR task** (PRD E5:
  "cost of quantization on YOUR task"), because JSON-emitting tasks can be
  more quantization-sensitive than chat: one corrupted brace = schema fail.

## Step 1 — Merge + export (on Colab, after DPO)

Unsloth does merge + convert + quantize in one call:

```python
model.save_pretrained_gguf(
    "expenselm-gguf",
    tokenizer,
    quantization_method="q4_k_m",   # also export "q8_0" as a control point:
)                                    # if q8 ≈ fp16 but q4 drops, the damage
                                     # curve is steep — report that.
```

Download the `.gguf` (~2.5GB) to your Mac.

Alternative (manual, worth knowing): merge with
`model.save_pretrained_merged("merged", tokenizer)`, then llama.cpp's
`convert_hf_to_gguf.py merged --outfile f16.gguf` and
`llama-quantize f16.gguf q4_k_m.gguf Q4_K_M`.

## Step 2 — LM Studio

1. Drop the `.gguf` into `~/.lmstudio/models/expenselm/expenselm-q4/`
2. Load it in LM Studio; chat with it manually first — enjoy this moment,
   it's your model.
3. **Developer tab → Start Server** (OpenAI-compatible, localhost:1234).

## Step 3 — E5 row

```bash
expenselm eval --system e5 --split data/splits/test.jsonl
expenselm report
```

Also record **tokens/sec** from LM Studio's UI — that's the local half of
metric #5 (cost/latency) vs. the Claude API's $/1k messages:

| | Claude API (E4) | ExpenseLM local (E5) |
|---|---|---|
| Cost per 1k messages | ~$7–10 (opus-4-8, ~1.2k in/300 out tok) | $0 marginal |
| Data leaves machine | yes | no |

That table is the business argument in one glance.

## Bonus (PRD §6) — one MLX-LoRA run on the M5 Pro

Worth one run just to have touched Apple's stack:

```bash
pip install mlx-lm
mlx_lm.lora --model Qwen/Qwen3-4B-Instruct-2507 --train \
    --data data/mlx/ --iters 600 --batch-size 2 --num-layers 16
```

(`mlx-lm` expects {"text": ...} JSONL — reuse the rendered ChatML text.)
Compare tokens/sec against the T4: unified memory vs discrete GPU is a
nice paragraph in the report's "compute" section.
