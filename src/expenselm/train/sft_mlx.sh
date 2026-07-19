#!/bin/zsh
# ============================================================================
# MLX-LoRA SFT on Apple Silicon — the PRD's local-compute path (§6).
# QLoRA-equivalent: the base is the community 4-bit MLX conversion (frozen),
# only LoRA adapters train. Runs on the M5 Pro in roughly 1-2 hours.
#
# Every flag explained (Week-4 rule applies here too):
#   --mask-prompt        loss ONLY on the assistant JSON, not on the long
#                        system prompt. Without this ~70% of gradient signal
#                        is wasted re-learning to parrot the prompt.
#   --iters 1200         batch 2 × 1200 = 2400 samples ≈ 1.5 epochs of 1600.
#   --num-layers 16      adapters on the last 16 (of 36) layers — the mlx-lm
#                        default trade-off; task adaptation concentrates in
#                        later layers, and this halves trainable memory.
#   --max-seq-length 2048  same as the CUDA recipe (schema+policy prompt fits).
#   --val-batches 25     50 dev examples per eval — cheap early-warning signal.
#   --save-every 200     checkpoints survive an interruption.
# ============================================================================
set -e
cd "$(dirname "$0")/../../.."   # repo root

uv run python -m mlx_lm lora \
    --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --train \
    --data data/mlx \
    --mask-prompt \
    --iters 1200 \
    --batch-size 2 \
    --num-layers 16 \
    --max-seq-length 2048 \
    --learning-rate 2e-4 \
    --val-batches 25 \
    --steps-per-eval 200 \
    --save-every 200 \
    --adapter-path models/sft-adapter-mlx

echo "adapter saved -> models/sft-adapter-mlx"
echo "next: EXPENSELM_BACKEND=mlx uv run expenselm eval --system e2 --split data/splits/test.jsonl"
