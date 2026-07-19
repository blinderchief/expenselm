# Optimization Landscape — what the big labs do, and what applies here

You asked about "the optimizations in DeepSeek models, DeepSpeed, and
others". This doc maps that landscape and — the part that matters —
sorts each technique into **use here / understand but don't use / v2**.
Knowing *when not* to add a tool is part of the skill (PRD §6 note).

---

## 1. DeepSeek's optimizations (V2 → V3 → R1)

DeepSeek's efficiency results come from co-designing architecture,
numerics, and RL. The four famous pieces:

### MLA — Multi-head Latent Attention
Standard attention caches a Key and Value vector per head per token (the
KV cache) — at long context this dominates GPU memory. MLA compresses K/V
into a shared low-rank *latent* vector and decompresses per head on the
fly: ~9.6× smaller KV cache, which is why DeepSeek can serve huge batches
cheaply.
**Relation to your project:** conceptually a cousin of LoRA — both exploit
low-rank structure, LoRA in *weight updates*, MLA in *attention state*.
You can't use it (it's baked into architecture at pretraining time), but
"low-rank compression shows up at every layer of the stack" is a great
report/interview line.

### DeepSeekMoE — fine-grained Mixture of Experts
Only a few small "expert" MLPs activate per token (V3: 671B total, ~37B
active). Train giant capacity, pay compute for a fraction. V2 saved ~42.5%
training cost vs. their dense 67B.
**Relation:** none directly — MoE solves a scale problem you don't have.
Qwen3-4B is dense; a 4B MoE would waste capacity on routing overhead.

### FP8 mixed-precision training
Most GEMMs in FP8 with fine-grained scaling (1×128 tiles for activations,
128×128 blocks for weights); sensitive ops (embeddings, attention, gating)
stay BF16/FP32. Roughly halves compute/memory vs BF16 training.
**Relation:** same *philosophy* as your stack, different tier of hardware.
Your QLoRA run is also mixed-precision: 4-bit frozen weights, fp16
compute, fp32 norms, 8-bit optimizer states. FP8 *training* needs H100-
class tensor cores — a T4 (2018, Turing) can't do it. Say "the principle
is 'spend precision only where gradients are fragile'" and you've captured
both.

### GRPO — Group Relative Policy Optimization (the R1 ingredient)
PPO needs a separate value network to estimate advantages. GRPO deletes
it: sample a *group* of responses per prompt, score them, and use each
response's score relative to the group mean as its advantage. Cheaper,
simpler, and the engine behind R1's emergent chain-of-thought.
**Relation:** GRPO/PPO and DPO are both preference-based post-training.
DPO is the offline, no-reward-model, no-sampling-loop version — the right
choice at your scale. A one-paragraph "why DPO and not GRPO" comparison in
your report shows you understand the design space:
- GRPO: online (samples during training), needs a reward signal, shines
  for *reasoning* where you can verify answers programmatically.
- DPO: offline pairs, no sampling loop, one GPU, stable — ideal for 500
  pairs on a T4.
- **v2 idea:** your grader IS a programmatic reward (schema valid? policy
  right?). GRPO/RLVR against your own grader would be a genuinely
  interesting follow-up — note it in "future work".

### R1-style distillation
DeepSeek distilled R1's reasoning into small dense models by SFT on
R1-generated traces — literally your pipeline (teacher generates data,
student SFTs on it), at nation-state scale. Your project is a miniature of
the same pattern; cite it as related work.

## 2. DeepSpeed (you likely meant this by "D spark")

Microsoft's distributed-training library. Its core is **ZeRO** (Zero
Redundancy Optimizer): in normal data-parallel training every GPU holds a
full copy of optimizer states, gradients, and params; ZeRO stages 1/2/3
*shard* them across GPUs, and ZeRO-Offload pushes them to CPU/NVMe.
**Relation: deliberately not used.** ZeRO solves "model too big for N
GPUs' combined memory". You have one GPU and a 4B model in 4-bit —
your memory problems are solved by QLoRA + checkpointing + 8-bit Adam.
Pulling in DeepSpeed here would be resume-driven engineering. Same verdict
for Axolotl/LLaMA-Factory/FSDP (the PRD's own note).

(If you instead meant **DSPy** — that's a prompt-*program* optimizer that
tunes prompts/few-shots against a metric. Orthogonal to fine-tuning; could
optimize the E0/E1 prompts, but that would blur the E0-vs-E2 comparison.
Out of scope, worth one line in related work.)

## 3. Optimizations you ARE using (know them cold)

| Technique | Attacks | Where in repo |
|---|---|---|
| NF4 4-bit quantization (+double quant) | frozen weight memory | `load_in_4bit` / BitsAndBytesConfig |
| LoRA rank-16 adapters | trainable params + optimizer memory | `get_peft_model` |
| Gradient checkpointing | activation memory (recompute in backward) | `use_gradient_checkpointing` |
| Gradient accumulation | effective batch on small VRAM | `gradient_accumulation_steps=8` |
| 8-bit AdamW | optimizer state memory | `optim="adamw_8bit"` |
| Fused Triton kernels | kernel-launch + memory traffic overhead | unsloth itself |
| Post-training quantization (Q4_K_M) | deployment size + CPU/Metal speed | GGUF export |
| Greedy decoding at eval | measurement variance (not speed) | `do_sample=False` |
| Prompt caching (Claude API) | data-gen cost — the long system prompt in `generate.py` is a cacheable prefix | anthropic SDK `cache_control` (optional add) |

## 4. Worth adding IF measurements say so (not before)

- **vLLM for eval inference** (Colab): paged-attention serving makes the
  200-example E2/E3 eval runs several× faster than `model.generate` loops.
  Add only if eval wall-time actually annoys you.
- **Flash Attention 2**: T4 doesn't support it (Ampere+); on Colab L4/A100
  upgrade days, unsloth uses it automatically. Nothing for you to do.
- **Batched generation** in `HFSystem` (batch 8 inputs per forward): ~5-8×
  eval speedup, slightly more code. First candidate if eval is slow.
- **Message Batches API** for data generation (50% off): worth it if you
  regenerate the dataset more than twice.

## 5. The meta-lesson

Every optimization above attacks one term of the same budget:

```
memory = weights + optimizer states + gradients + activations + KV cache
time   = FLOPs + memory traffic + kernel overhead + tokens generated
cost   = GPU-hours (training)  |  $/token (API)  |  latency (serving)
```

DeepSeek attacks it at pretraining scale (MLA, MoE, FP8), DeepSpeed at
multi-GPU scale (ZeRO), you at single-GPU scale (QLoRA + unsloth + GGUF).
Being able to place any technique someone names into this table — which
term it attacks, at which scale it pays off — is exactly the fluency the
"valuation things in the future" require.

Sources: [DeepSeek-V2 paper](https://arxiv.org/pdf/2405.04434) ·
[V3 hardware reflections](https://arxiv.org/pdf/2505.09343) ·
[MLA+FP8 in vLLM (Red Hat)](https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm) ·
[DeepSeek series overview (Fowler)](https://martinfowler.com/articles/deepseek-papers.html) ·
[Unsloth docs](https://unsloth.ai/docs) — the Qwen3-4B notebook there is
the closest working reference for your SFT run.
