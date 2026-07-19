"""The six systems E0..E5 behind one tiny interface: .generate(messages)->str.

The harness doesn't know or care whether it's talking to an API, a local
HF model, or a llama.cpp server — that's the whole point. Every system
receives the identical `messages` array from prompts.build_messages().

| System | Class            | Where it runs                          |
|--------|------------------|----------------------------------------|
| E0     | HFSystem         | Colab GPU (or Mac, slowly) — base model |
| E1     | HFSystem+few_shot| same, with 5 in-context examples        |
| E2     | HFSystem(adapter)| base + SFT LoRA adapter                 |
| E3     | HFSystem(adapter)| base + DPO LoRA adapter                 |
| E4     | GeminiSystem     | Gemini API (the frontier ceiling)       |
| E5     | LMStudioSystem   | your Mac via LM Studio's local server   |
"""

from __future__ import annotations

import json
import os


class GeminiSystem:
    """E4 baseline: frontier model, zero-shot, same prompt as everyone else.

    Ideally this is gemini-2.5-pro (the strongest ceiling), but the free
    tier has ZERO Pro quota ("limit: 0"), so the default is
    gemini-flash-latest — still a frontier-lab model, just the fast tier.
    Record which model filled the E4 row in your report; if you ever add
    billing, re-run with EXPENSELM_E4_MODEL=gemini-2.5-pro (test-set rule
    caveat: replacing E4's model is a NEW system row, not a second peek —
    the constraint is on tuning YOUR model against test, and E4 has no
    knobs to tune).
    """

    def __init__(self, model: str | None = None):
        from google import genai

        self.client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
        self.model = model or os.environ.get("EXPENSELM_E4_MODEL", "gemini-flash-latest")

    def generate(self, messages: list[dict]) -> str:
        from google.genai import types

        system = next(m["content"] for m in messages if m["role"] == "system")
        # Gemini's roles are user/model; few-shot turns map onto them 1:1.
        contents = [
            {"role": "user" if m["role"] == "user" else "model",
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        # Free-tier 429s are normal traffic — backoff instead of dying.
        # Combined with the harness's resume, a quota wall costs time, not work.
        import time

        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0,   # greedy, same rule as every other system
                        # Gemini's THINKING tokens count against this budget;
                        # 2048 truncated multi-expense answers mid-JSON.
                        max_output_tokens=8192,
                    ),
                )
                return resp.text or ""
            except Exception as e:
                if attempt == 5:
                    raise
                wait = min(90, 10 * 2**attempt)
                print(f"  API error ({e.__class__.__name__}), retry {attempt+1}/5 in {wait}s",
                      flush=True)
                time.sleep(wait)
        return ""  # unreachable


class ClaudeSystem:
    """Optional extra baseline (needs `pip install anthropic` + ANTHROPIC_API_KEY).

    Kept around in case you want a second frontier column in the table —
    two ceilings make the comparison more credible, but it's not required.
    """

    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, messages: list[dict]) -> str:
        system = next(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=chat,
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class HFSystem:
    """E0/E1/E2/E3: HuggingFace transformers, optionally with a LoRA adapter.

    Runs on Colab (fast) or CPU/MPS Mac (slow but works for smoke tests).
    For E1, pass `few_shot=` a list of dev examples — the harness picks it
    up automatically and prepends them as demonstration turns.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        adapter_path: str | None = None,
        few_shot: list[dict] | None = None,
        max_new_tokens: int = 1024,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load the tokenizer from the ADAPTER when one is given: a fine-tuned
        # adapter ships the exact chat template it was trained with, and using
        # the base model's template instead causes the model to ramble and
        # emit no JSON (train/inference template mismatch — LEARNING_GUIDE §6).
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name)
        if adapter_path:
            # Our LoRA adapters were trained on a 4-bit (NF4) base, so we MUST
            # eval on a 4-bit base too — applying them to a full-precision base
            # shifts the activations they were calibrated against and degrades
            # output badly (empty / non-JSON). Load 4-bit, then bolt on the LoRA.
            from transformers import BitsAndBytesConfig
            from peft import PeftModel

            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map="auto"
            )
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:  # E0/E1: plain base model, no adapter
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
        self.few_shot = few_shot
        self.max_new_tokens = max_new_tokens
        self.torch = torch

    def generate(self, messages: list[dict]) -> str:
        # apply_chat_template renders messages into the model's native
        # ChatML-style format — the SAME template used at training time.
        # Template mismatch between train and inference is the single most
        # common silent fine-tuning bug. See docs/LEARNING_GUIDE.md §6.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # Qwen's chat template exposes <tool_call> tokens; under greedy decoding
        # the SFT model sometimes collapses into repeating them instead of
        # emitting JSON (diagnosed in results/e2.json — 19% of the first E2 run).
        # Suppress those tokens so the model must produce an answer.
        suppress = []
        for t in ("<tool_call>", "</tool_call>"):
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                suppress.append(tid)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy: deterministic eval, no sampling noise
                pad_token_id=self.tokenizer.eos_token_id,
                suppress_tokens=suppress or None,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


class LMStudioSystem:
    """E5: the quantized GGUF served by LM Studio on your Mac.

    LM Studio exposes an OpenAI-compatible server at localhost:1234
    (Developer tab -> Start Server, after loading the GGUF). This closes
    the PRD's loop: your own quantized model, chatting locally, measured
    by the same harness.
    """

    def __init__(self, base_url: str | None = None, model: str = "expenselm"):
        self.base_url = base_url or os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
        self.model = model

    def generate(self, messages: list[dict]) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,      # greedy, same as HFSystem
                "max_tokens": 1024,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class MLXSystem:
    """E0/E1/E2 on Apple Silicon via MLX — the PRD's local-compute path.

    Uses the 4-bit community conversion of the same base model, so local
    LoRA training here is QLoRA-equivalent (frozen 4-bit base + adapters).
    Select with EXPENSELM_BACKEND=mlx; adapters live in models/*-mlx.
    """

    MODEL_4BIT = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

    def __init__(self, adapter_path: str | None = None,
                 few_shot: list[dict] | None = None, max_new_tokens: int = 1024):
        from mlx_lm import load

        self._model, self.tokenizer = load(self.MODEL_4BIT, adapter_path=adapter_path)
        # `.model` is the string the harness records in the metrics JSON
        self.model = self.MODEL_4BIT + (f" + {adapter_path}" if adapter_path else "")
        self.few_shot = few_shot
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict]) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self._model, self.tokenizer, prompt=prompt,
            max_tokens=self.max_new_tokens,
            sampler=make_sampler(temp=0.0),  # greedy — same rule as every system
            verbose=False,
        )


def build_system(name: str, dev_examples: list[dict] | None = None):
    """Factory the CLI uses. `name` in {e0,e1,e2,e3,e4,e5}.

    EXPENSELM_BACKEND=mlx routes the local rows through Apple-Silicon MLX
    (adapters under models/*-mlx); default is HF/CUDA for Colab.
    """
    name = name.lower()
    mlx = os.environ.get("EXPENSELM_BACKEND", "hf").lower() == "mlx"

    if name == "e0":
        return MLXSystem() if mlx else HFSystem()
    if name == "e1":
        assert dev_examples, "E1 needs --dev-file to draw 5 shots from"
        return (MLXSystem(few_shot=dev_examples[:5]) if mlx
                else HFSystem(few_shot=dev_examples[:5]))
    if name == "e2":
        return (MLXSystem(adapter_path="models/sft-adapter-mlx") if mlx
                else HFSystem(adapter_path="models/sft-adapter"))
    if name == "e3":
        return (MLXSystem(adapter_path="models/dpo-adapter-mlx") if mlx
                else HFSystem(adapter_path="models/dpo-adapter"))
    if name == "e4":
        return GeminiSystem()
    if name == "e5":
        return LMStudioSystem()
    raise ValueError(f"unknown system {name!r} (expected e0..e5)")
