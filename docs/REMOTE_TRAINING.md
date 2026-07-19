# Remote training — exact steps (Colab primary, Kaggle backup)

Your upload bundle is ready: **`expenselm-remote.zip`** (916 KB, repo root).
It contains code + frozen splits + ChatML files + current results. No
secrets (.env excluded) and no models — the notebook downloads the base
model itself.

One remote session covers the WHOLE remaining GPU pipeline:
**SFT → E2 eval → failure harvest → DPO → E3 eval → GGUF export.**
You run cells top-to-bottom; nothing else to write.

---

## Option A — Google Colab (recommended: the notebook is built for it)

1. Go to https://drive.google.com → upload `expenselm-remote.zip`.
2. Right-click the zip in Drive → Open with → Google Colaboratory is NOT
   needed — instead go to https://colab.research.google.com → File → Upload
   notebook → upload `notebooks/expenselm_colab.ipynb` from this Mac.
3. **Runtime → Change runtime type → T4 GPU → Save.**
4. Replace the notebook's cell 0.2 (the Drive copy) with:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip -q /content/drive/MyDrive/expenselm-remote.zip -d /content/expenselm
   %cd /content/expenselm
   ```
5. Run the install cell (0.3). Paste your W&B key when asked
   (get it at https://wandb.ai/authorize — free account).
6. Skip the Week-3 cells (E0/E1 already done on the Mac ✅) and run the
   **Week-4 SFT cells** onward, in order. SFT ≈ 1–2 h on the T4.
7. After each stage the notebook copies artifacts to
   `MyDrive/expenselm-runs/` — that's your download point.

**Colab survival tips:** keep the tab open (free tier disconnects idle
sessions); if disconnected mid-training, checkpoints are in Drive if you
set `output_dir` to Drive (the notebook's Week-4 note shows how); rerunning
the eval cells is safe — the harness resumes from saved predictions.

## Option B — Kaggle (if Colab won't give you a GPU)

1. https://kaggle.com → sign in → verify phone (required for GPU).
2. Create → New Dataset → upload `expenselm-remote.zip` → name it `expenselm`.
3. Create → New Notebook → File → Import Notebook → upload
   `expenselm_colab.ipynb`.
4. Right panel: **Accelerator → GPU T4 x2** (uses one), **Internet → On**.
5. Add Data → your `expenselm` dataset. Then replace cell 0.2 with:
   ```python
   !unzip -q /kaggle/input/expenselm/expenselm-remote.zip -d /kaggle/working/expenselm
   %cd /kaggle/working/expenselm
   ```
6. Continue exactly as Colab from step 5. Kaggle gives ~30 GPU-hours/week
   and 12-hour sessions — plenty.

## Option C — Unsloth Studio

It's a click-through web UI over the same unsloth engine. It can run the
SFT (upload `data/chatml/train.jsonl`, pick Qwen3-4B-Instruct, LoRA r=16,
2 epochs), but the eval/harvest/DPO steps still need our scripts — so
you'd end up in a notebook anyway. Fine for a first visual look at
training curves; Options A/B are the real path for the project.

## What to bring back to the Mac

Put these from `expenselm-runs/` into this repo:
| Artifact | Goes to |
|---|---|
| `results/*.json`, `results/*_predictions.jsonl` | `results/` (fills E2/E3 rows in `expenselm report`) |
| `sft-adapter/`, `dpo-adapter/` | `models/` |
| `expenselm-*.gguf` (Q4_K_M + Q8_0) | anywhere → load in LM Studio → E5 |

Then tell Claude "artifacts are back" and the E5 eval + failure analysis
run from here.

## The local 2.1 GB MLX model

Keep it until the remote E2 row lands (it's the fallback). After that:
```bash
uv run python -c "from huggingface_hub import scan_cache_dir; print('run:'); print('huggingface-cli delete-cache')"
# or simply:
rm -rf ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit
```
