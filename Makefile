# Convenience targets. `make help` lists them. Uses uv throughout.

.PHONY: help setup test gen splits decon baselines report clean

help:
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## create venv + install package with dev deps (uv)
	uv venv && uv pip install -e ".[dev]"

test: ## run the harness's own tests (keep green, always)
	uv run pytest -q

gen: ## generate synthetic data (needs GEMINI_API_KEY)
	uv run expenselm gen --n 1900 --out data/raw_generations/v1.jsonl

splits: ## perturb + deterministic train/dev/test split
	uv run expenselm perturb --in data/raw_generations/v1.jsonl --out data/master.jsonl
	uv run python -c "from expenselm.data.format import split; split('data/master.jsonl')"

decon: ## contamination check (removes offending TRAIN examples)
	uv run expenselm decontaminate --train data/splits/train.jsonl --test data/splits/test.jsonl

baselines: ## E4 (API) — E0/E1 need a GPU, run on Colab
	uv run expenselm eval --system e4 --split data/splits/test.jsonl

report: ## print the combined results table
	uv run expenselm report

clean:
	rm -rf .pytest_cache src/*.egg-info
