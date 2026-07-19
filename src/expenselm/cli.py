"""Command-line entry point. `pip install -e .` gives you the `expenselm` command.

    expenselm gen --n 300 --out data/raw_generations/batch1.jsonl
    expenselm perturb --in data/clean.jsonl --out data/perturbed.jsonl
    expenselm decontaminate --train data/splits/train.jsonl --test data/splits/test.jsonl
    expenselm format --in data/splits/train.jsonl --out data/chatml/train.jsonl
    expenselm eval --system e4 --split data/splits/test.jsonl
    expenselm report
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader — secrets live in a gitignored file, not your shell.

    setdefault means a real environment variable always wins over .env.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="expenselm")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="synthetic data generation via Gemini (GEMINI_API_KEY)")
    g.add_argument("--n", type=int, default=100, help="number of examples to generate")
    g.add_argument("--out", required=True)
    g.add_argument("--seed-file", default="data/seed/seed_examples.jsonl")

    pe = sub.add_parser("perturb", help="inject typos/OCR noise/currency variants")
    pe.add_argument("--infile", "--in", dest="infile", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--rate", type=float, default=0.5, help="fraction of examples to perturb")

    d = sub.add_parser("decontaminate", help="reject near-duplicates between train and test")
    d.add_argument("--train", required=True)
    d.add_argument("--test", required=True)
    d.add_argument("--threshold", type=float, default=85.0, help="rapidfuzz similarity 0-100")

    f = sub.add_parser("format", help="convert examples to ChatML messages JSONL for training")
    f.add_argument("--infile", "--in", dest="infile", required=True)
    f.add_argument("--out", required=True)

    e = sub.add_parser("eval", help="run one system on a split, save metrics")
    e.add_argument("--system", required=True, choices=["e0", "e1", "e2", "e3", "e4", "e5"])
    e.add_argument("--split", required=True, help="path to JSONL split")
    e.add_argument("--dev-file", default="data/splits/dev.jsonl", help="source of few-shots for e1")
    e.add_argument("--limit", type=int, default=None, help="cap examples (smoke tests)")

    sub.add_parser("report", help="print the combined E0..E5 results table")

    args = p.parse_args()

    if args.cmd == "gen":
        from expenselm.data.generate import generate_dataset

        generate_dataset(n=args.n, out_path=args.out, seed_path=args.seed_file)
    elif args.cmd == "perturb":
        from expenselm.data.perturb import perturb_file

        perturb_file(args.infile, args.out, rate=args.rate)
    elif args.cmd == "decontaminate":
        from expenselm.data.contamination import decontaminate

        decontaminate(args.train, args.test, threshold=args.threshold)
    elif args.cmd == "format":
        from expenselm.data.format import to_chatml_file

        to_chatml_file(args.infile, args.out)
    elif args.cmd == "eval":
        from expenselm.eval.harness import load_jsonl, run_eval
        from expenselm.systems import build_system

        dev = load_jsonl(args.dev_file) if args.system == "e1" else None
        system = build_system(args.system, dev_examples=dev)
        metrics = run_eval(system, load_jsonl(args.split), args.system, limit=args.limit)
        import json

        print(json.dumps(metrics, indent=2))
    elif args.cmd == "report":
        from expenselm.eval.harness import report

        report()


if __name__ == "__main__":
    main()
