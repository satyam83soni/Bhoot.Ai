"""Run baselines on eval datasets (repo tool, not shipped).

    python -m tools.baselines run --model openai/<model> --dataset evals/bank_statement --limit 5

Needs the provider's API key in the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY or
GEMINI_API_KEY). Pass --price-in/--price-out (USD per million tokens) to get costs.
Writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.baselines.base import run_dataset, summarise
from tools.baselines.raw_llm import RawLLMBaseline


def _run(args: argparse.Namespace) -> int:
    runner = RawLLMBaseline(
        args.model, api_base=args.api_base, price_per_mtok=(args.price_in, args.price_out)
    )
    reports = []
    for dataset in args.dataset:
        path = Path(dataset)
        if not (path / "manifest.yaml").exists():
            print(f"no manifest.yaml in {path}", file=sys.stderr)
            return 2
        results = run_dataset(runner, path, limit=args.limit)
        report = summarise(runner, path, results)
        reports.append(report)
        m = report["metrics"]
        print(
            f"{path.name:22s} cases={report['cases']:3d} errors={report['errors']:2d} "
            f"field_acc={m['field_accuracy']:.3f} wrong_rate={m['wrong_value_rate']:.3f} "
            f"row_recall={m['row_recall']:.3f} cost/doc=${report['cost_usd_per_doc']:.4f}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(reports, indent=2) + "\n").encode("utf-8"))
    print(f"report written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.baselines", description="Baseline runners"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the raw LLM baseline")
    run.add_argument("--model", required=True, help="provider/model, e.g. openai/<model>")
    run.add_argument("--api-base", default=None, help="any OpenAI-compatible server")
    run.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    run.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    run.add_argument("--dataset", action="append", required=True, help="repeatable")
    run.add_argument("--limit", type=int, default=None, help="first N cases per dataset")
    run.add_argument("--out", default="benchmarks/baseline_results.json")
    run.set_defaults(func=_run)
    args = parser.parse_args(argv)
    return int(args.func(args))
