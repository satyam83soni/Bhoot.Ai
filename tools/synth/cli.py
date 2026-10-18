"""Command line for the synthetic data generator (repo tool, not shipped to PyPI).

python -m tools.synth bank --count 30 --layouts all --seed 7 --out evals/bank_statement
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

from tools.synth.faker_providers import BANKS
from tools.synth.layouts import LAYOUTS
from tools.synth.ledger import LedgerGenerator
from tools.synth.truth import bank_statement_expected, write_case, write_manifest


def _bank(args: argparse.Namespace) -> int:
    layouts = sorted(LAYOUTS) if args.layouts == "all" else args.layouts.split(",")
    unknown = [name for name in layouts if name not in LAYOUTS]
    if unknown:
        print(
            f"unknown layout(s): {', '.join(unknown)}; choose from {', '.join(sorted(LAYOUTS))}",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cases = []
    for i in range(args.count):
        case_seed = args.seed * 10_000 + i
        layout = LAYOUTS[layouts[i % len(layouts)]]()
        # each layout is printed by its own fictional bank
        bank = BANKS[sorted(LAYOUTS).index(layout.name) % len(BANKS)]
        ledger = LedgerGenerator(case_seed).generate(months=args.months, bank=bank)
        pdf = layout.render(ledger)
        case_id = f"case_{i + 1:04d}"
        kinds = Counter(t.kind for t in ledger.transactions)
        meta = {
            "seed": case_seed,
            "layout": layout.name,
            "profile": "digital",
            "rows": len(ledger.transactions),
            "kinds": dict(sorted(kinds.items())),
        }
        write_case(out / case_id, document=pdf, expected=bank_statement_expected(ledger), meta=meta)
        cases.append(
            {"id": case_id, "tags": [layout.name, "digital"], "rows": len(ledger.transactions)}
        )
        print(f"{case_id}  {layout.name}  rows={len(ledger.transactions):4d}  bytes={len(pdf)}")

    write_manifest(out, schema="pack:lending.bank_statement", cases=cases)
    print(f"wrote {len(cases)} cases to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.synth", description="Synthetic lending documents"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bank = sub.add_parser("bank", help="bank statements")
    bank.add_argument("--count", type=int, default=30)
    bank.add_argument("--layouts", default="all", help="'all' or comma list, e.g. bank_a,bank_c")
    bank.add_argument(
        "--months", type=int, default=None, help="statement length; random 1-6 if omitted"
    )
    bank.add_argument("--seed", type=int, default=7)
    bank.add_argument("--out", default="evals/bank_statement")
    bank.add_argument("--clean", action="store_true", help="delete the output folder first")
    bank.set_defaults(func=_bank)

    args = parser.parse_args(argv)
    return int(args.func(args))
