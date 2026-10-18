"""Command line for the synthetic data generator (repo tool, not shipped to PyPI).

python -m tools.synth bank --count 30 --layouts all --seed 7 --out evals/bank_statement
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from tools.synth.faker_providers import BANKS
from tools.synth.kyc import KYC_TYPES, KycGenerator
from tools.synth.layouts import LAYOUTS
from tools.synth.layouts.kyc_layouts import render as render_kyc
from tools.synth.layouts.salary_layouts import SALARY_LAYOUTS
from tools.synth.ledger import LedgerGenerator
from tools.synth.salary import SalarySlipGenerator
from tools.synth.scan_sim import PROFILES, ScanSimulator
from tools.synth.truth import (
    bank_statement_expected,
    kyc_expected,
    salary_slip_expected,
    write_case,
    write_manifest,
)


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


def _scans(args: argparse.Namespace) -> int:
    profiles = sorted(PROFILES) if args.profiles == "all" else args.profiles.split(",")
    unknown = [name for name in profiles if name not in PROFILES]
    if unknown:
        print(
            f"unknown profile(s): {', '.join(unknown)}; choose from {', '.join(sorted(PROFILES))}",
            file=sys.stderr,
        )
        return 2

    source = Path(args.source)
    sources = sorted(p for p in source.glob("case_*") if (p / "document.pdf").exists())[
        : args.count
    ]
    if not sources:
        print(f"no cases with document.pdf under {source}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cases = []
    for src in sources:
        source_meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        expected = json.loads((src / "expected.json").read_text(encoding="utf-8"))
        pdf = (src / "document.pdf").read_bytes()
        for profile_name in profiles:
            seed = int(source_meta["seed"])
            scan = ScanSimulator(PROFILES[profile_name], seed).simulate(pdf)
            case_id = f"{src.name}_{profile_name}"
            meta = {**source_meta, "profile": profile_name, "source_case": src.name}
            write_case(out / case_id, document=scan, expected=expected, meta=meta)
            cases.append(
                {
                    "id": case_id,
                    "tags": [source_meta["layout"], profile_name, "scanned"],
                    "rows": source_meta["rows"],
                }
            )
            print(f"{case_id:28s} bytes={len(scan)}")

    write_manifest(out, schema="pack:lending.bank_statement", cases=cases)
    print(f"wrote {len(cases)} cases to {out}")
    return 0


def _salary(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    layouts = sorted(SALARY_LAYOUTS)
    cases = []
    for i in range(args.count):
        seed = args.seed * 10_000 + i
        layout = SALARY_LAYOUTS[layouts[i % len(layouts)]]()
        slip = SalarySlipGenerator(seed).generate()
        pdf = layout.render(slip)
        case_id = f"case_{i + 1:04d}"
        meta = {"seed": seed, "layout": layout.name, "profile": "digital"}
        write_case(out / case_id, document=pdf, expected=salary_slip_expected(slip), meta=meta)
        cases.append({"id": case_id, "tags": [layout.name, "digital"]})
        print(f"{case_id}  {layout.name:22s} net={slip.net_pay}")

    write_manifest(out, schema="pack:lending.salary_slip", cases=cases)
    print(f"wrote {len(cases)} cases to {out}")
    return 0


def _kyc(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cases = []
    for i in range(args.count):
        seed = args.seed * 10_000 + i
        kind = KYC_TYPES[i % len(KYC_TYPES)]
        renderer = "image" if (i // len(KYC_TYPES)) % 2 == 0 else "digital"
        doc = KycGenerator(seed).generate(kind)
        pdf = render_kyc(doc, renderer)
        case_id = f"case_{i + 1:04d}"
        meta = {"seed": seed, "document_type": kind, "renderer": renderer, "masked": doc.masked}
        write_case(out / case_id, document=pdf, expected=kyc_expected(doc), meta=meta)
        cases.append({"id": case_id, "tags": [kind, renderer]})
        print(f"{case_id}  {kind:16s} {renderer:8s} bytes={len(pdf)}")

    write_manifest(out, schema="pack:lending.kyc", cases=cases)
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

    scans = sub.add_parser("scans", help="scanned versions of existing digital cases")
    scans.add_argument("--source", default="evals/bank_statement")
    scans.add_argument("--profiles", default="all", help="'all' or comma list, e.g. clean,fax")
    scans.add_argument("--count", type=int, default=10, help="first N source cases")
    scans.add_argument("--out", default="evals/bank_statement_scans")
    scans.add_argument("--clean", action="store_true", help="delete the output folder first")
    scans.set_defaults(func=_scans)

    salary = sub.add_parser("salary", help="salary slips")
    salary.add_argument("--count", type=int, default=20)
    salary.add_argument("--seed", type=int, default=11)
    salary.add_argument("--out", default="evals/salary_slip")
    salary.add_argument("--clean", action="store_true", help="delete the output folder first")
    salary.set_defaults(func=_salary)

    kyc = sub.add_parser("kyc", help="KYC specimens (PAN, Aadhaar, passport, DL, voter ID)")
    kyc.add_argument("--count", type=int, default=20)
    kyc.add_argument("--seed", type=int, default=13)
    kyc.add_argument("--out", default="evals/kyc")
    kyc.add_argument("--clean", action="store_true", help="delete the output folder first")
    kyc.set_defaults(func=_kyc)

    args = parser.parse_args(argv)
    return int(args.func(args))
