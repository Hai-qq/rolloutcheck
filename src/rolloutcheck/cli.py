"""Local inspect and evidence export commands; neither executes case-provided code."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .case import MAX_CASE_BYTES, CaseError, load_case
from .evidence import export_bundle, export_trace_details, verify_evidence_details
from .history import inspect_case
from .text_report import render_text
from .trace import inspect_trace

EXIT_CODES = {"PASS": 0, "FAIL": 1, "ERROR": 2, "INCONCLUSIVE": 3, "NOT_APPLICABLE": 4}


def _export(source, destination, report):
    with source.open("rb") as stream:
        raw = stream.read(MAX_CASE_BYTES + 1)
    if hashlib.sha256(raw).hexdigest() != report["case_sha256"]:
        raise CaseError("Source changed after inspection; export aborted")
    export_bundle(destination, kind="case", raw=raw, report=report)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline rollout history-token contract checker")
    parser.add_argument("--version", action="version", version=f"rolloutcheck {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Check a local JSON case; print a JSON report")
    inspect.add_argument("case", type=Path)
    trace = sub.add_parser("inspect-trace", help="Check raw-token JSONL generation records")
    trace.add_argument("trace", type=Path)
    trace.add_argument("--cases-dir", type=Path, help="Export extracted cases to a new directory")
    export = sub.add_parser(
        "export-evidence", help="Export a witness-only bundle, never executable code"
    )
    export.add_argument("case", type=Path)
    export.add_argument("destination", type=Path)
    export_trace_parser = sub.add_parser(
        "export-trace-evidence",
        help="Bundle a complete trace and its report, including capture gaps",
    )
    export_trace_parser.add_argument("trace", type=Path)
    export_trace_parser.add_argument("destination", type=Path)
    verify = sub.add_parser("verify-evidence", help="Verify bundle hashes and recompute its report")
    verify.add_argument("directory", type=Path)
    for command in (inspect, trace, export, export_trace_parser, verify):
        command.add_argument(
            "--format",
            choices=("json", "text"),
            default="json",
            help="Output format (default: json); exit codes are unchanged",
        )
    args = parser.parse_args(argv)
    cases = []
    try:
        if args.command == "verify-evidence":
            report, cases = verify_evidence_details(args.directory)
        elif args.command == "export-trace-evidence":
            report, cases = export_trace_details(args.trace, args.destination)
            report["export"] = {"kind": "witness_only", "directory": str(args.destination)}
        elif args.command == "inspect-trace":
            report, cases = inspect_trace(args.trace, include_turn_ids=args.format == "text")
            if args.cases_dir is not None:
                args.cases_dir.mkdir(parents=True, exist_ok=False)
                for case in cases:
                    # Text rendering enriches identity only; preserve the existing
                    # extracted-case payload for either stdout format.
                    if args.format == "text":
                        case = {**case, "next": dict(case["next"])}
                        case["next"].pop("turn_id", None)
                    (args.cases_dir / (case["case_id"] + ".json")).write_text(
                        json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                (args.cases_dir / "trace.report.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                report["export"] = {"kind": "witness_only", "directory": str(args.cases_dir)}
        else:
            case, digest = load_case(args.case)
            report = dict(inspect_case(case), case_sha256=digest)
            cases = [case]
            if args.command == "export-evidence":
                _export(args.case, args.destination, report)
                report["export"] = {"kind": "witness_only", "directory": str(args.destination)}
    except (OSError, CaseError, RecursionError) as exc:
        report = {"status": "ERROR", "reason": str(exc)}
    print(
        render_text(report, cases)
        if args.format == "text"
        else json.dumps(report, indent=2, ensure_ascii=False)
    )
    return EXIT_CODES[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
