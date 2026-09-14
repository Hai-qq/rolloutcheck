"""Local inspect and evidence export commands; neither executes case-provided code."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .case import MAX_CASE_BYTES, CaseError, load_case
from .history import inspect_case

EXIT_CODES = {"PASS": 0, "FAIL": 1, "ERROR": 2, "INCONCLUSIVE": 3, "NOT_APPLICABLE": 4}


def _export(source, destination, report):
    with source.open("rb") as stream:
        raw = stream.read(MAX_CASE_BYTES + 1)
    if hashlib.sha256(raw).hexdigest() != report["case_sha256"]:
        raise CaseError("Source changed after inspection; export aborted")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "case.json").write_bytes(raw)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# RolloutCheck evidence bundle\n\n"
        "Status: **witness_only**. This bundle rechecks recorded IDs; it does not "
        "execute the original conversion or replay model sampling.\n\n"
        "With RolloutCheck 0.1.0a1 installed:\n\n"
        "```sh\nrolloutcheck inspect case.json\n```\n\n"
        "See report.json for the original case SHA-256. A hash is not proof of authenticity.\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline rollout history-token contract checker")
    parser.add_argument("--version", action="version", version="rolloutcheck 0.1.0a1")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Check a local JSON case; print a JSON report")
    inspect.add_argument("case", type=Path)
    export = sub.add_parser(
        "export-evidence", help="Export a witness-only bundle, never executable code"
    )
    export.add_argument("case", type=Path)
    export.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        case, digest = load_case(args.case)
        report = dict(inspect_case(case), case_sha256=digest)
        if args.command == "export-evidence":
            _export(args.case, args.destination, report)
            report["export"] = {"kind": "witness_only", "directory": str(args.destination)}
    except (OSError, CaseError, RecursionError) as exc:
        report = {"status": "ERROR", "reason": str(exc)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return EXIT_CODES[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
