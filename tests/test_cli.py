import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "rolloutcheck.cli", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    "filename,code,status",
    [
        ("pass", 0, "PASS"),
        ("drift", 1, "FAIL"),
        ("missing", 3, "INCONCLUSIVE"),
        ("rewrite", 4, "NOT_APPLICABLE"),
    ],
)
def test_cli_exit_contract(filename, code, status):
    result = cli("inspect", ROOT / "cases/synthetic" / f"{filename}.json")
    assert result.returncode == code
    assert json.loads(result.stdout)["status"] == status


def test_missing_file_is_structured_error(tmp_path):
    result = cli("inspect", tmp_path / "absent.json")
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "ERROR"


def test_export_preserves_original_and_refuses_overwrite(tmp_path):
    source = ROOT / "cases/synthetic/drift.json"
    destination = tmp_path / "bundle"
    result = cli("export-evidence", source, destination)
    assert result.returncode == 1  # Export success does not hide a failed contract.
    assert (destination / "case.json").read_bytes() == source.read_bytes()
    assert "witness_only" in (destination / "README.md").read_text()
    assert cli("inspect", destination / "case.json").returncode == 1
    assert cli("export-evidence", source, destination).returncode == 2


def test_core_does_not_import_training_or_tokenizer_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rolloutcheck; "
            "assert not {'torch','transformers','slime','requests'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        env=dict(os.environ, HF_HUB_OFFLINE="1"),
    )
    assert result.returncode == 0, result.stderr


def test_export_detects_source_change(tmp_path):
    from rolloutcheck.case import CaseError, load_case
    from rolloutcheck.cli import _export

    source = tmp_path / "source.json"
    source.write_bytes((ROOT / "cases/synthetic/drift.json").read_bytes())
    _, digest = load_case(source)
    source.write_text("{}")
    with pytest.raises(CaseError, match="Source changed"):
        _export(source, tmp_path / "bundle", {"case_sha256": digest})
    assert not (tmp_path / "bundle").exists()
