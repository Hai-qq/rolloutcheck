import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upstream_excerpt_has_not_been_changed():
    folder = ROOT / "integrations/slime"
    source = json.loads((folder / "source.json").read_text())
    text = (folder / "upstream_helpers.py").read_text()
    excerpt = text[text.index("def _render_token_ids(") :]
    assert hashlib.sha256(excerpt.encode()).hexdigest() == source["excerpt_sha256"]
    assert (
        (folder / "slime-LICENSE").read_text().startswith("                                 Apache")
    )
