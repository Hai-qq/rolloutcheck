"""Run audited upstream function excerpts with a real tokenizer and controlled input.

No sampling, servers, external tools, GPU, or arbitrary case-provided code.
Run from the repository root, after explicit asset preparation.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"

import transformers  # noqa: E402
from prepare_assets import verify_assets  # noqa: E402
from upstream_helpers import (  # noqa: E402
    _assert_append_only_prompt,
    _continue_from_canonical_turn,
    _reasoning_preserving_chat_template,
    _render_token_ids,
)

from rolloutcheck import inspect_case  # noqa: E402


def run(assets, output):
    lock = verify_assets(assets)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        assets, local_files_only=True, trust_remote_code=False
    )
    source = json.loads(Path(__file__).with_name("source.json").read_text())
    initial_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    response = {"role": "assistant", "content": "4.", "reasoning_content": "Two plus two is four."}
    messages = initial_messages + [response, {"role": "user", "content": "Explain why."}]
    # Explicitly hand-authored response suffix. NOT sampled or reconstructed as real evidence.
    frozen_output = "<think>\nTwo plus two is four.\n</think>\n\n4.<|im_end|>\n"
    previous_input = _render_token_ids(initial_messages, tokenizer, tools=None)
    previous_output = tokenizer.encode(frozen_output, add_special_tokens=False)
    canonical = previous_input + previous_output
    # Sanity-check that the authored response matches this tokenizer's serialization
    # before introducing a new user turn; this does not generate the expected IDs.
    assert (
        _render_token_ids(
            initial_messages + [response], tokenizer, tools=None, add_generation_prompt=False
        )
        == canonical
    )
    bad = _render_token_ids(messages, tokenizer, tools=None)
    template = _reasoning_preserving_chat_template(tokenizer)
    rerendered = _render_token_ids(messages, tokenizer, tools=None, chat_template=template)
    good = _continue_from_canonical_turn(
        messages,
        tokenizer,
        tools=None,
        chat_template=template,
        rendered_prompt_ids=rerendered,
        previous_turn_ids=canonical,
        previous_response_message=response,
    )
    # Independent identity check: expected is the frozen history, not the rerendered history.
    if bad[: len(canonical)] == canonical or good[: len(canonical)] != canonical:
        raise AssertionError("The declared fail/pass conversion contrast was not reproduced")
    try:
        _assert_append_only_prompt(canonical, bad)
    except RuntimeError as exc:
        upstream_detection = str(exc)
    else:
        raise AssertionError("Expected upstream's existing assertion to detect the same drift")
    _assert_append_only_prompt(canonical, good)
    space = "sha256:" + lock["files"]["tokenizer.json"]
    case = {
        "schema_version": 1,
        "case_id": "slime-qwen3-controlled-history-rerender",
        "evidence": {
            "kind": "controlled_upstream_transform",
            "description": "Captured outputs of pinned upstream function excerpts using a real "
            "Qwen3 tokenizer and hand-authored response; no model sampling.",
            "source": source,
            "tokenizer": lock,
            "initial_messages": initial_messages,
            "messages": messages,
            "frozen_response_text": frozen_output,
        },
        "contract": {
            "version": "history-prefix/v1",
            "mode": "append_only",
            "history_policy": "preserved",
        },
        "previous": {
            "session_id": "controlled",
            "branch_id": "main",
            "turn_id": "1",
            "token_space": space,
            "input_ids": previous_input,
            "output_ids": previous_output,
        },
        "next": {
            "session_id": "controlled",
            "branch_id": "main",
            "previous_turn_id": "1",
            "token_space": space,
            "input_ids": bad,
        },
        "boundaries": [
            {
                "name": "full_history_rerender",
                "origin": "captured",
                "token_space": space,
                "input_ids": bad,
            }
        ],
    }
    passed = copy.deepcopy(case)
    passed["case_id"] += "-canonical-continuation"
    passed["next"]["input_ids"] = good
    passed["boundaries"] = [
        {
            "name": "canonical_continuation",
            "origin": "captured",
            "token_space": space,
            "input_ids": good,
        }
    ]
    failed_report, passed_report = inspect_case(case), inspect_case(passed)
    assert failed_report["status"] == "FAIL" and passed_report["status"] == "PASS"
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ("failure.case.json", case),
        ("control.case.json", passed),
        ("failure.report.json", failed_report),
        ("control.report.json", passed_report),
    ):
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "experiment": "slime-qwen3-controlled-transform",
        "execution": "transform_reproduced",
        "scope": "unmodified upstream function excerpts, not full adapter or original issue run",
        "upstream_fix_attribution": "THUDM/slime PR #2287 by zy20031230",
        "failure_status": failed_report["status"],
        "control_status": passed_report["status"],
        "existing_upstream_assertion": upstream_detection,
        "incremental_value": (
            "structured evidence and export; practical advantage not yet validated"
        ),
        "first_difference": failed_report["first_difference"],
        "previous_input_tokens": len(previous_input),
        "frozen_output_tokens": len(previous_output),
        "failure_next_input_tokens": len(bad),
        "control_next_input_tokens": len(good),
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "model_sampling": False,
        "training_experiment": False,
        "external_adoption": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    run(args.assets, args.output)
