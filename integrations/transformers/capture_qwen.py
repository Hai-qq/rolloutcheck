"""Capture two real generation sessions: re-rendered history and upstream fixed control."""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))
import torch  # noqa: E402
import transformers  # noqa: E402
from prepare_assets import verify_assets  # noqa: E402
from prepare_model import REVISION, WEIGHT_SHA256, verify_weights  # noqa: E402
from upstream_helpers import (  # noqa: E402
    _assert_append_only_prompt,
    _continue_from_canonical_turn,
    _reasoning_preserving_chat_template,
    _render_token_ids,
)

from rolloutcheck.trace import TraceRecorder, inspect_trace  # noqa: E402
from rolloutcheck.transformers_capture import generate_recorded  # noqa: E402


def run(assets, output, device, max_new_tokens):
    asset_lock = verify_assets(assets)
    verify_weights(assets)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        assets, local_files_only=True, trust_remote_code=False
    )
    if device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    print(f"Loading fixed Qwen3-0.6B weights on {device}", flush=True)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            assets,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    space = "sha256:" + asset_lock["files"]["tokenizer.json"]
    contract = {
        "version": "history-prefix/v1",
        "mode": "append_only",
        "history_policy": "preserved",
    }
    metadata = {
        "model": "Qwen/Qwen3-0.6B",
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "transformers_version": transformers.__version__,
        "dtype": "float32",
        "attention_implementation": "eager",
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "experiment": "deliberate history re-render versus upstream canonical continuation",
        "source_issue": "https://github.com/THUDM/slime/issues/2288",
        "scope": "actual Transformers generation; not a complete slime/SGLang deployment",
    }
    measurements = []
    baseline = {}
    for mode in ("rerender", "canonical"):
        session = f"qwen-{mode}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2? Give a brief explanation."},
        ]
        config = transformers.GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            max_time=90,
        )
        first_input = _render_token_ids(messages, tokenizer, tools=None)
        with TraceRecorder(
            output / f"{mode}.trace.jsonl", trace_id=session, evidence_kind="observed_rollout"
        ) as recorder:
            first_output, measurement = generate_recorded(
                model,
                torch.tensor([first_input], device=device),
                recorder=recorder,
                session_id=session,
                branch_id="main",
                turn_id="1",
                parent_turn_id=None,
                token_space=space,
                contract=contract,
                generation_config=config,
                metadata=metadata,
            )
            measurements.append({"mode": mode, "turn": 1, **measurement})
            print(
                f"{mode}: first generation {len(first_output)} tokens, "
                f"stop={measurement['stop_reason']}",
                flush=True,
            )
            if measurement["stop_reason"] != "eos":
                raise RuntimeError(
                    "First response did not terminate with EOS; do not pretend "
                    "this is a completed multi-turn conversation"
                )
            # Strip only the final EOS for the client's structured content view.
            # The recorder and canonical history retain every original output ID.
            content = tokenizer.decode(
                first_output[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if "</think>" not in content:
                raise RuntimeError(
                    "Expected reasoning output was absent; target mechanism unverified"
                )
            response = {"role": "assistant", "content": content}
            messages += [response, {"role": "user", "content": "Reply with one word: done."}]
            canonical = first_input + first_output
            if mode == "rerender":
                next_input = _render_token_ids(messages, tokenizer, tools=None)
            else:
                template = _reasoning_preserving_chat_template(tokenizer)
                rendered = _render_token_ids(
                    messages, tokenizer, tools=None, chat_template=template
                )
                next_input = _continue_from_canonical_turn(
                    messages,
                    tokenizer,
                    tools=None,
                    chat_template=template,
                    rendered_prompt_ids=rendered,
                    previous_turn_ids=canonical,
                    previous_response_message=response,
                )
            try:
                _assert_append_only_prompt(canonical, next_input)
            except RuntimeError as exc:
                baseline[mode] = {"status": "FAIL", "message": str(exc)}
            else:
                baseline[mode] = {"status": "PASS"}
            config.max_new_tokens = 16
            _, measurement = generate_recorded(
                model,
                torch.tensor([next_input], device=device),
                recorder=recorder,
                session_id=session,
                branch_id="main",
                turn_id="2",
                parent_turn_id="1",
                token_space=space,
                contract=contract,
                generation_config=config,
                metadata=metadata,
                boundaries=[
                    {
                        "name": f"after_{mode}",
                        "origin": "captured",
                        "token_space": space,
                        "input_ids": next_input,
                    }
                ],
            )
            measurements.append({"mode": mode, "turn": 2, **measurement})
        report, cases = inspect_trace(output / f"{mode}.trace.jsonl")
        if report["status"] != ("FAIL" if mode == "rerender" else "PASS"):
            raise AssertionError("Target fail/pass contrast was not observed")
        (output / f"{mode}.report.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / f"{mode}.case.json").write_text(json.dumps(cases[0], indent=2) + "\n")
    summary = {
        "execution": "observed_model_generation",
        "decoding": "greedy",
        "scope": metadata["scope"],
        "device": device,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "model": metadata,
        "measurements": measurements,
        "upstream_baseline": baseline,
        "capture_timing_scope": "Host copies, validation and JSONL flush; excludes rendering; "
        "single run, not a before/after overhead benchmark",
        "training_experiment": False,
        "external_adoption": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    run(args.assets, args.output, args.device, args.max_new_tokens)
