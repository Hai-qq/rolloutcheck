"""Actual local Transformers/MPS generation through the pinned slime handoff probe.

This is a sequential verification service, NOT a native SGLang implementation.
It does not run an optimizer or test framework training performance.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

from prepare_assets import verify_assets
from verify_sglang import save
from verify_training_handoff import run

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformers"))
from prepare_model import REVISION, WEIGHT_SHA256, verify_weights  # noqa: E402


async def main(args):
    import torch
    import transformers
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    verify_assets(args.assets)
    verify_weights(args.assets)
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "This runner requires MPS; use the native or CPU-controlled runner instead"
        )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.assets, local_files_only=True, trust_remote_code=False
    )
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            args.assets,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to("mps")
        .eval()
    )
    requests = 0

    async def generate(request):
        nonlocal requests
        body = await request.json()
        inputs = torch.tensor([body["input_ids"]], device="mps")
        config = transformers.GenerationConfig(
            max_new_tokens=body["sampling_params"]["max_new_tokens"],
            do_sample=False,
            num_beams=1,
            use_cache=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=config,
                use_model_defaults=False,
            )
            ids = generated.sequences[0, inputs.shape[1] :].cpu().tolist()
            scores = (
                model.compute_transition_scores(
                    generated.sequences, generated.scores, normalize_logits=True
                )[0]
                .cpu()
                .tolist()
            )
        requests += 1
        return web.json_response(
            {
                "meta_info": {
                    "prompt_tokens": inputs.shape[1],
                    "completion_tokens": len(ids),
                    "finish_reason": {
                        "type": "stop" if ids[-1] == tokenizer.eos_token_id else "length"
                    },
                    "output_token_logprobs": [
                        [score, token, None] for score, token in zip(scores, ids, strict=True)
                    ],
                }
            }
        )

    app = web.Application()
    app.router.add_post("/generate", generate)
    async with TestServer(app) as server:
        await run(args.source, args.output, args.assets, str(server.make_url("")).rstrip("/"))
    save(
        args.output / "runtime.json",
        {
            "engine_implementation": "Transformers service using required /generate fields; "
            "NOT SGLang",
            "device": "mps",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "aiohttp", "httpx")
            },
            "model_revision": REVISION,
            "weight_sha256": WEIGHT_SHA256,
            "requests": requests,
            "optimizer_steps": 0,
            "source_sha256": {
                name: hashlib.sha256(
                    (Path(__file__).resolve().parents[2] / name).read_bytes()
                ).hexdigest()
                for name in [
                    "integrations/slime/verify_training_handoff_mps.py",
                    "integrations/slime/verify_training_handoff.py",
                    "src/rolloutcheck/sample_audit.py",
                    "src/rolloutcheck/__init__.py",
                    "src/rolloutcheck/slime_capture.py",
                    "src/rolloutcheck/trace.py",
                    "src/rolloutcheck/evidence.py",
                ]
            },
            "scope": "Actual greedy generation and upstream Sample construction; "
            "token/mask audit only.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
