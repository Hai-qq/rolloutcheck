"""Optional unpadded, single-sequence Transformers generation adapter.

Importing this module does not import torch or transformers. The call requires
the generation extra and an already-loaded decoder-only model.
"""

import time

from .case import CaseError


def generate_recorded(
    model,
    input_ids,
    *,
    recorder,
    session_id,
    branch_id,
    turn_id,
    parent_turn_id,
    token_space,
    contract,
    generation_config,
    metadata=None,
    boundaries=None,
):
    """Run generate and record untouched input/output IDs, retaining final EOS.

    Returns (generated_ids, measurement). Unsupported modes fail before generation.
    No model/tokenizer loading, downloading, logging of text, or global monkeypatching.
    """
    import torch

    if model.config.is_encoder_decoder:
        raise CaseError("Only decoder-only generation is supported")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise CaseError("Expected one nonempty, unpadded [1, sequence] input tensor")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise CaseError("input_ids must be an integer tensor")
    config = generation_config
    if config.num_beams != 1 or config.num_return_sequences != 1:
        raise CaseError("Beam search and multiple return sequences are not supported")
    if config.token_healing or config.stop_strings or config.penalty_alpha:
        raise CaseError("Token healing, stop strings and contrastive search are not supported")
    if config.return_dict_in_generate or config.output_scores or config.output_logits:
        raise CaseError("Only the plain sequences tensor is supported")
    if config.max_new_tokens is None or config.max_new_tokens <= 0:
        raise CaseError("An explicit positive max_new_tokens limit is required")
    if config.forced_eos_token_id is not None or config.forced_bos_token_id is not None:
        raise CaseError("Forced boundary tokens are not supported")
    eos = config.eos_token_id
    eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    if not eos or any(type(token) is not int or token < 0 for token in eos):
        raise CaseError("Explicit EOS token IDs are required for termination accounting")
    if type(config.pad_token_id) is not int or config.pad_token_id < 0:
        raise CaseError("An explicit pad_token_id is required; EOS alias is allowed")
    start = time.perf_counter()
    original = input_ids[0].detach().cpu().tolist()
    # With an explicit all-ones mask, historical EOS is content even when PAD aliases EOS.
    if min(original) < 0 or (config.pad_token_id not in eos and config.pad_token_id in original):
        raise CaseError("Negative IDs and padding in the input are not supported")
    prepare_seconds = time.perf_counter() - start
    start_generation = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=config,
            use_model_defaults=False,
        )
    if input_ids.device.type == "mps":
        torch.mps.synchronize()
    elif input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    generation_seconds = time.perf_counter() - start_generation
    capture_start = time.perf_counter()
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        raise CaseError("Generation returned an unsupported shape")
    full = sequences[0].detach().cpu().tolist()
    if full[: len(original)] != original:
        raise CaseError(
            "generate changed its returned prompt prefix; capture cannot assume alignment"
        )
    output = full[len(original) :]
    if not output:
        raise CaseError("Generation produced no output tokens")
    stop = (
        "eos"
        if output[-1] in eos
        else ("length" if len(output) == config.max_new_tokens else "other")
    )
    recorder.record(
        session_id=session_id,
        branch_id=branch_id,
        turn_id=turn_id,
        parent_turn_id=parent_turn_id,
        token_space=token_space,
        input_ids=original,
        output_ids=output,
        contract=contract,
        boundaries=boundaries,
        termination={
            "reason": stop,
            "eos_token_ids": eos,
            "terminal_eos_retained": stop == "eos",
            "output_padding": False,
            "stop_strings": False,
        },
        metadata={
            **(metadata or {}),
            "capture_point": "transformers.generate input/output tensors",
            "generation_config": config.to_dict(),
            "use_model_defaults": False,
            "torch_version": torch.__version__,
            "device": str(input_ids.device),
        },
    )
    return output, {
        "generation_seconds": generation_seconds,
        "capture_seconds": prepare_seconds + time.perf_counter() - capture_start,
        "input_tokens": len(original),
        "output_tokens": len(output),
        "stop_reason": stop,
    }
