# Extracted verbatim from THUDM/slime, Apache-2.0.
# Commit: 885a09e852e5272d497936370707cc604830f805 (PR #2287, not merged at review).
# Source: slime/agent/adapters/common.py; full adapter/server is NOT imported or tested.
# See NOTICE.md and slime-LICENSE. Upstream fixes are not RolloutCheck contributions.
from __future__ import annotations

import copy
from typing import Any

def _render_token_ids(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool = True,
    chat_template: str | None = None,
) -> list[int]:
    """Render a chat-message list to token ids with the served chat template."""
    kwargs = {
        "tools": tools,
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    enc = tokenizer.apply_chat_template(messages, **kwargs)
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


_QWEN_REASONING_HISTORY_BRANCH = "{%- if loop.index0 > ns.last_query_index %}"


def _reasoning_preserving_chat_template(tokenizer) -> str:
    """Return the Qwen chat template variant needed for append-only RL turns.

    Qwen3/3.5 normally drops reasoning from assistant messages before the most
    recent user message. That inference-time compaction rewrites already sampled
    tokens, which breaks a multi-turn RL trajectory. In this explicit mode all
    available ``reasoning_content`` is rendered while the normal open ``<think>``
    generation prompt remains unchanged.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or _QWEN_REASONING_HISTORY_BRANCH not in template:
        raise ValueError(
            "preserve_reasoning_history requires a Qwen chat template containing "
            f"{_QWEN_REASONING_HISTORY_BRANCH!r}"
        )
    replacement = "{%- if true %}"
    return template.replace(_QWEN_REASONING_HISTORY_BRANCH, replacement, 1)


def _assistant_visible_signature(message: dict) -> tuple[Any, Any]:
    return message.get("content"), message.get("tool_calls")


def _restore_canonical_assistant_history(messages: list[dict], assistant_history: list[dict]) -> int:
    """Replace client-echoed assistant turns with the server's canonical copy.

    Clients may omit reasoning and normalize text or tool-call arguments when
    replaying a response. The adapter generated those turns and owns their exact
    structured form, so align assistant messages from the newest turn backwards
    and restore them wholesale. User and tool messages remain client-owned.
    """
    positions = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
    restored = 0
    for position, canonical in zip(reversed(positions), reversed(assistant_history), strict=False):
        if messages[position] != canonical:
            messages[position] = copy.deepcopy(canonical)
            restored += 1
    return restored


def _assert_append_only_prompt(previous_turn_ids: list[int], prompt_ids: list[int]) -> None:
    """Require a new prompt to preserve every token sampled in the prior turn."""
    if not previous_turn_ids:
        return
    if prompt_ids[: len(previous_turn_ids)] == previous_turn_ids:
        return
    common = 0
    limit = min(len(previous_turn_ids), len(prompt_ids))
    while common < limit and previous_turn_ids[common] == prompt_ids[common]:
        common += 1
    raise RuntimeError(
        "non-append-only agent prompt: "
        f"previous_tokens={len(previous_turn_ids)} prompt_tokens={len(prompt_ids)} common_prefix={common}"
    )


def _continue_from_canonical_turn(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    chat_template: str,
    rendered_prompt_ids: list[int],
    previous_turn_ids: list[int],
    previous_response_message: dict,
) -> list[int]:
    """Append only the client messages added after the last sampled response.

    A parsed assistant response is semantically equivalent to the sampled text,
    but tool parsers and chat templates can normalize whitespace or argument
    formatting. The sampled token ids are therefore the canonical history. We
    use message identity only to locate the assistant boundary in the freshly
    rendered prompt, then append the new suffix to those canonical ids.
    """
    assistant_positions = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
    if not assistant_positions:
        raise RuntimeError("append-only agent prompt has no replayed assistant message")
    boundary = assistant_positions[-1]
    if _assistant_visible_signature(messages[boundary]) != _assistant_visible_signature(previous_response_message):
        raise RuntimeError("latest replayed assistant does not match the previous sampled response")

    rendered_through_assistant = _render_token_ids(
        messages[: boundary + 1],
        tokenizer,
        tools=tools,
        add_generation_prompt=False,
        chat_template=chat_template,
    )
    if rendered_prompt_ids[: len(rendered_through_assistant)] != rendered_through_assistant:
        raise RuntimeError("chat template rewrote messages before the latest assistant boundary")

    prompt_ids = previous_turn_ids + rendered_prompt_ids[len(rendered_through_assistant) :]
    _assert_append_only_prompt(previous_turn_ids, prompt_ids)
    return prompt_ids
