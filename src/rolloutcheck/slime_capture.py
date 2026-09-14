"""Opt-in slime debug_callback integration; imports neither slime nor a model runtime."""

from dataclasses import dataclass

from . import __version__
from .case import CaseError

SLIME_REVISION = "4c193f1f37509cca70f0e88807a9305b70f63f4e"


@dataclass(frozen=True)
class TurnContext:
    """Controller-supplied identity: never infer ancestry from arrival order or token equality."""

    session_id: str
    branch_id: str
    turn_id: str
    parent_turn_id: str | None
    token_space: str
    contract: dict


class SlimeDebugCapture:
    """Attach to BaseAdapter(debug_callback=...). Single-thread/event-loop usage only.

    resolve_context(sid, messages, tools, response, turn) returns TurnContext or None.
    Names should be public aliases: slime's raw sid can be an authentication bearer.
    Raw messages, tools, sid and response text are never persisted by this callback.
    Evidence kind is chosen explicitly on the recorder, not upgraded by this adapter.
    """

    def __init__(self, recorder, resolve_context, *, eos_token_ids=None):
        if not callable(resolve_context):
            raise CaseError("An explicit turn-context resolver is required")
        if eos_token_ids is not None and (
            not isinstance(eos_token_ids, list)
            or not eos_token_ids
            or any(type(x) is not int or x < 0 for x in eos_token_ids)
        ):
            raise CaseError("eos_token_ids must be a nonempty integer list or None")
        self.recorder = recorder
        self.resolve_context = resolve_context
        self.eos_token_ids = list(eos_token_ids) if eos_token_ids is not None else None
        self.invocations = 0
        self.captured = 0
        self.failed = 0
        self.persistence_failed = False

    def __call__(self, sid, messages, tools, response, turn):
        self.invocations += 1
        # Upstream catches callback exceptions. Persist omissions and keep a sticky
        # in-memory failure flag that the owner must check before accepting a run.
        try:
            context = self.resolve_context(sid, messages, tools, response, turn)
        except Exception:
            self._gap("context_resolver_failed")
            return
        if not isinstance(context, TurnContext):
            self._gap("turn_context_missing")
            return
        try:
            if not isinstance(turn.finish_reason, str) or not turn.finish_reason:
                raise CaseError("Missing upstream finish reason")
            self.recorder.record(
                session_id=context.session_id,
                branch_id=context.branch_id,
                turn_id=context.turn_id,
                parent_turn_id=context.parent_turn_id,
                token_space=context.token_space,
                input_ids=turn.prompt_ids,
                output_ids=turn.output_ids,
                contract=context.contract,
                termination={
                    "upstream_finish_reason": turn.finish_reason,
                    "eos_token_ids": self.eos_token_ids,
                    "terminal_eos_present": (
                        bool(turn.output_ids and turn.output_ids[-1] in self.eos_token_ids)
                        if self.eos_token_ids is not None
                        else None
                    ),
                    "engine_stop_token_retention": "unverified",
                },
                metadata={
                    "capture_point": "slime.BaseAdapter.debug_callback TurnRecord",
                    "collector_version": __version__,
                    "integration_reference_revision": SLIME_REVISION,
                    "reference_is_runtime_attestation": False,
                    "output_ids_source": "TurnRecord.output_ids; extracted from logprob tuple IDs",
                    "scope": "served adapter turn; engine and request completeness unverified",
                },
            )
        except Exception:
            self._gap("turn_record_capture_failed")
            return
        self.captured += 1

    def _gap(self, reason):
        self.failed += 1
        try:
            self.recorder.record_gap(reason)
        except Exception:
            # An unwritable trace cannot record its own failure. The explicit health
            # check is mandatory, including if slime has swallowed this exception.
            self.persistence_failed = True
            raise CaseError(
                "Could not persist capture gap; trace completeness is unknown"
            ) from None

    def raise_if_failed(self, *, expected_turns):
        """After requests drain, check coverage and finalize a version 2 recorder.

        Version 1 recorders retain the legacy in-memory health check only.
        """
        if type(expected_turns) is not int or expected_turns < 0:
            raise CaseError("expected_turns must be the controller's nonnegative served-turn count")
        if self.invocations != expected_turns:
            self._gap("served_turn_count_mismatch")
        if self.failed or self.persistence_failed:
            raise CaseError(
                f"slime capture incomplete: {self.failed} capture error(s); "
                f"gap persistence failed={self.persistence_failed}"
            )
        if self.recorder.trace_version == 2:
            self.recorder.finalize(expected_generations=expected_turns)
