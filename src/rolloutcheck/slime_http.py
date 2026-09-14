"""Explicit HTTP turn identity for a single-event-loop slime OpenAI adapter.

No ancestry is inferred from arrival order. aiohttp is imported only on attach.
"""

import asyncio
import re
from contextvars import ContextVar
from dataclasses import dataclass

from .case import CaseError, parse_object
from .slime_capture import SlimeDebugCapture, TurnContext
from .trace import validate_record

ALIAS = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")


def request_context(body, token_space):
    """Require public aliases and explicit parent/contract in request metadata."""
    metadata = body.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("rolloutcheck"), dict):
        raise CaseError("metadata.rolloutcheck is required")
    value = metadata["rolloutcheck"]
    keys = {"session_id", "branch_id", "turn_id", "parent_turn_id", "contract"}
    if set(value) != keys:
        raise CaseError(
            "rolloutcheck metadata requires exactly session, branch, turn, parent, contract"
        )
    for key in ("session_id", "branch_id", "turn_id"):
        if not isinstance(value[key], str) or ALIAS.fullmatch(value[key]) is None:
            raise CaseError("Identity fields must be public ASCII aliases, 1 to 128 characters")
    parent = value["parent_turn_id"]
    if parent is not None and (not isinstance(parent, str) or ALIAS.fullmatch(parent) is None):
        raise CaseError("parent_turn_id must be a public alias or null")
    if metadata.get("session_id") != value["session_id"]:
        raise CaseError("metadata.session_id must match the public rolloutcheck session alias")
    if body.get("stream", False) is not False:
        raise CaseError("This integration currently requires stream=false")
    context = TurnContext(token_space=token_space, **value)
    # Use the reader's contract and identity validation before any engine request.
    validate_record(
        {
            "trace_version": 2,
            "record_type": "generation",
            "sequence": 0,
            "trace_id": "validation",
            "evidence_kind": "synthetic",
            **value,
            "token_space": token_space,
            "input_ids": [0],
            "output_ids": [0],
            "metadata": {},
            "termination": {},
        }
    )
    return context


@dataclass
class _RequestState:
    context: TurnContext
    calls: int = 0
    captured: int = 0


class SlimeHTTPCapture:
    """Attach before serving; one writer/event loop, concurrent HTTP tasks supported.

    Parents must already have completed capture in the same session and branch.
    Finish stops admission, drains accepted HTTP handlers and checks independently
    counted successful responses against callback coverage. It never closes the
    caller-owned recorder. Limit counts received target requests, including errors.
    """

    def __init__(self, recorder, *, token_space, eos_token_ids=None, request_limit=None):
        if recorder.trace_version != 2:
            raise CaseError("HTTP capture requires a version 2 recorder")
        if not isinstance(token_space, str) or not token_space:
            raise CaseError("An explicit token space is required")
        if request_limit is not None and (type(request_limit) is not int or request_limit < 1):
            raise CaseError("request_limit must be a positive integer or None")
        self.recorder, self.token_space = recorder, token_space
        self.request_limit = request_limit
        self.received = self.succeeded = self.failed = self.active = 0
        self.accepting = True
        self._attached = self._finished = False
        self._gap_write_failed = False
        self._keys, self._completed = set(), set()
        self._state = ContextVar("rolloutcheck_request", default=None)
        self.idle, self.done = asyncio.Event(), asyncio.Event()
        self.idle.set()
        self.capture = SlimeDebugCapture(
            recorder,
            lambda *args: self._state.get().context if self._state.get() else None,
            eos_token_ids=eos_token_ids,
        )

    def _callback(self, *args):
        state = self._state.get()
        before = self.capture.captured
        if state is not None:
            state.calls += 1
        try:
            self.capture(*args)
        finally:
            if self.capture.persistence_failed:
                self.accepting = False
        if state is not None:
            state.captured += self.capture.captured - before

    def _gap(self, reason):
        self.failed += 1
        try:
            self.recorder.record_gap(reason)
        except Exception:
            self._gap_write_failed = True

    def attach(self, adapter):
        from aiohttp import web

        if self._attached or adapter.debug_callback is not None or adapter.app.frozen:
            raise CaseError("Attach once, before serving, to an adapter with no debug callback")

        @web.middleware
        async def middleware(request, handler):
            if request.method != "POST" or request.path != "/v1/chat/completions":
                return await handler(request)
            if not self.accepting:
                return web.json_response({"error": "capture admission closed"}, status=503)
            self.received += 1
            self.active += 1
            self.idle.clear()
            if self.request_limit is not None and self.received >= self.request_limit:
                self.accepting = False
            token = None
            try:
                try:
                    context = request_context(parse_object(await request.read()), self.token_space)
                except (CaseError, web.HTTPRequestEntityTooLarge):
                    self._gap("http_context_invalid")
                    return web.json_response(
                        {"error": "invalid rolloutcheck request metadata"}, status=422
                    )
                if "text/event-stream" in request.headers.get("Accept", ""):
                    self._gap("http_stream_unsupported")
                    return web.json_response({"error": "Streaming is unsupported"}, status=422)
                key = (context.session_id, context.branch_id, context.turn_id)
                authorization = request.headers.get("Authorization")
                if authorization is not None and authorization != f"Bearer {context.session_id}":
                    self._gap("http_session_auth_mismatch")
                    return web.json_response(
                        {"error": "Use the public session alias as the optional Bearer value"},
                        status=422,
                    )
                parent = (context.session_id, context.branch_id, context.parent_turn_id)
                if key in self._keys:
                    self._gap("http_duplicate_turn")
                    return web.json_response({"error": "turn identity already used"}, status=409)
                if context.parent_turn_id is not None and parent not in self._completed:
                    self._gap("http_parent_not_completed")
                    return web.json_response(
                        {"error": "declared parent has not completed capture"}, status=409
                    )
                self._keys.add(key)
                state = _RequestState(context)
                token = self._state.set(state)
                response = await handler(request)
                if 200 <= response.status < 300:
                    self.succeeded += 1
                    if state.calls != 1 or state.captured != 1:
                        self._gap("http_callback_coverage_mismatch")
                    else:
                        self._completed.add(key)
                else:
                    self._gap("http_unsuccessful_response")
                return response
            except (Exception, asyncio.CancelledError):
                self._gap("http_handler_failed_or_cancelled")
                raise
            finally:
                if token is not None:
                    self._state.reset(token)
                self.active -= 1
                if self.active == 0:
                    self.idle.set()
                    if not self.accepting:
                        self.done.set()

        adapter.app.middlewares.append(middleware)
        adapter.debug_callback = self._callback
        self._attached = True

    def status(self):
        return {
            "received": self.received,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "active": self.active,
            "callback_invocations": self.capture.invocations,
            "captured": self.capture.captured,
            "finalized": self._finished,
            "gap_write_failed": self._gap_write_failed,
            "scope": "Received non-streaming chat requests; not all intended client requests "
            "or engine/training coverage. Identity is client-declared.",
        }

    def abort(self, reason="http_owner_aborted"):
        """Stop admission and mark capture incomplete; drain handlers before closing."""
        if self._finished:
            raise CaseError("Cannot abort a finalized capture")
        self.accepting = False
        self._gap(reason)
        if self.active == 0:
            self.done.set()

    async def finish(self, *, timeout=30, expected_requests=None):
        if not self._attached or self._finished:
            raise CaseError("Capture must be attached and finalized exactly once")
        if expected_requests is not None and (
            type(expected_requests) is not int or expected_requests < 0
        ):
            raise CaseError("expected_requests must be a nonnegative integer")
        self.accepting = False
        try:
            await asyncio.wait_for(self.idle.wait(), timeout)
        except TimeoutError:
            self._gap("http_shutdown_timeout")
            raise CaseError("HTTP requests did not drain") from None
        expected = expected_requests if expected_requests is not None else self.request_limit
        if expected is not None and self.received != expected:
            self._gap("http_request_count_mismatch")
        if self.failed:
            raise CaseError("HTTP capture has failed or rejected requests; inspect gaps")
        self.capture.raise_if_failed(expected_turns=self.succeeded)
        self._finished = True
        return self.status()
