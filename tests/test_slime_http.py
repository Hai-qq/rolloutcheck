import asyncio
import json
from types import SimpleNamespace

import pytest

from rolloutcheck.case import CaseError
from rolloutcheck.slime_http import SlimeHTTPCapture, request_context
from rolloutcheck.trace import TraceRecorder, inspect_trace

CONTRACT = {"version": "history-prefix/v1", "mode": "append_only", "history_policy": "preserved"}


def body(session="public", turn="1", parent=None, inputs=None):
    return {
        "messages": [{"role": "user", "content": "PRIVATE-TEST-TEXT"}],
        "_inputs": inputs or [1],
        "metadata": {
            "session_id": session,
            "rolloutcheck": {
                "session_id": session,
                "branch_id": "main",
                "turn_id": turn,
                "parent_turn_id": parent,
                "contract": CONTRACT,
            },
        },
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.pop("metadata"),
        lambda x: x.update(stream=True),
        lambda x: x.update(stream="false"),
        lambda x: x["metadata"].update(session_id="different"),
        lambda x: x["metadata"]["rolloutcheck"].update(turn_id="../private"),
        lambda x: x["metadata"]["rolloutcheck"].pop("parent_turn_id"),
        lambda x: x["metadata"]["rolloutcheck"].update(parent_turn_id="1"),
        lambda x: x["metadata"]["rolloutcheck"].update(contract={"mode": "invented"}),
        lambda x: x["metadata"]["rolloutcheck"].update(token_space="override"),
    ],
)
def test_bad_request_context_rejected(mutate):
    value = body()
    mutate(value)
    with pytest.raises(CaseError):
        request_context(value, "pinned-token-space")


def test_context_uses_explicit_identity_and_server_token_space():
    value = body("session", "child", "root")
    value["metadata"]["rolloutcheck"]["contract"] = {}
    context = request_context(value, "pinned-token-space")
    assert context.parent_turn_id == "root" and context.token_space == "pinned-token-space"
    assert context.contract == {}  # Missing contract remains inconclusive, not invented.


def adapter(before=None, *, omit_callback=False):
    web = pytest.importorskip("aiohttp.web")
    result = SimpleNamespace(app=web.Application(), debug_callback=None)

    async def handler(request):
        value = await request.json()
        if before:
            await before(value)
        if not omit_callback:
            result.debug_callback(
                "PRIVATE-BEARER",
                value["messages"],
                None,
                {},
                SimpleNamespace(prompt_ids=value["_inputs"], output_ids=[2], finish_reason="stop"),
            )
        return web.json_response({"ok": True})

    result.app.router.add_post("/v1/chat/completions", handler)
    return result


def test_concurrent_sessions_keep_request_context_and_parent_identity(tmp_path):
    test_utils = pytest.importorskip("aiohttp.test_utils")
    path = tmp_path / "trace.jsonl"

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def before(value):
            ctx = value["metadata"]["rolloutcheck"]
            if ctx["session_id"] == "slow" and ctx["turn_id"] == "1":
                entered.set()
                await release.wait()

        a = adapter(before)
        with TraceRecorder(path, trace_id="concurrent", evidence_kind="synthetic") as writer:
            capture = SlimeHTTPCapture(writer, token_space="test", request_limit=4)
            capture.attach(a)
            async with test_utils.TestClient(test_utils.TestServer(a.app)) as client:
                slow = asyncio.create_task(client.post("/v1/chat/completions", json=body("slow")))
                await asyncio.wait_for(entered.wait(), 2)
                fast = await client.post("/v1/chat/completions", json=body("fast", inputs=[9]))
                assert fast.status == 200
                fast2 = await client.post(
                    "/v1/chat/completions", json=body("fast", "2", "1", [9, 2, 4])
                )
                assert fast2.status == 200
                release.set()
                assert (await slow).status == 200
                assert (
                    await client.post(
                        "/v1/chat/completions", json=body("slow", "2", "1", [1, 2, 3])
                    )
                ).status == 200
                await asyncio.wait_for(capture.done.wait(), 2)
                assert (await client.post("/v1/chat/completions", json=body("extra"))).status == 503
                status = await capture.finish()
                assert status["received"] == status["succeeded"] == status["captured"] == 4

    asyncio.run(scenario())
    report, cases = inspect_trace(path)
    assert report["status"] == "PASS" and len(cases) == 2
    assert {case["previous"]["session_id"] for case in cases} == {"fast", "slow"}
    assert "PRIVATE" not in path.read_text()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["session_id"] == "fast" and records[2]["session_id"] == "slow"


@pytest.mark.parametrize("mode,expected", [("duplicate", 409), ("parent", 409), ("invalid", 422)])
def test_rejected_requests_are_gaps_and_cannot_finalize(tmp_path, mode, expected):
    test_utils = pytest.importorskip("aiohttp.test_utils")
    path = tmp_path / "trace.jsonl"

    async def scenario():
        a = adapter()
        with TraceRecorder(path, trace_id="reject", evidence_kind="synthetic") as writer:
            capture = SlimeHTTPCapture(writer, token_space="test")
            capture.attach(a)
            async with test_utils.TestClient(test_utils.TestServer(a.app)) as client:
                assert (await client.post("/v1/chat/completions", json=body())).status == 200
                value = body() if mode == "duplicate" else body(turn="2", parent="unknown")
                if mode == "invalid":
                    value.pop("metadata")
                assert (await client.post("/v1/chat/completions", json=value)).status == expected
                with pytest.raises(CaseError, match="failed or rejected"):
                    await capture.finish()

    asyncio.run(scenario())
    report, _ = inspect_trace(path)
    assert report["status"] == "INCONCLUSIVE" and report["capture_gaps"]
    assert report["capture_completion"]["state"] == "missing"


def test_missing_callback_on_success_is_detected(tmp_path):
    test_utils = pytest.importorskip("aiohttp.test_utils")
    path = tmp_path / "trace.jsonl"

    async def scenario():
        a = adapter(omit_callback=True)
        with TraceRecorder(path, trace_id="missing", evidence_kind="synthetic") as writer:
            capture = SlimeHTTPCapture(writer, token_space="test")
            capture.attach(a)
            async with test_utils.TestClient(test_utils.TestServer(a.app)) as client:
                assert (await client.post("/v1/chat/completions", json=body())).status == 200
                with pytest.raises(CaseError):
                    await capture.finish()

    asyncio.run(scenario())
    assert inspect_trace(path)[0]["capture_gaps"][0]["reason"] == "http_callback_coverage_mismatch"


def test_cancel_during_request_read_is_a_persistent_gap(tmp_path):
    pytest.importorskip("aiohttp")
    path = tmp_path / "trace.jsonl"

    async def scenario():
        a = adapter()
        with TraceRecorder(path, trace_id="cancel", evidence_kind="synthetic") as writer:
            capture = SlimeHTTPCapture(writer, token_space="test")
            capture.attach(a)

            async def read():
                raise asyncio.CancelledError()

            request = SimpleNamespace(method="POST", path="/v1/chat/completions", read=read)
            with pytest.raises(asyncio.CancelledError):
                await a.app.middlewares[0](request, None)
            assert capture.active == 0
            with pytest.raises(CaseError):
                await capture.finish(expected_requests=1)

    asyncio.run(scenario())
    assert inspect_trace(path)[0]["capture_gaps"][0]["reason"] == "http_handler_failed_or_cancelled"


@pytest.mark.parametrize(
    "authorization,expected", [("Bearer public", 200), ("Bearer PRIVATE-KEY", 422)]
)
def test_bearer_must_match_public_session_and_never_persists(tmp_path, authorization, expected):
    test_utils = pytest.importorskip("aiohttp.test_utils")
    path = tmp_path / "trace.jsonl"

    async def scenario():
        a = adapter()
        with TraceRecorder(path, trace_id="auth", evidence_kind="synthetic") as writer:
            capture = SlimeHTTPCapture(writer, token_space="test", request_limit=1)
            capture.attach(a)
            async with test_utils.TestClient(test_utils.TestServer(a.app)) as client:
                response = await client.post(
                    "/v1/chat/completions", json=body(), headers={"Authorization": authorization}
                )
                assert response.status == expected
                if expected == 200:
                    await capture.finish()
                else:
                    assert capture.capture.captured == 0
                    with pytest.raises(CaseError):
                        await capture.finish()

    asyncio.run(scenario())
    assert "PRIVATE" not in path.read_text()


@pytest.mark.parametrize("mode", ["count", "abort", "timeout"])
def test_incomplete_owner_lifecycle_cannot_seal(tmp_path, mode):
    pytest.importorskip("aiohttp")
    path = tmp_path / "trace.jsonl"

    async def scenario():
        a = adapter()
        with TraceRecorder(path, trace_id="lifecycle", evidence_kind="synthetic") as writer:
            capture = SlimeHTTPCapture(writer, token_space="test", request_limit=1)
            capture.attach(a)
            if mode == "abort":
                capture.abort()
                assert capture.done.is_set()
            if mode == "timeout":
                capture.idle.clear()
            with pytest.raises(CaseError):
                await capture.finish(timeout=0.01)
            assert not capture.status()["finalized"] and not capture.accepting

    asyncio.run(scenario())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    reasons = {record.get("reason") for record in records}
    assert {
        "count": "http_request_count_mismatch",
        "abort": "http_owner_aborted",
        "timeout": "http_shutdown_timeout",
    }[mode] in reasons
    assert all(record["record_type"] != "capture_complete" for record in records)


def test_accept_header_cannot_bypass_non_streaming_contract(tmp_path):
    test_utils = pytest.importorskip("aiohttp.test_utils")
    path = tmp_path / "trace.jsonl"

    async def scenario():
        a = adapter()
        with TraceRecorder(path, trace_id="stream", evidence_kind="synthetic") as writer:
            capture = SlimeHTTPCapture(writer, token_space="test", request_limit=1)
            capture.attach(a)
            async with test_utils.TestClient(test_utils.TestServer(a.app)) as client:
                response = await client.post(
                    "/v1/chat/completions", json=body(), headers={"Accept": "text/event-stream"}
                )
                assert response.status == 422 and capture.capture.captured == 0
                with pytest.raises(CaseError):
                    await capture.finish()

    asyncio.run(scenario())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["reason"] == "http_stream_unsupported"
