"""A separate, standard-library HTTP client with explicit multi-turn identity.

Uses two concurrent sessions by default. No model or RolloutCheck imports.
"""

import argparse
import concurrent.futures
import json
import urllib.request
from pathlib import Path


def conversation(url, session):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2? Give a brief explanation."},
    ]
    results = []
    for number, limit in ((1, 512), (2, 16)):
        context = {
            "session_id": session,
            "branch_id": "main",
            "turn_id": str(number),
            "parent_turn_id": str(number - 1) if number > 1 else None,
            "contract": {
                "version": "history-prefix/v1",
                "mode": "append_only",
                "history_policy": "preserved",
            },
        }
        body = {
            "model": "local-pinned-qwen",
            "messages": messages,
            "max_tokens": limit,
            "temperature": 0,
            "stream": False,
            "metadata": {"session_id": session, "rolloutcheck": context},
        }
        request = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=600) as reply:
            raw = reply.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise RuntimeError("Adapter response exceeds size limit")
        data = json.loads(raw)
        if number == 1 and data["choices"][0]["finish_reason"] != "stop":
            raise RuntimeError("First generation did not finish; keep partial evidence")
        results.append({"turn_id": str(number), "response": data})
        messages += [
            data["choices"][0]["message"],
            {"role": "user", "content": "Reply with one word: done."},
        ]
    return {"session_id": session, "turns": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=2, choices=range(1, 9))
    args = parser.parse_args()
    # This public demonstration is deliberately local and ignores system proxy settings.
    from urllib.parse import urlsplit

    parts = urlsplit(args.adapter_url)
    if (
        parts.scheme != "http"
        or parts.hostname != "127.0.0.1"
        or parts.port is None
        or parts.username
        or parts.password
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        parser.error("Use http://127.0.0.1:PORT")
    args.output.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions) as executor:
        futures = {
            executor.submit(conversation, args.adapter_url, f"public-session-{i + 1}"): i + 1
            for i in range(args.sessions)
        }
        for future in concurrent.futures.as_completed(futures):
            value = future.result()
            path = args.output / f"session-{futures[future]}.json"
            path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {args.sessions} sessions / {args.sessions * 2} requests")


if __name__ == "__main__":
    main()
