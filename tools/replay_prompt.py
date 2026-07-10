#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay a prompt dump captured via VLLM_DEBUG_DUMP_PROMPTS_DIR.

The server (serve_glm.sh with GLM_DUMP_PROMPTS=1, or any launch with
VLLM_DEBUG_DUMP_PROMPTS_DIR=<dir>) writes one JSON file per request into the
dump dir, containing:
  - api_request:       the ORIGINAL client request body (messages, tools,
                        sampling fields) — replayable verbatim
  - rendered_prompt:   the post-chat-template prompt text
  - prompt_token_ids:  the exact token ids sent to the engine
  - sampling_params:   the resolved engine SamplingParams (repr)

Usage:
  # List what a dump contains (prompt preview, tools, params):
  ./replay_prompt.py dump.json --show

  # Print the FULL rendered prompt (what the model actually saw):
  ./replay_prompt.py dump.json --show-prompt

  # Replay the original chat request (full pipeline: template + tool parser):
  ./replay_prompt.py dump.json

  # Replay at temperature 0 with a different question swapped in:
  ./replay_prompt.py dump.json --set temperature=0

  # Token-exact replay of the rendered prompt via /v1/completions
  # (bypasses chat template + tool-call parser — see the RAW model output):
  ./replay_prompt.py dump.json --mode tokens

Stdlib-only; no venv needed.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def load_dump(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_set(kvs: list[str]) -> dict:
    out = {}
    for kv in kvs:
        if "=" not in kv:
            sys.exit(f"--set expects key=value, got: {kv!r}")
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)  # numbers/bools/null/quoted strings/objects
        except json.JSONDecodeError:
            out[k] = v  # bare string
    return out


def post(url: str, body: dict, timeout: float, stream: bool):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url}:\n{e.read().decode(errors='replace')}")

    if not stream:
        return json.loads(resp.read().decode("utf-8"))

    # SSE: print deltas as they arrive, return the reassembled pieces.
    acc = {"content": "", "reasoning": "", "tool_calls": {}}
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            for key in ("reasoning", "reasoning_content"):
                if delta.get(key):
                    acc["reasoning"] += delta[key]
                    sys.stderr.write(delta[key])
                    sys.stderr.flush()
            if delta.get("content"):
                acc["content"] += delta["content"]
                sys.stdout.write(delta["content"])
                sys.stdout.flush()
            # /v1/completions streams put text at the top level
            if choice.get("text"):
                acc["content"] += choice["text"]
                sys.stdout.write(choice["text"])
                sys.stdout.flush()
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = acc["tool_calls"].setdefault(
                    idx, {"name": "", "arguments": ""}
                )
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    print()
    return acc


def show(dump: dict, full_prompt: bool) -> None:
    api = dump.get("api_request") or {}
    if full_prompt:
        print(dump.get("rendered_prompt") or "<no rendered_prompt in dump>")
        return
    print(f"request_id:  {dump.get('request_id')}")
    print(f"endpoint:    {dump.get('endpoint')}")
    print(f"timestamp:   {dump.get('timestamp_utc')}")
    print(f"model:       {dump.get('model')}")
    print(f"tokens:      {dump.get('num_prompt_tokens')}")
    print(f"sampling:    {dump.get('sampling_params')}")
    print(f"stream:      {api.get('stream')}")
    tools = api.get("tools") or []
    print(f"tools:       {len(tools)}")
    for t in tools:
        fn = t.get("function", t)
        print(f"  - {fn.get('name')}: {str(fn.get('description', ''))[:80]}")
    msgs = api.get("messages")
    if msgs:
        print(f"messages:    {len(msgs)}")
        for m in msgs:
            content = m.get("content")
            if isinstance(content, list):  # multimodal-style parts
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            content = (content or "").replace("\n", "\\n")
            print(f"  [{m.get('role')}] ({len(content)} ch) {content[:120]}")
    rp = dump.get("rendered_prompt") or ""
    print(f"rendered_prompt: {len(rp)} chars (--show-prompt to print)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump", help="dump JSON file written by the server")
    ap.add_argument("--url", default="http://127.0.0.1:8080",
                    help="server base URL (default %(default)s)")
    ap.add_argument("--mode", choices=["chat", "tokens", "text"], default="chat",
                    help="chat = replay original API request (default); "
                    "tokens = token-exact rendered prompt via /v1/completions; "
                    "text = rendered prompt text via /v1/completions")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override request fields (JSON values), repeatable; "
                    "e.g. --set temperature=0 --set max_tokens=512")
    ap.add_argument("--stream", action="store_true",
                    help="replay with stream=true like Open WebUI does "
                    "(default: force stream=false for a single JSON response)")
    ap.add_argument("--show", action="store_true",
                    help="just summarize the dump, no request")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the full rendered prompt, no request")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    dump = load_dump(args.dump)
    if args.show or args.show_prompt:
        show(dump, args.show_prompt)
        return

    overrides = parse_set(getattr(args, "set"))

    if args.mode == "chat":
        body = dict(dump.get("api_request") or {})
        if not body:
            sys.exit("dump has no api_request")
        endpoint = dump.get("endpoint") or "/v1/chat/completions"
    else:
        api = dump.get("api_request") or {}
        if args.mode == "tokens":
            prompt = dump.get("prompt_token_ids")
            if not prompt:
                sys.exit("dump has no prompt_token_ids")
        else:
            prompt = dump.get("rendered_prompt")
            if not prompt:
                sys.exit("dump has no rendered_prompt")
        body = {"model": dump.get("model") or api.get("model"), "prompt": prompt}
        # carry over the client's sampling fields where present
        for k in ("temperature", "top_p", "top_k", "max_tokens", "stop",
                  "presence_penalty", "frequency_penalty", "seed"):
            if k in api:
                body[k] = api[k]
        if "max_tokens" not in body and "max_completion_tokens" in api:
            body["max_tokens"] = api["max_completion_tokens"]
        endpoint = "/v1/completions"

    body["stream"] = args.stream
    if not args.stream:
        body.pop("stream_options", None)
    body.update(overrides)

    url = args.url.rstrip("/") + endpoint
    ntok = dump.get("num_prompt_tokens")
    print(f"POST {url}  (mode={args.mode}, prompt tokens={ntok}, "
          f"stream={body['stream']})", file=sys.stderr)

    result = post(url, body, args.timeout, args.stream)

    if args.stream:
        if result["tool_calls"]:
            print("\n=== TOOL CALLS (assembled from stream) ===")
            for idx in sorted(result["tool_calls"]):
                tc = result["tool_calls"][idx]
                print(f"  [{idx}] {tc['name']}({tc['arguments']})")
        else:
            print("\n=== NO TOOL CALLS in stream ===")
        return

    # Non-stream: pretty-print the interesting parts, then the full JSON.
    for choice in result.get("choices", []):
        msg = choice.get("message")
        if msg is not None:  # chat
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            if reasoning:
                print(f"--- reasoning ---\n{reasoning}\n")
            print(f"--- content (finish_reason={choice.get('finish_reason')}) ---")
            print(msg.get("content"))
            tcs = msg.get("tool_calls")
            if tcs:
                print("=== TOOL CALLS ===")
                for tc in tcs:
                    fn = tc.get("function", {})
                    print(f"  {fn.get('name')}({fn.get('arguments')})")
            else:
                print("=== NO TOOL CALLS ===")
        else:  # completions
            print(f"--- text (finish_reason={choice.get('finish_reason')}) ---")
            print(choice.get("text"))
    print("\n--- full response JSON ---", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
