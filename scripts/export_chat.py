"""
Export a Cursor agent-transcript JSONL file to a human-readable Markdown log.

Usage:
    python scripts/export_chat.py <transcript.jsonl> <out.md>

Filters out tool-call noise and just keeps user + assistant text so the file
reads like a chat. Code fences inside assistant messages are preserved.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


HEADER = """# Refactor to FastAPI + React frontend — chat transcript

Exported from Cursor on the fly. Only user prompts and the assistant's
final natural-language messages are included; intermediate tool calls and
file reads are omitted for readability.

---
"""


def extract_text_blocks(message: dict) -> list[str]:
    """Pull all 'text' blocks out of a Cursor message payload."""
    out: list[str] = []
    content = (message or {}).get("content")
    if not isinstance(content, list):
        return out
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            txt = block.get("text") or ""
            out.append(txt)
    return out


USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)


def clean_user_text(text: str) -> str:
    """Strip Cursor's <user_query>…</user_query> wrapper."""
    match = USER_QUERY_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def clean_assistant_text(text: str) -> str:
    """Drop placeholder [REDACTED] segments Cursor may inject."""
    if text.strip() in {"[REDACTED]", ""}:
        return ""
    text = re.sub(r"\[REDACTED\]\s*", "", text)
    return text.strip()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python scripts/export_chat.py <in.jsonl> <out.md>")
        return 2

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if not src.exists():
        print(f"transcript not found: {src}")
        return 1

    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()

    parts: list[str] = [HEADER]
    exchange = 0

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        role = obj.get("role")
        texts = extract_text_blocks(obj.get("message") or {})
        joined = "\n\n".join(t for t in texts if t).strip()
        if not joined:
            continue

        if role == "user":
            exchange += 1
            cleaned = clean_user_text(joined)
            if not cleaned:
                continue
            parts.append(f"## {exchange}. User\n\n{cleaned}\n")
        elif role == "assistant":
            cleaned = clean_assistant_text(joined)
            if not cleaned:
                continue
            parts.append(f"### Assistant\n\n{cleaned}\n")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {dst} ({dst.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
