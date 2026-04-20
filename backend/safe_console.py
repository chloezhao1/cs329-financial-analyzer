"""
Windows-safe console output for legacy pipeline code.

The ingestion stack prints Unicode (checkmarks, arrows) and raw SEC titles.
When ``sys.stdout`` is still cp1252, ``print()`` raises ``UnicodeEncodeError``.
Reconfiguring streams sometimes fails under uvicorn; patching ``builtins.print``
to coerce strings for the active stream's encoding is reliable.
"""
from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def patch_print_for_console() -> Iterator[None]:
    """Replace ``print`` so strings are encodable by the target stream."""
    real_print = builtins.print

    def safe_print(*args: Any, **kwargs: Any) -> None:
        file = kwargs.get("file", sys.stdout)
        enc = getattr(file, "encoding", None) or "utf-8"
        if enc.lower() in ("utf-8", "utf8", "utf_8"):
            return real_print(*args, **kwargs)

        fixed: list[Any] = []
        for value in args:
            if isinstance(value, str):
                try:
                    value.encode(enc)
                except UnicodeEncodeError:
                    value = value.encode(enc, errors="replace").decode(enc)
            fixed.append(value)
        return real_print(*fixed, **kwargs)

    builtins.print = safe_print
    try:
        yield
    finally:
        builtins.print = real_print
