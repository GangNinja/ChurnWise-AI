"""Structured logging helpers.

Log records carry identifiers and counts only. URLs are logged without query
strings, and credentials are never available to this module in the first place.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from churnwise.contracts import utc_now_iso

_SAFE_EXTRA = ("source_id", "candidate_id", "status", "duration_ms", "request_count", "stage")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key in _SAFE_EXTRA:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Error"
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(
    *,
    verbose: bool = False,
    log_file: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger("churnwise")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(JsonFormatter())
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    return logger
