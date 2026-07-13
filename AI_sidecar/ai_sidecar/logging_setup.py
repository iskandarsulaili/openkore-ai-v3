from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include ALL extra fields from the LogRecord
        for key in dir(record):
            if key.startswith("_") or key in ("args", "msg", "exc_info", "exc_text", "stack_info",
                                                "filename", "funcName", "levelname", "levelno",
                                                "lineno", "module", "msecs", "name", "pathname",
                                                "process", "processName", "relativeCreated",
                                                "thread", "threadName", "created", "message",
                                                "getMessage", "prefix", "kwargs"):
                continue
            value = getattr(record, key, None)
            if value is not None:
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str, use_json: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(stream=sys.stdout)
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    root.addHandler(handler)

