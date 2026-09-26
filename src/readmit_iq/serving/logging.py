"""Structured application events with a fixed allowlist; never serialize request payloads."""

import json
import logging
from datetime import UTC, datetime

FIELDS = (
    "model_version",
    "record_count",
    "latency_ms",
    "status_code",
    "error_code",
    "error_count",
    "endpoint",
)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        event.update({key: getattr(record, key) for key in FIELDS if hasattr(record, key)})
        return json.dumps(event)


def configure_logging():
    logger = logging.getLogger("readmit_iq.serving")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
