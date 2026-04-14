import json
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any

logger = logging.getLogger("lumen.telemetry")


@dataclass
class _MetricRow:
    requests: int = 0
    errors: int = 0
    latency_ms_sum: float = 0.0
    latency_ms_max: float = 0.0


class InferenceTelemetry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._rows: dict[tuple[str, str], _MetricRow] = {}

    def record(self, *, endpoint: str, model: str, status_code: int, latency_ms: float) -> None:
        key = (endpoint, model)
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                row = _MetricRow()
                self._rows[key] = row
            row.requests += 1
            if status_code >= 400:
                row.errors += 1
            row.latency_ms_sum += latency_ms
            row.latency_ms_max = max(row.latency_ms_max, latency_ms)

        logger.info(
            json.dumps(
                {
                    "event": "inference_request",
                    "endpoint": endpoint,
                    "model": model,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 3),
                }
            )
        )

    def snapshot(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        with self._lock:
            for (endpoint, model), row in sorted(self._rows.items()):
                avg_latency = row.latency_ms_sum / row.requests if row.requests else 0.0
                rows.append(
                    {
                        "endpoint": endpoint,
                        "model": model,
                        "requests": row.requests,
                        "errors": row.errors,
                        "error_rate": round(row.errors / row.requests, 6) if row.requests else 0.0,
                        "avg_latency_ms": round(avg_latency, 3),
                        "max_latency_ms": round(row.latency_ms_max, 3),
                    }
                )
        return {"rows": rows}


inference_telemetry = InferenceTelemetry()

