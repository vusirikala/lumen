import json
import logging

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger("lumen.telemetry")

# Latency buckets in seconds covering fast cache hits through slow 2-minute generations.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)


class InferenceTelemetry:
    """Prometheus-backed telemetry for inference requests.

    Each instance owns a private CollectorRegistry so that unit tests can
    create isolated instances without cross-test counter accumulation.
    The module-level `inference_telemetry` singleton is what the app uses.
    """

    def __init__(self) -> None:
        self._registry = CollectorRegistry(auto_describe=True)

        self._requests_total = Counter(
            "lumen_requests_total",
            "Total inference requests by endpoint, model, and HTTP status code.",
            ["endpoint", "model", "status_code"],
            registry=self._registry,
        )
        self._request_duration_seconds = Histogram(
            "lumen_request_duration_seconds",
            "Inference request duration in seconds.",
            ["endpoint", "model"],
            buckets=_LATENCY_BUCKETS,
            registry=self._registry,
        )

    def record(self, *, endpoint: str, model: str, status_code: int, latency_ms: float) -> None:
        """Record one completed request. Safe to call from async context (no blocking I/O)."""
        self._requests_total.labels(
            endpoint=endpoint, model=model, status_code=str(status_code)
        ).inc()
        self._request_duration_seconds.labels(endpoint=endpoint, model=model).observe(
            latency_ms / 1000.0
        )
        logger.info(
            json.dumps({
                "event": "inference_request",
                "endpoint": endpoint,
                "model": model,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
            })
        )

    def metrics_output(self) -> tuple[bytes, str]:
        """Return (content_bytes, content_type) ready for a /metrics HTTP response."""
        return generate_latest(self._registry), CONTENT_TYPE_LATEST


inference_telemetry = InferenceTelemetry()
