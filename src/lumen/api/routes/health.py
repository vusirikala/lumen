from typing import Any

import httpx
import redis.asyncio as redis
from fastapi import APIRouter, Request
from redis.exceptions import RedisError

from lumen.settings import Settings, get_settings

router = APIRouter(tags=["health"])


async def _inference_readiness(settings: Settings) -> dict[str, Any]:
    if settings.inference_base_url is None:
        return {"status": "skipped", "detail": "inference backend not configured"}

    url = f"{str(settings.inference_base_url).rstrip('/')}/health"
    timeout = httpx.Timeout(2.0, connect=1.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            return {"status": "unreachable", "detail": str(exc)}

    if response.status_code >= 400:
        return {"status": "unhealthy", "detail": f"backend health returned {response.status_code}"}
    return {"status": "ok"}


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness and optional Redis readiness."""
    settings = get_settings()
    inference = await _inference_readiness(settings)

    client: redis.Redis | None = getattr(request.app.state, "redis", None)
    if client is None:
        return {"status": "ok", "redis": "skipped", "inference": inference["status"]}
    try:
        await client.ping()
    except RedisError:
        return {"status": "degraded", "redis": "unreachable", "inference": inference["status"]}
    return {"status": "ok", "redis": "ok", "inference": inference["status"]}


@router.get("/health/inference")
async def inference_health() -> dict[str, Any]:
    """Dedicated inference backend readiness endpoint."""
    settings = get_settings()
    readiness = await _inference_readiness(settings)
    return {"status": readiness["status"], "detail": readiness.get("detail")}
