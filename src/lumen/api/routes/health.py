from typing import Any

import redis.asyncio as redis
from fastapi import APIRouter, Request
from redis.exceptions import RedisError

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness and optional Redis readiness."""
    client: redis.Redis | None = getattr(request.app.state, "redis", None)
    if client is None:
        return {"status": "ok", "redis": "skipped"}
    try:
        await client.ping()
    except RedisError:
        return {"status": "degraded", "redis": "unreachable"}
    return {"status": "ok", "redis": "ok"}
