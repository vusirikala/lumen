from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import redis.asyncio as redis
from fastapi import FastAPI

from lumen.api.routes import health, v1_inference
from lumen.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.redis = None
    if settings.redis_url is not None:
        client = redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )
        app.state.redis = client
    try:
        yield
    finally:
        r: redis.Redis | None = getattr(app.state, "redis", None)
        if r is not None:
            await r.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Lumen",
        description="LLM inference control plane with OpenAI-compatible APIs for self-hosted backends.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(v1_inference.router, prefix="/v1")
    return app


app: FastAPI = create_app()
