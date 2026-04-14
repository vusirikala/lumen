"""
Exact-prefix KV cache — canonical serialization, prefix hashing, and
Redis metadata store.

Lumen stores a mapping:
    lumen:cache:exact:{model_id}:{sha256} → {backend_url, model_id, created_at, hit_count}

On a cache hit the request is routed to the recorded backend, where the engine's
internal KV blocks are more likely still resident (reducing recompute cost).

Design decisions:
- Flat Redis Hash per prefix (not a radix tree). The engine owns prefix-sharing
  within a sequence; Lumen only needs "did backend X serve this exact prefix?".
- SHA-256 of a deterministically serialized (model, messages, tools) tuple.
  model_id is included in the key, not just the hash, to keep keys scannable.
- Tools are sorted before hashing so tool-definition ordering doesn't create
  spurious cache misses.
- TTL is absolute (not sliding). Hot prefixes re-register on each completion,
  keeping them warm naturally.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as aioredis

from lumen.models.openai_compat import ChatMessage

_KEY_NS = "lumen:cache:exact"


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------

def _stable_message(msg: ChatMessage) -> dict[str, Any]:
    """Minimal, stable dict representation of a ChatMessage for hashing."""
    out: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        out["content"] = msg.content
    if msg.name is not None:
        out["name"] = msg.name
    if msg.tool_calls:
        out["tool_calls"] = msg.tool_calls
    return out


def canonical_prefix(
    model_id: str,
    messages: list[ChatMessage],
    tools: list[dict[str, Any]] | None = None,
) -> bytes:
    """Return deterministic UTF-8 bytes for a (model, messages, tools) tuple.

    Guarantee: same logical request → same bytes, regardless of Python dict
    insertion order or whitespace differences in the caller.

    Tools are sorted by their JSON representation so that two requests
    specifying the same tools in different orders hash identically.
    """
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [_stable_message(m) for m in messages],
    }
    if tools:
        payload["tools"] = sorted(tools, key=lambda t: json.dumps(t, sort_keys=True))

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def prefix_hash(canonical_bytes: bytes) -> str:
    """SHA-256 hex digest of canonical prefix bytes."""
    return hashlib.sha256(canonical_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Redis key helpers
# ---------------------------------------------------------------------------

def _safe_model(model_id: str) -> str:
    """Replace characters that would make Redis keys awkward to scan or parse."""
    return model_id.replace("/", "_").replace(":", "_").replace(" ", "_")


def cache_key(model_id: str, hash_hex: str) -> str:
    """Fully-qualified Redis key for an exact-prefix cache entry.

    Format: lumen:cache:exact:{safe_model_id}:{sha256_hex}
    """
    return f"{_KEY_NS}:{_safe_model(model_id)}:{hash_hex}"


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    backend_url: str
    model_id: str
    created_at: int
    hit_count: int

    @classmethod
    def from_redis(cls, data: dict[str, str]) -> "CacheEntry":
        return cls(
            backend_url=data["backend_url"],
            model_id=data["model_id"],
            created_at=int(data.get("created_at", "0")),
            hit_count=int(data.get("hit_count", "0")),
        )


# ---------------------------------------------------------------------------
# Redis operations
# ---------------------------------------------------------------------------

async def get_entry(
    client: aioredis.Redis,
    model_id: str,
    hash_hex: str,
) -> CacheEntry | None:
    """Return the cache entry for this prefix hash, or None on miss.

    Increments hit_count on a hit so operators can identify hot prefixes.
    The increment is non-critical: a Redis error here does not fail the request.
    """
    key = cache_key(model_id, hash_hex)
    data: dict[str, str] = await client.hgetall(key)
    if not data or "backend_url" not in data:
        return None
    try:
        await client.hincrby(key, "hit_count", 1)
    except Exception:
        pass
    return CacheEntry.from_redis(data)


async def put_entry(
    client: aioredis.Redis,
    model_id: str,
    hash_hex: str,
    backend_url: str,
    ttl_seconds: int,
) -> None:
    """Write (or refresh) a cache entry atomically via pipeline.

    Re-registering an existing key resets hit_count to 0 and refreshes TTL,
    which is correct: the backend_url may have changed if the pool was
    rebalanced since the last registration.
    """
    key = cache_key(model_id, hash_hex)
    pipe = client.pipeline(transaction=False)
    pipe.hset(
        key,
        mapping={
            "backend_url": backend_url,
            "model_id": model_id,
            "created_at": str(int(time.time())),
            "hit_count": "0",
        },
    )
    pipe.expire(key, ttl_seconds)
    await pipe.execute()


async def delete_entry(
    client: aioredis.Redis,
    model_id: str,
    hash_hex: str,
) -> None:
    """Explicitly invalidate one cache entry (e.g. on model revision change)."""
    await client.delete(cache_key(model_id, hash_hex))
