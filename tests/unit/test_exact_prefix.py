"""
Unit tests for exact-prefix cache: canonical serialization, hashing, key
generation, and Redis get/put operations (using a mock Redis client).

No network calls. No real Redis.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lumen.models.openai_compat import ChatMessage
from lumen.services.cache.exact_prefix import (
    CacheEntry,
    cache_key,
    canonical_prefix,
    delete_entry,
    get_entry,
    prefix_hash,
    put_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str | None = None, **kwargs) -> ChatMessage:
    return ChatMessage(role=role, content=content, **kwargs)


def _hash(model: str, messages: list[ChatMessage], tools=None) -> str:
    return prefix_hash(canonical_prefix(model, messages, tools))


# ---------------------------------------------------------------------------
# canonical_prefix: same logical request → same bytes
# ---------------------------------------------------------------------------

class TestCanonicalPrefix:
    def test_identical_inputs_produce_identical_bytes(self):
        msgs = [_msg("user", "hello")]
        assert canonical_prefix("m", msgs) == canonical_prefix("m", msgs)

    def test_model_id_is_included(self):
        msgs = [_msg("user", "hello")]
        assert canonical_prefix("model-a", msgs) != canonical_prefix("model-b", msgs)

    def test_message_content_change_changes_bytes(self):
        a = canonical_prefix("m", [_msg("user", "hello")])
        b = canonical_prefix("m", [_msg("user", "world")])
        assert a != b

    def test_message_role_change_changes_bytes(self):
        a = canonical_prefix("m", [_msg("user", "hi")])
        b = canonical_prefix("m", [_msg("assistant", "hi")])
        assert a != b

    def test_additional_message_changes_bytes(self):
        base = [_msg("system", "You are helpful."), _msg("user", "Hi")]
        extended = base + [_msg("assistant", "Hello!"), _msg("user", "Follow up")]
        assert canonical_prefix("m", base) != canonical_prefix("m", extended)

    def test_message_ordering_matters(self):
        a = canonical_prefix("m", [_msg("system", "S"), _msg("user", "U")])
        b = canonical_prefix("m", [_msg("user", "U"), _msg("system", "S")])
        assert a != b

    def test_tools_included_when_provided(self):
        msgs = [_msg("user", "hi")]
        tool = {"type": "function", "function": {"name": "f", "description": "d"}}
        without = canonical_prefix("m", msgs)
        with_tools = canonical_prefix("m", msgs, tools=[tool])
        assert without != with_tools

    def test_tool_ordering_is_normalised(self):
        msgs = [_msg("user", "hi")]
        tool_a = {"type": "function", "function": {"name": "a"}}
        tool_b = {"type": "function", "function": {"name": "b"}}
        ab = canonical_prefix("m", msgs, tools=[tool_a, tool_b])
        ba = canonical_prefix("m", msgs, tools=[tool_b, tool_a])
        assert ab == ba

    def test_none_and_empty_tools_treated_identically(self):
        msgs = [_msg("user", "hi")]
        assert canonical_prefix("m", msgs, tools=None) == canonical_prefix("m", msgs, tools=[])

    def test_output_is_valid_utf8_json(self):
        msgs = [_msg("user", "héllo wörld")]
        raw = canonical_prefix("m", msgs)
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["messages"][0]["content"] == "héllo wörld"

    def test_optional_fields_omitted_when_none(self):
        msg = _msg("user", "hi")
        raw = canonical_prefix("m", [msg])
        parsed = json.loads(raw)
        assert "name" not in parsed["messages"][0]
        assert "tool_calls" not in parsed["messages"][0]

    def test_name_included_when_set(self):
        msg = _msg("tool", "result", name="my_tool")
        raw = canonical_prefix("m", [msg])
        parsed = json.loads(raw)
        assert parsed["messages"][0]["name"] == "my_tool"


# ---------------------------------------------------------------------------
# prefix_hash: determinism and sensitivity
# ---------------------------------------------------------------------------

class TestPrefixHash:
    def test_same_bytes_same_hash(self):
        msgs = [_msg("user", "hello")]
        h1 = _hash("m", msgs)
        h2 = _hash("m", msgs)
        assert h1 == h2

    def test_different_content_different_hash(self):
        assert _hash("m", [_msg("user", "a")]) != _hash("m", [_msg("user", "b")])

    def test_hash_is_64_hex_chars(self):
        h = _hash("m", [_msg("user", "hi")])
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# cache_key: format and sanitisation
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_format(self):
        k = cache_key("mymodel", "abc123")
        assert k == "lumen:cache:exact:mymodel:abc123"

    def test_slash_in_model_id_is_sanitised(self):
        k = cache_key("Qwen/Qwen2.5-7B-Instruct", "deadbeef")
        assert "/" not in k
        assert k.startswith("lumen:cache:exact:")

    def test_colon_in_model_id_is_sanitised(self):
        k = cache_key("org:model", "deadbeef")
        # lumen:cache:exact:org_model:deadbeef → 4 colons from the namespace + separator + hash
        assert "org_model" in k
        assert k == "lumen:cache:exact:org_model:deadbeef"

    def test_different_models_different_keys(self):
        assert cache_key("model-a", "hash") != cache_key("model-b", "hash")

    def test_different_hashes_different_keys(self):
        assert cache_key("m", "hash1") != cache_key("m", "hash2")


# ---------------------------------------------------------------------------
# CacheEntry.from_redis
# ---------------------------------------------------------------------------

class TestCacheEntry:
    def test_from_redis_parses_fields(self):
        data = {
            "backend_url": "http://backend:8000",
            "model_id": "mymodel",
            "created_at": "1713100000",
            "hit_count": "5",
        }
        entry = CacheEntry.from_redis(data)
        assert entry.backend_url == "http://backend:8000"
        assert entry.model_id == "mymodel"
        assert entry.created_at == 1713100000
        assert entry.hit_count == 5

    def test_from_redis_defaults_missing_optional_fields(self):
        data = {"backend_url": "http://b:8000", "model_id": "m"}
        entry = CacheEntry.from_redis(data)
        assert entry.created_at == 0
        assert entry.hit_count == 0


# ---------------------------------------------------------------------------
# get_entry: Redis interaction
# ---------------------------------------------------------------------------

class TestGetEntry:
    def _make_client(self, hgetall_return: dict) -> AsyncMock:
        client = AsyncMock()
        client.hgetall = AsyncMock(return_value=hgetall_return)
        client.hincrby = AsyncMock(return_value=1)
        return client

    @pytest.mark.asyncio
    async def test_returns_none_on_miss(self):
        client = self._make_client({})
        result = await get_entry(client, "mymodel", "abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_backend_url_missing(self):
        client = self._make_client({"model_id": "m", "created_at": "0"})
        result = await get_entry(client, "mymodel", "abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_entry_on_hit(self):
        data = {
            "backend_url": "http://b:8000",
            "model_id": "mymodel",
            "created_at": "1000",
            "hit_count": "2",
        }
        client = self._make_client(data)
        result = await get_entry(client, "mymodel", "abc")
        assert result is not None
        assert result.backend_url == "http://b:8000"
        assert result.hit_count == 2

    @pytest.mark.asyncio
    async def test_increments_hit_count_on_hit(self):
        data = {
            "backend_url": "http://b:8000",
            "model_id": "mymodel",
            "created_at": "0",
            "hit_count": "0",
        }
        client = self._make_client(data)
        await get_entry(client, "mymodel", "abc")
        client.hincrby.assert_called_once()
        args = client.hincrby.call_args
        assert args[0][1] == "hit_count"
        assert args[0][2] == 1

    @pytest.mark.asyncio
    async def test_does_not_increment_on_miss(self):
        client = self._make_client({})
        await get_entry(client, "mymodel", "abc")
        client.hincrby.assert_not_called()

    @pytest.mark.asyncio
    async def test_hincrby_error_does_not_raise(self):
        data = {"backend_url": "http://b:8000", "model_id": "m", "created_at": "0", "hit_count": "0"}
        client = self._make_client(data)
        client.hincrby = AsyncMock(side_effect=Exception("redis error"))
        # Should not raise; result should still be returned
        result = await get_entry(client, "m", "abc")
        assert result is not None


# ---------------------------------------------------------------------------
# put_entry: Redis interaction
# ---------------------------------------------------------------------------

class TestPutEntry:
    def _make_client(self) -> AsyncMock:
        client = AsyncMock()
        pipe = AsyncMock()
        pipe.hset = MagicMock(return_value=pipe)
        pipe.expire = MagicMock(return_value=pipe)
        pipe.execute = AsyncMock(return_value=[1, True])
        client.pipeline = MagicMock(return_value=pipe)
        return client, pipe

    @pytest.mark.asyncio
    async def test_calls_pipeline_hset_and_expire(self):
        client, pipe = self._make_client()
        await put_entry(client, "mymodel", "abc123", "http://b:8000", ttl_seconds=300)
        pipe.hset.assert_called_once()
        pipe.expire.assert_called_once()
        pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_hset_mapping_contains_required_fields(self):
        client, pipe = self._make_client()
        await put_entry(client, "mymodel", "abc123", "http://b:8000", ttl_seconds=60)
        call_kwargs = pipe.hset.call_args[1]
        mapping = call_kwargs["mapping"]
        assert mapping["backend_url"] == "http://b:8000"
        assert mapping["model_id"] == "mymodel"
        assert "created_at" in mapping

    @pytest.mark.asyncio
    async def test_expire_uses_correct_ttl(self):
        client, pipe = self._make_client()
        await put_entry(client, "mymodel", "abc123", "http://b:8000", ttl_seconds=120)
        expire_args = pipe.expire.call_args[0]
        assert expire_args[1] == 120


# ---------------------------------------------------------------------------
# delete_entry
# ---------------------------------------------------------------------------

class TestDeleteEntry:
    @pytest.mark.asyncio
    async def test_deletes_correct_key(self):
        client = AsyncMock()
        client.delete = AsyncMock(return_value=1)
        await delete_entry(client, "mymodel", "abc123")
        expected_key = cache_key("mymodel", "abc123")
        client.delete.assert_called_once_with(expected_key)
