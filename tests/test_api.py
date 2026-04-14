import pytest
from pydantic import ValidationError

from lumen.api.routes import health as health_route
from lumen.settings import get_settings


@pytest.mark.asyncio
async def test_health_ok(client) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["redis"] == "skipped"
    assert body["inference"] == "skipped"


@pytest.mark.asyncio
async def test_inference_health_ok(client, monkeypatch) -> None:
    async def _ok_readiness(_settings):
        return {"status": "ok"}

    monkeypatch.setattr(health_route, "_inference_readiness", _ok_readiness)
    r = await client.get("/health/inference")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["detail"] is None


@pytest.mark.asyncio
async def test_inference_health_unreachable(client, monkeypatch) -> None:
    async def _unreachable_readiness(_settings):
        return {"status": "unreachable", "detail": "connection refused"}

    monkeypatch.setattr(health_route, "_inference_readiness", _unreachable_readiness)
    r = await client.get("/health/inference")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "unreachable"
    assert body["detail"] == "connection refused"


@pytest.mark.asyncio
async def test_list_models(client) -> None:
    r = await client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert any(m["id"] == "Qwen/Qwen2.5-7B-Instruct" for m in data["data"])


@pytest.mark.asyncio
async def test_chat_completion_non_stream(client) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Lumen" in (data["choices"][0]["message"]["content"] or "")


@pytest.mark.asyncio
async def test_chat_completion_stream(client) -> None:
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        lines: list[str] = []
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                lines.append(line)
        assert any("[DONE]" in ln for ln in lines)


@pytest.mark.asyncio
async def test_embeddings(client) -> None:
    r = await client.post(
        "/v1/embeddings",
        json={"model": "Qwen/Qwen2.5-7B-Instruct", "input": "hello"},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 8


@pytest.mark.asyncio
async def test_chat_completion_rejects_unknown_model(client) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "unknown/model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert "not allowed" in body["detail"]


@pytest.mark.asyncio
async def test_chat_completion_auto_uses_default_model(client, monkeypatch) -> None:
    original = get_settings()

    class _SettingsProxy:
        redis_url = original.redis_url
        inference_base_url = original.inference_base_url
        inference_api_key = original.inference_api_key
        inference_model_ids = original.inference_model_ids
        default_model_id = original.inference_model_ids[1]
        allow_unknown_models = original.allow_unknown_models

    monkeypatch.setattr("lumen.api.routes.v1_inference.get_settings", lambda: _SettingsProxy())
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == original.inference_model_ids[1]


def test_settings_reject_invalid_default_model(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_MODEL_IDS", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("DEFAULT_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
    get_settings.cache_clear()
    with pytest.raises(ValidationError, match="DEFAULT_MODEL_ID must be one of INFERENCE_MODEL_IDS"):
        get_settings()
    get_settings.cache_clear()
