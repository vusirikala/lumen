import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from lumen.api.routes import health as health_route
from lumen.api.routes import v1_inference
from lumen.settings import get_settings
from lumen.telemetry import inference_telemetry


def _proxy_settings():
    original = get_settings()

    class _SettingsProxy:
        redis_url = original.redis_url
        inference_base_url = "http://backend.local"
        inference_api_key = None
        inference_model_ids = original.inference_model_ids
        default_model_id = original.default_model_id
        allow_unknown_models = original.allow_unknown_models
        proxy_chat_timeout_seconds = 10.0
        proxy_completion_timeout_seconds = 10.0
        proxy_embedding_timeout_seconds = 10.0
        proxy_max_retries = 0
        proxy_retry_backoff_seconds = 0.0

    return _SettingsProxy()


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


@pytest.mark.asyncio
async def test_proxy_sets_request_id_header(client, monkeypatch) -> None:
    async def _fake_proxy_request(path: str, payload: dict[str, object], request_id: str):
        assert path == "/v1/chat/completions"
        assert payload["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert request_id == "req-123"
        return type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": staticmethod(lambda: {"id": "chatcmpl-1", "object": "chat.completion", "model": payload["model"], "choices": []}),
            },
        )()

    monkeypatch.setattr(v1_inference, "get_settings", _proxy_settings)
    monkeypatch.setattr(v1_inference, "_proxy_request", _fake_proxy_request)
    r = await client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-123"},
        json={"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "req-123"


@pytest.mark.asyncio
async def test_proxy_error_payload_includes_request_id(client, monkeypatch) -> None:
    async def _failing_proxy_request(_path: str, _payload: dict[str, object], _request_id: str):
        raise HTTPException(status_code=502, detail={"message": "Inference backend error", "request_id": _request_id})

    monkeypatch.setattr(v1_inference, "get_settings", _proxy_settings)
    monkeypatch.setattr(v1_inference, "_proxy_request", _failing_proxy_request)
    r = await client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-999"},
        json={"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["request_id"] == "req-999"


@pytest.mark.asyncio
async def test_proxy_completions_success(client, monkeypatch) -> None:
    async def _fake_proxy_request(path: str, payload: dict[str, object], request_id: str):
        assert path == "/v1/completions"
        assert payload["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert request_id == "cmp-1"
        return type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": staticmethod(lambda: {"id": "cmpl-1", "object": "text_completion", "model": payload["model"], "choices": [{"text": "ok"}]}),
            },
        )()

    monkeypatch.setattr(v1_inference, "get_settings", _proxy_settings)
    monkeypatch.setattr(v1_inference, "_proxy_request", _fake_proxy_request)
    r = await client.post(
        "/v1/completions",
        headers={"x-request-id": "cmp-1"},
        json={"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "hello"},
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "cmp-1"
    assert r.json()["object"] == "text_completion"


@pytest.mark.asyncio
async def test_proxy_embeddings_success(client, monkeypatch) -> None:
    async def _fake_proxy_request(path: str, payload: dict[str, object], request_id: str):
        assert path == "/v1/embeddings"
        assert payload["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert request_id == "emb-1"
        return type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": staticmethod(
                    lambda: {
                        "object": "list",
                        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                        "model": payload["model"],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                ),
            },
        )()

    monkeypatch.setattr(v1_inference, "get_settings", _proxy_settings)
    monkeypatch.setattr(v1_inference, "_proxy_request", _fake_proxy_request)
    r = await client.post(
        "/v1/embeddings",
        headers={"x-request-id": "emb-1"},
        json={"model": "Qwen/Qwen2.5-7B-Instruct", "input": "hello"},
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "emb-1"
    assert r.json()["data"][0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_proxy_chat_backend_http_error_maps_detail(client, monkeypatch) -> None:
    class _ErrorResponse:
        status_code = 503
        text = "backend unavailable"

        @staticmethod
        def json() -> dict[str, str]:
            return {"message": "backend unavailable"}

    async def _fake_proxy_request(_path: str, _payload: dict[str, object], _request_id: str):
        return _ErrorResponse()

    monkeypatch.setattr(v1_inference, "get_settings", _proxy_settings)
    monkeypatch.setattr(v1_inference, "_proxy_request", _fake_proxy_request)
    r = await client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "err-1"},
        json={"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["message"] == "backend unavailable"
    assert detail["request_id"] == "err-1"


@pytest.mark.asyncio
async def test_proxy_chat_stream_passthrough(client, monkeypatch) -> None:
    async def _fake_proxy_stream(_path: str, _payload: dict[str, object], _request_id: str):
        yield b"data: {\"id\":\"chunk-1\"}\n\n"
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(v1_inference, "get_settings", _proxy_settings)
    monkeypatch.setattr(v1_inference, "_proxy_stream", _fake_proxy_stream)
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-request-id": "stream-1"},
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["x-request-id"] == "stream-1"
        lines = [line async for line in r.aiter_lines() if line.startswith("data: ")]
    assert any("[DONE]" in line for line in lines)


@pytest.mark.asyncio
async def test_inference_metrics_rows_present(client) -> None:
    await client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    r = await client.get("/metrics/inference")
    assert r.status_code == 200
    body = r.json()
    assert "rows" in body
    assert any(
        row["endpoint"] == "/v1/chat/completions" and row["model"] == "Qwen/Qwen2.5-7B-Instruct"
        for row in body["rows"]
    )


def test_inference_telemetry_records_error_count() -> None:
    inference_telemetry.record(endpoint="/v1/chat/completions", model="model-a", status_code=500, latency_ms=12.0)
    snapshot = inference_telemetry.snapshot()
    matching_rows = [row for row in snapshot["rows"] if row["endpoint"] == "/v1/chat/completions" and row["model"] == "model-a"]
    assert matching_rows
    assert matching_rows[0]["errors"] >= 1
