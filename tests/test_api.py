import pytest

from lumen.api.routes import health as health_route


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
