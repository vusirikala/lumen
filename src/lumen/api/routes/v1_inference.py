import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lumen.models.openai_compat import (
    ChatCompletionChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    ModelsListResponse,
)
from lumen.settings import get_settings

router = APIRouter(tags=["inference"])

_DUMMY_CHAT_REPLY = (
    "This is a dummy chat completion from Lumen. "
    "Set INFERENCE_BASE_URL to forward to your vLLM inference service."
)
_DUMMY_EMBEDDING_DIM = 8
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def _models_from_settings() -> list[ModelInfo]:
    settings = get_settings()
    return [ModelInfo(id=model_id, created=0, owned_by="self-hosted") for model_id in settings.inference_model_ids]


def _effective_model_id(requested: str) -> str:
    settings = get_settings()
    if requested in ("", "auto"):
        selected = settings.default_model_id or settings.inference_model_ids[0]
    else:
        selected = requested
    if not settings.allow_unknown_models and selected not in settings.inference_model_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model {selected!r} is not allowed. "
                "Choose one from INFERENCE_MODEL_IDS or set ALLOW_UNKNOWN_MODELS=true."
            ),
        )
    return selected


def _correlation_id(request: Request) -> str:
    incoming = request.headers.get("x-request-id")
    if incoming:
        return incoming
    return uuid.uuid4().hex


def _backend_url(path: str) -> str:
    settings = get_settings()
    if settings.inference_base_url is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Inference backend is not configured. "
                "Set INFERENCE_BASE_URL to route /v1 requests to vLLM."
            ),
        )
    return f"{str(settings.inference_base_url).rstrip('/')}{path}"


def _proxy_headers(request_id: str) -> dict[str, str]:
    settings = get_settings()
    headers: dict[str, str] = {}
    if settings.inference_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.inference_api_key}"
    headers["X-Request-ID"] = request_id
    return headers


def _timeout_for_path(path: str) -> httpx.Timeout:
    settings = get_settings()
    if path == "/v1/chat/completions":
        total = settings.proxy_chat_timeout_seconds
    elif path == "/v1/completions":
        total = settings.proxy_completion_timeout_seconds
    else:
        total = settings.proxy_embedding_timeout_seconds
    return httpx.Timeout(total, connect=min(total, 5.0))


def _proxy_error_detail(response: httpx.Response, request_id: str) -> dict[str, Any]:
    try:
        backend_payload = response.json()
    except ValueError:
        backend_payload = {"message": response.text}
    if not isinstance(backend_payload, dict):
        backend_payload = {"message": str(backend_payload)}
    backend_payload["request_id"] = request_id
    return backend_payload


async def _proxy_request(path: str, payload: dict[str, Any], request_id: str) -> httpx.Response:
    settings = get_settings()
    url = _backend_url(path)
    headers = _proxy_headers(request_id)
    timeout = _timeout_for_path(path)
    attempts = settings.proxy_max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            if attempt < attempts:
                await asyncio.sleep(settings.proxy_retry_backoff_seconds * attempt)
                continue
            raise HTTPException(
                status_code=502,
                detail={"message": f"Inference backend error: {exc}", "request_id": request_id},
            ) from exc
        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < attempts:
            await asyncio.sleep(settings.proxy_retry_backoff_seconds * attempt)
            continue
        return response
    raise HTTPException(status_code=502, detail={"message": "Inference backend retry exhaustion", "request_id": request_id})


async def _proxy_stream(path: str, payload: dict[str, Any], request_id: str) -> AsyncIterator[bytes]:
    settings = get_settings()
    url = _backend_url(path)
    headers = _proxy_headers(request_id)
    timeout = _timeout_for_path(path)
    attempts = settings.proxy_max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code in _RETRYABLE_STATUS_CODES and attempt < attempts:
                        await response.aread()
                        await asyncio.sleep(settings.proxy_retry_backoff_seconds * attempt)
                        continue
                    if response.status_code >= 400:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=_proxy_error_detail(response, request_id),
                        )
                    async for chunk in response.aiter_bytes():
                        yield chunk
                    return
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            if attempt < attempts:
                await asyncio.sleep(settings.proxy_retry_backoff_seconds * attempt)
                continue
            raise HTTPException(
                status_code=502,
                detail={"message": f"Inference backend error: {exc}", "request_id": request_id},
            ) from exc
    raise HTTPException(status_code=502, detail={"message": "Inference backend retry exhaustion", "request_id": request_id})


def _prompt_preview(req: ChatCompletionRequest) -> str:
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content:
            return msg.content[:120]
    return ""


@router.get("/models")
async def list_models() -> ModelsListResponse:
    return ModelsListResponse(data=_models_from_settings())


@router.get("/models/{model_id}")
async def retrieve_model(model_id: str) -> ModelInfo:
    for m in _models_from_settings():
        if m.id == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    selected_model = _effective_model_id(body.model)
    request_id = _correlation_id(request)
    settings = get_settings()
    if settings.inference_base_url is not None:
        payload = body.model_dump(exclude_none=True)
        payload["model"] = selected_model
        if body.stream:
            stream_response = StreamingResponse(
                _proxy_stream("/v1/chat/completions", payload, request_id),
                media_type="text/event-stream",
            )
            stream_response.headers["X-Request-ID"] = request_id
            return stream_response
        response = await _proxy_request("/v1/chat/completions", payload, request_id)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=_proxy_error_detail(response, request_id))
        return JSONResponse(content=response.json(), headers={"X-Request-ID": request_id})

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    preview = _prompt_preview(body)
    content = _DUMMY_CHAT_REPLY
    if preview:
        content = f"{_DUMMY_CHAT_REPLY} (last user message preview: {preview!r})"

    if body.stream:
        return StreamingResponse(
            _chat_completion_sse(req_id, created, selected_model, content),
            media_type="text/event-stream",
        )

    return ChatCompletionResponse(
        id=req_id,
        created=created,
        model=selected_model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(content=content),
            )
        ],
    )


async def _chat_completion_sse(
    req_id: str,
    created: int,
    model: str,
    full_text: str,
) -> AsyncIterator[str]:
    """OpenAI-style SSE stream (dummy tokens)."""
    first = ChatCompletionStreamResponse(
        id=req_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                delta=ChatCompletionDelta(role="assistant", content=""),
            )
        ],
    )
    yield f"data: {first.model_dump_json()}\n\n"
    await asyncio.sleep(0)

    words = full_text.split()
    for i, word in enumerate(words):
        piece = word if i == 0 else f" {word}"
        chunk = ChatCompletionStreamResponse(
            id=req_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    delta=ChatCompletionDelta(content=piece),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0)

    final = ChatCompletionStreamResponse(
        id=req_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                delta=ChatCompletionDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/completions")
async def completions(body: CompletionRequest, request: Request) -> Any:
    selected_model = _effective_model_id(body.model)
    request_id = _correlation_id(request)
    settings = get_settings()
    if settings.inference_base_url is not None:
        payload = body.model_dump(exclude_none=True)
        payload["model"] = selected_model
        if body.stream:
            stream_response = StreamingResponse(
                _proxy_stream("/v1/completions", payload, request_id),
                media_type="text/event-stream",
            )
            stream_response.headers["X-Request-ID"] = request_id
            return stream_response
        response = await _proxy_request("/v1/completions", payload, request_id)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=_proxy_error_detail(response, request_id))
        return JSONResponse(content=response.json(), headers={"X-Request-ID": request_id})

    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    if isinstance(body.prompt, str):
        prompt_text = body.prompt
    else:
        prompt_text = " ".join(body.prompt[:3])
    text = (
        f"[dummy completion] continuation for: {prompt_text[:100]!r}..."
        if prompt_text
        else "[dummy completion]"
    )

    if body.stream:
        return StreamingResponse(
            _completion_sse(req_id, created, selected_model, text),
            media_type="text/event-stream",
        )

    return CompletionResponse(
        id=req_id,
        created=created,
        model=selected_model,
        choices=[CompletionChoice(text=text)],
    )


async def _completion_sse(
    req_id: str,
    created: int,
    model: str,
    full_text: str,
) -> AsyncIterator[str]:
    """Legacy completions SSE (simplified JSON lines)."""
    for i, ch in enumerate(full_text):
        payload = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"text": ch, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0)
    done = {
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/embeddings")
async def embeddings(body: EmbeddingRequest, request: Request) -> Any:
    selected_model = _effective_model_id(body.model)
    request_id = _correlation_id(request)
    settings = get_settings()
    if settings.inference_base_url is not None:
        payload = body.model_dump(exclude_none=True)
        payload["model"] = selected_model
        response = await _proxy_request("/v1/embeddings", payload, request_id)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=_proxy_error_detail(response, request_id))
        return JSONResponse(
            content=EmbeddingResponse.model_validate(response.json()).model_dump(),
            headers={"X-Request-ID": request_id},
        )

    if isinstance(body.input, str):
        inputs = [body.input]
    else:
        inputs = body.input

    data: list[EmbeddingData] = []
    for idx, text in enumerate(inputs):
        # Deterministic tiny dummy vector from text length (not semantic)
        base = float((len(text) % 97) / 97.0)
        vec = [round((base + i * 0.01) % 1.0, 6) for i in range(_DUMMY_EMBEDDING_DIM)]
        data.append(EmbeddingData(embedding=vec, index=idx))

    return EmbeddingResponse(
        data=data,
        model=selected_model,
        usage={"prompt_tokens": sum(len(t) for t in inputs), "total_tokens": sum(len(t) for t in inputs)},
    )

