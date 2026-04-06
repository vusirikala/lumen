import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from lumen.models.openai import (
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

router = APIRouter(tags=["openai"])

_DUMMY_MODELS: list[ModelInfo] = [
    ModelInfo(id="lumen-dummy", created=0, owned_by="lumen"),
]

_DUMMY_CHAT_REPLY = (
    "This is a dummy chat completion from Lumen. "
    "Wire this endpoint to your inference engine when ready."
)
_DUMMY_EMBEDDING_DIM = 8


def _prompt_preview(req: ChatCompletionRequest) -> str:
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content:
            return msg.content[:120]
    return ""


@router.get("/models")
async def list_models() -> ModelsListResponse:
    return ModelsListResponse(data=list(_DUMMY_MODELS))


@router.get("/models/{model_id}")
async def retrieve_model(model_id: str) -> ModelInfo:
    for m in _DUMMY_MODELS:
        if m.id == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> Any:
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    preview = _prompt_preview(body)
    content = _DUMMY_CHAT_REPLY
    if preview:
        content = f"{_DUMMY_CHAT_REPLY} (last user message preview: {preview!r})"

    if body.stream:
        return StreamingResponse(
            _chat_completion_sse(req_id, created, body.model, content),
            media_type="text/event-stream",
        )

    return ChatCompletionResponse(
        id=req_id,
        created=created,
        model=body.model,
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
async def completions(body: CompletionRequest) -> Any:
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
            _completion_sse(req_id, created, body.model, text),
            media_type="text/event-stream",
        )

    return CompletionResponse(
        id=req_id,
        created=created,
        model=body.model,
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
async def embeddings(body: EmbeddingRequest) -> EmbeddingResponse:
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
        model=body.model,
        usage={"prompt_tokens": sum(len(t) for t in inputs), "total_tokens": sum(len(t) for t in inputs)},
    )
