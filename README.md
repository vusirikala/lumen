# Lumen

Lumen is an LLM inference **control plane**: a FastAPI service that exposes OpenAI-compatible HTTP APIs. Responses are **dummy placeholders** today; wiring to real inference backends (vLLM, TGI, and similar) is the next step.

## Requirements

- Python 3.11 or newer
- Optional: [Redis](https://redis.io/) if you set `REDIS_URL` (used for `/health` checks and future state)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the server

From the repository root, put the `src` tree on `PYTHONPATH` so the `lumen` package resolves:

```bash
PYTHONPATH=src uvicorn lumen.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive OpenAPI documentation.

## Configuration

Environment variables are loaded via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) (optional `.env` in the project root).

| Variable     | Description |
| ------------ | ----------- |
| `REDIS_URL`  | If set, e.g. `redis://localhost:6379/0`, the app opens an async Redis client and `/health` reports Redis status. If unset, health reports Redis as skipped. |

## HTTP API (current)

All paths below are served by the app; behavior is **stub** unless noted.

| Method | Path | Notes |
| ------ | ---- | ----- |
| `GET`  | `/health` | Liveness; optional Redis ping when `REDIS_URL` is set. |
| `GET`  | `/v1/models` | Lists a dummy model `lumen-dummy`. |
| `GET`  | `/v1/models/{model_id}` | Returns model metadata or 404. |
| `POST` | `/v1/chat/completions` | Dummy chat completion; supports `stream: true` (SSE). |
| `POST` | `/v1/completions` | Dummy text completion; supports streaming. |
| `POST` | `/v1/embeddings` | Dummy embeddings (fixed small dimension). |

## Tests

```bash
PYTHONPATH=src pytest tests/ -q
```

Tests use [pytest](https://docs.pytest.org/) and [httpx](https://www.python-httpx.org/) against the ASGI app.

## Project layout

```text
src/lumen/
  main.py              # FastAPI app factory and lifespan
  settings.py          # Settings from environment
  api/routes/          # HTTP routers (health, OpenAI v1 stubs)
  models/              # Pydantic request/response schemas
tests/                 # API smoke tests
```
