# Lumen

Lumen is an LLM inference **control plane**: a FastAPI service that exposes OpenAI-compatible HTTP APIs and can route requests to a self-hosted inference backend (for example vLLM).

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
You can start from [`.env.example`](.env.example) and adapt values for your cluster.

| Variable | Description |
| --- | --- |
| `REDIS_URL` | If set, e.g. `redis://localhost:6379/0`, the app opens an async Redis client and `/health` reports Redis status. If unset, health reports Redis as skipped. |
| `INFERENCE_BASE_URL` | Base URL for an OpenAI-compatible self-hosted backend (for example `http://vllm-service:8000`). If unset, Lumen returns local dummy responses for completion and embedding routes. |
| `INFERENCE_API_KEY` | Optional bearer token forwarded to the backend as `Authorization: Bearer ...`. |
| `INFERENCE_MODEL_IDS` | Comma-separated model IDs exposed by `/v1/models` (for example `Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3`). |
| `DEFAULT_MODEL_ID` | Optional default model used when requests set `model` to `auto` or empty. |
| `ALLOW_UNKNOWN_MODELS` | Defaults to `false`. When `false`, request models must be in `INFERENCE_MODEL_IDS`; when `true`, unknown model IDs are passed through to backend. |
| `PROXY_CHAT_TIMEOUT_SECONDS` | Timeout for `/v1/chat/completions` backend calls (default `120`). |
| `PROXY_COMPLETION_TIMEOUT_SECONDS` | Timeout for `/v1/completions` backend calls (default `120`). |
| `PROXY_EMBEDDING_TIMEOUT_SECONDS` | Timeout for `/v1/embeddings` backend calls (default `60`). |
| `PROXY_MAX_RETRIES` | Retry count for retryable backend failures (`429`, `502`, `503`, `504`), default `2`. |
| `PROXY_RETRY_BACKOFF_SECONDS` | Linear backoff base between retries, default `0.2`. |

Example `.env` for vLLM:

```bash
INFERENCE_BASE_URL=http://127.0.0.1:8001
INFERENCE_MODEL_IDS=Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct,mistralai/Mistral-7B-Instruct-v0.3
DEFAULT_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
```

### Deployment Profiles (Ampere-focused)

Use these as starting points, then tune with production traffic.

| Profile | Recommended model class | Typical GPUs | Suggested Lumen proxy tuning |
| --- | --- | --- | --- |
| Small / low-latency | 7B-8B instruct (`Qwen2.5-7B`, `Mistral-7B`) | A10/L4/A100 | `CHAT=120`, `COMPLETION=90`, `EMBED=45`, `RETRIES=1` |
| Medium / quality | 14B-32B instruct (`Qwen2.5-14B/32B`, `Gemma-2-27B`) | A100 multi-GPU | `CHAT=180`, `COMPLETION=120`, `EMBED=60`, `RETRIES=2` |
| Large / high-quality | MoE or larger dense models | A100 cluster | `CHAT=300`, `COMPLETION=180`, `EMBED=90`, `RETRIES=2-3` |

Where `CHAT/COMPLETION/EMBED/RETRIES` map to:
- `PROXY_CHAT_TIMEOUT_SECONDS`
- `PROXY_COMPLETION_TIMEOUT_SECONDS`
- `PROXY_EMBEDDING_TIMEOUT_SECONDS`
- `PROXY_MAX_RETRIES`

### Model Swap Runbook

1. Update catalog/default model in `.env`:
   - set `INFERENCE_MODEL_IDS=...`
   - set `DEFAULT_MODEL_ID=...`
2. Restart Lumen and verify:
   - `GET /v1/models` includes the new model
   - `POST /v1/chat/completions` with `model: "auto"` resolves to the default
3. Confirm backend compatibility:
   - run `GET /health/inference`
   - send a smoke prompt to `/v1/chat/completions`

Quick examples:
- Qwen default:
  - `DEFAULT_MODEL_ID=Qwen/Qwen2.5-14B-Instruct`
- Mistral default:
  - `DEFAULT_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3`
- Gemma default:
  - `DEFAULT_MODEL_ID=google/gemma-2-9b-it`

Model governance behavior:
- `INFERENCE_MODEL_IDS` must contain at least one model.
- `DEFAULT_MODEL_ID` must belong to `INFERENCE_MODEL_IDS` when set.
- Requests with `model: "auto"` (or empty model) resolve to `DEFAULT_MODEL_ID` if set, otherwise the first configured model.
- Unknown request model IDs return `400` unless `ALLOW_UNKNOWN_MODELS=true`.
- Correlation IDs are propagated via `X-Request-ID` to the backend and returned in proxy responses/errors.
- Inference routes emit structured JSON logs with `event`, `endpoint`, `model`, `status_code`, and `latency_ms`.

## HTTP API (current)

All paths below are served by the app. When `INFERENCE_BASE_URL` is set, generation and embedding calls are proxied to that backend.

| Method | Path | Notes |
| ------ | ---- | ----- |
| `GET`  | `/health` | Liveness plus Redis and inference summary readiness statuses. |
| `GET`  | `/health/inference` | Dedicated inference backend readiness check (`skipped`, `ok`, `unreachable`, or `unhealthy`). |
| `GET`  | `/metrics/inference` | Baseline in-process request/error/latency metrics grouped by endpoint+model. |
| `GET`  | `/v1/models` | Lists model IDs from `INFERENCE_MODEL_IDS`. |
| `GET`  | `/v1/models/{model_id}` | Returns model metadata or 404. |
| `POST` | `/v1/chat/completions` | Proxies to backend when configured; supports `stream: true` (SSE). |
| `POST` | `/v1/completions` | Proxies to backend when configured; supports streaming. |
| `POST` | `/v1/embeddings` | Proxies to backend when configured. |

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
  api/routes/          # HTTP routers (health and /v1 inference routes)
  models/              # Pydantic request/response schemas
tests/                 # API smoke tests
```
