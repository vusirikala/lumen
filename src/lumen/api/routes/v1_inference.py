"""Inference-facing /v1 routes.

This module name reflects current behavior: OpenAI-compatible request/response
shape, backed by configurable self-hosted inference providers.
"""

from .v1_openai import router

