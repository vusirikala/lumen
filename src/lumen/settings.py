from functools import lru_cache

from pydantic import AnyHttpUrl, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process configuration loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    redis_url: RedisDsn | None = None
    inference_base_url: AnyHttpUrl | None = None
    inference_api_key: str | None = None
    inference_model_ids: list[str] = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    default_model_id: str | None = None
    allow_unknown_models: bool = False

    @field_validator("inference_model_ids", mode="before")
    @classmethod
    def parse_inference_model_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def validate_model_governance(self) -> "Settings":
        if not self.inference_model_ids:
            raise ValueError("INFERENCE_MODEL_IDS must contain at least one model ID")
        if self.default_model_id is not None and self.default_model_id not in self.inference_model_ids:
            raise ValueError("DEFAULT_MODEL_ID must be one of INFERENCE_MODEL_IDS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
