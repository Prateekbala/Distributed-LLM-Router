"""Gateway settings (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RoutingStrategy = Literal["round_robin", "least_loaded", "latency_based"]


class Settings(BaseSettings):
    """Environment-driven gateway configuration."""

    model_config = SettingsConfigDict(
        env_file="gateway/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vllm_nodes: list[str] = Field(
        default_factory=lambda: ["http://localhost:8001"],
        validation_alias="VLLM_NODES",
    )
    routing_strategy: RoutingStrategy = Field(default="least_loaded", validation_alias="ROUTING_STRATEGY")
    node_timeout_seconds: float = Field(default=120.0, validation_alias="NODE_TIMEOUT_SECONDS")
    max_retries: int = Field(default=2, validation_alias="MAX_RETRIES")
    health_check_interval_seconds: float = Field(default=15.0, validation_alias="HEALTH_CHECK_INTERVAL_SECONDS")
    health_failure_threshold: int = Field(default=3, validation_alias="HEALTH_FAILURE_THRESHOLD")

    gateway_port: int = Field(default=8000, validation_alias="GATEWAY_PORT")
    auth_token: str = Field(default="your-secret-token-here", validation_alias="AUTH_TOKEN")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    model_name: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        validation_alias="MODEL_NAME",
    )
    connect_timeout_seconds: float = Field(default=10.0, validation_alias="CONNECT_TIMEOUT_SECONDS")
    max_concurrent_requests: int = Field(default=16, validation_alias="MAX_CONCURRENT_REQUESTS")
    max_queue_size: int = Field(default=32, validation_alias="MAX_QUEUE_SIZE")

    @field_validator("vllm_nodes", mode="before")
    @classmethod
    def _parse_nodes(cls, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(u).strip().rstrip("/") for u in value if str(u).strip()]
        if value is None or value == "":
            return ["http://localhost:8001"]
        return [p.strip().rstrip("/") for p in str(value).split(",") if p.strip()]

    @model_validator(mode="after")
    def _ensure_nodes(self) -> Self:
        if not self.vllm_nodes:
            self.vllm_nodes = ["http://localhost:8001"]
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
