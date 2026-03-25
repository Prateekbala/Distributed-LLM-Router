"""Shared helpers for chat completion proxying."""

from __future__ import annotations


def extract_usage(payload: dict) -> tuple[int, int]:
    usage = payload.get("usage") or {}
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
