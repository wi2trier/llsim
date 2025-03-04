import os
from dataclasses import dataclass
from functools import partial

import cbrkit
import httpx
import ollama
from openai import AsyncOpenAI
from pydantic import BaseModel

# https://github.com/openai/openai-python/blob/main/src/openai/_constants.py
openai_client = AsyncOpenAI(
    max_retries=3,
    http_client=httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(timeout=120, connect=5),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
    ),
)
openrouter_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_retries=3,
    http_client=httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(timeout=120, connect=5),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
    ),
)


@dataclass(slots=True, frozen=True)
class Provider[T: BaseModel]:
    name: str

    def build(
        self,
        response_type: type[T],
        system_message: str | None = None,
        default_response: T | None = None,
    ) -> cbrkit.typing.BatchConversionFunc[str, T]:
        openai_provider = partial(
            cbrkit.synthesis.providers.openai,
            client=openai_client,
            response_type=response_type,
            system_message=system_message,
            default_response=default_response,
        )
        openrouter_provider = partial(
            cbrkit.synthesis.providers.openai,
            client=openrouter_client,
            response_type=response_type,
            system_message=system_message,
            default_response=default_response,
        )

        match self.name:
            case "o1":
                return openai_provider(
                    model="o1-2024-12-17",
                    reasoning_effort="high",
                )
            case "o3-mini-high":
                return openai_provider(
                    model="o3-mini-2025-01-31",
                    reasoning_effort="high",
                )
            case "4o":
                return openai_provider(model="gpt-4o-2024-11-20")
            case "4o-mini":
                return openai_provider(model="gpt-4o-mini-2024-07-18")

            case "gemini-flash":
                return openrouter_provider(model="google/gemini-2.0-flash-001")
            case "gemini-flash-lite":
                return openrouter_provider(model="google/gemini-2.0-flash-lite-001")

            case "llama-405b":
                return openrouter_provider(
                    model="meta-llama/llama-3.1-405b-instruct",
                    tool_choice=response_type,
                    system_message=(system_message or "") + "\nRespond with JSON.",
                )
            case "llama-70b":
                return openrouter_provider(
                    model="meta-llama/llama-3.3-70b-instruct",
                    tool_choice=response_type,
                    system_message=(system_message or "") + "\nRespond with JSON.",
                )
            case "llama-8b":
                return openrouter_provider(model="meta-llama/llama-3.1-8b-instruct")
            case "llama-3b":
                return openrouter_provider(model="meta-llama/llama-3.2-3b-instruct")

            case "qwen-turbo":
                return openrouter_provider(model="qwen/qwen-turbo")
            case "qwen-plus":
                return openrouter_provider(model="qwen/qwen-plus")
            case "qwen-max":
                return openrouter_provider(model="qwen/qwen-max")

            case "deepseek-r1":
                return openrouter_provider(model="deepseek/deepseek-r1")
            case "deepseek-v3":
                return openrouter_provider(model="deepseek/deepseek-chat")

            case "claude-thinking":
                return openrouter_provider(model="anthropic/claude-3.7-sonnet:thinking")
            case "claude":
                return openrouter_provider(model="anthropic/claude-3.7-sonnet")

            case "ollama-llama-8b":
                return cbrkit.synthesis.providers.ollama(
                    model="llama3.1:8b",
                    response_type=response_type,
                    system_message=system_message,
                    options=ollama.Options(num_ctx=32000),
                )
            case "ollama-llama-3b":
                return cbrkit.synthesis.providers.ollama(
                    model="llama3.2:3b",
                    response_type=response_type,
                    system_message=system_message,
                    options=ollama.Options(num_ctx=32000),
                )

        raise ValueError(f"Unknown provider: {self.name}")
