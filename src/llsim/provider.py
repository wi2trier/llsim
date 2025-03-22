import os
from dataclasses import dataclass
from functools import partial

import cbrkit
import httpx
import instructor
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
instructor_client = instructor.from_openai(openrouter_client, mode=instructor.Mode.JSON)


@dataclass(slots=True, frozen=True)
class Provider[T: BaseModel]:
    name: str

    def build(
        self,
        response_type: type[T],
        system_message: str | None = None,
        default_response: T | None = None,
        retries: int = 0,
    ) -> cbrkit.typing.BatchConversionFunc[str, cbrkit.synthesis.providers.Response[T]]:
        openai_provider = partial(
            cbrkit.synthesis.providers.openai,
            client=openai_client,
            response_type=response_type,
            system_message=system_message,
            default_response=default_response,
            retries=retries,
        )
        openrouter_provider = partial(
            cbrkit.synthesis.providers.openai,
            client=openrouter_client,
            response_type=response_type,
            system_message=system_message,
            default_response=default_response,
            retries=retries,
            extra_body={
                "provider": {
                    "require_parameters": True,
                },
            },
        )
        instructor_provider = partial(
            cbrkit.synthesis.providers.instructor,
            client=instructor_client,
            response_type=response_type,
            system_message=system_message,
            default_response=default_response,
            retries=retries,
            extra_kwargs={
                "extra_body": {
                    "provider": {
                        "require_parameters": True,
                    },
                },
            },
        )
        instructor_provider_filtered = partial(
            instructor_provider,
            extra_kwargs={
                "extra_body": {
                    "provider": {
                        "require_parameters": True,
                        # those provide large context sizes for input/output
                        "order": [
                            "InferenceNet",
                            "Lambda",
                            "Nebius",
                            "Parasail",
                            "Friendli",
                            "Kluster",
                            "Fireworks",
                        ],
                    },
                },
            },
        )

        match self.name:
            case "o1":
                return openai_provider(
                    model="o1-2024-12-17",
                    reasoning_effort="high",
                )
            case "o3-mini":
                return openai_provider(
                    model="o3-mini-2025-01-31",
                    reasoning_effort="high",
                )
            case "gpt-4-5":
                return openai_provider(model="gpt-4.5-preview-2025-02-27")
            case "gpt-4o":
                return openai_provider(model="gpt-4o-2024-11-20")
            case "gpt-4o-mini":
                return openai_provider(model="gpt-4o-mini-2024-07-18")

            case "gemini-flash":
                return openrouter_provider(model="google/gemini-2.0-flash-001")
            case "gemini-flash-lite":
                return openrouter_provider(
                    model="google/gemini-2.0-flash-lite-001",
                    max_completion_tokens=3,
                )

            case "llama-405b":
                return instructor_provider_filtered(
                    model="meta-llama/llama-3.1-405b-instruct"
                )
            case "llama-70b":
                return instructor_provider(model="meta-llama/llama-3.3-70b-instruct")
            # case "llama-70b":
            #     return openrouter_provider(
            #         model="meta-llama/llama-3.3-70b-instruct",
            #         tool_choice=response_type,
            #         system_message=(system_message or "") + "\nRespond with JSON.",
            #     )
            case "llama-8b":
                return openrouter_provider(
                    model="meta-llama/llama-3.1-8b-instruct",
                    max_completion_tokens=3,
                )
            case "llama-3b":
                return openrouter_provider(
                    model="meta-llama/llama-3.2-3b-instruct",
                    max_completion_tokens=3,
                )

            case "qwen-turbo":
                return openrouter_provider(
                    model="qwen/qwen-turbo",
                    max_completion_tokens=3,
                )
            case "qwen-plus":
                return instructor_provider(model="qwen/qwen-plus")
            case "qwen-max":
                return instructor_provider(model="qwen/qwen-max")
            case "qwen-72b":
                return instructor_provider(model="qwen/qwen-2.5-72b-instruct")
            case "qwen-7b":
                return openrouter_provider(
                    model="qwen/qwen-2.5-7b-instruct",
                    max_completion_tokens=3,
                )

            case "deepseek-r1":
                return instructor_provider_filtered(model="deepseek/deepseek-r1")
            case "deepseek-v3":
                return instructor_provider_filtered(model="deepseek/deepseek-chat")

            case "command-r-7b":
                return openrouter_provider(
                    model="cohere/command-r7b-12-2024",
                    max_completion_tokens=3,
                )
            case "command-r":
                return instructor_provider(model="cohere/command-r-08-2024")
            case "command-r-plus":
                return instructor_provider(model="cohere/command-r-plus-08-2024")

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
