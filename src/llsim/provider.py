import cbrkit
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

type Provider[T] = cbrkit.typing.BatchConversionFunc[str, T]


def openai_provider[T: BaseModel](
    model: str,
    response_type: type[T],
    system_message: str | None = None,
    temperature: float = 1.0,
) -> Provider[T]:
    return cbrkit.synthesis.providers.openai(
        model=model,
        response_type=response_type,
        temperature=temperature,
        system_message=system_message,
        # https://github.com/openai/openai-python/blob/main/src/openai/_constants.py
        client=AsyncOpenAI(
            max_retries=3,
            http_client=httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(timeout=120, connect=5),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
            ),
        ),
    )
