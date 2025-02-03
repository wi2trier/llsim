from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cbrkit
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel


class SimModel[K](BaseModel):
    similarities: dict[K, float]


class RankModel[K](BaseModel):
    ranking: list[K]


def from_sim_model[K](response: SimModel[K]) -> dict[K, float]:
    return response.similarities


def from_rank_model[K](response: RankModel[K]) -> dict[K, float]:
    return {
        key: 1.0 - i / len(response.ranking) for i, key in enumerate(response.ranking)
    }


@dataclass(slots=True, frozen=True)
class Retriever[R: BaseModel, V]:
    synthesis_func: cbrkit.typing.SynthesizerFunc[R, str, V, float]
    conversion_func: cbrkit.typing.ConversionFunc[R, Mapping[str, float]]

    def __call__(
        self,
        batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    ) -> Sequence[Mapping[str, float]]:
        func = cbrkit.synthesis.transpose(self.synthesis_func, self.conversion_func)

        return func([(casebase, query, None) for casebase, query in batches])


PROMPT = cbrkit.synthesis.prompts.default(
    "Given a list of documents and a query, generate a ranking of the documents with respect to the query. "
    "Only include documents IDs that were provided in the list. "
    "The IDs are given as markdown headings. "
)


def Synthesizer[T: BaseModel](
    model: str,
    response_type: type[T],
) -> cbrkit.typing.SynthesizerFunc[T, str, str, float]:
    provider = cbrkit.synthesis.providers.openai(
        model=model,
        response_type=response_type,
        temperature=1.0,
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
    return cbrkit.synthesis.build(provider, PROMPT)


SIM_RETRIEVER = Retriever(
    synthesis_func=Synthesizer("gpt-4o-mini-2024-07-18", SimModel),
    conversion_func=from_sim_model,
)

RANK_RETRIEVER = Retriever(
    synthesis_func=Synthesizer("gpt-4o-mini-2024-07-18", RankModel),
    conversion_func=from_rank_model,
)
