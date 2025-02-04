from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated

import cbrkit
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class SimModelEntry(BaseModel):
    id: Annotated[str, Field(description="The ID of the document")]
    similarity: Annotated[
        float,
        Field(
            description="The similarity of the document to the query. Must be between 0.0 and 1.0"
        ),
    ]


class SimModel(BaseModel):
    similarities: list[SimModelEntry]


class RankModel(BaseModel):
    ranking: list[str]


def from_sim_model[K](response: SimModel) -> dict[str, float]:
    return {entry.id: entry.similarity for entry in response.similarities}


def from_rank_model[K](response: RankModel) -> dict[str, float]:
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
    "It should include all the documents in the list and must not include IDs that were not provided. "
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
    synthesis_func=Synthesizer("o3-mini-2025-01-31", SimModel),
    conversion_func=from_sim_model,
)

RANK_RETRIEVER = Retriever(
    synthesis_func=Synthesizer("o3-mini-2025-01-31", RankModel),
    conversion_func=from_rank_model,
)
