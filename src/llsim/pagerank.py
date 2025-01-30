import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cbrkit
import httpx
import rustworkx
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_CONFIDENCE = 5


class SynthesisPreference(BaseModel):
    winner_id: str
    loser_id: str
    # confidence: Annotated[
    #     int, Field(description=f"Must be between 0 and {MAX_CONFIDENCE}")
    # ]


class SynthesisResponse(BaseModel):
    preferences: list[SynthesisPreference]


@dataclass(slots=True, frozen=True)
class PageRankSim(cbrkit.typing.StructuredValue[float]):
    value: float
    preferences: list[str]


@dataclass(slots=True, frozen=True)
class Retriever[V]:
    synthesis_func: cbrkit.typing.SynthesizerFunc[
        SynthesisResponse, str, V, PageRankSim
    ]

    def __call__(
        self,
        batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    ) -> Sequence[Mapping[str, PageRankSim]]:
        responses = self.synthesis_func([(*batch, None) for batch in batches])
        similarities: list[Mapping[str, PageRankSim]] = []

        for res, (casebase, _) in zip(responses, batches, strict=True):
            g = rustworkx.PyDiGraph()

            id_map = {key: g.add_node(key) for key in casebase.keys()}

            for entry in res.preferences:
                if entry.winner_id not in id_map:
                    logger.error(f"KeyError: {entry.winner_id}")
                if entry.loser_id not in id_map:
                    logger.error(f"KeyError: {entry.loser_id}")

                if (winner_id := id_map.get(entry.winner_id)) and (
                    loser_id := id_map.get(entry.loser_id)
                ):
                    g.add_edge(loser_id, winner_id, None)

            pagerank = rustworkx.pagerank(g, alpha=0.85)
            scores = {key: pagerank[id_map[key]] for key in casebase.keys()}

            similarities.append(
                {
                    key: PageRankSim(
                        value=value,
                        preferences=[
                            f"{entry.winner_id} > {entry.loser_id}"
                            for entry in res.preferences
                            if entry.winner_id == key or entry.loser_id == key
                        ],
                    )
                    for key, value in scores.items()
                }
            )

        return similarities


def openai_provider(model: str):
    return cbrkit.synthesis.providers.openai(
        model=model,
        response_type=SynthesisResponse,
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


PROMPT = cbrkit.synthesis.prompts.default(
    "Given a list of documents, provide preferences between them. "
    "Only include documents IDs that were provided in the list. "
    "The IDs are given as markdown headings. "
    "Do not include attributes or nodes from invididual documents in the preferences. "
)
POOLING_PROMPT = cbrkit.synthesis.prompts.pooling(
    "The following pairwise preferences were predicted. "
    "As not all documents were available, the preferences are split into multiple chunks. "
    "Please combine the partial results and create missing transitive preferences to get the final ranking. "
    "This will later be used to calculate the PageRank scores, so a complete matrix is required. "
)
POOLING_FUNC = cbrkit.synthesis.pooling(
    openai_provider("gpt-4o-2024-11-20"), POOLING_PROMPT
)

retriever_overlap = Retriever(
    cbrkit.synthesis.chunks(
        cbrkit.synthesis.build(openai_provider("gpt-4o-mini-2024-07-18"), PROMPT),
        POOLING_FUNC,
        size=1,
        overlap=1,
    )
)

retriever_complete = Retriever(
    cbrkit.synthesis.chunks(
        cbrkit.synthesis.build(openai_provider("gpt-4o-mini-2024-07-18"), PROMPT),
        POOLING_FUNC,
        size=20,
    )
)
