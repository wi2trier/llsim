import itertools
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cbrkit
import httpx
import rustworkx
from openai import AsyncOpenAI
from pydantic import BaseModel
from rustworkx.rustworkx import PyDiGraph

random.seed(42)

logger = logging.getLogger(__name__)

MAX_CONFIDENCE = 5


class SynthesisPreference(BaseModel):
    winner_id: str
    loser_id: str


class SynthesisResponse(BaseModel):
    preferences: list[SynthesisPreference]


@dataclass(slots=True, frozen=True)
class PageRankSim(cbrkit.typing.StructuredValue[float]):
    value: float
    preferences: list[str]


def preferences2graph[V](
    res: SynthesisResponse, casebase: cbrkit.typing.Casebase[str, V]
) -> tuple[PyDiGraph[str, None], dict[str, int]]:
    g: PyDiGraph[str, None] = rustworkx.PyDiGraph()

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

    return g, id_map


def graph2preferences(
    g: PyDiGraph[str, None], id_map: dict[str, int]
) -> SynthesisResponse:
    preferences = []
    id_map_inv = {v: k for k, v in id_map.items()}

    for node in g.node_indices():
        for successor in g.successor_indices(node):
            preferences.append(
                SynthesisPreference(
                    winner_id=id_map_inv[successor],
                    loser_id=id_map_inv[node],
                )
            )

    return SynthesisResponse(preferences=preferences)


def request[V](
    provider: cbrkit.typing.BatchConversionFunc[str, SynthesisResponse],
    batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    prev_responses: Sequence[SynthesisResponse],
    max_cases: int,
) -> Sequence[SynthesisResponse]:
    requests: list[str | None] = []

    for res, (casebase, query) in zip(prev_responses, batches, strict=True):
        if len(casebase) > max_cases:
            sampled_cases = random.sample(list(casebase.items()), max_cases)
            casebase = dict(sampled_cases)

        all_combinations = itertools.combinations(casebase.keys(), 2)
        predicted_combinations = [
            (entry.winner_id, entry.loser_id) for entry in res.preferences
        ]
        missing_combinations = set(all_combinations) - set(predicted_combinations)
        missing_keys = {key[0] for key in missing_combinations} | {
            key[1] for key in missing_combinations
        }

        if missing_combinations:
            requests.append(
                "\n\n".join(
                    [
                        prompt_combinations(list(missing_combinations)),
                        cbrkit.synthesis.prompts.default()(
                            {
                                key: value
                                for key, value in casebase.items()
                                if key in missing_keys
                            },
                            query,
                            None,
                        ),
                    ]
                )
            )
        else:
            requests.append(None)

    next_responses_raw = provider([batch for batch in requests if batch is not None])
    next_responses = [
        next_responses_raw[i]
        if batch is not None
        else SynthesisResponse(preferences=[])
        for i, batch in enumerate(requests)
    ]

    return [
        SynthesisResponse(
            preferences=response.preferences + retried_response.preferences
        )
        for response, retried_response in zip(
            prev_responses, next_responses, strict=True
        )
    ]


def infer_missing[V](
    batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    prev_responses: Sequence[SynthesisResponse],
) -> Sequence[SynthesisResponse]:
    new_responses: list[SynthesisResponse] = []

    for res, (casebase, _) in zip(prev_responses, batches, strict=True):
        new_preferences: list[SynthesisPreference] = []
        all_combinations = itertools.combinations(casebase.keys(), 2)
        predicted_combinations = [
            (entry.winner_id, entry.loser_id) for entry in res.preferences
        ]
        missing_combinations = set(all_combinations) - set(predicted_combinations)

        if missing_combinations:
            g, id_map = preferences2graph(res, casebase)
            for source, target in missing_combinations:
                if (source_id := id_map.get(source)) and (
                    target_id := id_map.get(target)
                ):
                    if rustworkx.has_path(g, source_id, target_id):
                        new_preferences.append(
                            SynthesisPreference(loser_id=source, winner_id=target)
                        )
                    elif rustworkx.has_path(g, target_id, source_id):
                        new_preferences.append(
                            SynthesisPreference(loser_id=target, winner_id=source)
                        )

        new_responses.append(
            SynthesisResponse(preferences=res.preferences + new_preferences)
        )

    return new_responses


@dataclass(slots=True, frozen=True)
class Retriever[V]:
    provider: cbrkit.typing.BatchConversionFunc[str, SynthesisResponse]
    tries: int = 1
    infer_missing: bool = True
    max_cases: int = 100

    def __call__(
        self,
        batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    ) -> Sequence[Mapping[str, PageRankSim]]:
        similarities: list[Mapping[str, PageRankSim]] = []
        responses = [SynthesisResponse(preferences=[]) for _ in batches]

        for _ in range(self.tries):
            responses = request(self.provider, batches, responses, self.max_cases)

            if self.infer_missing:
                responses = infer_missing(batches, responses)

        for res, (casebase, _) in zip(responses, batches, strict=True):
            g, id_map = preferences2graph(res, casebase)
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


# openai_o3_mini = openai_provider("o3-mini-2025-01-31")
openai_4o_mini = openai_provider("gpt-4o-mini-2024-07-18")


def prompt_combinations(combinations: list[tuple[str, str]]) -> str:
    dumper = cbrkit.dumpers.markdown()

    return (
        "Given a list of documents and a query, generate preferences between the documents with respect to the query. "
        "The IDs are given as markdown headings. "
        "We want a full pairwise preference matrix, so please provide the following preferences: "
        f"\n{dumper(combinations)}"
    )


def prompt_instruction_builder(
    casebase: cbrkit.typing.Casebase[str, Any], *args, **kwargs
) -> str:
    return prompt_combinations(list(itertools.combinations(casebase.keys(), 2)))


def chunks_pooler(responses: Sequence[SynthesisResponse]) -> SynthesisResponse:
    return SynthesisResponse(
        preferences=list(
            itertools.chain.from_iterable(
                response.preferences for response in responses
            )
        )
    )


RETRIEVER = Retriever(
    provider=openai_4o_mini,
    tries=3,
    infer_missing=True,
)
