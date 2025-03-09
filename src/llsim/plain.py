import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated

import cbrkit
from pydantic import BaseModel, Field, OnErrorOmit

from llsim.provider import Provider

logger = logging.getLogger(__name__)


class SimModelEntry(BaseModel):
    id: Annotated[str, Field(description="The ID of the document")]
    similarity: Annotated[
        float,
        Field(
            description="The similarity of the document to the query. Must be between 0.0 and 1.0"
        ),
    ]


class SimModel(BaseModel):
    similarities: list[OnErrorOmit[SimModelEntry]]


class RankModel(BaseModel):
    ranking: list[OnErrorOmit[str]]


def from_sim_model[K](response: SimModel) -> dict[str, float]:
    return {entry.id: entry.sim for entry in response.value.similarities}


def from_rank_model[K](response: RankModel) -> dict[str, float]:
    return {
        key: 1.0 - i / len(response.value.ranking)
        for i, key in enumerate(response.value.ranking)
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
        numeric_batches: list[tuple[cbrkit.typing.Casebase[str, V], V, None]] = []
        # numeric -> string
        batches_lookup: list[dict[str, str]] = []

        for casebase, query in batches:
            numeric_casebase: cbrkit.typing.Casebase[str, V] = {}
            lookup: dict[str, str] = {}

            for i, (key, value) in enumerate(casebase.items(), start=1):
                numeric_casebase[str(i)] = value
                lookup[str(i)] = key

            numeric_batches.append((numeric_casebase, query, None))
            batches_lookup.append(lookup)

        raw_sims = func(numeric_batches)
        parsed_sims: list[dict[str, float]] = []

        for raw_sim, (casebase, _), lookup in zip(
            raw_sims, batches, batches_lookup, strict=True
        ):
            parsed_sim: dict[str, float] = {}
            # string -> numeric
            inverse_lookup = {v: k for k, v in lookup.items()}

            for key in casebase.keys():
                numeric_key = inverse_lookup[key]

                if sim := raw_sim.get(numeric_key):
                    parsed_sim[key] = sim
                else:
                    logger.info(f"Key {key} not in response")

            for numeric_key in raw_sim.keys():
                key = lookup.get(numeric_key)

                if key is None or key not in casebase:
                    logger.info(f"Key {key} not in casebase")

            parsed_sims.append(parsed_sim)

        return parsed_sims


def Synthesizer[T: BaseModel](
    provider: Provider,
    response_type: type[T],
) -> cbrkit.typing.SynthesizerFunc[T, str, str, float]:
    generation_func = provider.build(
        response_type=response_type,
        system_message=(
            "You are a helpful assistant with the following task: "
            "Given a list of documents and a query, generate a ranking of the documents with respect to the query. "
            "It should include **all** the documents in the list and must not include IDs that were not provided. "
            "The IDs are given as markdown headings. "
        ),
    )

    return cbrkit.synthesis.build(generation_func, cbrkit.synthesis.prompts.default())


def SIM_RETRIEVER(model: str):
    return Retriever(
        synthesis_func=Synthesizer(Provider(model), SimModel),
        conversion_func=from_sim_model,
    )


def RANK_RETRIEVER(model: str):
    return Retriever(
        synthesis_func=Synthesizer(Provider(model), RankModel),
        conversion_func=from_rank_model,
    )
