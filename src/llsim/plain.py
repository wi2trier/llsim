import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated

import cbrkit
from pydantic import BaseModel, Field, OnErrorOmit

from llsim.provider import Provider

logger = logging.getLogger(__name__)


class Entry(BaseModel):
    id: Annotated[str, Field(description="The ID of the document")]
    sim: Annotated[
        float,
        Field(
            description="The similarity of the document to the query. Must be between 0.0 and 1.0"
        ),
    ]


class SimModel(BaseModel):
    similarities: Annotated[
        list[OnErrorOmit[Entry]],
        Field(description="List of similarity scores for each document"),
    ]


class RankModel(BaseModel):
    ranking: Annotated[
        list[OnErrorOmit[str]],
        Field(
            description="Ranking of all documents by similarity, the first element is the most similar"
        ),
    ]


def from_sim_model[K](response: SimModel) -> dict[str, float]:
    return {entry.id: entry.sim for entry in response.similarities}


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
        numeric_batches: list[tuple[cbrkit.typing.Casebase[str, V], V, None]] = []
        # numeric -> string
        batches_lookup: list[dict[str, str]] = []

        for casebase, query in batches:
            numeric_casebase: cbrkit.typing.Casebase[str, V] = {}
            lookup: dict[str, str] = {}

            for i, (key, value) in enumerate(casebase.items(), start=1000):
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
                    logger.info(f"Key {numeric_key} not in casebase")

            parsed_sims.append(parsed_sim)

        return parsed_sims


@dataclass(slots=True, frozen=True)
class Synthesizer[R: BaseModel, V](cbrkit.typing.SynthesizerFunc[R, str, V, float]):
    provider: Provider[R]
    response_type: type[R]
    retry: bool

    def __call__(
        self,
        batches: Sequence[tuple[Mapping[str, V], V | None, Mapping[str, float] | None]],
    ) -> Sequence[R]:
        generation_func = self.provider.build(
            response_type=self.response_type,
            system_message=(
                "You are a helpful assistant with the following task: "
                "Given a list of documents and a query, generate a ranking of the documents with respect to the query. "
                "It should include **all** the documents in the list and must not include IDs that were not provided. "
                "The IDs are given as markdown headings. "
                "Do not consider IDs found inside markdown code blocks such as node or edge names. "
            ),
        )
        prompt_func = cbrkit.synthesis.prompts.default()
        generation_input = [prompt_func(*batch) for batch in batches]

        results = cbrkit.helpers.unpack_values(generation_func(generation_input))

        if not self.retry:
            return results

        retry_instructions: list[str | None] = []

        for batch, result in zip(batches, results, strict=True):
            if isinstance(result, SimModel):
                missing_cases = [
                    key for key in batch[0].keys() if key not in result.similarities
                ]
                if missing_cases:
                    retry_instructions.append(
                        f"In a previous run, you predicted the following ranking: {result.model_dump_json()} "
                        f"The following documents are missing: {', '.join(missing_cases)} "
                    )
                else:
                    retry_instructions.append(None)

            elif isinstance(result, RankModel):
                missing_cases = [
                    key for key in batch[0].keys() if key not in result.ranking
                ]
                if missing_cases:
                    retry_instructions.append(
                        f"In a previous run, you predicted the following ranking: {result.model_dump_json()} "
                        f"The following documents are missing: {', '.join(missing_cases)} "
                        "Complete the result by adding all missing items and generating a new ranking. "
                    )
                else:
                    retry_instructions.append(None)

            else:
                raise ValueError(
                    f"Unsupported response type: {type(result)}. Expected {self.response_type}"
                )

        retry_input = [
            cbrkit.synthesis.prompts.default(instruction)(*batch)
            for batch, instruction in zip(batches, retry_instructions)
            if instruction is not None
        ]
        retried_results = cbrkit.helpers.unpack_values(generation_func(retry_input))
        retried_results_iter = iter(retried_results)

        final_results: list[R] = []

        for idx, instruction in enumerate(retry_instructions):
            if instruction is not None:
                final_results.append(next(retried_results_iter))
            else:
                final_results.append(results[idx])

        return final_results


def SimRetriever(model: str, retry: str = "false"):
    return Retriever(
        synthesis_func=Synthesizer(
            Provider(model), SimModel, retry=True if retry == "true" else False
        ),
        conversion_func=from_sim_model,
    )


def RankRetriever(model: str, retry: str = "false"):
    return Retriever(
        synthesis_func=Synthesizer(
            Provider(model), RankModel, retry=True if retry == "true" else False
        ),
        conversion_func=from_rank_model,
    )
