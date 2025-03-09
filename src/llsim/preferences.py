import itertools
import logging
import random
from collections.abc import Collection, Sequence
from typing import override

import cbrkit
import rustworkx
from pydantic import BaseModel, OnErrorOmit

from llsim.provider import Provider

random.seed(42)

logger = logging.getLogger(__name__)


class Pref(BaseModel):
    winner: str
    loser: str

    @override
    def __str__(self) -> str:
        return f"{self.winner} > {self.loser}"


class Response(BaseModel):
    preferences: list[OnErrorOmit[Pref]]


def combinations2instructions(combinations: list[tuple[str, str]]) -> str:
    dumper = cbrkit.dumpers.markdown()

    return (
        "We want a full pairwise preference matrix, so please provide the following preferences: "
        f"\n{dumper(combinations)}"
    )


def preferences2graph[V](
    res: Response, casebase: cbrkit.typing.Casebase[str, V]
) -> tuple[rustworkx.PyDiGraph[str, None], dict[str, int]]:
    g: rustworkx.PyDiGraph[str, None] = rustworkx.PyDiGraph()

    id_map = {key: g.add_node(key) for key in casebase.keys()}

    for entry in res.preferences:
        if entry.winner not in id_map:
            logger.error(f"KeyError: {entry.winner}")
        if entry.loser not in id_map:
            logger.error(f"KeyError: {entry.loser}")

        if (winner := id_map.get(entry.winner)) and (loser := id_map.get(entry.loser)):
            g.add_edge(loser, winner, None)

    return g, id_map


def graph2preferences(
    g: rustworkx.PyDiGraph[str, None], id_map: dict[str, int]
) -> Response:
    preferences = []
    id_map_inv = {v: k for k, v in id_map.items()}

    for node in g.node_indices():
        for successor in g.successor_indices(node):
            preferences.append(
                Pref(
                    winner=id_map_inv[successor],
                    loser=id_map_inv[node],
                )
            )

    return Response(preferences=preferences)


def get_missing_combinations(
    all: Collection[str], prev: Collection[Pref]
) -> set[tuple[str, str]]:
    all_combinations = itertools.combinations(all, 2)
    predicted_combinations = [(entry.winner, entry.loser) for entry in prev] + [
        (entry.loser, entry.winner) for entry in prev
    ]
    return set(all_combinations) - set(predicted_combinations)


def request_pairwise[V](
    provider: Provider,
    batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    prev_responses: Sequence[Response],
    max_cases: float,
) -> Sequence[Response]:
    generation_func = provider.build(
        str,
        system_message=(
            "Given two documents and a query, which one is more relevant to the query? "
            "Give only the ID of the winner document. "
            "Answer either 'first' or 'second', do not generate anything else. "
        ),
        default_response="",
    )
    synthesis_func = cbrkit.synthesis.build(
        generation_func, cbrkit.synthesis.prompts.default()
    )
    responses: list[Response] = []

    for prev_res, (casebase, query) in zip(prev_responses, batches, strict=True):
        missing_combinations = get_missing_combinations(
            casebase.keys(), prev_res.preferences
        )
        missing_keys = set(itertools.chain.from_iterable(missing_combinations))

        if max_cases < 1:
            max_cases = int(len(casebase) * max_cases)

        if len(missing_keys) > max_cases:
            missing_keys = set(random.sample(list(casebase.keys()), int(max_cases)))

        pairwise_batches = {
            (case1, case2): (
                {"first": casebase[case1], "second": casebase[case2]},
                query,
                None,
            )
            for case1, case2 in missing_combinations
            if case1 in missing_keys and case2 in missing_keys
        }
        pairwise_responses = cbrkit.synthesis.apply_batches(
            pairwise_batches,
            synthesis_func,
        )
        pairwise_preferences: list[Pref] = []

        for (case1, case2), result in pairwise_responses.queries.items():
            parsed_result = (
                result.response.value.strip()
                .lower()
                .replace("'", "")
                .replace('"', "")
                .replace(".", "")
            )

            if parsed_result == "first":
                pairwise_preferences.append(Pref(winner=case1, loser=case2))
            elif parsed_result == "second":
                pairwise_preferences.append(Pref(winner=case2, loser=case1))
            else:
                logger.warning(f"Got '{parsed_result}' for ({case1},{case2})")

        responses.append(
            Response(preferences=prev_res.preferences + pairwise_preferences)
        )

    return responses


def request[V](
    provider: Provider,
    batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    prev_responses: Sequence[Response],
    max_cases: float,
) -> Sequence[Response]:
    generation_func = provider.build(
        Response,
        system_message=(
            "Given a list of documents and a query, generate preferences between the documents with respect to the query. "
            "The IDs are given as markdown headings. "
            "Do not consider IDs found inside markdown code blocks such as node or edge names. "
        ),
        default_response=Response(preferences=[]),
    )
    requests: list[str | None] = []
    numeric_batches: list[tuple[cbrkit.typing.Casebase[str, V], V]] = []
    # numeric -> string
    batches_lookup: list[dict[str, str]] = []

    for casebase, query in batches:
        numeric_casebase: cbrkit.typing.Casebase[str, V] = {}
        lookup: dict[str, str] = {}
        for i, (key, value) in enumerate(casebase.items(), start=1000):
            numeric_casebase[str(i)] = value
            lookup[str(i)] = key
        numeric_batches.append((numeric_casebase, query))
        batches_lookup.append(lookup)

    for prev_res, (numeric_casebase, query) in zip(
        prev_responses, numeric_batches, strict=True
    ):
        missing_combinations = get_missing_combinations(
            numeric_casebase.keys(), prev_res.preferences
        )
        missing_keys = set(itertools.chain.from_iterable(missing_combinations))

        if max_cases < 1:
            max_cases = int(len(numeric_casebase) * max_cases)

        if len(missing_keys) > max_cases:
            missing_keys = set(
                random.sample(list(numeric_casebase.keys()), int(max_cases))
            )

        if missing_combinations:
            prompt_func = cbrkit.synthesis.prompts.default(
                instructions=combinations2instructions(list(missing_combinations))
            )
            requests.append(
                prompt_func(
                    {key: numeric_casebase[key] for key in missing_keys},
                    query,
                    None,
                )
            )
        else:
            requests.append(None)

    next_responses_numeric_raw = generation_func(
        [req for req in requests if req is not None]
    )
    next_responses_numeric = [
        next_responses_numeric_raw[i]
        if req is not None
        else cbrkit.synthesis.providers.Response(Response(preferences=[]))
        for i, req in enumerate(requests)
    ]
    next_responses: list[Response] = []

    for next_res_numeric, lookup in zip(
        next_responses_numeric, batches_lookup, strict=True
    ):
        new_preferences: list[Pref] = []

        for entry_numeric in next_res_numeric.value.preferences:
            if entry_numeric.winner not in lookup:
                logger.error(f"KeyError: {entry_numeric.winner}")
            if entry_numeric.loser not in lookup:
                logger.error(f"KeyError: {entry_numeric.loser}")

            if (winner := lookup.get(entry_numeric.winner)) and (
                loser := lookup.get(entry_numeric.loser)
            ):
                new_preferences.append(Pref(winner=winner, loser=loser))

        next_responses.append(Response(preferences=new_preferences))

    return [
        Response(preferences=prev_response.preferences + next_response.preferences)
        for prev_response, next_response in zip(
            prev_responses, next_responses, strict=True
        )
    ]


def infer_missing[V](
    batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    prev_responses: Sequence[Response],
) -> Sequence[Response]:
    new_responses: list[Response] = []

    for prev_res, (casebase, _) in zip(prev_responses, batches, strict=True):
        new_preferences: list[Pref] = []
        missing_combinations = get_missing_combinations(
            casebase.keys(), prev_res.preferences
        )

        if missing_combinations:
            g, id_map = preferences2graph(prev_res, casebase)
            for source, target in missing_combinations:
                if (source_id := id_map.get(source)) and (
                    target_id := id_map.get(target)
                ):
                    if rustworkx.has_path(g, source_id, target_id):
                        new_preferences.append(Pref(loser=source, winner=target))
                    elif rustworkx.has_path(g, target_id, source_id):
                        new_preferences.append(Pref(loser=target, winner=source))

        new_responses.append(
            Response(preferences=prev_res.preferences + new_preferences)
        )

    return new_responses
