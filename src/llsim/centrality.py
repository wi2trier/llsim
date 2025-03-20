import json
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Callable

import cbrkit
import rustworkx

from llsim.preferences import Response, preferences2graph

logger = logging.getLogger(__name__)

type CentralityMeasure = Callable[[rustworkx.PyDiGraph[str, None]], Mapping[int, float]]


@dataclass(slots=True, frozen=True)
class CentralitySim(cbrkit.typing.StructuredValue[float]):
    value: float
    preferences: list[str]


centrality_measures: dict[str, CentralityMeasure] = {
    # centrality measures
    "betweenness_centrality": rustworkx.betweenness_centrality,
    "closeness_centrality": rustworkx.closeness_centrality,
    "eigenvector_centrality": rustworkx.eigenvector_centrality,
    "katz_centrality": rustworkx.katz_centrality,
    # in-degree centrality, not normalized between 0 and 1
    "in_degree_centrality": lambda g: {
        key: value / g.num_edges()
        for key, value in rustworkx.in_degree_centrality(g).items()
    },
    # link analysis
    "pagerank": partial(rustworkx.pagerank),
    # result is of type tuple[authority_scores, hub_scores], we want authorities
    "hits": lambda g: rustworkx.hits(g)[0],
}


@dataclass(slots=True, frozen=True)
class Retriever[V](cbrkit.typing.RetrieverFunc[str, V, CentralitySim]):
    measures: str
    file: str

    def __call__(
        self,
        batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    ) -> Sequence[Mapping[str, CentralitySim]]:
        similarities: list[Mapping[str, CentralitySim]] = []
        functions = {
            measure: centrality_measures[measure]
            for measure in self.measures.split(",")
        }

        with open(self.file) as fp:
            obj = json.load(fp)
            responses = [Response.model_validate(entry) for entry in obj["responses"]]

        for res, (casebase, _) in zip(responses, batches, strict=True):
            g, id_map = preferences2graph(res, casebase, remove_orphans=True)

            casebase_scores: dict[str, list[float]] = {key: [] for key in id_map.keys()}

            for function_name, function in functions.items():
                try:
                    scores = function(g)

                    for casebase_key, score_key in id_map.items():
                        casebase_scores[casebase_key].append(scores[score_key])

                except rustworkx.FailedToConverge:
                    logger.warning(f"Failed to converge for {function_name}, skipping")

                    for key in id_map.keys():
                        casebase_scores[key].append(0.0)

            similarities.append(
                {
                    key: CentralitySim(
                        value=statistics.mean(scores),
                        preferences=[
                            str(entry)
                            for entry in res.preferences
                            if entry.winner == key or entry.loser == key
                        ],
                    )
                    for key, scores in casebase_scores.items()
                }
            )

        return similarities
