from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cbrkit
from cbrkit.typing import BatchSimFunc
from frozendict import deepfreeze

type NodeData = Mapping[str, Any]


def load(
    path: Path,
) -> cbrkit.typing.Casebase[str, cbrkit.sim.graphs.Graph[str, NodeData, None, None]]:
    return cbrkit.sim.graphs.load(cbrkit.loaders.file(path), node_converter=deepfreeze)


NODE_SIM: BatchSimFunc[
    cbrkit.sim.graphs.Node[str, NodeData],
    cbrkit.typing.Float,
] = cbrkit.sim.cache(
    cbrkit.sim.transpose_value(
        cbrkit.sim.attribute_table(
            {
                "workflow": cbrkit.sim.attribute_value(
                    {
                        "name": cbrkit.sim.cache(cbrkit.sim.strings.levenshtein()),
                        "preparation time (min)": cbrkit.sim.numbers.linear(1000),
                        "calories": cbrkit.sim.numbers.linear(2500),
                    }
                ),
                "task": cbrkit.sim.attribute_value(
                    {
                        "name": cbrkit.sim.cache(
                            cbrkit.sim.taxonomy.build(
                                Path("./data/taxonomies/activity.json"),
                                cbrkit.sim.taxonomy.weights("user", "optimistic"),
                            )
                        )
                    }
                ),
                "data": cbrkit.sim.attribute_value(
                    {
                        "name": cbrkit.sim.cache(
                            cbrkit.sim.taxonomy.build(
                                Path("./data/taxonomies/ingredient.json"),
                                cbrkit.sim.taxonomy.weights("user", "optimistic"),
                            )
                        ),
                        "amount": cbrkit.sim.cache(
                            cbrkit.sim.attribute_value(
                                {
                                    "value": cbrkit.sim.numbers.linear(1000),
                                    "unit": cbrkit.sim.generic.equality(),
                                }
                            )
                        ),
                    },
                    aggregator=cbrkit.sim.aggregator(pooling_weights={"name": 2}),
                    # some data nodes do not have an amount
                    default=0.0,
                ),
            },
            attribute="type",
            default=cbrkit.sim.generic.static(0.0),
        )
    )
)
GRAPH_SIM: cbrkit.typing.AnySimFunc[
    cbrkit.sim.graphs.Graph[str, NodeData, None, None],
    cbrkit.sim.graphs.GraphSim[str],
] = cbrkit.sim.graphs.astar.build(
    past_cost_func=cbrkit.sim.graphs.astar.g1(NODE_SIM),
    future_cost_func=cbrkit.sim.graphs.astar.h3(NODE_SIM),
    selection_func=cbrkit.sim.graphs.astar.select3(
        cbrkit.sim.graphs.astar.h3(NODE_SIM)
    ),
    init_func=cbrkit.sim.graphs.astar.init2[str, NodeData, None, None](),
    queue_limit=1,
)

Retriever: cbrkit.typing.RetrieverFunc[
    str,
    cbrkit.sim.graphs.Graph[str, NodeData, None, None],
    cbrkit.sim.graphs.GraphSim[str],
] = cbrkit.retrieval.build(GRAPH_SIM, multiprocessing=True)
