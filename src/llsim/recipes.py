import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cbrkit
from cbrkit.model.graph import Graph
from cbrkit.sim.graphs import GraphSim
from cbrkit.typing import BatchSimFunc
from frozendict import deepfreeze
from pydantic import ValidationError

type NodeData = Mapping[str, Any]
logger = logging.getLogger(__name__)


def load(
    path: Path,
) -> cbrkit.typing.Casebase[str, Graph[str, NodeData, None, None]]:
    return {
        key: cbrkit.model.graph.from_dict(value, node_converter=deepfreeze)
        for key, value in cbrkit.loaders.file(path).items()
    }


try:
    ACTIVITY_SIM = cbrkit.sim.taxonomy.build(
        Path("./data/taxonomies/activity.json"),
        cbrkit.sim.taxonomy.weights("user", "optimistic"),
    )
except ValidationError:
    ACTIVITY_SIM = cbrkit.sim.generic.equality()
    logger.warning("Failed to load taxonomy for activity. Using equality instead.")

try:
    INGREDIENT_SIM = cbrkit.sim.taxonomy.build(
        Path("./data/taxonomies/ingredient.json"),
        cbrkit.sim.taxonomy.weights("user", "optimistic"),
    )
except ValidationError:
    INGREDIENT_SIM = cbrkit.sim.generic.equality()
    logger.warning("Failed to load taxonomy for ingredient. Using equality instead.")


WORKFLOW_NODE_SIM = cbrkit.sim.attribute_value(
    {
        "name": cbrkit.sim.strings.levenshtein(),
        "preparation time (min)": cbrkit.sim.numbers.linear(1000),
        "calories": cbrkit.sim.numbers.linear(2500),
    }
)

TASK_NODE_SIM = cbrkit.sim.attribute_value({"name": ACTIVITY_SIM})

DATA_NODE_SIM = cbrkit.sim.attribute_value(
    {
        "name": INGREDIENT_SIM,
        "amount": cbrkit.sim.attribute_value(
            {
                "value": cbrkit.sim.numbers.linear(1000),
                "unit": cbrkit.sim.generic.equality(),
            }
        ),
    },
    aggregator=cbrkit.sim.aggregator(pooling_weights={"name": 2}),
    # some data nodes do not have an amount
    default=0.0,
)

NODE_SIM: BatchSimFunc[NodeData, cbrkit.typing.Float] = cbrkit.sim.attribute_table(
    {
        "workflow": WORKFLOW_NODE_SIM,
        "task": TASK_NODE_SIM,
        "data": DATA_NODE_SIM,
    },
    attribute="type",
)


def node_matcher(x: Mapping[str, Any], y: Mapping[str, Any]) -> bool:
    return x["type"] == y["type"]


GRAPH_SIM: cbrkit.typing.AnySimFunc[Graph[str, NodeData, None, None], GraphSim[str]] = (
    cbrkit.sim.graphs.astar.build(
        node_sim_func=NODE_SIM,
        node_matcher=node_matcher,
        beam_width=1,
    )
)


def Retriever() -> cbrkit.typing.RetrieverFunc[
    str, Graph[str, NodeData, None, None], GraphSim[str]
]:
    return cbrkit.retrieval.build(GRAPH_SIM, multiprocessing=True, chunksize=1)
