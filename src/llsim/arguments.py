from collections.abc import Mapping
from pathlib import Path
from typing import Any

import arguebuf
import cbrkit
from cbrkit.model.graph import (
    Graph,
    SerializedEdge,
    SerializedGraph,
)
from frozendict import deepfreeze

type NodeData = Mapping[str, Any]
type GraphData = Mapping[str, Any]


def load(path: Path) -> Graph[str, NodeData, None, GraphData]:
    g = arguebuf.load.file(path)

    atom_nodes: dict[str, NodeData] = {
        key: {
            "type": "atom",
            "text": value.plain_text,
        }
        for key, value in g.atom_nodes.items()
    }
    scheme_nodes: dict[str, NodeData] = {
        key: {
            "type": "scheme",
            "text": type(value.scheme).__name__ if value.scheme is not None else "None",
        }
        for key, value in g.scheme_nodes.items()
    }
    edges: dict[str, SerializedEdge[str, None]] = {
        key: SerializedEdge(
            source=value.source.id,
            target=value.target.id,
            value=None,
        )
        for key, value in g.edges.items()
    }

    if len(g.resources) > 0:
        graph_text = next(iter(g.resources.values())).plain_text
    else:
        graph_text = " ".join(value.plain_text for value in g.atom_nodes.values())

    if "cbrEvaluations" in g.userdata:
        ranking = g.userdata["cbrEvaluations"][0]["ranking"]
        max_rank = max(ranking.values())
        qrels = {
            key.split("/")[-1]: max_rank - value + 1 for key, value in ranking.items()
        }
    else:
        qrels = None

    return Graph.load(
        SerializedGraph(
            nodes={**atom_nodes, **scheme_nodes},
            edges=edges,
            value={
                "text": graph_text,
                "qrels": qrels,
            },
        ),
        node_converter=deepfreeze,
    )


def graph2text(g: Graph[str, NodeData, None, GraphData]) -> str:
    return g.value["text"]


EMBED_FUNC = cbrkit.sim.embed.cache(cbrkit.sim.embed.openai("text-embedding-3-small"))
SEMANTIC_SIM = cbrkit.sim.embed.build(EMBED_FUNC, cbrkit.sim.embed.cosine())
ATOM_SIM = cbrkit.sim.attribute_value(
    {
        "text": SEMANTIC_SIM,
    }
)
SCHEME_SIM = cbrkit.sim.attribute_value(
    {
        "text": cbrkit.sim.generic.equality(),
    }
)
NODE_SIM = cbrkit.sim.attribute_table(
    {
        "atom": ATOM_SIM,
        "scheme": SCHEME_SIM,
    },
    attribute="type",
    default=cbrkit.sim.generic.static(0.0),
)


def node_matcher(x: Mapping[str, Any], y: Mapping[str, Any]) -> bool:
    return x["type"] == y["type"]


def GRAPH_SIM_FACTORY() -> cbrkit.typing.AnySimFunc[
    Graph[str, NodeData, None, GraphData],
    cbrkit.sim.graphs.GraphSim[str],
]:
    return cbrkit.sim.graphs.astar.build(
        past_cost_func=cbrkit.sim.graphs.astar.g1(NODE_SIM),
        future_cost_func=cbrkit.sim.graphs.astar.h3(NODE_SIM),
        selection_func=cbrkit.sim.graphs.astar.select3(
            cbrkit.sim.graphs.astar.h3(NODE_SIM)
        ),
        init_func=cbrkit.sim.graphs.astar.init2(
            node_matcher=node_matcher
        ),
        queue_limit=1,
        node_matcher=node_matcher,
    )


# GRAPH_MAC = cbrkit.retrieval.build(
#     cbrkit.sim.transpose(SEMANTIC_SIM, graph2text),
# )
GRAPH_FAC_PRECOMPUTE = cbrkit.retrieval.build(
    cbrkit.sim.graphs.precompute(
        cbrkit.sim.attribute_table(
            {"atom": ATOM_SIM},
            attribute="type",
            default=cbrkit.sim.generic.static(0.0),
        )
    )
)
GRAPH_FAC = cbrkit.retrieval.build(GRAPH_SIM_FACTORY, multiprocessing=True)


def Retrievers() -> cbrkit.typing.MaybeFactories[
    cbrkit.typing.RetrieverFunc[
        str,
        Graph[str, NodeData, None, GraphData],
        float | cbrkit.sim.graphs.GraphSim[str],
    ]
]:
    # return [GRAPH_FAC_PRECOMPUTE, GRAPH_FAC]
    return [GRAPH_FAC]
