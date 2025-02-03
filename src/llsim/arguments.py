from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arguebuf
import cbrkit
from cbrkit.model.graph import (
    Graph,
    SerializedEdge,
    SerializedGraph,
)


@dataclass(frozen=True, slots=True)
class SchemeData:
    scheme: arguebuf.Scheme | None


type AtomData = str
type NodeData = AtomData | SchemeData
type EdgeData = None
type GraphData = dict[str, Any]


def unpack_scheme(node: SchemeData) -> arguebuf.Scheme | None:
    return node.scheme


def load(path: Path) -> Graph[str, NodeData, EdgeData, GraphData]:
    g = arguebuf.load.file(path)

    atom_nodes: dict[str, NodeData] = {
        key: value.plain_text for key, value in g.atom_nodes.items()
    }
    scheme_nodes: dict[str, NodeData] = {
        key: SchemeData(value.scheme) for key, value in g.scheme_nodes.items()
    }
    edges: dict[str, SerializedEdge[str, EdgeData]] = {
        key: SerializedEdge(source=value.source.id, target=value.target.id, value=None)
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
        )
    )


def graph2text(g: Graph[str, NodeData, EdgeData, GraphData]) -> str:
    return g.value["text"]


EMBED_FUNC = cbrkit.sim.embed.cache(
    cbrkit.sim.embed.openai("text-embedding-3-small"),
    "./data/embeddings-cache.npz",
    autodump=True,
)
SEMANTIC_SIM = cbrkit.sim.embed.build(EMBED_FUNC, cbrkit.sim.embed.cosine())
SCHEME_SIM = cbrkit.sim.transpose(cbrkit.sim.generic.type_equality(), unpack_scheme)
NODE_SIM = cbrkit.sim.type_table(
    {
        str: SEMANTIC_SIM,
        SchemeData: SCHEME_SIM,
    },
    default=cbrkit.sim.generic.static(0.0),
)


def GRAPH_SIM_FACTORY() -> cbrkit.typing.AnySimFunc[
    Graph[str, NodeData, EdgeData, GraphData],
    cbrkit.sim.graphs.GraphSim[str],
]:
    return cbrkit.sim.graphs.astar.build(
        past_cost_func=cbrkit.sim.graphs.astar.g1(NODE_SIM),
        future_cost_func=cbrkit.sim.graphs.astar.h3(NODE_SIM),
        selection_func=cbrkit.sim.graphs.astar.select3(
            cbrkit.sim.graphs.astar.h3(NODE_SIM)
        ),
        init_func=cbrkit.sim.graphs.astar.init2[str, NodeData, EdgeData, GraphData](),
        queue_limit=1,
    )


GRAPH_MAC = cbrkit.retrieval.build(cbrkit.sim.transpose(SEMANTIC_SIM, graph2text))
GRAPH_FAC_PRECOMPUTE = cbrkit.retrieval.build(
    cbrkit.sim.graphs.precompute(
        cbrkit.sim.type_table(
            {str: SEMANTIC_SIM},
            default=cbrkit.sim.generic.static(0.0),
        )
    )
)
GRAPH_FAC = cbrkit.retrieval.build(GRAPH_SIM_FACTORY, multiprocessing=True)


RETRIEVER: cbrkit.typing.MaybeFactories[
    cbrkit.typing.RetrieverFunc[
        str,
        Graph[str, NodeData, EdgeData, GraphData],
        float | cbrkit.sim.graphs.GraphSim[str],
    ]
] = [GRAPH_FAC_PRECOMPUTE, GRAPH_FAC]
