import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
)

import cbrkit
from pydantic import BaseModel, Field, OnErrorOmit

from llsim.provider import Provider

type SimFuncGenerator[T] = Callable[
    ..., cbrkit.typing.AnySimFunc[T, cbrkit.typing.Float]
]


@dataclass(slots=True)
class embedding:
    __doc__ = cbrkit.sim.embed.openai.__doc__
    func: cbrkit.typing.AnySimFunc[str, cbrkit.typing.Float] = field(init=False)

    def __post_init__(self):
        self.func = cbrkit.sim.embed.build(
            cbrkit.sim.embed.cache(
                cbrkit.sim.embed.openai("text-embedding-3-small"),
            ),
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


@dataclass(slots=True)
class ngram:
    n: int
    case_sensitive: bool
    __doc__ = cbrkit.sim.strings.ngram.__doc__
    func: cbrkit.typing.AnySimFunc[str, cbrkit.typing.Float] = field(init=False)

    def __post_init__(self):
        self.func = cbrkit.sim.strings.ngram(self.n, case_sensitive=self.case_sensitive)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class TaxonomyNode(BaseModel):
    name: Annotated[str, Field(description="The value of the node")]
    children: Annotated[
        list[OnErrorOmit["TaxonomyNode"]],
        Field(description="The children of the node to build a hierarchical taxonomy"),
    ]


TaxonomyNode.model_rebuild()

type TaxonomyDistanceFunc = Literal["wu_palmer", "paths", "levels", "weights"]

taxonomy_distance_functions: dict[
    TaxonomyDistanceFunc, cbrkit.sim.taxonomy.TaxonomySimFunc
] = {
    "wu_palmer": cbrkit.sim.taxonomy.wu_palmer(),
    "paths": cbrkit.sim.taxonomy.paths(),
    "levels": cbrkit.sim.taxonomy.levels(strategy="average"),
    "weights": cbrkit.sim.taxonomy.weights(strategy="average", source="auto"),
}


@dataclass(slots=True)
class taxonomy:
    root_node: TaxonomyNode = field(
        metadata={
            "description": (
                "The root node of the taxonomy. "
                "Its children cannot be empty, instead the taxonomy must be a tree with more than one node."
            )
        }
    )
    distance_function: TaxonomyDistanceFunc
    __doc__ = cbrkit.sim.taxonomy.Taxonomy.__doc__
    func: cbrkit.typing.AnySimFunc[str, cbrkit.typing.Float] = field(init=False)

    def __post_init__(self):
        taxonomy = cbrkit.sim.taxonomy.SerializedTaxonomyNode.model_validate(
            self.root_node.model_dump
            if isinstance(self.root_node, TaxonomyNode)
            else self.root_node
        )
        self.func = cbrkit.sim.taxonomy.build(
            taxonomy,
            taxonomy_distance_functions[self.distance_function],
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class TableEntry(BaseModel):
    source: Annotated[str, Field(description="The value of the source")]
    target: Annotated[str, Field(description="The value of the target")]
    similarity: Annotated[
        float,
        Field(
            description="The similarity of the source and target values. Must be between 0.0 and 1.0"
        ),
    ]

    def dump(self) -> tuple[str, str, float]:
        return self.source, self.target, self.similarity


@dataclass(slots=True)
class table:
    entries: Annotated[
        list[OnErrorOmit[TableEntry]],
        Field(description="List of table entries to build a similarity matrix"),
    ]
    symmetric: Annotated[
        bool, Field(description="Whether the table is symmetric or not")
    ]
    default: Annotated[
        float, Field(description="The default similarity value for missing entries")
    ]
    __doc__ = cbrkit.sim.table.__doc__
    func: cbrkit.typing.AnySimFunc[str, cbrkit.typing.Float] = field(init=False)

    def __post_init__(self):
        for idx, entry in enumerate(self.entries):
            if not isinstance(entry, TableEntry):
                self.entries[idx] = TableEntry(**entry)

        self.func = cbrkit.sim.strings.table(
            [entry.dump() for entry in self.entries],
            symmetric=self.symmetric,
            default=self.default,
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def measure_lookup[T](
    measure_type: UnionType | Any,
) -> dict[str, SimFuncGenerator[Any]]:
    if get_origin(measure_type) is UnionType:
        return {measure.__name__: measure for measure in get_args(measure_type)}

    measure_type = cast(type, measure_type)
    return {measure_type.__name__: measure_type}


NumberMeasure = (
    cbrkit.sim.generic.equality
    | cbrkit.sim.numbers.exponential
    | cbrkit.sim.numbers.linear_interval
    | cbrkit.sim.numbers.sigmoid
    | cbrkit.sim.numbers.threshold
    | cbrkit.sim.numbers.linear
)

StringMeasure = (
    cbrkit.sim.generic.equality
    | embedding
    | table
    | taxonomy
    | ngram
    | cbrkit.sim.strings.jaro
    | cbrkit.sim.strings.jaro_winkler
    | cbrkit.sim.strings.levenshtein
)


class SerializedConfigEntry(TypedDict):
    name: str
    kind: str
    kwargs: dict[str, Any]


type SerializedConfig = Mapping[str, SerializedConfigEntry]


class AttributeTableConfig(TypedDict):
    attribute: str
    table: dict[Any, SerializedConfig]


def build[V](
    casebase: cbrkit.typing.Casebase[str, V],
    attributes: list[str],
    attribute_table: str | None,
    provider: Provider,
) -> SerializedConfig | AttributeTableConfig:
    cases: dict[str, dict[str, Any]] = {}

    if any(isinstance(case, cbrkit.model.graph.Graph) for case in casebase.values()):
        for case_key, case in cast(
            cbrkit.typing.Casebase[str, cbrkit.model.graph.Graph], casebase
        ).items():
            for node_key, node in case.nodes.items():
                cases[f"{case_key}-{node_key}"] = node.value

    else:
        cases = {key: cast(dict[str, Any], case) for key, case in casebase.items()}

    if not attribute_table:
        return build_part(cases, attributes, provider)

    attribute_table_values: set[Any] = (
        {case[attribute_table] for case in cases.values()} if attribute_table else set()
    )

    return AttributeTableConfig(
        attribute=attribute_table,
        table={
            value: build_part(
                {
                    key: {k: v for k, v in case.items() if k != attribute_table}
                    for key, case in cases.items()
                    if case[attribute_table] == value
                },
                attributes,
                provider,
            )
            for value in attribute_table_values
        },
    )


def build_part(
    cases: Mapping[str, Mapping[str, Any]],
    attributes: list[str],
    provider: Provider,
) -> SerializedConfig:
    attribute_kinds: defaultdict[str, list[str]] = defaultdict(list)
    kind_lookup: defaultdict[str, set[str]] = defaultdict(set)
    configurations: dict[str, SerializedConfigEntry] = {}

    for case in cases.values():
        for name, value in case.items():
            if not attributes or name in attributes:
                kind = type(value)

                if issubclass(kind, str):
                    attribute_kinds[name].append("string")
                elif issubclass(kind, float | int):
                    attribute_kinds[name].append("number")

    for name, kinds in attribute_kinds.items():
        if len(set(kinds)) > 1:
            raise ValueError(f"Attribute {name} has different types")

        kind = kinds[0]
        kind_lookup[kind].add(name)

    for kind, names in kind_lookup.items():
        if kind == "number":
            measures = measure_lookup(NumberMeasure).values()
        elif kind == "string":
            measures = measure_lookup(StringMeasure).values()
        else:
            raise ValueError(f"Unsupported type {kind}")

        measure_models = [
            cbrkit.helpers.callable2model(measure, with_default=False)
            for measure in measures
        ]

        MeasureModel = Union[*measure_models]
        generation_func = provider.build(
            MeasureModel,  # pyright: ignore
            (
                "Compute the similarity for the given documents by calling the most suitable tool with the correct arguments. "
                "The response should be based on the presented values and not their keys/ids. "
                "Make sure to **always** call exactly one of the provided tools. "
            ),
        )

        requests: list[str] = [
            cbrkit.synthesis.prompts.default(
                instructions=f"The values are derived from an attribute with the following name: {name}"
            )(
                {key: value[name] for key, value in cases.items()},
                None,
                None,
            )
            for name in names
        ]

        responses = cbrkit.helpers.unpack_values(generation_func(requests))

        for name, response in zip(names, responses):
            configurations[name] = SerializedConfigEntry(
                name=response.__class__.__name__,
                kind=kind,
                kwargs=response.model_dump(),
            )

    return configurations


@dataclass(slots=True, frozen=True)
class AttributeValueSimFactory:
    config: SerializedConfig

    def __call__(self):
        number_lookup = measure_lookup(NumberMeasure)
        string_lookup = measure_lookup(StringMeasure)

        functions: dict[str, cbrkit.typing.AnySimFunc[Any, cbrkit.typing.Float]] = {
            name: number_lookup[config["name"]](**config["kwargs"])
            if config["kind"] == "number"
            else string_lookup[config["name"]](**config["kwargs"])
            for name, config in self.config.items()
        }
        cached_funcs = {
            name: cbrkit.sim.cache(func) for name, func in functions.items()
        }

        return cbrkit.sim.attribute_value(cached_funcs, default=0.0)


@dataclass(slots=True, frozen=True)
class AttributeTableSimFactory:
    config: AttributeTableConfig

    def __call__(self):
        return cbrkit.sim.attribute_table(
            entries={
                key: AttributeValueSimFactory(value)()
                for key, value in self.config["table"].items()
            },
            attribute=self.config["attribute"],
            default=cbrkit.sim.generic.static(0.0),
        )


@dataclass(slots=True, frozen=True)
class GraphSimFactory:
    node_sim_func: cbrkit.typing.Factory[
        cbrkit.typing.AnySimFunc[Any, cbrkit.typing.Float]
    ]

    def __call__(self):
        node_sim_func = self.node_sim_func()
        return cbrkit.sim.graphs.astar.build(
            past_cost_func=cbrkit.sim.graphs.astar.g1(node_sim_func),
            future_cost_func=cbrkit.sim.graphs.astar.h3(node_sim_func),
            selection_func=cbrkit.sim.graphs.astar.select3(
                cbrkit.sim.graphs.astar.h3(node_sim_func)
            ),
            init_func=cbrkit.sim.graphs.astar.init2(),
            queue_limit=1,
        )


@dataclass(slots=True, init=False)
class Retriever[V](cbrkit.typing.RetrieverFunc[str, V, cbrkit.typing.Float]):
    sim_func: cbrkit.typing.Factory[
        cbrkit.typing.AnySimFunc[V, cbrkit.typing.Float]
    ] = field(init=False)

    def __init__(self, file: str):
        with open(file) as fp:
            obj = json.load(fp)
            measures: SerializedConfig | AttributeTableConfig = obj["response"]

        if "table" in measures:
            measures = cast(AttributeTableConfig, measures)
            self.sim_func = AttributeTableSimFactory(measures)
        else:
            measures = cast(SerializedConfig, measures)
            self.sim_func = AttributeValueSimFactory(measures)

    def __call__(
        self,
        batches: Sequence[tuple[cbrkit.typing.Casebase[str, V], V]],
    ) -> Sequence[Mapping[str, cbrkit.typing.Float]]:
        first_query = batches[0][1]

        if isinstance(first_query, cbrkit.model.graph.Graph):
            graph_batches = cast(
                Sequence[
                    tuple[
                        cbrkit.typing.Casebase[str, cbrkit.model.graph.Graph],
                        cbrkit.model.graph.Graph,
                    ]
                ],
                batches,
            )

            retriever_func = cbrkit.retrieval.build(
                GraphSimFactory(self.sim_func), multiprocessing=True
            )

            return retriever_func(graph_batches)

        return cbrkit.retrieval.build(self.sim_func)(batches)
