import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, TypedDict, Union, cast, get_args, get_origin

import cbrkit
from pydantic import BaseModel

from llsim.provider import openai_provider

type SimFuncGenerator[T] = Callable[
    ..., cbrkit.typing.AnySimFunc[T, cbrkit.typing.Float]
]


@dataclass(slots=True)
class embedding:
    model: str
    __doc__ = cbrkit.sim.embed.sentence_transformers.__doc__
    func: cbrkit.typing.AnySimFunc[str, cbrkit.typing.Float] = field(init=False)

    def __post_init__(self):
        self.func = cbrkit.sim.embed.build(
            cbrkit.sim.embed.sentence_transformers(self.model)
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class TableEntry(BaseModel):
    source: str
    target: str
    similarity: float

    def dump(self) -> tuple[str, str, float]:
        return self.source, self.target, self.similarity


@dataclass(slots=True)
class table:
    entries: list[TableEntry]
    symmetric: bool
    default: float
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
    cbrkit.sim.numbers.exponential
    | cbrkit.sim.numbers.linear_interval
    | cbrkit.sim.numbers.sigmoid
    | cbrkit.sim.numbers.threshold
    | cbrkit.sim.numbers.linear
)

StringMeasure = (
    embedding
    | table
    | cbrkit.sim.strings.glob
    | cbrkit.sim.strings.jaro
    | cbrkit.sim.strings.jaro_winkler
    | cbrkit.sim.strings.levenshtein
    # | cbrkit.sim.strings.ngram # tokenizer cannot be handled by pydantic
    | cbrkit.sim.strings.regex
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
    queries: cbrkit.typing.Casebase[str, V],
    attributes: list[str],
    attribute_table: str | None,
) -> SerializedConfig | AttributeTableConfig:
    raw_cases = {key: casebase[key] for key in casebase}
    raw_cases.update({key: queries[key] for key in queries})
    cases: dict[str, dict[str, Any]] = {}

    if any(isinstance(case, cbrkit.model.graph.Graph) for case in raw_cases.values()):
        raw_cases = cast(dict[str, cbrkit.model.graph.Graph], raw_cases)

        for case_key, case in raw_cases.items():
            for node_key, node in case.nodes.items():
                cases[f"{case_key}-{node_key}"] = node.value

    else:
        cases = {key: cast(dict[str, Any], case) for key, case in raw_cases.items()}

    if not attribute_table:
        return build_part(cases, attributes)

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
            )
            for value in attribute_table_values
        },
    )


def build_part(
    cases: Mapping[str, Mapping[str, Any]],
    attributes: list[str],
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
        provider = openai_provider(
            "o3-mini-2025-01-31",
            MeasureModel,  # pyright: ignore
            (
                "Compute the similarityfor the given documents by calling the most suitable tool with the correct arguments. "
                "The response should be based on the presented values and not their keys/ids. "
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

        responses = provider(requests)

        for name, response in zip(names, responses):
            configurations[name] = SerializedConfigEntry(
                name=response.__class__.__name__,
                kind=kind,
                kwargs=response.model_dump(),
            )

    return configurations


def Retriever(file: str):
    with open(file) as fp:
        measures: SerializedConfig | AttributeTableConfig = json.load(fp)

    if "table" in measures:
        measures = cast(AttributeTableConfig, measures)
        sim = cbrkit.sim.attribute_table(
            {
                value: AttributeValueSim(configs)
                for value, configs in measures["table"].items()
            },
            measures["attribute"],
        )
    else:
        measures = cast(SerializedConfig, measures)
        sim = AttributeValueSim(measures)

    return cbrkit.retrieval.build(sim)


def AttributeValueSim(configs: SerializedConfig):
    number_lookup = measure_lookup(NumberMeasure)
    string_lookup = measure_lookup(StringMeasure)

    attribute_functions: dict[
        str, cbrkit.typing.AnySimFunc[Any, cbrkit.typing.Float]
    ] = {
        name: number_lookup[config["name"]](**config["kwargs"])
        if config["kind"] == "number"
        else string_lookup[config["name"]](**config["kwargs"])
        for name, config in configs.items()
    }

    return cbrkit.sim.attribute_value(attribute_functions)
