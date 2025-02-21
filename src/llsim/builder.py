import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Number
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


class SerializedConfig(TypedDict):
    name: str
    kind: str
    kwargs: dict[str, Any]


def build(
    casebase: cbrkit.typing.Casebase[str, dict[str, Any]],
    queries: cbrkit.typing.Casebase[str, dict[str, Any]],
    attributes: list[str],
) -> dict[str, Any]:
    cases = {key: casebase[key] for key in casebase}
    cases.update({key: queries[key] for key in queries})
    attribute_kinds: defaultdict[str, list[str]] = defaultdict(list)
    configurations: dict[str, SerializedConfig] = {}
    kind_lookup: defaultdict[str, set[str]] = defaultdict(set)

    for case in cases.values():
        for name, value in case.items():
            if not attributes or name in attributes:
                kind = type(value)

                if issubclass(kind, str):
                    attribute_kinds[name].append("string")
                elif issubclass(kind, Number):
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
            configurations[name] = SerializedConfig(
                name=response.__class__.__name__,
                kind=kind,
                kwargs=response.model_dump(),
            )

    return configurations


def Retriever(file: str):
    with open(file) as fp:
        attribute_measures = json.load(fp)

    number_lookup = measure_lookup(NumberMeasure)
    string_lookup = measure_lookup(StringMeasure)

    attribute_functions: dict[
        str, cbrkit.typing.AnySimFunc[Any, cbrkit.typing.Float]
    ] = {
        name: number_lookup[config["name"]](**config["kwargs"])
        if config["kind"] == "number"
        else string_lookup[config["name"]](**config["kwargs"])
        for name, config in attribute_measures.items()
    }

    return cbrkit.retrieval.build(cbrkit.sim.attribute_value(attribute_functions))
