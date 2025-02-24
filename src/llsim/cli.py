import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import cbrkit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typer import Option, Typer

from llsim import builder, preferences

app = Typer(pretty_exceptions_enable=False)


def load_cases(
    loader: Callable[[Path], Any], path: Path, pattern: str | None
) -> dict[str, Any]:
    if pattern is None and path.is_file():
        return loader(path)
    elif pattern is not None and path.is_dir():
        return {path.stem: loader(path) for path in path.glob(pattern)}

    raise ValueError("Invalid path or pattern")


@app.command()
def build_preferences(
    cases: Annotated[Path, Option()],
    loader: Annotated[str, Option()],
    out: Annotated[Path, Option()],
    query_name: Annotated[list[str], Option(default_factory=list)],
    tries: Annotated[int, Option()] = 1,
    infer_missing: Annotated[bool, Option()] = True,
    max_cases: Annotated[int, Option()] = 100,
    queries: Annotated[Path | None, Option()] = None,
    cases_pattern: str | None = None,
    queries_pattern: str | None = None,
):
    _loader: Callable[..., Any] = cbrkit.helpers.load_callable(loader)
    out.parent.mkdir(parents=True, exist_ok=True)

    if queries is None:
        queries = cases

    if queries_pattern is None:
        queries_pattern = cases_pattern

    _cases = load_cases(_loader, cases, cases_pattern)
    _queries = load_cases(_loader, queries, queries_pattern)

    if query_name:
        _queries = {key: _queries[key] for key in query_name}

    batches = [(_cases, query) for query in _queries.values()]

    responses = [preferences.SynthesisResponse(preferences=[]) for _ in batches]

    for _ in range(tries):
        responses = preferences.request(batches, responses, max_cases)

        if infer_missing:
            responses = preferences.infer_missing(batches, responses)

    with open(out, "w") as fp:
        json.dump([entry.model_dump() for entry in responses], fp, indent=2)


@app.command()
def build_similarity(
    cases: Annotated[Path, Option()],
    loader: Annotated[str, Option()],
    out: Annotated[Path, Option()],
    query_name: Annotated[list[str], Option(default_factory=list)],
    attribute: Annotated[list[str], Option(default_factory=list)],
    queries: Annotated[Path | None, Option()] = None,
    cases_pattern: str | None = None,
    queries_pattern: str | None = None,
    attribute_table: str | None = None,
):
    _loader: Callable[..., Any] = cbrkit.helpers.load_callable(loader)
    out.parent.mkdir(parents=True, exist_ok=True)

    if queries_pattern is None:
        queries_pattern = cases_pattern

    _cases = load_cases(_loader, cases, cases_pattern)

    if queries is None:
        _queries = {}
    else:
        _queries = load_cases(_loader, queries, queries_pattern)

    if query_name:
        _queries = {key: _queries[key] for key in query_name}

    result = builder.build(_cases, _queries, attribute, attribute_table)

    with out.open("w") as fp:
        json.dump(result, fp, indent=2)


@app.command()
def retrieve(
    cases: Annotated[Path, Option()],
    retriever: Annotated[str, Option()],
    out: Annotated[Path, Option()],
    retriever_arg: Annotated[list[str], Option(default_factory=dict)],
    loader: Annotated[str, Option()],
    query_name: Annotated[list[str], Option(default_factory=list)],
    queries: Annotated[Path | None, Option()] = None,
    cases_pattern: str | None = None,
    queries_pattern: str | None = None,
):
    retriever_kwargs: dict[str, str] = {}

    for arg in retriever_arg:
        key, value = arg.split("=")
        retriever_kwargs[key] = value

    _retriever: Callable[..., Any] = cbrkit.helpers.load_object(retriever)(
        **retriever_kwargs
    )
    _loader: Callable[..., Any] = cbrkit.helpers.load_callable(loader)

    out.parent.mkdir(parents=True, exist_ok=True)

    if queries is None:
        queries = cases

    if queries_pattern is None:
        queries_pattern = cases_pattern

    _cases = load_cases(_loader, cases, cases_pattern)
    _queries = load_cases(_loader, queries, queries_pattern)

    if query_name:
        _queries = {key: _queries[key] for key in query_name}

    result = cbrkit.retrieval.apply_queries(_cases, _queries, _retriever)

    with out.open("w") as fp:
        json.dump(result.model_dump(), fp, indent=2)


@app.command()
def evaluate_run(
    result: Annotated[Path, Option()],
    baseline: Annotated[Path, Option()],
    k: Annotated[list[int], Option(default_factory=list)],
    max_qrel: int | None = None,
    min_qrel: int = 0,
):
    with baseline.open("r") as fp:
        _baseline = cbrkit.model.Result.model_validate(json.load(fp))

    with result.open("r") as fp:
        _result = cbrkit.model.Result.model_validate(json.load(fp))

    metrics = cbrkit.eval.retrieval_step(
        cbrkit.eval.retrieval_step_to_qrels(_baseline.final_step, max_qrel, min_qrel),
        _result.final_step,
        metrics=cbrkit.eval.generate_metrics(ks=k + [None]),
    )

    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")

    # compute mean average error between similarities from llm and benchmark
    baseline_scores = {
        key: entry.similarities for key, entry in _baseline.final_step.queries.items()
    }
    result_scores = {
        key: entry.similarities for key, entry in _result.final_step.queries.items()
    }

    # TODO: Currently does not handle k values
    mse = cbrkit.eval.compute_score_metrics(
        baseline_scores,
        result_scores,
        {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
        },
    )

    for key, value in mse.items():
        print(f"{key}: {value:.3f}")


@app.command()
def evaluate_qrels(
    result: Annotated[Path, Option()],
    queries: Annotated[Path, Option()],
    loader: Annotated[str, Option()],
    k: Annotated[list[int], Option(default_factory=list)],
    queries_pattern: str | None = None,
):
    with result.open("r") as fp:
        _result = cbrkit.model.Result.model_validate(json.load(fp))

    _loader = cbrkit.helpers.load_callable(loader)
    _queries = load_cases(_loader, queries, queries_pattern)

    qrels: dict[str, dict[str, int]] = {
        key: query.value["qrels"] for key, query in _queries.items()
    }

    metrics = cbrkit.eval.retrieval_step(
        qrels,
        _result.final_step,
        metrics=cbrkit.eval.generate_metrics(ks=k),
    )

    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
