from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import cbrkit
import orjson
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typer import Option, Typer

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
def retrieve(
    cases: Annotated[Path, Option()],
    out: Annotated[Path, Option()],
    retriever: Annotated[str, Option()],
    loader: Annotated[str, Option()],
    query_name: Annotated[list[str], Option(default_factory=list)],
    queries: Annotated[Path | None, Option()] = None,
    cases_pattern: str | None = None,
    queries_pattern: str | None = None,
):
    _retriever = cbrkit.helpers.load_callable(retriever)
    _loader = cbrkit.helpers.load_callable(loader)
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

    with out.open("wb") as fp:
        fp.write(orjson.dumps(result.model_dump()))


@app.command()
def evaluate_run(
    result: Annotated[Path, Option()],
    baseline: Annotated[Path, Option()],
    k: Annotated[list[int], Option(default_factory=list)],
    max_qrel: int | None = None,
    min_qrel: int = 0,
):
    with baseline.open("rb") as fp:
        _baseline = cbrkit.model.Result.model_validate(orjson.loads(fp.read()))

    with result.open("rb") as fp:
        _result = cbrkit.model.Result.model_validate(orjson.loads(fp.read()))

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
    with result.open("rb") as fp:
        _result = cbrkit.model.Result.model_validate(orjson.loads(fp.read()))

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
