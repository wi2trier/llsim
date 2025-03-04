import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import cbrkit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typer import Option, Typer

from llsim import builder, preferences
from llsim.provider import Provider

app = Typer(pretty_exceptions_enable=False)


def load_cases(
    loader: Callable[[Path], Any], path: Path, pattern: str | None
) -> dict[str, Any]:
    if pattern is None and path.is_file():
        return loader(path)
    elif pattern is not None and path.is_dir():
        return {path.stem: loader(path) for path in path.glob(pattern)}

    raise ValueError("Invalid path or pattern")


def load_domain(
    name: str,
) -> tuple[str, str, Path, str | None, Path | None, str | None]:
    match name:
        case "recipes":
            return (
                "llsim.recipes:load",
                "llsim.recipes.RETRIEVER",
                Path("data/cases/recipes.json"),
                None,
                None,
                None,
            )
        case "cars":
            return (
                "llsim.cars:load",
                "llsim.cars.RETRIEVER",
                Path("data/cases/cars.json"),
                None,
                None,
                None,
            )
        case "arguments":
            return (
                "llsim.arguments:load",
                "llsim.arguments.RETRIEVERS",
                Path("data/cases/arguments"),
                "*.json",
                Path("data/cases/arguments"),
                "*.json",
            )

    raise ValueError(f"Unknown domain: {name}")


@app.command()
def build_preferences(
    out: Annotated[Path, Option()],
    model: Annotated[str, Option()],
    query_name: Annotated[list[str], Option(default_factory=list)],
    tries: Annotated[int, Option()] = 3,
    infer_missing: Annotated[bool, Option()] = True,
    max_cases: Annotated[int, Option()] = 100,
    queries: Annotated[Path | None, Option()] = None,
    domain: str | None = None,
    loader: str | None = None,
    cases: Path | None = None,
    cases_pattern: str | None = None,
    queries_pattern: str | None = None,
    pairwise: bool = False,
):
    if domain is not None:
        loader, _, cases, cases_pattern, queries, queries_pattern = load_domain(domain)

    assert loader is not None, "loader is required"
    assert cases is not None, "cases is required"

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
    provider = Provider(model)

    for _ in range(tries):
        if pairwise:
            responses = preferences.request_pairwise(
                provider, batches, responses, max_cases
            )
        else:
            responses = preferences.request(provider, batches, responses, max_cases)

        if infer_missing:
            responses = preferences.infer_missing(batches, responses)

    with open(out, "w") as fp:
        json.dump([entry.model_dump() for entry in responses], fp, indent=2)


@app.command()
def build_similarity(
    out: Annotated[Path, Option()],
    model: Annotated[str, Option()],
    attribute: Annotated[list[str], Option(default_factory=list)],
    domain: str | None = None,
    loader: str | None = None,
    cases: Path | None = None,
    cases_pattern: str | None = None,
    attribute_table: str | None = None,
):
    if domain is not None:
        loader, _, cases, cases_pattern, _, _ = load_domain(domain)

        if domain == "recipes":
            attribute_table = "type"

    assert loader is not None, "loader is required"
    assert cases is not None, "cases is required"

    _loader: Callable[..., Any] = cbrkit.helpers.load_callable(loader)
    out.parent.mkdir(parents=True, exist_ok=True)

    _cases = load_cases(_loader, cases, cases_pattern)

    result = builder.build(_cases, attribute, attribute_table, Provider(model))

    with out.open("w") as fp:
        json.dump(result, fp, indent=2)


@app.command()
def retrieve(
    out: Annotated[Path, Option()],
    retriever_arg: Annotated[list[str], Option(default_factory=dict)],
    query_name: Annotated[list[str], Option(default_factory=list)],
    retriever: str | None = None,
    domain: str | None = None,
    loader: str | None = None,
    cases: Path | None = None,
    queries: Path | None = None,
    cases_pattern: str | None = None,
    queries_pattern: str | None = None,
):
    baseline_retriever: str | None = None
    if domain is not None:
        loader, baseline_retriever, cases, cases_pattern, queries, queries_pattern = (
            load_domain(domain)
        )

    if retriever is None and baseline_retriever is not None:
        retriever = baseline_retriever

    assert loader is not None, "loader is required"
    assert cases is not None, "cases is required"
    assert retriever is not None, "retriever is required"

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


def normalize_similarities(
    result: cbrkit.model.Result[Any, Any, Any, float],
) -> dict[Any, dict[Any, float]]:
    min_sim = min(result.final_step.similarities.values())
    max_sim = max(result.final_step.similarities.values())

    return {
        query: {
            case: (sim - min_sim) / (max_sim - min_sim)
            for case, sim in entry.similarities.items()
        }
        for query, entry in result.final_step.queries.items()
    }


def print_metrics(metrics: dict[str, dict[str, float]]):
    metric_names = sorted(list(metrics.values())[0].keys())

    print("\\toprule")
    print("name & " + " & ".join(metric_names) + " \\\\")
    print("\\midrule")

    for key, values in metrics.items():
        pretty_value = " & ".join(f"{values[metric]:.3f}" for metric in metric_names)
        print(f"{key} & {pretty_value} \\\\")

    print("\\bottomrule")


# TODO: Add coverage (i.e., the number of results where the run actually contains the required values)


@app.command()
def evaluate_run(
    directory: Path,
    k: Annotated[list[int], Option(default_factory=list)],
    max_qrel: int | None = None,
    min_qrel: int = 0,
):
    baseline_path = directory / "baseline.json"
    run_paths = directory.glob("*.json")
    metric_funcs = cbrkit.eval.generate_metrics(ks=k + [None])
    metrics: dict[str, dict[str, float]] = {}

    with baseline_path.open("r") as fp:
        baseline = cbrkit.model.Result.model_validate(json.load(fp))

    baseline_qrels = cbrkit.eval.retrieval_step_to_qrels(
        baseline.final_step, max_qrel, min_qrel
    )
    baseline_sims = normalize_similarities(baseline)

    for run_path in run_paths:
        if run_path == baseline_path or run_path.stem.endswith("-config"):
            continue

        with run_path.open("r") as fp:
            run = cbrkit.model.Result.model_validate(json.load(fp))

        metrics[run_path.stem] = cbrkit.eval.retrieval_step(
            baseline_qrels,
            run.final_step,
            metrics=metric_funcs,
        )

        error_metrics = cbrkit.eval.compute_score_metrics(
            baseline_sims,
            normalize_similarities(run),
            {
                "mse": mean_squared_error,
                "mae": mean_absolute_error,
            },
        )
        metrics[run_path.stem].update(cast(dict[str, float], error_metrics))
        metrics[run_path.stem]["duration"] = run.final_step.duration

    print_metrics(metrics)


@app.command()
def evaluate_qrels(
    directory: Path,
    k: Annotated[list[int], Option(default_factory=list)],
    domain: str | None = None,
    loader: str | None = None,
    queries: Path | None = None,
    queries_pattern: str | None = None,
):
    if domain is not None:
        loader, _, _, _, queries, queries_pattern = load_domain(domain)

    assert loader is not None, "loader is required"
    assert queries is not None, "queries is required"

    _loader = cbrkit.helpers.load_callable(loader)
    _queries = load_cases(_loader, queries, queries_pattern)

    qrels: dict[str, dict[str, int]] = {
        key: query.value["qrels"] for key, query in _queries.items()
    }
    run_paths = directory.glob("*.json")
    metrics: dict[str, dict[str, float]] = {}

    for run_path in run_paths:
        if run_path.stem.endswith("-config"):
            continue

        with run_path.open("r") as fp:
            run = cbrkit.model.Result.model_validate(json.load(fp))

        metrics[run_path.stem] = cbrkit.eval.retrieval_step(
            qrels,
            run.final_step,
            metrics=cbrkit.eval.generate_metrics(ks=k),
        )
        metrics[run_path.stem]["duration"] = run.final_step.duration

    print_metrics(metrics)


if __name__ == "__main__":
    app()
