import json
from collections.abc import Callable
from pathlib import Path
from timeit import default_timer
from typing import Annotated, Any

import cbrkit
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
                "llsim.recipes.Retriever",
                Path("data/cases/recipes.json"),
                None,
                None,
                None,
            )
        case "cars":
            return (
                "llsim.cars:load",
                "llsim.cars.Retriever",
                Path("data/cases/cars.json"),
                None,
                None,
                None,
            )
        case "arguments":
            return (
                "llsim.arguments:load",
                "llsim.arguments.Retrievers",
                Path("data/cases/arguments"),
                "*.json",
                Path("data/queries/arguments"),
                "*.json",
            )

    raise ValueError(f"Unknown domain: {name}")


@app.command()
def build_preferences(
    out: Annotated[Path, Option()],
    model: Annotated[str, Option()],
    query_name: Annotated[list[str], Option(default_factory=list)],
    retries: Annotated[int, Option()] = 1,
    infer_missing: Annotated[bool, Option()] = True,
    max_cases: Annotated[float, Option()] = 100,
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

    responses = [preferences.Response(preferences=[]) for _ in batches]
    provider = Provider(model)
    start_time = default_timer()

    for _ in range(retries + 1):
        if pairwise:
            responses = preferences.request_pairwise(
                provider, batches, responses, max_cases
            )
        else:
            responses = preferences.request(provider, batches, responses, max_cases)

        if infer_missing:
            responses = preferences.infer_missing(batches, responses)

    with open(out, "w") as fp:
        json.dump(
            {
                "duration": default_timer() - start_time,
                "responses": [entry.model_dump() for entry in responses],
            },
            fp,
            indent=2,
        )


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

    assert loader is not None, "loader is required"
    assert cases is not None, "cases is required"

    _loader: Callable[..., Any] = cbrkit.helpers.load_callable(loader)
    out.parent.mkdir(parents=True, exist_ok=True)

    _cases = load_cases(_loader, cases, cases_pattern)
    start_time = default_timer()

    response = builder.build(_cases, attribute, attribute_table, Provider(model))

    with out.open("w") as fp:
        json.dump(
            {
                "duration": default_timer() - start_time,
                "response": response,
            },
            fp,
            indent=2,
        )


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
    result_step: cbrkit.model.ResultStep[Any, Any, Any, cbrkit.typing.Float],
) -> dict[Any, dict[Any, float]]:
    all_similarities = cbrkit.helpers.unpack_floats(
        sim
        for query in result_step.queries.values()
        for sim in query.similarities.values()
    )
    min_sim = min(all_similarities)
    max_sim = max(all_similarities)

    return {
        query: {
            case: (cbrkit.helpers.unpack_float(sim) - min_sim) / (max_sim - min_sim)
            for case, sim in entry.similarities.items()
        }
        for query, entry in result_step.queries.items()
    }


def format_float(x: float) -> str:
    if x <= 1:
        return f"{x:.3f}"
    elif x <= 10:
        return f"{x:.2f}"
    elif x <= 100:
        return f"{x:.1f}"

    return f"{x:.0f}"


def print_metrics(metrics: dict[str, dict[str, float]], max_qrels: list[int | None]):
    unique_metric_names = [
        "completeness",
        "correctness",
        # "ndcg",
    ]
    metric_names = [
        f"{metric}#{qrel}" if qrel else metric
        for metric in unique_metric_names
        for qrel in max_qrels
    ] + ["duration"]
    metric_commands = [
        f"\\{metric}{{{qrel or 'n'}}}"
        for metric in unique_metric_names
        for qrel in max_qrels
    ] + ["\\duration{}"]

    print("\\begin{tabular}{l" + "c" * len(metric_names) + "}")
    print("\\toprule")
    print("\\experiment{} & " + " & ".join(metric_commands) + " \\\\")
    print("\\midrule")

    for key, values in sorted(metrics.items(), key=lambda x: x[0]):
        pretty_value = " & ".join(
            format_float(values[metric]) for metric in metric_names
        )
        print(f"{key} & {pretty_value} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


@app.command()
def evaluate_run(
    directory: Path,
    k: Annotated[list[int], Option(default_factory=list)],
    max_qrel: Annotated[list[int], Option(default_factory=list)],
    min_qrel: int = 0,
    baseline_name: str = "baseline.json",
):
    max_qrels = [None] + max_qrel
    ks = [None] + k

    baseline_path = directory / baseline_name
    run_paths = directory.glob("*.json")
    metric_funcs = cbrkit.eval.generate_metrics(
        [
            "completeness",
            "correctness",
            # "ndcg",
        ],
        ks=ks,
    )
    # error_metric_funcs = {
    #     "mse": mean_squared_error,
    #     "mae": mean_absolute_error,
    # }
    metrics: dict[str, dict[str, float]] = {}

    with baseline_path.open("r") as fp:
        baseline = cbrkit.model.Result.model_validate(json.load(fp))

    baseline_qrels = {
        max_qrel_entry: cbrkit.eval.retrieval_step_to_qrels(
            baseline.final_step,
            max_qrel=max_qrel_entry,
            min_qrel=min_qrel,
        )
        for max_qrel_entry in max_qrels
    }
    # baseline_sims = normalize_similarities(baseline.final_step)

    for run_path in run_paths:
        if run_path == baseline_path or run_path.stem.endswith("-config"):
            continue

        with run_path.open("r") as fp:
            run = cbrkit.model.Result.model_validate(json.load(fp))

        metrics[run_path.stem] = {}

        for qrel, baseline_qrel in baseline_qrels.items():
            current_metrics = cbrkit.eval.retrieval_step(
                baseline_qrel,
                run.final_step,
                metrics=metric_funcs,
            )

            if qrel is None:
                metrics[run_path.stem].update(
                    {f"{metric}": value for metric, value in current_metrics.items()}
                )
            else:
                metrics[run_path.stem].update(
                    {
                        f"{metric}#{qrel}": value
                        for metric, value in current_metrics.items()
                    }
                )

        # try:
        #     error_metrics = cbrkit.eval.compute_score_metrics(
        #         baseline_sims,
        #         normalize_similarities(run.final_step),
        #         error_metric_funcs,  # pyright: ignore
        #     )
        #     metrics[run_path.stem].update(cast(dict[str, float], error_metrics))
        # except ValueError:
        #     metrics[run_path.stem].update(
        #         {metric: float("nan") for metric in error_metric_funcs.keys()}
        #     )

        duration = run.final_step.duration
        run_config_path = directory / f"{run_path.stem}-config.json"

        if run_config_path.exists() and not run_path.stem.startswith("builder-"):
            with run_config_path.open("r") as fp:
                run_config = json.load(fp)

            duration += run_config["duration"]

        metrics[run_path.stem]["duration"] = duration

    print_metrics(metrics, max_qrels)


@app.command()
def evaluate_qrels(
    directory: Path,
    k: Annotated[list[int], Option(default_factory=list)],
    domain: str | None = None,
    loader: str | None = None,
    queries: Path | None = None,
    queries_pattern: str | None = None,
):
    ks = [None] + k

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
            metrics=cbrkit.eval.generate_metrics(ks=ks),
        )

        duration = run.final_step.duration
        run_config_path = directory / f"{run_path.stem}-config.json"

        if run_config_path.exists() and not run_path.stem.startswith("builder-"):
            with run_config_path.open("r") as fp:
                run_config = json.load(fp)

            duration += run_config["duration"]

        metrics[run_path.stem]["duration"] = duration

    print_metrics(metrics, [None])


if __name__ == "__main__":
    app()
