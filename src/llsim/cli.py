import pickle
from pathlib import Path
from typing import Annotated

import cbrkit
import orjson
from typer import Option, Typer

app = Typer(pretty_exceptions_enable=False)


def dump_result(result: cbrkit.retrieval.Result, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.with_suffix(".json").open("wb") as fp:
        fp.write(orjson.dumps(result.as_dict()))

    with output_path.with_suffix(".pkl").open("wb") as fp:
        pickle.dump(result, fp)


@app.command()
def retrieve_dir(
    cases_path: Path,
    queries_path: Path,
    output_path: Path,
    retriever: str,
    loader: str,
    cases_pattern: str = "*.json",
    queries_pattern: str = "*.json",
):
    _retriever = cbrkit.helpers.load_callable(retriever)
    _loader = cbrkit.helpers.load_callable(loader)

    cases = {path.stem: _loader(path) for path in cases_path.glob(cases_pattern)}
    queries = {path.stem: _loader(path) for path in queries_path.glob(queries_pattern)}
    result = cbrkit.retrieval.apply_queries(cases, queries, _retriever)

    dump_result(result, output_path)


@app.command()
def retrieve_file(
    cases_path: Path,
    queries_path: Path,
    output_path: Path,
    retriever: str,
    loader: str,
):
    _retriever = cbrkit.helpers.load_callable(retriever)
    _loader = cbrkit.helpers.load_callable(loader)

    cases = _loader(cases_path)
    queries = _loader(queries_path)
    # TODO: change this again to include all queries!
    result = cbrkit.retrieval.apply_queries(cases, {"W40": queries["W40"]}, _retriever)

    dump_result(result, output_path)


@app.command()
def evaluate_run(
    result_path: Path,
    baseline_path: Path,
    k: Annotated[list[int], Option(default_factory=list)],
    max_qrel: int | None = None,
    min_qrel: int = 0,
):
    with baseline_path.open("rb") as fp:
        baseline: cbrkit.retrieval.Result = pickle.load(fp)

    with result_path.open("rb") as fp:
        result: cbrkit.retrieval.Result = pickle.load(fp)

    metrics = cbrkit.eval.retrieval_step(
        cbrkit.eval.retrieval_step_to_qrels(baseline.final_step, max_qrel, min_qrel),
        result.final_step,
        metrics=cbrkit.eval.generate_metrics(ks=k),
    )

    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")


@app.command()
def evaluate_qrels(
    result_path: Path,
    k: Annotated[list[int], Option(default_factory=list)],
):
    with result_path.open("rb") as fp:
        result: cbrkit.retrieval.Result = pickle.load(fp)

    qrels: dict[str, dict[str, int]] = {
        key: entry.query.value["qrels"]
        for key, entry in result.first_step.queries.items()
    }

    metrics = cbrkit.eval.retrieval_step(
        qrels,
        result.final_step,
        metrics=cbrkit.eval.generate_metrics(ks=k),
    )

    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
