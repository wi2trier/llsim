from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cbrkit
from cbrkit.typing import BatchSimFunc

type NodeData = Mapping[str, Any]


def load(path: Path) -> cbrkit.typing.Casebase[str, Mapping[str, Any]]:
    return cbrkit.loaders.file(path)


SIM: BatchSimFunc[Mapping[str, Any], cbrkit.typing.Float] = cbrkit.sim.attribute_value(
    {
        "price": cbrkit.sim.numbers.linear(100000),
        "year": cbrkit.sim.numbers.linear(100),
        "manufacturer": cbrkit.sim.cache(
            cbrkit.sim.taxonomy.build(
                Path("./data/taxonomies/manufacturer.json"),
                cbrkit.sim.taxonomy.paths(),
            )
        ),
        "make": cbrkit.sim.cache(cbrkit.sim.strings.levenshtein()),
        "fuel": cbrkit.sim.cache(cbrkit.sim.strings.levenshtein()),
        "miles": cbrkit.sim.numbers.linear(500000),
        "title_status": cbrkit.sim.cache(cbrkit.sim.strings.levenshtein()),
        "transmission": cbrkit.sim.cache(cbrkit.sim.strings.levenshtein()),
        "drive": cbrkit.sim.cache(cbrkit.sim.strings.levenshtein()),
        "type": cbrkit.sim.cache(cbrkit.sim.strings.levenshtein()),
        "paint_color": cbrkit.sim.cache(cbrkit.sim.strings.levenshtein()),
    },
)

RETRIEVER = cbrkit.retrieval.build(SIM, multiprocessing=False)
