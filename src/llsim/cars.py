import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cbrkit
from cbrkit.typing import BatchSimFunc
from pydantic import ValidationError

type NodeData = Mapping[str, Any]
logger = logging.getLogger(__name__)


def load(path: Path) -> cbrkit.typing.Casebase[str, Mapping[str, Any]]:
    return cbrkit.loaders.file(path)


try:
    MANUFACTURER_SIM = cbrkit.sim.taxonomy.build(
        Path("./data/taxonomies/manufacturer.json"),
        cbrkit.sim.taxonomy.paths(),
    )
except ValidationError:
    MANUFACTURER_SIM = cbrkit.sim.generic.equality()
    logger.warning("Failed to load taxonomy for manufacturer. Using equality instead.")


SIM: BatchSimFunc[Mapping[str, Any], cbrkit.typing.Float] = cbrkit.sim.attribute_value(
    {
        "price": cbrkit.sim.numbers.linear(100000),
        "year": cbrkit.sim.numbers.linear(100),
        "manufacturer": MANUFACTURER_SIM,
        "make": cbrkit.sim.strings.levenshtein(),
        "fuel": cbrkit.sim.strings.levenshtein(),
        "miles": cbrkit.sim.numbers.linear(500000),
        "title_status": cbrkit.sim.strings.levenshtein(),
        "transmission": cbrkit.sim.strings.levenshtein(),
        "drive": cbrkit.sim.strings.levenshtein(),
        "type": cbrkit.sim.strings.levenshtein(),
        "paint_color": cbrkit.sim.strings.levenshtein(),
    },
)


def Retriever():
    return cbrkit.retrieval.build(SIM, multiprocessing=False)
