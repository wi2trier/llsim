import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("cbrkit.sim.graphs").setLevel(logging.DEBUG)
logging.getLogger("cbrkit.sim.attribute_value").setLevel(logging.DEBUG)
logging.getLogger("cbrkit.retrieval").setLevel(logging.DEBUG)
