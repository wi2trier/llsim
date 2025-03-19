import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("llsim").setLevel(logging.DEBUG)
logging.getLogger("cbrkit.sim.graphs").setLevel(logging.DEBUG)
logging.getLogger("cbrkit.retrieval").setLevel(logging.DEBUG)
logging.getLogger("cbrkit.synthesis.providers").setLevel(logging.DEBUG)
