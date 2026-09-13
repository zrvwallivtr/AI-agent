from src.config import models

from src.agent.models.ollama import ollama_clt
from src import logger


app_log = logger.app_logger(f"{__name__}.app")

EMBED_MODEL = models.EMBED_MODEL


def embedding_content(cont: str) -> tuple[str, list[float], int] | None:
    """Generate embedding from given texts."""
    try:
        response = ollama_clt.embed(model=EMBED_MODEL, input=cont)
        embdings = response["embeddings"][0]
        if not embdings:
            app_log.warning("Error: Model failed to generate vector embedding")
            return

        cont_tkns = response.get("prompt_eval_count", 0)
        return cont, embdings, cont_tkns

    except Exception as e:
        app_log.warning("Error: %s", e)
        return
