from src.config.models import EMBED_MODEL
from src.agent.models.ollama import olma_client
from models_database import EMB_MODEL_DIMENSION


class Embed:
    def __init__(self):
        self.model      = EMBED_MODEL
        self.emb_dim    = EMB_MODEL_DIMENSION[self.model]

    def embedding_content(self, cont: str) -> tuple[str, list[float], int]:
        """Generate embedding from given texts."""
        try:
            response = olma_client.embed(model=self.model, input=cont)
            embeddings = response["embeddings"][0]
            if not embeddings:
                return "Error: Model failed to generate vector embedding", [], 0

            cont_tkns = response.get("prompt_eval_count", 0)

            return cont, embeddings, cont_tkns

        except Exception as e:
            return f"Error: {e}", [], 0
