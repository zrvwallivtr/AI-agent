from models_database import EMB_MODEL_DIMENSION
import ollama
import hashlib

from src import config


class Embed:
    def __init__(self):
        self.model      = config.EMBED_MODEL
        self.emb_dim    = EMB_MODEL_DIMENSION[self.model]

    def embedding_content(self, cont: str) -> tuple[str, list[float]]:
        """Generate embedding from given texts."""
        try:
            response = ollama.embed(model=self.model, input=cont)
            embeddings = response["embeddings"][0]
            if not embeddings:
                return "Error: Model failed to generate vector embedding", []
            return cont, embeddings

        except Exception as e:
            return f"Error: {e}", []
