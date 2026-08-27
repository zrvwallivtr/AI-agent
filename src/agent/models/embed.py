import ollama
import hashlib

from src import config


class Embed:
    def __init__(self):
        self.emb_model = config.EMBED_MODEL

    def embedding_content(self, cont: str) -> tuple[str, list[float]]:
        """Generate embedding from given texts."""
        try:
            response = ollama.embed(model=self.emb_model, input=cont)
            embeddings = response["embeddings"][0]
            if not embeddings:
                return "Error: Model failed to generate vector embedding", []
            return cont, embeddings

        except Exception as e:
            return f"Error: {e}", []
