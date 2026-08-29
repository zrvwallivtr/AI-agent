from tokenizers import Tokenizer

from src import models_database
from src.logger import app_logger


app_log = app_logger(__name__)


class Tokens:
    def __init__(self, model: str, model_max_tokens: int | None = None):
        self.model = model

        # === SET MODEL MAX TOKENS ====================================

        # User specified max tokens
        if model_max_tokens is not None:
            self.model_max_tokens = model_max_tokens

        # Fallback to predefined list
        elif self.model in models_database.MODEL_MAX:
            self.model_max_tokens = models_database.MODEL_MAX[model]

        # Neither option is available
        else:
            raise ValueError(
                f"Model '{model}' not in MODEL_MAX and no max tokens provided. "
                f"Add it to MODEL_MAX or set chat_max tokens in config.toml."
            )

        # === ENCODER =================================================

        try:
            # Exact token vocabulary mapped to specific model name
            self.tokenizer = Tokenizer.from_pretrained(self.model)
            app_log.debug("Loaded exact token encoding for model: %s", self.model)

        except Exception as e:
            # Fall back to a common general-purpose tokenizer
            self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
            app_log.debug(
                "Exact token encoding not found for model '%s': %s. Falling back to 'bert-base-uncased'",
                self.model,
                e
            )


    def estimate(self, text: str) -> int:
        """Estimates token count base on charater length (4 chars ~ 1 token)."""
        return len(text) // 4


    def check_fit(self, text: str, reserve: int = 50) -> bool:
        """Check if text fits within the model's bounds."""
        return (self.estimate(text) + reserve) <= self.model_max_tokens


    # =================================
    # Count tokens
    # =================================

    def count_string_tokens(self, text: str) -> int:
        """Counts tokens in string."""
        if not text:
            # logger.warning("Cannot count tokens in string: No text provided")
            return 0
        return len(self.encoder.encode(text))


    def count_history_tokens(self, msgs: list[dict]) -> int:
        """Calculates total token weight."""
        total_tokens = 0

        for msg in msgs:
            # Account for message content text
            total_tokens += self.count_string_tokens(msg.get("content", ""))

            # For tags (e.g. 'assistant'/'user')
            total_tokens += 4

        total_tokens += 3 # Assistant indicator tokens
        return total_tokens
