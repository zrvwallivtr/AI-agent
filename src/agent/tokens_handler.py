import tiktoken


MODEL_MAX = {
    # Conversational / Generation Models
    "gemma3:1b":            4096,
    "ministral_3b:latest":  4096,

    # Embedding Models
    "nomic-embed-text": 2048,
    "embeddinggemma":   2048,
    "all-minilm":       512,
}


# chat_tokens = Tokens(MODEL, MODEL_MAX_TOKENS)
# pm_tokens = Tokens(PM_MODEL, PM_MAX_TOKENS)

class Tokens:
    def __init__(self, model: str, max_tokens: int | None = None):
        self.model = model

        # ========================================
        # Max tokens
        # ========================================

        # User specified max tokens
        if max_tokens is not None:
            self.max_tokens = max_tokens

        # Fallback to predefined list
        elif self.model in MODEL_MAX:
            self.max_tokens = MODEL_MAX[model]

        # Neither option is available
        else:
            raise ValueError(
                f"Model '{model}' not in MODEL_MAX and no max tokens provided. "
                f"Add it to MODEL_MAX or set chat_max tokens in config.toml."
            )

        # ========================================
        # Encoder
        # ========================================

        try:
            # Looks up exact token vocabulary mapped to specific model name
            self.encoder = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # If unknown, fall back to 'cl100k_base'
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def estimate(self, text: str) -> int:
        """Estimates token count base on charater length (4 chars ~ 1 token)."""
        return len(text) // 4

    def check_fit(self, text: str, reserve: int = 50) -> bool:
        """Check if text fits within the model's bounds."""
        return (self.estimate(text) + reserve) <= self.max_tokens

    # =================================
    # Count tokens
    # =================================

    def count_string_tokens(self, text: str) -> int:
        """Counts tokens in string."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def count_history_tokens(self, messages: list[dict]) -> int:
        """Calculates total token weight."""
        total_tokens = 0

        for message in messages:
            # Account for message content text
            total_tokens += self.count_string_tokens(message.get("content", ""))

            # For tags (e.g. 'assistant'/'user')
            total_tokens += 4

        total_tokens += 3 # Assistant indicator tokens
        return total_tokens
