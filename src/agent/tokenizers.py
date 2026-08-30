import ollama
from pathlib import Path
from tokenizers import Tokenizer

from src.config.files_and_directories import TOKENIZERS_DIR, _cfg
from src.config.models import FALLBACK_TOKENIZER
from src import models_database
from src.logger import app_logger


MODEL_NAME_TO_HF_TOKENIZER = {
    # Mistral
    "mistral":              "mistralai/Mistral-7B-Instruct-v0.2",
    "dolphin-mistral":      "mistralai/Mistral-7B-Instruct-v0.2",
    "ministral_3b":         "ministral/Ministral-3b-instruct",

    "nomic-embed-text":     "nomic-ai/nomic-embed-text-v1.5",

    # Llama
    "llama3.1":             "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2":             "meta-llama/Llama-3.2-3B-Instruct",

    # Gemma
    "gemma3":               "google/gemma-3-4b-it",

    # Phi
    "dolphin-phi":          "microsoft/phi-2",
}

app_log = app_logger(__name__)


def install_tokenizers():
    """Install tokenizers from model list, internet required."""
    # === FALLBACK TOKENIZER INSTALL ===================

    fb_tknizr_path = TOKENIZERS_DIR / f"{FALLBACK_TOKENIZER}.json"

    if fb_tknizr_path.exists():
        app_log.info(
            "'%s' tokenizer already saved in '%s' as fallback tokenizer",
            FALLBACK_TOKENIZER,
            fb_tknizr_path
        )
    else:
        try:
            print(f"Installing fallback tokenizer '{FALLBACK_TOKENIZER}'...")
            fb_tknizr = Tokenizer.from_pretrained(FALLBACK_TOKENIZER)
            fb_tknizr.save(str(fb_tknizr_path))
            app_log.info(
                "'%s' tokenizer saved to '%s' as fallback tokenizer",
                FALLBACK_TOKENIZER,
                fb_tknizr_path
            )
            print(f"'{FALLBACK_TOKENIZER}' tokenizer saved to {fb_tknizr_path} as fallback tokenizer")

        except Exception as e:
            app_log.error(
                "Failed to install fallback tokenizer (model '%s'): %s. Skipping",
                FALLBACK_TOKENIZER,
                e
            )

    # === TOKENIZER INSTALL FROM LIST ==================

    installed_models = ollama.list().models

    for installed_model in installed_models:
        model_name = installed_model.model
        base_name = model_name.split(":")[0]

        hf_repo = MODEL_NAME_TO_HF_TOKENIZER.get(base_name)
        if hf_repo is None:
            app_log.warning(
                "No known tokenizer mapping for model '%s' (base '%s'). Skipping",
                model_name,
                base_name
            )
            print(f"No known tokenizer mapping for model '{model_name}'. Skipping")
            continue

        tknizr_path = TOKENIZERS_DIR / f"{model_name}.json"
        if tknizr_path.exists():
            app_log.info("Tokenizer already installed for '%s'. Skipping", model_name)
            continue

        try:
            print(f"Installing '{model_name}' tokenizer...")
            tknizr = Tokenizer.from_pretrained(hf_repo)
            tknizr.save(str(tknizr_path)) # Writes local file
            print(f"'{model_name}' tokenizer saved to '{tknizr_path}'")
            app_log.info("%s tokenizer saved to '%s'", model_name, tknizr_path)

        except Exception as e:
            app_log.error(
                "Failed to install '%s' (repo=%s) tokenizer: %s. Skipping",
                model_name,
                hf_repo,
                e
            )


class Tknizr:
    def __init__(self, model: str, model_max_tokens: int | None = None):
        self.model = model
        self.tknizr = self._load_tokenizer()

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
            self.model_max_tokens = None


    def _load_tokenizer(self) -> Tokenizer | None:
        """Load local installed tokenizer."""
        file = TOKENIZERS_DIR / f"{self.model}.json"

        try:
            # Load pre-installed tokenizer
            tknizr = Tokenizer.from_file(str(file))
            app_log.debug("'%s' tokenizer loaded from '%s'", self.model, file)
            return tknizr

        except Exception as e:
            try:
                # Load pre-installed fallback tokenizer
                tknizr = Tokenizer.from_file(str(TOKENIZERS_DIR / f"{FALLBACK_TOKENIZER}.json"))
                app_log.warning(
                    "Failed to load '%s' tokenizer: %s. Falling back to '%s' tokenizer",
                    self.model,
                    e,
                    FALLBACK_TOKENIZER
                )
                return tknizr

            except Exception as fallback_err:
                # Disable tokenizer feature
                app_log.warning(
                    "Failed to load '%s' fallback tokenizer: %s. Tokenizer feature disabled",
                    FALLBACK_TOKENIZER,
                    fallback_err
                )
                tknizr = None


    # def estimate(self, text: str) -> int:
    #     """Estimates token count base on charater length (4 chars ~ 1 token)."""
    #     return len(text) // 4


    # def check_fit(self, text: str, reserve: int = 50) -> bool:
    #     """Check if text fits within the model's bounds."""
    #     return (self.estimate(text) + reserve) <= self.model_max_tokens


    def count_string_tokens(self, text: str) -> int | None:
        """Counts tokens in string."""
        if self.tknizr:
            if not text:
                # logger.warning("Cannot count tokens in string: No text provided")
                return 0
            return len(self.tknizr.encode(text))


    def count_history_tokens(self, msgs: list[dict]) -> int | None:
        """Calculates total token weight."""
        if self.tknizr:
            total_tkns = 0

            for msg in msgs:
                # Account for message content text
                add_tkns = self.count_string_tokens(msg.get("content", ""))
                if not add_tkns:
                    return
                total_tkns += add_tkns

                # For tags (e.g. 'assistant'/'user')
                total_tkns += 4

            total_tkns += 3 # Assistant indicator tokens
            return total_tkns
