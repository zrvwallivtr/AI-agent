from pathlib import Path
from tokenizers import Tokenizer

from src.config import models
from src.config import files_and_directories as fls_n_dir

from src.agent import ollama
from src import models_database
from src.logger import app_logger


MODEL_NAME_TO_HF_TOKENIZER = {
    # ==============        ===============================
    # | MODEL NAME |        | HUGGINGFACE REPOSITORY NAME |
    # ==============        ===============================

    # === LLMS ============================================
    # Mistral
    "mistral":              "mistralai/Mistral-7B-Instruct-v0.2",
    "dolphin-mistral":      "mistralai/Mistral-7B-Instruct-v0.2",
    "ministral_3b":         "ministral/Ministral-3b-instruct",

    # Llama
    "llama3.1":             "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2":             "meta-llama/Llama-3.2-3B-Instruct",

    # Gemma
    "gemma3":               "google/gemma-3-4b-it",

    # Phi
    "dolphin-phi":          "microsoft/phi-2",

    # === EMBEDDING MODELS ================================
    # Nomic
    "nomic-embed-text":     "nomic-ai/nomic-embed-text-v1.5",
}


app_log = app_logger(__name__)

TOKENIZERS_DIR      = fls_n_dir.TOKENIZERS_DIR
FALLBACK_TOKENIZER  = models.FALLBACK_TOKENIZER
MODEL_MAX_TOKENS    = models.MODEL_MAX_TOKENS
MODEL_MAX           = models_database.MODEL_MAX

ollama_clt = ollama.ollama_clt


# =============================================================
# INSTALLATION
# =============================================================

def install_tokenizers():
    """Install tokenizers from model list, internet required."""
    # === FALLBACK TOKENIZER INSTALL ======================
    fb_tknizr_path = TOKENIZERS_DIR / f"{FALLBACK_TOKENIZER}.json"

    if fb_tknizr_path.exists():
        app_log.debug(
            "Model '%s' tokenizer already saved in '%s' as fallback tokenizer",
            FALLBACK_TOKENIZER,
            fb_tknizr_path
        )

    else:
        app_log.info("Installing fallback tokenizer '%s'", FALLBACK_TOKENIZER)
        try:
            fb_tknizr = Tokenizer.from_pretrained(FALLBACK_TOKENIZER)
            fb_tknizr.save(str(fb_tknizr_path))
            app_log.info(
                "Model '%s' tokenizer has been installed and saved to '%s' as fallback tokenizer",
                FALLBACK_TOKENIZER,
                fb_tknizr_path
            )

        except Exception as e:
            app_log.warning(
                "Failed to install model '%s' tokenizer as fallback tokenizer: %s. Skipping",
                FALLBACK_TOKENIZER,
                e
            )

    # === TOKENIZER INSTALL FROM LIST =====================
    installed_models = ollama_clt.list().models

    count = 0
    for installed_model in installed_models:
        count += 1
        model_name = installed_model.model
        base_name = model_name.split(":")[0]
        app_log.info(
            "[%d/%d] Installing model '%s' tokenizer", model_name, count, len(installed_models)
        )

        # HUGGINGFACE REPO NOT FOUND
        hf_repo = MODEL_NAME_TO_HF_TOKENIZER.get(base_name)
        if hf_repo is None:
            app_log.warning(
                "No known tokenizer mapping for model '%s' (base '%s'). Skipping",
                model_name,
                base_name
            )
            continue

        # TOKENIZER ALREADY INSTALLED
        tknizr_path = TOKENIZERS_DIR / f"{model_name}.json"
        if tknizr_path.exists():
            app_log.info(
                "Model '%s' tokenizer already installed. Skipping", model_name
            )
            continue

        try:
            tknizr = Tokenizer.from_pretrained(hf_repo)
            tknizr.save(str(tknizr_path)) # Writes local file
            app_log.info("Model '%s' tokenizer saved to '%s'", model_name, tknizr_path)

        except Exception as e:
            app_log.error(
                "Failed to install '%s' (repo=%s) tokenizer: %s. Skipping",
                model_name,
                hf_repo,
                e
            )


# =============================================================
# TOKENIZER
# =============================================================

class Tknizr:
    def __init__(self, model: str):
        self.model          = model
        self.load_tknizr    = self._load_tokenizer()

        # === SET MODEL MAX TOKENS ====================================
        # User specified max tokens
        if MODEL_MAX_TOKENS:
            self.model_max_tkns = MODEL_MAX_TOKENS
            app_log.debug(
                "Imported user specified max token limit for model '%s' tokenizer",
                self.model
            )

        # Fallback to preset list
        elif self.model in models_database.MODEL_MAX:
            self.model_max_tkns = MODEL_MAX[self.model]
            app_log.debug(
                "Max token limit not specified by user. Loaded preset limits for model '%s' tokenizer",
                self.model
            )

        # Neither option is available
        else:
            app_log.warning(
                "Model '%s' not in 'MODEL_MAX' and no max tokens provided. Add it to 'MODEL_MAX' or set 'chat_max tokens' in '~/.agent_app/config.toml'.",
                self.model
            )
            self.model_max_tkns = None


    # =========================================================
    # LOAD INSTALLED TOKENIZER
    # =========================================================

    def _load_tokenizer(self) -> Tokenizer | None:
        """Load local installed tokenizer."""
        file = TOKENIZERS_DIR / f"{self.model}.json"

        # === LOAD PRE-INSTALLED TOKENIZER ===================================
        app_log.debug("Loading model '%s' tokenizer from file '%s'", self.model, file)
        try:
            tknizr = Tokenizer.from_file(str(file))
            app_log.debug("Model '%s' tokenizer loaded from '%s'", self.model, file)
            return tknizr

        except Exception as e:

            # === LOAD PRE-INSTALLED FALLBACK TOKENIZER ======================
            try:
                tknizr = Tokenizer.from_file(str(TOKENIZERS_DIR / f"{FALLBACK_TOKENIZER}.json"))
                app_log.warning(
                    "Failed to load model '%s' tokenizer: %s. Falling back to model '%s' tokenizer",
                    self.model,
                    e,
                    FALLBACK_TOKENIZER
                )
                return tknizr

            # === DISABLE TOKENIZER FEATURE ==================================
            except Exception as fallback_err:
                app_log.warning(
                    "Failed to load model '%s' fallback tokenizer: %s. Tokenizer feature disabled",
                    FALLBACK_TOKENIZER,
                    fallback_err
                )
                return


    def count_string_tokens(self, text: str) -> int | None:
        """Counts tokens in string."""
        if not self.load_tknizr:
            app_log.warning("Unable to count tokens: Tokenizer feature has been disabled")
            return
        app_log.debug("Counting tokens from the provided text")
        return len(self.load_tknizr.encode(text))


    def count_history_tokens(self, msgs: list[dict]) -> int | None:
        """Calculates total token weight."""
        if not self.load_tknizr:
            app_log.warning("Unable to count tokens: Tokenizer feature has been disabled")
            return

        tol_tkns = 0

        count = 0
        for msg in msgs:
            count += 1
            app_log.debug("Retrieving message (%d/%d) from the message list", count, len(msgs))
            # Account for message content text
            add_tkns = self.count_string_tokens(msg.get("content", ""))
            if not add_tkns:
                return
            tol_tkns += add_tkns

            # For tags (e.g. 'assistant'/'user')
            tol_tkns += 4

        tol_tkns += 3 # Assistant indicator tokens
        app_log.debug("All messages has been processed: Total token counts = %d tokens", tol_tkns)
        return tol_tkns


    def encode_text(self, txt: str) -> list[int] | None:
        """Encode text using the loaded tokenizer."""
        if not self.load_tknizr:
            app_log.warning("Unable to encode text: Tokenizer feature has been disabled")
            return
        app_log.debug("Encoding from provided text")
        return self.load_tknizr.encode(txt)
