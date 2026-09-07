import ollama

from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")


def validate_model(model: str):
    """Check if model is installed via Ollama."""
    try:
        # Fetch all downloaded models
        local_models = [m['model'] for m in ollama.list().get('models', [])]

        # Check match
        unknown_models = []
        if model not in local_models:
            unknown_models += model
            raise ValueError(
                f"Error: Unknown or unavailable model '{model}'."
                f"Run 'ollama pull {model}' to install model."
            )
        known_models = set(local_models) - set(unknown_models)
        app_log.debug("Detected %s known model(s) and %s unknown model(s)", known_models, unknown_models)

    except Exception as e:
        # If ollama is down
        if isinstance(e, ValueError):
            raise e
        raise RuntimeError(f"Error: Could not connect to local Ollama service '{e}'")
