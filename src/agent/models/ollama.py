import ollama
import subprocess

from src.config import models

from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

OLLAMA_HOST = models.OLLAMA_HOST

ollama_clt = ollama.Client(host=OLLAMA_HOST)


def _is_model_installed(model: str) -> bool:
    """Return True if model is already installed."""
    inst = [m["model"] for m in ollama_clt.list()["models"]]

    if model in inst:
        app_log.debug("Model '%s' has already been installed locally", model)
        return True
    app_log.info("Model '%s' has not been installed", model)
    return False


def _ollama_pull_via_docker(model: str, vol_name: str = "agent_app_ollama_models") -> bool:
    """
    Pull model using temporary Docker container with internet access,
    writing into the same volume the isolated ollama service mounts.
    """
    app_log.info("Pulling '%s' via host Docker", model)
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{vol_name}:/root/.ollama",
                "--dns", "1.1.1.1",
                "--entrypoint", "sh",
                "ollama/ollama",
                "-c",
                f"ollama serve & sleep 2 && ollama pull {model}"
            ],
            check=True,
            capture_output=False
        )
        app_log.info("Model '%s' has been installed", model)
        return result.returncode == 0

    except subprocess.CalledProcessError as e:
        app_log.warning("Failed to pull model '%s' vai host Docker: %s", model, e)
        return False

    except FileNotFoundError:
        app_log.warning("Failed to pull model: Docker CLI not found on host")
        return False


def ollama_pull_model(model: str) -> bool:
    """Pull an Ollama model, print progress in the terminal."""
    if _is_model_installed(model):
        return True

    try:
        return _ollama_pull_via_docker(model)

    except Exception as e:
        app_log.warning("Failed to pull model '%s': %s", model, e)
        return False
