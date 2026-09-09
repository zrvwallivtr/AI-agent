import ollama
import subprocess

from config.models import OLLAMA_HOST 


olma_client = ollama.Client(host=OLLAMA_HOST)


def _is_model_installed(model: str) -> bool:
    """Return True if model is already installed."""
    inst = [m["model"] for m in olma_client.list()["models"]]

    if model in inst:
        return True
    return False


def _ollama_pull_via_docker(model: str, vol_name: str = "agent_app_ollama_models") -> bool:
    """
    Pull model using temporary Docker container with internet access,
    writing into the same volume the isolated ollama service mounts.
    """
    try:
        print(f"Pulling '{model}' via host Docker...")
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
        return result.returncode == 0

    except subprocess.CalledProcessError as e:
        print(f"\nFailed to pull model '{model}' vai host Docker: {e}")
        return False

    except FileNotFoundError:
        print("\nFailed to pull model: Docker CLI not found on host")
        return False


def ollama_pull_model(model: str) -> bool:
    """Pull an Ollama model, print progress in the terminal."""
    if _is_model_installed(model):
        return True

    try:
        return _ollama_pull_via_docker(model)

    except Exception as e:
        print(f"\nFailed to pull model '{model}': {e}")
        return False
