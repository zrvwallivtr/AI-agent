from typing_extensions import Doc
import ollama
from pathlib import Path

from src import config
from src.agent.chat import Chat
from src.agent.core import Agent
from src.tools.doc_knowledge_base import DocKnowledgeBase


def _delete_session_files(session: str | None = None):
    """
    Delete session related files. If 'session'
    is not specified, delete default session 
    related files.
    """
    chat        = Chat(session)
    file_reader = DocKnowledgeBase(session)

    # Clear active conversation
    clear_active_conv_response = chat.delete_active_conv()
    print(clear_active_conv_response) if clear_active_conv_response else None

    # Clear chat history
    clear_chat_history_response = chat.delete_chat_history()
    print(clear_chat_history_response) if clear_chat_history_response else None

    # Clear all files in dropbox
    clear_session_dropbox_response  = file_reader.clear_session_dropbox()
    print(clear_session_dropbox_response) if clear_session_dropbox_response else None
    

# =========================================================
# General
#
# Main tools:
# - call Agent to answer question
# - delete default chat file in DEFAULT_PATH
# =========================================================

class General:
    @staticmethod
    def question(
        model: str,
        prompt: str,
        session: str | None = None,
        project: str | None = None
    ):
        """Ask question only, not flags."""
        agent   = Agent(model=model, session=session, project=project)
        answer  = agent.ask(prompt=prompt)
        return

    @staticmethod
    def reset_default():
        """Clear active conversation and chat history."""
        _delete_session_files()

    @staticmethod
    def installed_models():
        model_list = ollama.list()

        print("Installed models:")
        for model in model_list.get("models", []):
            print(f"- {model['model']}")


# =========================================================
# Session class
#
# Main tools:
# - call Agent to answer question
# - delete default chat file in DEFAULT_PATH
# =========================================================

class Session:
    def __init__(self, session: str):
        self.session    = session
        self.chat       = Chat(session)

    def create_session(self, model: str, prompt: str | None = None) -> None:
        """
        If no question asked:
        - Create session.

        If question was asked:
        - Create session, model response to prompt.
        """
        if self.chat.active_conv_path.exists() or self.chat.chat_history_path.exists():
            print(f"Error: Session {self.session} already exist.")
            return

        if not prompt:
            self.chat.save()
            print(f"Created new session: {self.session}")
            return

        else:
            return General.question(prompt=prompt, model=model, session=self.session)

    def delete_session(self):
        """Delete session's related files."""
        _delete_session_files(self.session)

    @staticmethod
    def list_session():
        """List all user created sessions, do not display session's chat history."""
        exclude         = {config.DEFAULT_PATH.name, config.DEFAULT_CHAT_HISTORY_PATH.name}
        all_sessions    = [
            p.name.replace(".json", "")
            for p in config.CHATS_DIR.glob("*.json")
            if not p.name.endswith("_chat_history.json")
            if p.name not in exclude
        ]

        print("Available session:")
        for sessions in all_sessions:
            print(sessions)
        print("\n")

# =========================================================
# File class
#
#
# =========================================================

class File:
    def __init__(self, session: str):
        self.session = session

    def files_with_prompt(
        self,
        model: str,
        prompt: str,
        file_paths: list[Path],
        project: str | None = None
    ):
        """Combine contents in file(s) with user prompt."""
        agent   = Agent(model=model, session=self.session, project=project)
        answer  = agent.ask(
            prompt=prompt,
            enable_attachments=True,
            file_paths=file_paths
        )
        return
