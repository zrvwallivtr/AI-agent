import psycopg2
import ollama
from typing_extensions import Doc
from pathlib import Path

from src import config

from src.agent.chat_logs import ChatLogs
from src.agent.core import Agent
# from src.tools.doc_knowledge_base import DocKnowledgeBase

conn = psycopg2.connect(
    dbname=config.DBNAME,
    user=config.USER,
    password=config.PASSWORD,
    host=config.HOST,
    port=config.PORT
)
cur = conn.cursor()

def _del_sess(sess_name: str | None = None) -> str:
    """
    Delete session related files. If 'session'
    is not specified, delete default session 
    related files.
    """
    chat_logs = ChatLogs(sess_name=sess_name)
    sess_id = chat_logs.get_sess_id()
    if not sess_id:
        return f"Failed to delete session: Session '{sess_name}' does not exist"

    # Clear session chat logs
    response, has_del_chat = chat_logs.clear_sess_chat_logs()
    if response:
        print(response)

    # file_reader = DocKnowledgeBase(session)

    # # Clear all files in dropbox
    # clear_session_dropbox_response  = file_reader.clear_session_dropbox()
    # print(clear_session_dropbox_response) if clear_session_dropbox_response else None

    # DELETE SESSION
    # Ensure all session related contents are cleared
    if has_del_chat:
        cur.execute(
            """
            DELETE FROM chat_sessions
            WHERE session_id = %s;
            """,
            (sess_id,)
        )
        conn.commit()
        del_count = cur.rowcount
        conn.commit()
        if del_count == 0:
            return f"Failed to delete session: session={sess_name}"
        return f"Session deleted: session={sess_name}"

    # Error message
    del_chat_err = ""
    if not has_del_chat:
        del_chat_err = f"- Unable to clear session chat logs\n"
    err_msg = (
        f"Failed to delete session:\n"
        f"\tSession: {sess_name}\n"
        f"\tError:\n"
        f"\t{del_chat_err}\n"
    )
    return err_msg
    

# =========================================================
# GENERAL
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
        sess_name: str | None = None,
        project: str | None = None
    ):
        """Ask question only, not flags."""
        agent   = Agent(model=model, sess_name=sess_name, project=project)
        answer  = agent.ask(prompt=prompt)
        return

    @staticmethod
    def reset_default():
        """Clear active conversation and chat history."""
        _del_sess()

    @staticmethod
    def installed_models():
        model_list = ollama.list()

        print("Installed models:")
        for model in model_list.get("models", []):
            print(f"- {model['model']}")


# =========================================================
# SESSION CLASS
# =========================================================

class Session:
    def __init__(self, sess_name: str):
        self.sess_name  = sess_name

    def create_session(
        self,
        model: str,
        prompt: str | None = None
    ) -> str | None:
        """Create session and response to user question."""
        self.chat_logs  = ChatLogs(sess_name=self.sess_name)

        if not prompt:
            return f"New session created: session={self.sess_name}"
        return General.question(prompt=prompt, model=model, sess_name=self.sess_name)

    def delete_session(self) -> str:
        """Delete session's related files."""
        response = _del_sess(self.sess_name)
        return response

    def list_session(self):
        """List all user created sessions, do not display session's chat history."""
        self.chat_logs  = ChatLogs()
        sess_dict = self.chat_logs.get_all_existing_sess_metadata()

        print("AVAILABLE SESSION(S)")
        print("--------------------")
        print("CREATED AT\t\t\t\tSESSION NAME")
        for sess in sess_dict:
            print(f"{sess_dict[sess]["created_at"]}\t{sess_dict[sess]["session_name"]}")
        print("\n")

# =========================================================
# File class
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
        agent   = Agent(model=model, sess_name=self.session, project=project)
        answer  = agent.ask(
            prompt=prompt,
            enable_attachments=True,
            file_paths=file_paths
        )
        return
