import psycopg2
from typing_extensions import Doc
from pathlib import Path

from src.config import postgres

from src.agent import ollama
from src.agent import chat_logs
from src.core import Agent
from src.rag import knowledge_base
from src import logger
from src.cli import interface


conn = postgres.conn
cur = conn.cursor()

app_log = logger.app_logger(f"{__name__}.app")

ChatLogs        = chat_logs.ChatLogs
KnowledgeBase   = knowledge_base.KnowledgeBase


def del_sess(sess_name: str | None = None) -> None:
    """
    Delete session related chat logs and database contents. If 'session'
    is not specified, delete default session related contents.
    """
    chat_logs   = ChatLogs(conn=conn, sess_name=sess_name)
    kw_bs       = KnowledgeBase(conn=conn, chat_logs=chat_logs, sess_name=sess_name)

    sess_id = chat_logs.get_sess_id()
    if not sess_id:
        app_log.warning("Failed to delete session: Session '%s' does not exist", sess_name)
        return

    # Clear session chat logs
    has_del_chat = chat_logs.clear_sess_chat_logs()

    # Clear session knowledge base
    response = kw_bs.clear_sess_kw_bs("document")
    # response = kw_bs.clear_sess_kw_bs("web_search")

    # Delete session - ensure all session related contents are cleared
    if not has_del_chat:
        app_log.warning("Failed to clear session '%s' chat logs", sess_name)

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
        app_log.warning("Failed to delete session: Session '%s' does not exist", sess_name)
        return
    app_log.info("Session '%s' and related data deleted from database", sess_name)
    return
