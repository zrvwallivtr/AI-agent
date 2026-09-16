import argparse
from pathlib import Path
from rich.console import Console
from agent import tokenizers
from rich_argparse import RichHelpFormatter

from src.config import files_and_directories as files_n_dir
from src.config import models
from src.config import postgres

from src.cli import interface
from src import logger


MODEL = models.MODEL


def build_parser() -> None:
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, description="Local RAG agent")

    # === GENERAL ===================================================================
    parser.add_argument("prompt", nargs="?", help="Prompt agent")
    parser.add_argument("--delete-default-session", action="store_true", help="Reset default session")
    parser.add_argument("--dashboard", action="store_true", help="Show the status menu")

    # === SETTINGS ==================================================================
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show debug-level logs")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--width", type=int, default=None, help="Force console width (useful for non-TTY output)")

    # === MODELS ====================================================================
    parser.add_argument("--model", "-m", default=MODEL, metavar=("MODEL_NAME"), help=f"Select model (defualt: {MODEL})")
    parser.add_argument("--list-models", "-lm", action="store_true", help="List all installed ollama models")
    parser.add_argument("--install-model", metavar=("MODEL_NAME"), help=f"Install specified model")

    # === SESSION FLAGS =============================================================
    parser.add_argument("--session", "-s", default=None, metavar=("SESSION_NAME"), help="Continue a selected session")
    parser.add_argument("--list-session", "-ls", action="store_true", help="List all existing session")
    parser.add_argument("--new-session", default=None, metavar=("SESSION_NAME"), help="Create a new session and prompt agent (optional)")
    parser.add_argument("--delete-session", default=None, metavar=("SESSION_NAME"), help="Delete a specified session")

    # === ATTACHMENTS ===============================================================
    parser.add_argument("--attachments", "-a", nargs="+", type=Path, default=None, metavar=("FILE_PATH"), help=f"Read specified attachments (It must be uploaded to {files_n_dir.UPLOAD_DIR})")
    parser.add_argument("--list-attachments", "-la", action="store_true", help="List all uploaded attachments in the database")

    # === INITIALISE TOKENIZERS =====================================================
    parser.add_argument("--install-tokenizers", action="store_true", help="Install tokenizers for current installed models")

    args = parser.parse_args()
    interface.init_logger(args)
    logger.set_verbose(verbose=args.verbose)

    from agent.models import ollama
    from src.cli import flag_functions
    from src.agent.chat_logs import ChatLogs
    from src.core import Agent
    from src.agent import tokenizers

    # =================================================================
    # FUNCTIONS
    # =================================================================

    # === DELETE ======================================================
    if args.delete_default_session:
        flag_functions.del_sess()
        return

    # === INSTALL MODELS ==============================================
    if args.install_model:
        ollama.ollama_pull_model(args.install_model)
        return

    if args.list_models:
        model_dict = ollama.ollama_clt.list()
        interface.installed_models_table(model_dict)
        return

    # === SESSIONS ====================================================
    if args.new_session:
        chat_logs = ChatLogs(conn=postgres.conn, sess_name=args.new_session)
        chat_logs.create_sess()
        if not args.prompt:
            return
        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.prompt)
        return

    if args.delete_session:
        response = flag_functions.del_sess(args.delete_session)
        if response:
            print(response)
        return

    if args.list_session:
        chat_logs = ChatLogs(conn=postgres.conn)
        sess_dict = chat_logs.get_all_existing_sess_metadata()
        interface.sessions_table(sess_dict)
        return

    # === ATTACHMENTS =================================================
    if args.attachments:
        if not args.prompt:
            print("Error: Prompt required")
            return

        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.prompt, is_attchmnt=True, paths=args.attachments)
        return

    if args.list_attachments:
        from src.rag import document_knowledge_base
        chat_logs = ChatLogs(conn=postgres.conn, sess_name=args.session)
        doc_kw_bs = document_knowledge_base.DocumentKnowledgeBase(
            conn=postgres.conn, chat_logs=chat_logs, sess_name=args.session
        )
        print(doc_kw_bs.list_all_uploaded_documents())
        return

    # === TOKENIZERS ==================================================
    if args.install_tokenizers:
        from src.agent.tokenizers import install_tokenizers
        install_tokenizers()

    # === WHEN QUESTION IS ASKED ======================================
    if args.prompt:
        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.prompt)

    # === STATUS ======================================================
    if args.dashboard:
        model_dict = ollama.ollama_clt.list()
        chat_logs = ChatLogs(conn=postgres.conn)
        sess_dict = chat_logs.get_all_existing_sess_metadata()
        tknizr_dict = tokenizers.fetch_all_installed_tokenizers()
        latest = chat_logs.latest_modified_chat_session()
        if latest:
            latest_sess, latest_sess_dt = latest
        else:
            latest_sess, latest_sess_dt = None, None

        interface.dashboard(
            model_dict=model_dict,
            chat_model=models.MODEL,
            mem_model=models.MEM_MODEL,
            sear_model=models.SEAR_MODEL,
            embed_model=models.EMBED_MODEL,
            tknizr_dict=tknizr_dict,
            fallback_tknizr=models.FALLBACK_TOKENIZER,
            sess_dict=sess_dict,
            latest_sess=latest_sess,
            latest_sess_dt=latest_sess_dt
        )
        return

    # === NO ARGUMENTS AND QUESTION ===================================
    if not args.prompt and not any([args.attachments]):
        from src.tui.app import StartTUI
        StartTUI().run()
        return
