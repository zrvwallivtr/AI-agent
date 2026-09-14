import argparse
from pathlib import Path
from rich.console import Console
from rich_argparse import RichHelpFormatter

from src.config import models
from src.config import postgres

from src.cli import interface
from src import logger


MODEL = models.MODEL


def build_parser() -> None:
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, description="Local RAG agent")

    # === GENERAL ===================================================================
    parser.add_argument("question", nargs="?", help="Ask questions")
    parser.add_argument("--model", "-m", default=MODEL, metavar=("MODEL_NAME"), help=f"Select model (defualt: {MODEL})")
    parser.add_argument("--reset-default", "-rd", action="store_true", help="Reset default session")
    parser.add_argument("--install-model", "-i", metavar=("MODEL_NAME"), help=f"Install specified model")
    parser.add_argument("--installed-models", action="store_true", help="List all installed ollama models")

    # === SETTINGS ==================================================================
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show debug-level logs")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--width", type=int, default=None, help="Force console width (useful for non-TTY output)")

    # === SESSION FLAGS =============================================================
    parser.add_argument("--session", "-s", default=None, metavar=("SESSION_NAME"), help="Continue a selected session")
    parser.add_argument("--list-session", "-ls", action="store_true", help="List all existing session")
    parser.add_argument("--new-session", "-ns", default=None, metavar=("SESSION_NAME"), help="Create a new session")
    parser.add_argument("--delete-session", "-d", default=None, metavar=("SESSION_NAME"), help="Delete a selected session")

    # === READ ATTACHMENTS ==========================================================
    parser.add_argument("--file", "-f", nargs="+", type=Path, default=None, metavar=("FILE_PATH"), help="Read selected file")
    parser.add_argument("--list-files", "-lf", action="store_true", help="List all files in dropbox")

    # === PROJECT MANAGER ===========================================================
    parser.add_argument("--project-summary", "-ps", nargs=3, metavar=("PROJECT_NAME", "SESSION_NAME", "TEXT"), help="Edit project tasklist")
    parser.add_argument("--project-task", "-pt", nargs=2, metavar=("PROJECT_NAME", "SESSION_NAME"), help="Edit project decisions")
    parser.add_argument("--project-dec", "-pd", nargs=3, metavar=("PROJECT_NAME", "SESSION_NAME", "TEXT"), help="Edit project milestone")
    parser.add_argument("--new-project", "-np", default=None, metavar=("PROJECT_NAME"), help="Create new project directory containing all the required files")

    # === INITIALISE TOKENIZERS =====================================================
    parser.add_argument("--load-tokenizers", "-lt", action="store_true", help="Install tokenizers for current installed models")

    args = parser.parse_args()
    interface.init_logger(args)
    logger.set_verbose(verbose=args.verbose)

    from agent.models import ollama
    from src.cli import flag_functions
    from src.agent.chat_logs import ChatLogs
    from src.core import Agent

    # =================================================================
    # DELETE
    # =================================================================

    if args.reset_default:
        flag_functions.del_sess()
        return

    # =================================================================
    # INSTALL MODELS
    # =================================================================

    if args.install_model:
        ollama.ollama_pull_model(args.install_model)
        return

    if args.installed_models:
        model_list = ollama.ollama_clt.list()
        interface.installed_models_table(model_list)
        return

    # =================================================================
    # SESSION
    # =================================================================

    if args.new_session:
        chat_logs = ChatLogs(conn=postgres.conn, sess_name=args.new_session)
        chat_logs.create_sess()
        if not args.question:
            return
        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.question)
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

    # =================================================================
    # READ
    # =================================================================

    if args.file:
        if not args.question:
            print("Error: question required")
            return

        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.question, is_attchmnt=True, paths=args.file)
        return

    if args.list_files:
        from src.rag import document_knowledge_base
        chat_logs = ChatLogs(conn=postgres.conn, sess_name=args.session)
        doc_kw_bs = document_knowledge_base.DocumentKnowledgeBase(
            conn=postgres.conn, chat_logs=chat_logs, sess_name=args.session
        )
        print(doc_kw_bs.list_all_uploaded_documents())
        return

    # =================================================================
    # TOKENIZERS
    # =================================================================

    if args.load_tokenizers:
        from src.agent.tokenizers import install_tokenizers
        install_tokenizers()

    # =================================================================
    # WHEN QUESTION IS ASKED
    #
    # agent -s SESSION_NAME -m MODEL_NAME PROMPT
    # =================================================================

    if args.question:
        agent = Agent(sess_name=args.session)
        answer = agent.ask(prompt=args.question)

    # =================================================================
    # HELP / NO ARGUMENTS
    # =================================================================

    if not args.question and not any([args.file]):
        parser.print_help()
        # interface.menu()
        return
