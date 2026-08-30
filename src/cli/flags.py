import argparse
from pathlib import Path

from src.config.models import MODEL
from src.cli.flag_functions import General, Session, File


def main():
    parser = argparse.ArgumentParser(prog="agent", description="AI Agent")

    # === GENERAL ===================================================================

    parser.add_argument(
        "question",
        nargs="?",
        help="Ask questions"
    )

    parser.add_argument(
        "--model",
        "-m",
        default=MODEL,
        metavar=("MODEL_NAME"),
        help=f"Select model (defualt: {MODEL})"
    )

    parser.add_argument(
        "--reset-default",
        "-rd",
        action="store_true",
        help="Reset default session"
    )

    parser.add_argument(
        "--installed-models",
        "-i",
        action="store_true",
        help="List all installed ollama models"
    )

    # === SESSION FLAGS =============================================================

    parser.add_argument(
        "--session",
        "-s",
        default=None,
        metavar=("SESSION_NAME"),
        help="Continue a selected session"
    )

    parser.add_argument(
        "--list-session",
        "-ls",
        action="store_true",
        help="List all existing session"
    )

    parser.add_argument(
        "--new-session",
        "-ns",
        default=None,
        metavar=("SESSION_NAME"),
        help="Create a new session"
    )

    parser.add_argument(
        "--delete-session",
        "-d",
        default=None,
        metavar=("SESSION_NAME"),
        help="Delete a selected session"
    )

    # === READ ATTACHMENTS ==========================================================

    parser.add_argument(
        "--file",
        "-f",
        nargs="+",
        type=Path,
        default=None,
        metavar=("FILE_PATH"),
        help="Read selected file"
    )

    parser.add_argument(
        "--list-files",
        "-lf",
        action="store_true",
        help="List all files in dropbox"
    )

    # === PROJECT MANAGER ===========================================================

    parser.add_argument(
        "--project-summary",
        "-ps",
        nargs=3,
        metavar=("PROJECT_NAME", "SESSION_NAME", "TEXT"),
        help="Edit project tasklist"
    )

    parser.add_argument(
        "--project-task",
        "-pt",
        nargs=2,
        metavar=("PROJECT_NAME", "SESSION_NAME"),
        help="Edit project decisions"
    )

    parser.add_argument(
        "--project-dec",
        "-pd",
        nargs=3,
        metavar=("PROJECT_NAME", "SESSION_NAME", "TEXT"),
        help="Edit project milestone"
    )

    parser.add_argument(
        "--new-project",
        "-np",
        default=None,
        metavar=("PROJECT_NAME"),
        help="Create new project directory containing all the required files"
    )

    # === INITIALISE TOKENIZERS =====================================================

    parser.add_argument(
        "--load-tokenizers",
        "-lt",
        action="store_true",
        help="Install tokenizers for current installed models"
    )

    args = parser.parse_args()

    # =================================================================
    # DELETE
    # =================================================================

    if args.reset_default:
        General.reset_default()
        return True

    # =================================================================
    # LIST INSTALLED MODELS
    # =================================================================

    if args.installed_models:
        General.installed_models()
        return

    # =================================================================
    # SESSION
    # =================================================================

    if args.new_session:
        session = Session(args.new_session)
        response = session.create_session(model=args.model, prompt=args.question)
        if response:
            print(response)
        return

    if args.delete_session:
        from src.cli.flag_functions import del_sess
        response = del_sess(args.delete_session)
        if response:
            print(response)
        return

    if args.list_session:
        from src.agent.chat_logs import ChatLogs
        from src.config.postgres import conn
        chat_logs = ChatLogs(conn=conn)
        sess_dict = chat_logs.get_all_existing_sess_metadata()

        print("AVAILABLE SESSION(S)")
        print("====================")
        print("CREATED AT\t\t\t\tSESSION NAME")
        for sess in sess_dict:
            print(f"{sess_dict[sess]["created_at"]}\t{sess_dict[sess]["session_name"]}")
        print("\n")
        return

    # =================================================================
    # READ
    # =================================================================

    if args.file:
        if not args.question:
            print("Error: question required")
            return

        file = File(args.session)
        file.attachments_with_prompt(
            model=args.model,
            prompt=args.question,
            paths=args.file,
            #project=args.project
        )
        return

    if args.list_files:
        from src.agent.chat_logs import ChatLogs
        from src.tools import DocumentKnowledgeBase
        from src.config.postgres import conn
        chat_logs = ChatLogs(conn=conn, sess_name=args.session)
        doc_kw_bs = DocumentKnowledgeBase(conn=conn, chat_logs=chat_logs, sess_name=args.session)
        print(doc_kw_bs.list_all_uploaded_documents())
        return

    # =================================================================
    # PROJECT
    # =================================================================

    # if args.new_project:
    #     pm = ProjectManager(args.new_project)
    #     pm.create_project_workspace()
    #     return

    # if args.project_summary:
    #     project_name, session, question = args.project_summary  # requires 3 args
    #     pm          = ProjectManager(project_name)
    #     response    = pm.summarize_project(question, session=session)
    #     return

    # if args.project_task:
    #     project_name, session   = args.project_task  # requires 2 args
    #     pm                      = ProjectManager(project_name)
    #     response                = pm.generate_tasklist(session=session)
    #     return

    # if args.project_dec:
    #     project_name, session, question = args.project_dec  # requires 3 args
    #     pm                              = ProjectManager(project_name)
    #     response                        = pm.add_decisions(question, session=session)
    #     return

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
        General.question(
            prompt=args.question,
            model=args.model,
            sess_name=args.session,
        )

    # =================================================================
    # HELP / NO ARGUMENTS
    # =================================================================

    action_flags = [args.file]

    if not args.question and not any(action_flags):
        parser.print_help()
        return
