import argparse
from pathlib import Path

from src.agent.core import Agent
from src.agent.memory import Memory
from src import config

from src.tools.file_reader import FileReader
from src.tools.project_manager import ProjectManager
from src.tools.search import SearchAgent

from src.cli.flag_functions import General, Session, File
from src.tools import file_reader


def main():
    parser = argparse.ArgumentParser(prog="agent", description="AI Agent")

    # General
    parser.add_argument("question",                 nargs="?",              help="Ask questions")
    parser.add_argument("--model", "-m",            default=config.MODEL,   metavar=("MODEL_NAME"),                     help=f"Select model (defualt: {config.MODEL})")
    parser.add_argument("--reset-default", "-rd",   action="store_true",    help="Reset default session")
    parser.add_argument("--installed-models", "-i", action="store_true",    help="List all installed ollama models")

    # Session flags
    parser.add_argument("--session", "-s",          default=None,           metavar=("SESSION_NAME"),                   help="Continue a selected session")
    parser.add_argument("--list-session", "-ls",    action="store_true",                                                help="List all existing session")
    parser.add_argument("--new-session", "-ns",     default=None,           metavar=("SESSION_NAME"),                   help="Create a new session")
    parser.add_argument("--delete-session", "-d",   default=None,           metavar=("SESSION_NAME"),                   help="Delete a selected session")

    # Read files
    parser.add_argument("--file", "-f",         nargs="+",      type=Path,     default=None,    metavar=("FILE_PATH"),      help="Read selected file")
    parser.add_argument("--list-files", "-lf",   action="store_true",                                                        help="List all files in dropbox")

    # Project manager
    parser.add_argument("--project-summary", "-ps",     nargs=3,        metavar=("PROJECT_NAME", "SESSION_NAME", "TEXT"),       help="Edit project tasklist")
    parser.add_argument("--project-task", "-pt",        nargs=2,        metavar=("PROJECT_NAME", "SESSION_NAME"),               help="Edit project decisions")
    parser.add_argument("--project-dec", "-pd",         nargs=3,        metavar=("PROJECT_NAME", "SESSION_NAME", "TEXT"),       help="Edit project milestone")
    parser.add_argument("--new-project", "-np",         default=None,   metavar=("PROJECT_NAME"),                               help="Create new project directory containing all the required files")


    args = parser.parse_args()

    agent       = Agent(model=args.model, session=args.session)
    session     = Session(args.session)
    file        = File(args.session)
    file_reader = FileReader(args.session)

    # =================================================================
    # Delete
    # =================================================================
    if args.reset_default:
        General.reset_default()
        return True

    # =================================================================
    # List installed models
    # =================================================================
    if args.installed_models:
        General.installed_models()
        return

    # =================================================================
    # Session
    # =================================================================
    if args.new_session:
        answer = session.create_session(model=args.model, prompt=args.question)
        return

    if args.delete_session:
        session.delete_session()
        return

    if args.list_session:
        Session.list_session()
        return

    # =================================================================
    # Read
    # =================================================================
    if args.file:
        if not args.question:
            print("Error: question required")
            return

        if args.file:
            file.files_with_prompt(
                model=args.model,
                prompt=args.question,
                file_paths=args.file,
                #project=args.project
            )
            return

    if args.list_files:
        file_reader.list_available_files()

    # =================================================================
    # Project
    # =================================================================
    if args.new_project:
        pm = ProjectManager(args.new_project)
        pm.create_project_workspace()
        return

    if args.project_summary:
        project_name, session, question = args.project_summary  # requires 3 args
        pm          = ProjectManager(project_name)
        response    = pm.summarize_project(question, session=session)
        return

    if args.project_task:
        project_name, session   = args.project_task  # requires 2 args
        pm                      = ProjectManager(project_name)
        response                = pm.generate_tasklist(session=session)
        return

    if args.project_dec:
        project_name, session, question = args.project_dec  # requires 3 args
        pm                              = ProjectManager(project_name)
        response                        = pm.add_decisions(question, session=session)
        return

    # =================================================================
    # When question is asked
    #
    # agent -s SESSION_NAME -m MODEL_NAME PROMPT
    # =================================================================
    if args.question:
        General.question(
            prompt=args.question,
            model=args.model,
            session=args.session,
        )

    # =================================================================
    # Help
    # =================================================================
    action_flags = [args.file]

    if not args.question and not any(action_flags):
        parser.print_help()
        return
