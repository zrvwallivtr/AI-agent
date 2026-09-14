import argparse
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout


console: Console | None = None
console_handler: RichHandler | None = None


def build_console(args: argparse.Namespace) -> Console:
    return Console(
        no_color=args.no_color,
        width=args.width,
        quiet=args.quiet,
        force_terminal=None,
    )


def init_logger(args: argparse.Namespace) -> None:
    """Call once right after argparse.parse_args(), before anything else."""
    global console, console_handler

    console = build_console(args)

    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        markup=True,
    )
    console_handler.setFormatter(
        logging.Formatter("%(message)s",)
    )
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)


def safe_print(*args, **kwargs) -> None:
    """Print via shared console if initialised, else fallback to normal print()."""
    if console:
        console.print(*args, **kwargs)
    else:
        print(*args)


def menu() -> None:
    """Menu for the agent, display when no flags or 'help' flag are used."""
    layout = Layout()
    layout.split_column(
        Layout(name="RAG AGENT"),
        Layout(name="Options")
    )
    safe_print(layout)


def sessions_table(sess_dict: dict | None) -> None:
    """List all user created sessions."""
    if not sess_dict:
        safe_print("No session in the database")
        return

    table = Table(title="[bright_cyan bold not italic]AVAILABLE SESSIONS")
    table.add_column("[bright_cyan]Created at", justify="center", no_wrap=True, vertical="middle", style="bright_black")
    table.add_column("[bright_cyan]Session name", justify="center", vertical="middle")

    for sess in sess_dict:
        table.add_row(f"{sess_dict[sess]['created_at']}", f"{sess_dict[sess]['session_name']}")

    safe_print(table)


def installed_models_table(model_list: list):
    """List all installed models."""
    if not model_list:
        safe_print("No model installed locally")

    table = Table(title="[bright_cyan not italic]INSTALLED MODELS")
    table.add_column("[bright_cyan]Modified at", justify="center", vertical="middle", style="bright_black")
    table.add_column("[bright_cyan]Model name", justify="center", vertical="middle")
    table.add_column("[bright_cyan]Family", justify="center", vertical="middle")
    table.add_column("[bright_cyan]Parameter size", justify="center", vertical="middle")
    table.add_column("[bright_cyan]Quantization level", justify="center", vertical="middle")

    for model in model_list.models:
        table.add_row(f"{model.modified_at}", f"{model.model}", f"{model.details.family}", f"{model.details.parameter_size}", f"{model.details.quantization_level}")
    
    safe_print(table)
