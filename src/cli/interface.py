import argparse
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.columns import Columns
from rich import box

from src.config import models
from assests.icons import app_icon_ascii


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


def dashboard(
    model_dict: dict | None,
    chat_model: str,
    mem_model: str,
    sear_model: str,
    embed_model: str,
    tknizr_dict: list[dict] | None,
    fallback_tknizr: str,
    sess_dict: dict | None,
    latest_sess: str | None,
    latest_sess_dt: str | None,
):
    """
    Dashboard for the agent includes:
    - App icon
    - Description
    - Status
      - Models
      - Tokenizers
      - Sessions
    """
    icon_panel = Panel(
        f"[bright_cyan bold not italic]{app_icon_ascii.RAG}",
        height=11,
        width=26,
        box=box.SQUARE
    )

    des_panel = Panel(
        (
            "A local Command-Line Interface (CLI) AI assistant featuring long-term memory, "
            "file context injection, (isolated web crawling / search and automated token management)."
        ),
        title=f"[bright_cyan bold not italic]DESCRIPTION",
        height=11,
        width=60,
        box=box.SQUARE
    )

    status_str = ""

    if model_dict:
        status_str += (
            f"[bold]Models[/bold] ({len(model_dict.models)} installed)\n"
            f"[bright_black]│[/bright_black]\tChat model:\t\t[yellow]{chat_model}[/yellow]\n"
            f"[bright_black]│[/bright_black]\tMemory model:\t\t[yellow]{mem_model}[/yellow]\n"
            f"[bright_black]│[/bright_black]\tWeb search model:\t[yellow]{sear_model}[/yellow]\n"
            f"[bright_black]│[/bright_black]\tEmbedding model:\t[yellow]{embed_model}[/yellow]\n\n"
        )

    if tknizr_dict:
        status_str += (
            f"[bold]Tokenizers[/bold] ({len(tknizr_dict)} installed)\n"
            f"[bright_black]│[/bright_black]\tFallback tokenizer: [yellow]{fallback_tknizr}[/yellow]\n\n"
        )

    if sess_dict:
        status_str += (
            f"[bold]Sessions[/bold] ({len(sess_dict)} created)\n"
        )

    if latest_sess_dt and latest_sess:
        status_str += (
            f"[bright_black]│[/bright_black]\tLatest: [blue]{latest_sess_dt}[/blue] [yellow]{latest_sess}[/yellow]"
        )

    status_panel = Panel(
            (status_str),
            title=f"[bright_cyan bold not italic]STATUS",
            width=26+60+1,
            box=box.SQUARE
    )

    safe_print(Columns([icon_panel, des_panel]))
    safe_print(status_panel)


def sessions_table(sess_dict: dict | None) -> None:
    """List all user created sessions."""
    if not sess_dict:
        safe_print("No session in the database")
        return

    table = Table(title="[bright_cyan bold not italic]AVAILABLE SESSIONS", box=box.SIMPLE)
    table.add_column("[bright_cyan not bold]Created at", justify="center", no_wrap=True, vertical="middle", style="blue")
    table.add_column("[bright_cyan not bold]Session name", justify="center", vertical="middle")

    for sess in sess_dict:
        table.add_row(f"{sess_dict[sess]['created_at']}", f"{sess_dict[sess]['session_name']}")
    safe_print(table)


def installed_models_table(model_dict: dict | None) -> None:
    """List all installed models."""
    if not model_dict:
        safe_print("No model installed locally")
        return

    table = Table(title="[bright_cyan bold not italic]INSTALLED MODELS", box=box.SIMPLE)
    table.add_column("[bright_cyan not bold]Modified at", justify="center", vertical="middle", style="blue")
    table.add_column("[bright_cyan not bold]Model name", justify="center", vertical="middle")
    table.add_column("[bright_cyan not bold]Family", justify="center", vertical="middle")
    table.add_column("[bright_cyan not bold]Parameter size", justify="center", vertical="middle")
    table.add_column("[bright_cyan not bold]Quantization level", justify="center", vertical="middle")

    for model in model_dict.models:
        table.add_row(f"{model.modified_at}", f"{model.model}", f"{model.details.family}", f"{model.details.parameter_size}", f"{model.details.quantization_level}")
    safe_print(table)
