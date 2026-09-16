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
from src.config import postgres
from src.agent.chat_logs import ChatLogs
from src.agent.models import ollama
from src.agent import tokenizers
from assests.icons import app_icon_ascii


def app_icon() -> Panel:
    icon_panel = Panel(
        f"[bright_cyan bold not italic]{app_icon_ascii.RAG}",
        height=11,
        width=26,
        box=box.SQUARE
    )
    return icon_panel


def app_description() -> Panel:
    dsrption_panel = Panel(
        (
            "A local Command-Line Interface (CLI) AI assistant featuring long-term memory, "
            "file context injection, (isolated web crawling / search and automated token management)."
        ),
        title=f"[bright_cyan bold not italic]DESCRIPTION",
        box=box.SQUARE
    )
    return dsrption_panel


def app_status() -> Panel:
    model_dict = ollama.ollama_clt.list()
    chat_logs = ChatLogs(conn=postgres.conn)
    sess_dict = chat_logs.get_all_existing_sess_metadata()
    tknizr_dict = tokenizers.fetch_all_installed_tokenizers()
    latest = chat_logs.latest_modified_chat_session()
    if latest:
        latest_sess, latest_sess_dt = latest
    else:
        latest_sess, latest_sess_dt = None, None

    status_panel = ""

    if model_dict:
        status_panel += (
            f"[bold]Models[/bold] ({len(model_dict.models)} installed)\n"
            f"[bright_black]│[/bright_black]\tChat model:\t\t[yellow]{models.MODEL}[/yellow]\n"
            f"[bright_black]│[/bright_black]\tMemory model:\t\t[yellow]{models.MEM_MODEL}[/yellow]\n"
            f"[bright_black]│[/bright_black]\tWeb search model:\t[yellow]{models.SEAR_MODEL}[/yellow]\n"
            f"[bright_black]│[/bright_black]\tEmbedding model:\t[yellow]{models.EMBED_MODEL}[/yellow]\n\n"
        )

    if tknizr_dict:
        status_panel += (
            f"[bold]Tokenizers[/bold] ({len(tknizr_dict)} installed)\n"
            f"[bright_black]│[/bright_black]\tFallback tokenizer: [yellow]{models.FALLBACK_TOKENIZER}[/yellow]\n\n"
        )

    if sess_dict:
        status_panel += (
            f"[bold]Sessions[/bold] ({len(sess_dict)} created)\n"
        )

    if latest_sess_dt and latest_sess:
        status_panel += (
            f"[bright_black]│[/bright_black]\tLatest: [blue]{latest_sess_dt}[/blue] [yellow]{latest_sess}[/yellow]"
        )

    status_panel = Panel(
            (status_panel),
            title=f"[bright_cyan bold not italic]STATUS",
            box=box.SQUARE
    )

    return status_panel
