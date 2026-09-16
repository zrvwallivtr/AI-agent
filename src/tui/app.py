from itertools import accumulate
from textual.app import App, Screen, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Footer, RichLog, Input, Markdown
from textual import work

from src.core import Agent
from src.tui import menu


class StartTUI(App):
    CSS_PATH = "tcss/menu.tcss"
    BINDINGS = [
        ("ctrl+c", "quit"),
        ("ctrl+d", "start_default_session"),
        ("ctrl+m", "to_menu")
    ]


    def __init__(self):
        super().__init__()


    def compose(self) -> ComposeResult:
        with Horizontal(id="top_container"):
            yield Static(id="icon", classes="box")
            yield Static(id="description", classes="box")
        yield Static(id="status", classes="box")


    def on_mount(self) -> None:
        self.show_menu()


    def show_menu(self):
        icon_panel = menu.app_icon()
        dsrption_panel = menu.app_description()
        status_panel = menu.app_status()

        self.query_one("#icon", Static).update(icon_panel)
        self.query_one("#description", Static).update(dsrption_panel)
        self.query_one("#status", Static).update(status_panel)


    def action_to_menu(self) -> None:
        self.show_menu()


    def action_start_default_session(self, sess_name: str | None = None) -> None:
        self.push_screen(ChatInterface(sess_name=sess_name))


    def action_quit(self) -> None:
        self.exit()


class ChatInterface(Screen):
    CSS_PATH = "tcss/chat_interface.tcss"


    def __init__(self, sess_name: str | None):
        super().__init__()
        self.agent = Agent(sess_name=sess_name)
        self.sess_name = sess_name or "Default session"


    def compose(self) -> ComposeResult:
        with Vertical():
            with VerticalScroll(id="chat_container"):
                pass
            yield Input(placeholder="Write a message...", id="prompt_input")


    def on_mount(self) -> None:
        self.query_one("#prompt_input", Input).focus()
        self.load_chat_history()


    def load_chat_history(self) -> None:
        chat_container = self.query_one("#chat_container", VerticalScroll)

        chat_hist = self.agent.chat_logs.get_chat_history(filter="all")

        if not chat_hist:
            return

        for msg in chat_hist:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                chat_container.mount(Static(f"[bold]USER:[/] {content}"))
            elif role == "assistant":
                chat_container.mount(Markdown(content))

        chat_container.scroll_end(animate=False)


    def on_input_submitted(self, msg: Input.Submitted) -> None:
        prompt = msg.value.strip()
        if not prompt:
            return

        chat_container = self.query_one("#chat_container", VerticalScroll)

        user_msg = Static(f"[bold]USER[/]: {prompt}")
        chat_container.mount(user_msg)
        chat_container.scroll_end(animate=False)

        self.query_one("#prompt_input", Input).clear()

        self.fetch_agent_response(prompt)


    @work(exclusive=True, thread=True)
    def fetch_agent_response(self, prompt: str) -> None:
        chat_container = self.query_one("#chat_container", VerticalScroll)

        md_widget = Markdown("")
        self.app.call_from_thread(chat_container.mount, md_widget)

        full_txt = ""

        def on_token(tkn: str) -> None:
            nonlocal full_txt
            full_txt += tkn
            self.app.call_from_thread(md_widget.update, full_txt)
            self.app.call_from_thread(chat_container.scroll_end, animate=False)

        self.agent.ask(prompt=prompt, callback=on_token)
