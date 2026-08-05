import json
from pathlib import Path
from typing import Literal

from src.config import (
    SYS_PROMPT,
    CHAT_DIR,
    DEFAULT_PATH,
    DEFAULT_CHAT_HISTORY_PATH
)
from src.agent.llm import LLM


def _session_path(session: str = None) -> tuple[Path, Path]:
    if session is not None:
        return CHAT_DIR / f"{session}.json", CHAT_DIR / f"{session}_chat_history.json"
    else:
        return DEFAULT_PATH, DEFAULT_CHAT_HISTORY_PATH


class Chat:
    def __init__(self, session: str = None):
        self.active_conv_path, self.chat_history_path   = _session_path(session)
        self.prompt                                     = SYS_PROMPT
        self._messages                                  = self._load_chat()

    def _load_chat(self) -> list[dict]:
        """Fetch/Start messages in a session."""
        if self.active_conv_path.exists():
            # Load the entire conversation
            messages = json.loads(self.active_conv_path.read_text())

            # Update system prompt if it was changed 
            if messages and messages[0]["role"] == "system":
                if messages[0]["content"] != self.prompt:
                    messages[0]["content"] = self.prompt
                    self.active_conv_path.write_text(json.dumps(messages, indent=4))
            return messages

        else:
            # Start fresh, if system prompt file exist,
            # load system prompt only.
            return [{"role": "system", "content": self.prompt, "state": "internal"}]

    def all(self) -> list[dict]:
        """
        Entire history, including all conversations
        and token counts.
        """
        return list(self._messages)

    def to_llm(self) -> list[dict]:
        """Trim 'self._messages' for the model to read."""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self._messages
        ]

    def clear(self):
        """Empty stored messages and clear contents in path."""
        self._messages = []
        self.active_conv_path.write_text("[]")

    def save(self, msg: dict = None):
        """
        Create file(s) if it does not exist already.

        Two files will be saved:

        1. Active conversation -> 'self.active_conv_path'
        2. Chat history -> 'self.chat_history_path'
        """
        # Active conversation:
        #
        # - Purpose: Read by the MODEL.
        # - Writes messages stored in 'self._messages'.
        #
        # Note: File can still be created with no questions asked.
        self.active_conv_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_conv_path.write_text(json.dumps(self._messages, indent=4))

        # Chat history:
        #
        # - Purpose: Read by the USER.
        # - Writes messages stored in 'msg'.
        #
        # Note: Only called when message is present.
        if msg:
            self.chat_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.chat_history_path, "a") as f:
                f.write(json.dumps(msg) + "\n")

    def clear_active_conv(self) -> str:
        """Deleted conversation file."""
        if self.active_conv_path.exists():
            self.active_conv_path.unlink()
            return f"Deleted session {self.active_conv_path.name}"
        else:
            return f"Error: Session {self.active_conv_path.name} not found."

    def clear_chat_history(self) -> str:
        """Delete chat history file."""
        if self.chat_history_path.exists():
            self.chat_history_path.unlink()
            return f"Deleted session {self.chat_history_path.name}'s chat history."
        else:
            return f"Error: Session {self.chat_history_path.name}'s chat history not found."

    def append_message(
            self,
            role: str,
            content: str,
            state: Literal["internal", "external"],
            prompt_tokens: int = 0,
            output_tokens: int = 0
    ):
        """
        Every message contains 'role', 'content', 'state' entires.

        If 'role' = 'assistant', the stored message will contains
        'prompt_tokens' and 'output_tokens' as well.

        State:
        - 'internal': Pre-written prompt.
        - 'external': User/model interactions.
        """
        msg = {"role": role, "content": content, "state": state}

        if role == "assistant" and prompt_tokens:
            msg["prompt_tokens"] = prompt_tokens
            msg["output_tokens"] = output_tokens

        self._messages.append(msg)
        self.save(msg)

    def compression(self, model: str):
        """
        Summarise all conversations:

        The compression steps depends on the size of the
        model.
        """
        # if max_token < set number:
        #
        # Applicable for micro models (~ 1-3B).

        # Could add token check to ensure the output 
        # summary is under (max_token * 0.1).
        messages = self.to_llm() + [LLM.user(
            "Create a concise summary of the conversation above. "
            "The summary will replace this conversation as context for future messages. "
            "Include: key topics discussed, conclusions reached, and any important details mentioned. "
            "Be factual and specific. Do not add any commentary or explanation — output the summary only."
        )]
        content, prompt_tokens, output_tokens = LLM.model_response(messages, model)

        # If error occurs (model returns nothing or
        # suspiciously low token count), keep history.
        if not content.strip() or output_tokens <= 5:
            print("Error: Compression failed")
            return

        instruction = (
            "Summarise context from previous conversations."
        )

        summary = (
            "[CONVERSATION SUMMARY]\n"
            "The following is a condensed record of previous conversations. "
            "Treat all information below as established context:\n\n"
            + content
        )

        self.clear()
        self.append_message("system", self.prompt, "internal")
        self.append_message("user", instruction, "internal")
        self.append_message("assistant", summary, "internal", prompt_tokens, output_tokens)

        # if max_token < a higher set number (~ 7-13B model max)
        #   keep ~ 3-6 conversations, or the sum of
        #   conversations that is under (max_token * 0.2).

        # if max_token > very high set number (~ 30B model max)
        #   keep ~ 10-20 conversations, or the sum of
        #   conversations that is under (max_token * 0.3).
