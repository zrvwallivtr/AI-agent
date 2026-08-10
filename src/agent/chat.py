import json
import mimetypes
from pathlib import Path
from typing import Literal, Optional, Any
from datetime import datetime, timezone

from src import config
from src.agent.llm import LLM


TOOL_LIST = Literal["web_search", "read_files"]


def _session_path(session: str | None = None) -> tuple[Path, Path]:
    if session is not None:
        return config.CHATS_DIR / f"{session}" / f"chat.json", config.CHATS_DIR / f"{session}" / f"chat_history.json"
    else:
        return config.DEFAULT_PATH, config.DEFAULT_CHAT_HISTORY_PATH


class Chat:
    def __init__(self, session: str | None = None):
        self.session                                    = session
        self.active_conv_path, self.chat_history_path   = _session_path(session)
        self.prompt                                     = config.SYS_PROMPT
        self._messages                                  = self._load_chat()
        self.compression_prompt                         = config.COMPRESS_PROMPT

    def _load_chat(self) -> list[dict]:
        """Fetch/Start messages in a session."""
        if self.active_conv_path.exists():
            # Load the entire conversation
            messages = json.loads(self.active_conv_path.read_text(encoding="utf-8"))

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

    def save(self, msg: dict | None = None):
        """
        Create file(s) if it does not exist already.

        Two files will be saved:

        1. Active conversation -> 'self.active_conv_path'
        2. Chat history -> 'self.chat_history_path'
        """
        # Active conversation:
        # - Purpose: Read by the MODEL.
        # - Overwrites entire file with messages stored
        #   in 'self._messages'.
        #
        # Note: File can still be created with no questions asked.
        self.active_conv_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_conv_path.write_text(json.dumps(self._messages, indent=4))

        # Chat history:
        # - Purpose: Read by the user.
        # - Append new entry with contents stored in 'msg'.
        #
        # Note: Only called when message is present.
        if msg:
            self.chat_history_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.chat_history_path, "a") as f:
                f.write(json.dumps(msg, indent=4) + "\n")

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

    # ========================================================
    # Metadata
    # ========================================================

    def _attachments_metadata(self, attachments: list[Path] | None) -> dict[str, dict[str, Any]]:
        """
        Add metadata to every filename in the list of filenames.

        {
            "filename": {
                "mime_type": "type",
                "size_bytes": size,
            }
        }
        """
        if attachments:

            attachment_dict = {}

            for attachment in attachments:

                mime_type, _ = mimetypes.guess_type(attachment)
                
                attachment_dict[attachment.name] = {
                    "mime_type": mime_type,
                    "size_bytes": attachment.stat().st_size if attachment.exists else 0
                }

            return attachment_dict

        return {"": {"": []}}

    def _web_search_metadata(
        self,
        queries_with_urls: list[dict[str, list[str]]] | None,
    ) -> dict[str, list[str]]:
        """
        Add metadata to every query in the list of queries.

        {
            "query_name": [
                "query_url",
            ]
        }
        """
        if queries_with_urls:
            web_search_dict = {}
            
            for query_dict in queries_with_urls:
                web_search_dict.update(query_dict)

            return web_search_dict
        
        return {"": [""]}

    def _tool_calls_metadata(
        self,
        msg: dict[str, Any],
        attachments: list[Path] | None = None,
        read_dropbox: bool = False,
        queries_with_urls: list[dict[str, list[str]]] | None = None,
        search: bool = False
    ) -> dict[str, dict[str, Any]]:
        """
        Add metadata to every tool call in the list of tool calls.

        "tool_calls": {
            "attachments": {
                "filename": {
                    "mime_type": "type",
                    "size_bytes": size,
                },
            },
            "web_search": {
                "query_name": [
                    "query_url",
                ],
            }
        }
        """
        tool_entries = {}

        # Read files
        if read_dropbox == True:
            tool_entries["attachments"] = self._attachments_metadata(attachments)

        # Web search
        if search == True:
            tool_entries["web_search"] = self._web_search_metadata(queries_with_urls)

        msg["tool_calls"] = tool_entries

        return tool_entries

    def append_user_message_with_metadata(
        self,
        content: str,
        state: Literal["internal", "external"],
        attachments: list[Path] | None = None,
    ):
        """
        State:
        - 'internal': Pre-written prompt.
        - 'external': User/model interactions.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        msg: dict[str, Any] = {
            "timestamp": timestamp,
            "role": "user",
            "content": content,
            "state": state
        }

        # Attachments metadata processing
        if attachments:
            msg["attachments"] = self._attachments_metadata(attachments)

        self._messages.append(msg)
        self.save(msg)

    def append_assistant_message_with_metadata(
        self,
        content: str,
        state: Literal["internal", "external"],

        attachments: list[Path] | None = None,
        read_dropbox: bool = False,

        query_with_urls: list[dict[str, list[str]]] | None = None,
        search: bool = False,

        prompt_tokens: int = 0,
        output_tokens: int = 0
    ):
        """
        State:
        - 'internal': Pre-written prompt.
        - 'external': User/model interactions.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Applicable for all
        msg: dict[str, Any] = {
            "timestamp": timestamp,
            "role": "assistant",
            "content": content,
            "state": state,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens
        }

        # Tool calls metadata processing
        self._tool_calls_metadata(msg, attachments, read_dropbox, query_with_urls, search)

        self._messages.append(msg)
        self.save(msg)

    def append_system_prompt(self, content: str):
        """Add system prompt to messages."""
        msg = {"role": "system", "content": content, "state": "internal"}

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
        history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in self.to_llm()])

        messages = [
            LLM.system(self.compression_prompt),
            LLM.user(f"Summarise the following conversation, The summary will replace this conversation as context for future messages.: \n\n{history}")
        ]
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
        self.append_system_prompt(self.prompt)
        self.append_user_message_with_metadata(
            content=instruction,
            state="internal"
        )
        self.append_assistant_message_with_metadata(
            content=summary,
            state="internal",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )

        # if max_token < a higher set number (~ 7-13B model max)
        #   keep ~ 3-6 conversations, or the sum of
        #   conversations that is under (max_token * 0.2).

        # if max_token > very high set number (~ 30B model max)
        #   keep ~ 10-20 conversations, or the sum of
        #   conversations that is under (max_token * 0.3).
