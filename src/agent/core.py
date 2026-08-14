import ollama
from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat import Chat
from src.agent.memory import Memory
from src.agent.cmd_functions import Command
from src.agent.attachments import Attachments
from src.tools.search import is_connected, SearchAgent
from src.tools.file_reader import FileReader
from src.logger import get_logger
from src import config


logger = get_logger(__name__)


def _last_token_usage(messages: list[dict]) -> tuple[int, int]:
    """Returns prompt_tokens and output_tokens from the latest assistant message."""
    for msg in reversed(messages):

        if msg["role"] == "assistant":
            prompt_tokens = msg.get("prompt_tokens", 0)
            output_tokens = msg.get("output_tokens", 0)
            logger.info("Latest assistant response found (prompt_tokens=%d, output_tokens=%d)", prompt_tokens, output_tokens)
            return prompt_tokens, output_tokens

    logger.warning("Latest assistant response not found")
    return 0, 0 # no previous assistant message yet (first turn)

def _detect_cmd(prompt: str) -> tuple[str | None, str]:
    """Extracts shortcut if detected."""
    question_trimmed = prompt.strip()

    if question_trimmed.startswith(f"/"):
        parts           = question_trimmed.split(" ", 1)
        cmd             = parts[0]
        cleaned_text    = parts[1].strip() if len(parts) > 1 else ""
        logger.info("Command detected: %s", cmd)
        return cmd, cleaned_text

    logger.info("No command detected")
    return None, prompt


class Agent:
    def __init__(
        self,
        model: str | None = None,
        session: str | None = None,
        project: str | None = None
    ):
        model = config.MODEL if model is None else model

        self._validate_model(model)

        if config.MODEL_MAX_TOKENS:
            self.tokens = Tokens(model, config.MODEL_MAX_TOKENS)
        else:
            self.tokens = Tokens(model)

        self.model          = model  # model is specified in the config file
        self.session        = session
        self.project        = project
        self.chat           = Chat(session=session)
        self.memory         = Memory(project=project)
        self.file_reader    = FileReader(session=session)
        self.command        = Command(model=model, session=session, project=project)
        self.attachments    = Attachments(session=session)
        self.search_agent   = SearchAgent()

    @property
    def get_model_max_tokens(self) -> int:
        """Dynamically fetches the current token ceiling from 'self.token'."""
        return self.tokens.model_max_tokens

    # ===================================
    # Model validation
    # ===================================

    def _validate_model(self, model: str):
        """Check if model is installed via Ollama."""
        try:
            # Fetch all downloaded models
            local_models = [m['model'] for m in ollama.list().get('models', [])]

            # Check match
            unknown_models = []
            if model not in local_models:
                unknown_models += model
                logger.error("Unknown or unavailable model detected: %s", model)
                raise ValueError(
                    f"Error: Unknown or unavailable model: '{model}'."
                    f"Run 'ollama pull {model}' to install model."
                )
            known_models = set(local_models) - set(unknown_models)
            logger.info("Fetched all local models:\n Known models = %s\n Unknown model = %s", known_models, unknown_models)

        except Exception as e:
            # If ollama is down
            if isinstance(e, ValueError):
                raise e
            logger.critical("Could not connect to local Ollama service: %s", e)
            raise RuntimeError(f"Error: Could not connect to local Ollama service: {e}")

    # ===================================
    # Token management
    # ===================================

    def _manage_token_budget(self, prompt: str):
        """
        Reserves extra tokens for model response. If exceeds
        maximum tokens, the model summarise previous messages
        to free up token space.
        """
        reserve                     = 1000 # (tokens)
        current_history_tokens      = self.tokens.count_history_tokens(self.chat.to_llm())
        estimate_next               = current_history_tokens + (len(prompt) // 4)

        if self.tokens.model_max_tokens - estimate_next -reserve < 0:
            logger.info("Current tokens exceeds threshold. Triggering compression")
            print("Current tokens exceeds threshold. Compressing session...")
            self.chat.compression(self.model)
            print("Continue session...")
            return
        logger.info("Current tokens within threshold. Continue session")

    # ===================================
    # Execution
    # ===================================

    def ask(
        self,
        prompt: str,
        enable_attachments: bool = False,
        enable_auto_memory_retrieve: bool = config.ENABLE_AUTO_MEMORY_RETRIEVE,
        enable_auto_memory_store: bool = config.ENABLE_AUTO_MEMORY_STORE,
        enable_auto_web_search: bool = config.ENABLE_AUTO_WEB_SEARCH,
        enable_auto_read_dropbox: bool = config.ENABLE_AUTO_READ_DROPBOX,
        file_paths: list[Path] | None = None
    ) -> None | str:
        """
        Model decide what memories to read.

        Manage tokens, compress session if needed.

        Note:
        - Only the user question and LLM response will be
          stored into chat history.

        Process:
        1. Model decides from {prompt} --> {memory_entries}
        2. Model decides from {memory_entries} + {prompt} --> {file_content} from dropbox
        3. Model decides from {memory_entries} + {file_content} + {prompt} --> {search_results}
        """
        self._manage_token_budget(prompt)

        cmd, user_prompt = _detect_cmd(prompt)

        if cmd:
            # Only use 'user_prompt' as 'prompt' here

            if cmd == "/forget":
                logger.info("'/forget' command triggered")
                response = self.command.cmd_forget(user_prompt)

                if response:
                    print(response)
                return
                # ==============
                # // End here //
                # ==============

            if cmd == "/memorise":
                logger.info("'/memorise' command triggered")
                response = self.command.cmd_memorise(
                    prompt=user_prompt,
                    enable_attachments=enable_attachments,
                    file_paths=file_paths
                )

                if response:
                    print(response)
                return
                # ==============
                # // End here //
                # ==============

            if cmd == "/recall":
                logger.info("'/recall' command triggered")
                # User's question and agent's response were saved
                response = self.command.cmd_recall(
                    prompt=user_prompt,
                    enable_attachments=enable_attachments,
                    enable_auto_memory_retrieve=enable_auto_memory_retrieve,
                    file_paths=file_paths
                )

                if response:
                    print(response)
                return
                # ==============
                # // End here //
                # ==============

            if cmd == "/search":
                logger.info("'/search' command tirggered")
                # User's question were saved
                response = self.command.cmd_search(
                    prompt=user_prompt,
                    enable_attachments=enable_attachments,
                    file_paths=file_paths
                )

                if response:
                    print(response)
                self.memory.toggle_auto_store_memory_entries(
                    enable_auto_memory_store= True,
                    model_max_tokens=self.get_model_max_tokens,
                    context=self.chat.to_llm()
                )
                return
                # ==============
                # // End here //
                # ==============

        messages = self.chat.to_llm()

        attachments_content = self.attachments.files(
            messages=messages,
            enable_attachments=enable_attachments,
            file_paths=file_paths
        )

        memory_entries = self.memory.toggle_auto_retrive_memory_entry(
            enable_auto_memory_retrieve=enable_auto_memory_retrieve, 
            prompt=prompt
        )

        file_contents, found_file_paths, read_dropbox = self.file_reader.toggle_auto_read_dropbox(
            messages=messages, 
            prompt=prompt,
            memory_entries=memory_entries, 
            enable_attachments=enable_attachments,
            attachments_content=attachments_content,
            enable_auto_read_dropbox=enable_auto_read_dropbox
        )

        search_results, query_with_urls, search = self.search_agent.toggle_auto_web_search(
            messages=messages,
            prompt=prompt,
            memory_entries=memory_entries,
            file_contents=file_contents,
            enable_attachments=enable_attachments,
            added_file_contents=attachments_content,
            enable_auto_web_search=enable_auto_web_search
        )
        
        # Model answer
        if enable_attachments == True:
            messages.append({
                "role": "user",
                "content": f"# Retrieved memory entries\n\n{memory_entries}\n\n---\n\n# Previously uploaded files\n\n{file_contents}\n\n---\n\n# Web search results\n\n{search_results}\n\n---\n\n# User input\n\n{prompt}\n\n## Uploaded files\n\n{attachments_content}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"# Retrieved memory entries\n\n{memory_entries}\n\n---\n\n# Previously uploaded files\n\n{file_contents}\n\n---\n\n# Web search results\n\n{search_results}\n\n---\n\n# User input\n\n{prompt}\n\n"
            })
        answer, prompt_tokens, output_tokens = LLM.model_response(messages=messages, model = self.model)

        # Save messages
        self.chat.append_user_message_with_metadata(content=prompt, state="external")
        self.chat.append_assistant_message_with_metadata(
            content=answer,
            state="external",
            attachments=found_file_paths,
            query_with_urls=query_with_urls,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )

        # Update dropbox metadata
        if file_paths:
            logger.info("Updating dropbox metadata for: %s", str(file_paths))
            print("Updataing dropbox metadata...")
            self.file_reader._add_metadata_and_summary(file_paths)
        
        self.memory.toggle_auto_store_memory_entries(
            enable_auto_memory_store= True,
            model_max_tokens=self.get_model_max_tokens,
            context=self.chat.to_llm()
        )

        self.memory.toggle_auto_store_memory_entries(
            enable_auto_memory_store= True,
            model_max_tokens=self.get_model_max_tokens,
            context=self.chat.to_llm()
        )

        return
        # ==============
        # // End here //
        # ==============
