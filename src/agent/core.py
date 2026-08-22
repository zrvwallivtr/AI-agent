from operator import is_
import ollama
from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat_logs import ChatLogs
from src.agent.memory import Memory
from src.agent.cmd_functions import Command
#from src.tools.search import is_connected, SearchAgent
from src.tools.documents import Document
from src.tools.knowledge_base import KnowledgeBase
from src.logger import get_logger
from src import config


logger = get_logger(__name__)


def _last_token_usage(msgs: list[dict]) -> tuple[int, int]:
    """Returns p_tkns and o_tkns from the latest assistant message."""
    for msg in reversed(msgs):

        if msg["role"] == "assistant":
            p_tkns = msg.get("p_tkns", 0)
            o_tkns = msg.get("o_tkns", 0)
            logger.info("Latest assistant response found (p_tkns=%d, o_tkns=%d)", p_tkns, o_tkns)
            return p_tkns, o_tkns

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
        sess_name: str | None = None,
        project: str | None = None
    ):
        model = config.MODEL if model is None else model
        self._validate_model(model)

        if config.MODEL_MAX_TOKENS:
            self.tokens = Tokens(model, config.MODEL_MAX_TOKENS)
        else:
            self.tokens = Tokens(model)

        self.model          = model
        self.sess_name      = sess_name
        self.project        = project
        self.chat_logs      = ChatLogs(sess_name=self.sess_name)
        self.mem            = Memory(project=project)
        self.kw_bs          = KnowledgeBase(sess_name=self.sess_name)
        self.cmd            = Command(model=model, sess_name=self.sess_name, project=project)
        self.doc            = Document(sess_name=self.sess_name)
        # self.search_agent   = SearchAgent()

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
        current_history_tokens      = self.tokens.count_history_tokens(self.chat_logs.get_entire_conv())
        estimate_next               = current_history_tokens + (len(prompt) // 4)

        # if self.tokens.model_max_tokens - estimate_next -reserve < 0:
        #     logger.info("Current tokens exceeds threshold. Triggering compression")
        #     print("Current tokens exceeds threshold. Compressing session...")
        #     self.chat_logs.compression(self.model)
        #     print("Continue session...")
        #     return
        # logger.info("Current tokens within threshold. Continue session")

    # ===================================
    # Execution
    # ===================================

    def ask(
        self,
        prompt: str,
        is_auto_mem_rtve: bool = True,
        is_attchmnt: bool = False,
        paths: list[Path] | None = None
    ) -> None | str:
        """
        Model decide what memories to read.
        Manage tokens, compress session if needed.

        Note:
        - Only the user question and LLM response will be
          stored into chat history.
        """
        msgs = self.chat_logs.get_entire_conv()

        cmd, user_prompt = _detect_cmd(prompt)

        # == SLASH COMMANDS ====================================

        if cmd:
            # Only use 'user_prompt' as 'prompt' here
            if cmd == "/memorise":
                logger.info("'/memorise' command triggered")
                msgs.append({"role": "user", "content": user_prompt})
                response = self.cmd.cmd_memorise(
                    prompt=user_prompt,
                    is_attchmnt=is_attchmnt,
                    paths=paths
                )
                if response:
                    print(response)
                return
                # // END HERE //

            if cmd == "/recall":
                logger.info("'/recall' command triggered")
                msgs.append({"role": "user", "content": user_prompt})
                response = self.cmd.cmd_recall(
                    prompt=user_prompt,
                    is_attchmnt=is_attchmnt,
                    paths=paths
                )
                if response:
                    print(response)
                return
                # // END HERE //

            # if cmd == "/search":
            #     logger.info("'/search' command tirggered")
            #     # User's question were saved
            #     messages.append({"role": "user", "content": user_prompt})
            #     response = self.command.cmd_search(
            #         prompt=user_prompt,
            #         enable_attachments=enable_attachments,
            #         file_paths=file_paths
            #     )
            #     if response:
            #         print(response)
            #     self.memory.toggle_auto_store_memory_entries(
            #         enable_auto_memory_store= True,
            #         model_max_tokens=self.get_model_max_tokens,
            #         context=self.chat.to_llm()
            #     )
            #     return
                # // End here //

        # == FULL CONTEXT ======================================

        # Auto retrieve memory
        mem_list = self.mem.toggle_auto_retrive_memory_entries(
            is_auto_mem_rtve=is_auto_mem_rtve, prompt=prompt
        )
        mem_sect = "# Retrieved memory(s)\n\n".join(
            f"## Memory\n"
            f"Content: {item['content']}\n"
            f"Similarity score: {item['similarity']}\n"
            for item in mem_list
        ).join("---\n\n") if isinstance(mem_list, list) and mem_list else ""

        # Attachments (manual call by user)
        attchmnt_dict = self.doc.get_attachments_content(
            is_attchmnt=is_attchmnt, doc_paths=paths
        )
        attach_sect = "# Attachment(s)\n\n".join(
            f"## Document name: {doc_name}\n"
            f"Content:\n"
            f"{content}\n\n"
            for doc_name, content in attchmnt_dict.items()
        ).join("---\n\n") if attchmnt_dict else ""

        # Auto read knowledge base
        # - Could add user specify retrieve list no. during session, if not specified use default.
        # dropbox_file_data = self.file_reader.toggle_auto_read_dropbox(
        #     messages=messages, 
        #     prompt=prompt,
        #     memory_entries=memory_entries, 
        #     enable_attachments=enable_attachments,
        #     attach_file_data=attach_files_data,
        #     enable_auto_read_dropbox=enable_auto_read_dropbox
        # )
        # dropbox_sect = "# Previously uploaded file(s)\n\n".join(
        #     f"Filename: {filename}\n"
        #     f"Content: {content}\n\n"
        #     for filename, content in dropbox_file_data.items()
        # ).join("---\n\n")

        # Auto web search
        # search_results, query_with_urls, search = self.search_agent.toggle_auto_web_search(
        #     messages=messages,
        #     prompt=prompt,
        #     enable_attachments=enable_attachments,
        #     enable_auto_web_search=enable_auto_web_search,
        #     memory_entries=memory_entries,
        #     file_contents=dropbox_file_data,
        #     attach_file_data=attach_files_data,
        # )
        # web_search_sect = (
        #     f"# Web search results\n\n"
        #     f"{search_results}\n\n"
        #     f"---\n\n"
        # )

        # User prompt
        prompt_sect = (
            f"# User prompt\n\n"
            f"{prompt}"
        )

        cmbind_prompt = mem_sect + attach_sect + prompt_sect
        msgs.append({"role": "user", "content": cmbind_prompt})

        # == MODEL ANSWER ======================================

        response, p_tkns, o_tkns = LLM.model_response(
            msgs=msgs, model=self.model
        )
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=response,
            state="external",
            attchmnts=paths,
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )

        # == STORE MEMORY(S) ===================================

        # self.memory.toggle_auto_store_memory_entries(
        #     enable_auto_memory_store=True,
        #     model_max_tokens=self.get_model_max_tokens,
        #     context=self.chat.to_llm()
        # )

        # == STORE ATTACHMENT(S) ===============================

        if paths:
            for path in paths:
                notify = self.kw_bs.embed_and_add_doc_to_kw_bs(path)
                print(notify)
        return
        # // END HERE //
