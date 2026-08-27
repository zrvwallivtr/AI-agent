from operator import is_
from agent import format_context
import ollama
from pathlib import Path

from src.config.models import MODEL, MODEL_MAX_TOKENS
from src.config.postgres import conn
from src.agent.models.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat_logs import ChatLogs
from src.agent.memory import Memory
from src.agent.cmd_functions import Command
from src.agent import format_context as fmt_cont
from src.tools import KnowledgeBase, DocumentKnowledgeBase
from src.logger import get_logger


log = get_logger(__name__)


# def _last_token_usage(msgs: list[dict]) -> tuple[int, int]:
#     """Returns p_tkns and o_tkns from the latest assistant message."""
#     for msg in reversed(msgs):
# 
#         if msg["role"] == "assistant":
#             p_tkns = msg.get("p_tkns", 0)
#             o_tkns = msg.get("o_tkns", 0)
#             log.info("Latest assistant response found (p_tkns=%d, o_tkns=%d)", p_tkns, o_tkns)
#             return p_tkns, o_tkns
# 
#     log.warning("Latest assistant response not found")
#     return 0, 0 # no previous assistant message yet (first turn)


def _detect_cmd(prompt: str) -> tuple[str | None, str]:
    """Extracts shortcut if detected."""
    question_trimmed = prompt.strip()

    if question_trimmed.startswith(f"/"):
        parts           = question_trimmed.split(" ", 1)
        cmd             = parts[0]
        cleaned_text    = parts[1].strip() if len(parts) > 1 else ""
        log.debug("Command detected: %s", cmd)
        return cmd, cleaned_text

    return None, prompt


class Agent:
    def __init__(
        self,
        model: str | None = None,
        sess_name: str | None = None,
        project: str | None = None
    ):
        model = MODEL if model is None else model
        self._validate_model(model)

        if MODEL_MAX_TOKENS:
            self.tokens = Tokens(model, MODEL_MAX_TOKENS)
        else:
            self.tokens = Tokens(model)

        self.conn       = conn
        self.model      = model
        self.sess_name  = sess_name
        self.project    = project

        self.chat_logs  = ChatLogs(conn=self.conn, sess_name=self.sess_name)
        self.mem        = Memory(
            conn=self.conn, chat_logs=self.chat_logs, project=project
        )
        self.kw_bs      = KnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.cmd        = Command(
            conn=self.conn, chat_logs=self.chat_logs, model=model, sess_name=self.sess_name, project=project
        )
        self.doc_kw_bs  = DocumentKnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
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
                raise ValueError(
                    f"Error: Unknown or unavailable model: '{model}'."
                    f"Run 'ollama pull {model}' to install model."
                )
            known_models = set(local_models) - set(unknown_models)
            log.info("Detected %s known model(s) and %s unknown model(s)", known_models, unknown_models)

        except Exception as e:
            # If ollama is down
            if isinstance(e, ValueError):
                raise e
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
        is_auto_doc_rtve: bool = True,
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

        # === SLASH COMMANDS ====================================

        if cmd:
            # Only use 'user_prompt' as 'prompt' here
            if cmd == "/memorise":
                log.debug("'/memorise' command triggered")
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
                log.debug("'/recall' command triggered")
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

        # === FULL CONTEXT ======================================

        # Auto retrieve relevant memories
        mem_list = self.mem.toggle_auto_retrive_memory_entries(
            is_auto_mem_rtve=is_auto_mem_rtve, prompt=prompt
        )

        # Auto retrieve relevant session documents
        doc_list = self.doc_kw_bs.toggle_auto_retrieve_sess_docs(
            is_auto_doc_rtve=is_auto_doc_rtve, prompt=prompt
        )

        # Attachments (manual call by user)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )

        # Auto web search

        cmbind_prompt = fmt_cont.build_prompt(
            prompt=prompt, mem_list=mem_list, doc_list=doc_list, attchmnt_dict=attchmnt_dict
        )
        msgs.append({"role": "user", "content": cmbind_prompt})
        log.debug("Appended new message to current messages")

        # === MODEL ANSWER ======================================

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

        # === STORE MEMORY(S) ===================================

        # self.memory.toggle_auto_store_memory_entries(
        #     enable_auto_memory_store=True,
        #     model_max_tokens=self.get_model_max_tokens,
        #     context=self.chat.to_llm()
        # )

        # === STORE ATTACHMENT(S) ===============================

        if attchmnt_dict:
            log.info(
                "Storing %d uploaded attachment(s) to session '%s' knowledge base",
                len(attchmnt_dict),
                self.sess_name
            )
            for doc_path, cont in attchmnt_dict.items():
                notify = self.doc_kw_bs.embed_txt_and_add_doc_to_kw_bs(doc_path, cont)
                print(notify)
        return
        # // END HERE //
