from operator import is_
from agent import format_context
import ollama
from pathlib import Path

from src.config.models import MODEL, MODEL_MAX_TOKENS
from src.config.postgres import conn
from src.agent.models.llm import LLM
from src.agent.models.embed import Embed
from src.agent.tokenizers import Tknizr
from src.agent.chat_logs import ChatLogs
from src.agent.memory import Memory
from src.agent.cmd_functions import Command
from src.agent import format_context as fmt_cont
from src.tools import KnowledgeBase, DocumentKnowledgeBase
from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")


def _detect_cmd(prompt: str) -> tuple[str | None, str]:
    """Extracts shortcut if detected."""
    question_trimmed = prompt.strip()

    if question_trimmed.startswith(f"/"):
        parts           = question_trimmed.split(" ", 1)
        cmd             = parts[0]
        cleaned_text    = parts[1].strip() if len(parts) > 1 else ""
        app_log.debug("Command detected: %s", cmd)
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
            self.tknizr = Tknizr(model, MODEL_MAX_TOKENS)
        else:
            self.tknizr = Tknizr(model)

        self.conn       = conn
        self.model      = model
        self.sess_name  = sess_name
        self.project    = project

        self.embed = Embed()

        self.chat_logs = ChatLogs(conn=self.conn, sess_name=self.sess_name)

        self.mem = Memory(
            conn=self.conn,
            chat_logs=self.chat_logs,
            project=project
        )

        self.kw_bs = KnowledgeBase(
            conn=self.conn,
            chat_logs=self.chat_logs,
            sess_name=self.sess_name
        )

        self.cmd = Command(
            conn=self.conn,
            chat_logs=self.chat_logs,
            model=model,
            sess_name=self.sess_name,
            project=project
        )

        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn,
            chat_logs=self.chat_logs,
            sess_name=self.sess_name
        )

        # self.search_agent   = SearchAgent()


    @property
    def get_model_max_tokens(self) -> int | None:
        """Dynamically fetches the current token ceiling from 'self.token'."""
        return self.tknizr.model_max_tokens


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
                    f"Error: Unknown or unavailable model '{model}'."
                    f"Run 'ollama pull {model}' to install model."
                )
            known_models = set(local_models) - set(unknown_models)
            app_log.debug("Detected %s known model(s) and %s unknown model(s)", known_models, unknown_models)

        except Exception as e:
            # If ollama is down
            if isinstance(e, ValueError):
                raise e
            raise RuntimeError(f"Error: Could not connect to local Ollama service '{e}'")


    # ===================================
    # Token management
    # ===================================

    def _manage_token_budget(self, prompt: str):
        """
        Reserves extra tokens for model response. If exceeds
        maximum tokens, the model summarise previous messages
        to free up token space.
        """
        reserve         = 1000 # (tokens)
        curr_hstry_tkns = self.tknizr.count_history_tokens(self.chat_logs.get_actv_convs())
        if not curr_hstry_tkns:
            return

        estimate_next = curr_hstry_tkns + (len(prompt) // 4)

        if self.tknizr.model_max_tokens - estimate_next -reserve < 0:
            app_log.info("Current tokens exceeds threshold. Compressing session '%s'", self.sess_name)
            print("Current tokens exceeds threshold. Compressing session...")
            self.chat_logs.auto_compresss_active_conv()
            app_log.info("Compression complete. Continue session '%s'", self.sess_name)
            print("Compression complete. Continue session...")
            return

        app_log.debug("Current tokens within threshold. Continue session")


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
        self._manage_token_budget(prompt)

        msgs = self.chat_logs.get_actv_convs()

        cmd, user_prompt = _detect_cmd(prompt)

        # === SLASH COMMANDS ====================================

        if cmd:
            # Only use 'user_prompt' as 'prompt' here
            if cmd == "/memorise":
                app_log.debug("'/memorise' command triggered")
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
                app_log.debug("'/recall' command triggered")
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

            if cmd == "/compress":
                app_log.debug("'/compress' command triggered")
                msgs.append({"role": "user", "content": user_prompt})
                response = self.cmd.cmd_compress(
                    prompt=user_prompt
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

        prompt, prompt_embeddings, prompt_tkns = self.embed.embedding_content(prompt)

        # Auto retrieve relevant memories
        mem_list = self.mem.toggle_auto_retrive_memory_entries(
            is_auto_mem_rtve=is_auto_mem_rtve,
            prompt=prompt,
            prompt_embeddings=prompt_embeddings
        )

        # Auto retrieve relevant session documents
        doc_list = self.doc_kw_bs.toggle_auto_retrieve_sess_docs(
            is_auto_doc_rtve=is_auto_doc_rtve,
            prompt=prompt,
            prompt_embeddings=prompt_embeddings
        )

        # Attachments (manual call by user)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )

        # Auto web search

        # All context combined
        cmbind_prompt = fmt_cont.build_prompt(
            prompt=prompt, mem_list=mem_list, doc_list=doc_list, attchmnt_dict=attchmnt_dict
        )
        msgs.append({"role": "user", "content": cmbind_prompt})
        app_log.debug("Appended new message to current messages")

        # === MODEL ANSWER ======================================

        response, p_tkns, o_tkns = LLM.model_response(
            model=self.model, msgs=msgs
        )

        # Calculate total tokens
        total_p_tkns = prompt_tkns + p_tkns
        total_o_tkns = o_tkns

        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=response,
            state="external",
            attchmnts=paths,
            p_tkns=total_p_tkns,
            o_tkns=total_o_tkns
        )

        # === STORE MEMORY(S) ===================================

        # self.memory.toggle_auto_store_memory_entries(
        #     enable_auto_memory_store=True,
        #     model_max_tokens=self.get_model_max_tokens,
        #     context=self.chat.to_llm()
        # )

        # === STORE ATTACHMENT(S) ===============================

        if attchmnt_dict:
            app_log.info(
                "Storing %d uploaded attachment(s) to session '%s' knowledge base",
                len(attchmnt_dict),
                self.sess_name
            )
            for doc_path, cont in attchmnt_dict.items():
                notify, _ = self.doc_kw_bs.embed_txt_and_add_doc_to_kw_bs(doc_path, cont)
                print(notify)
        return
        # // END HERE //
