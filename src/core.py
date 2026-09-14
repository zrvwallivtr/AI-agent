import socket
from pathlib import Path

from src.config.postgres import conn
from src.config import models

from src import format_context
from src.agent import (
    ollama,
    llm,
    embed,
    chat_logs,
    tokenizers
)
from src.rag import (
    memory,
    knowledge_base,
    document_knowledge_base
)
from src import slash_cmds
from src import logger


app_log = logger.app_logger(f"{__name__}.app")

MODEL               = models.MODEL
MODEL_MAX_TOKENS    = models.MODEL_MAX_TOKENS
EMBED_MODEL         = models.EMBED_MODEL
EMBED_MAX_TOKENS    = models.EMBED_MAX_TOKENS

Tknizr                  = tokenizers.Tknizr
ChatLogs                = chat_logs.ChatLogs
Memory                  = memory.Memory
KnowledgeBase           = knowledge_base.KnowledgeBase
DocumentKnowledgeBase   = document_knowledge_base.DocumentKnowledgeBase


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


def _is_connected(host="1.1.1.1", port=53, timeout=3) -> bool:
    """
    Returns True if the system can connect to the host/port,
    otherwise returns false.
    Host (Cloudflare DNS):  1.1.1.1
    Port (DNS traffic):     53
    """
    try:
        # Create socket object with connection timeout
        socket.setdefaulttimeout(timeout)

        # Attempt to connect to the host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True

    except (socket.timeout, OSError):
        return False


class Agent:
    def __init__(
        self,
        sess_name: str | None = None,
        project: str | None = None
    ):
        self.tknizr = Tknizr(MODEL)

        self.sess_name  = sess_name
        self.project    = project

        self.conn       = conn
        self.chat_logs = ChatLogs(
            conn=self.conn, sess_name=self.sess_name
        )
        self.mem = Memory(
            conn=self.conn, chat_logs=self.chat_logs, project=project
        )
        self.kw_bs = KnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        self.slash_cmd = slash_cmds.SlashCmds(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name, project=project
        )
        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )
        # self.search_agent   = SearchAgent()


    # ===================================
    # Token management
    # ===================================

    def _manage_token_budget(self, prompt: str) -> None:
        """
        Reserves extra tokens for model response. If exceeds
        maximum tokens, the model summarise previous messages
        to free up token space.
        """
        app_log.debug("Estimating token usage for session '%s'", self.sess_name)
        reserve = 1000 # (tokens)
        curr_hist_tkns = self.tknizr.count_history_tokens(self.chat_logs.get_actv_convs())
        if not curr_hist_tkns:
            return

        est_next = curr_hist_tkns + (len(prompt) // 4)

        if not self.tknizr.model_max_tkns:
            app_log.warning("Failed to load 'manage token budget feature': Model maximum token limit not set")
            return

        if self.tknizr.model_max_tkns - est_next - reserve < 0:
            app_log.info("Current tokens exceeds threshold")
            self.chat_logs.auto_compresss_active_conv()
            return


    # ===================================
    # Execution
    # ===================================

    def ask(
        self,
        prompt: str,
        is_auto_mem_rtve: bool = True,
        is_auto_mem_store: bool = False,
        is_auto_doc_rtve: bool = True,
        is_auto_web_sear: bool = False,
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

        # === SLASH COMMANDS ====================================
        cmd, user_prompt = _detect_cmd(prompt)

        if cmd: # Only use 'user_prompt' as 'prompt' here
            if cmd == "/memorise":
                app_log.debug("'/memorise' command triggered")
                msgs.append(llm.user_message(user_prompt))
                self.slash_cmd.cmd_memorise(
                    prompt=user_prompt, is_attchmnt=is_attchmnt, paths=paths
                )
                return
                # // END HERE //

            if cmd == "/recall":
                app_log.debug("'/recall' command triggered")
                msgs.append(llm.user_message(user_prompt))
                self.slash_cmd.cmd_recall(
                    prompt=user_prompt, is_attchmnt=is_attchmnt, paths=paths
                )
                return
                # // END HERE //

            if cmd == "/compress":
                app_log.debug("'/compress' command triggered")
                msgs.append(llm.user_message(user_prompt))
                self.slash_cmd.cmd_compress(
                    prompt=user_prompt, is_attchmnt=is_attchmnt, paths=paths
                )
                return
                # // END HERE //

            if cmd == "/search":
                app_log.debug("'/search' command tirggered")
                # User's question were saved
                self.slash_cmd.cmd_search(
                    prompt=user_prompt, is_attchmnt=is_attchmnt, paths=paths
                )
                return
                # // END HERE //

        # === FULL CONTEXT ======================================
        embed_response = embed.embedding_content(prompt)
        if embed_response:
            prompt, prompt_embdings, prompt_tkns = embed_response
        else:
            app_log.warning(
                "Turning off features that requires embeddings: "
                "Auto memory retrieval, Auto document content retrieval"
            )
            prompt_embdings = [0.0]
            prompt_tkns = 0
            is_auto_mem_rtve = False
            is_auto_doc_rtve = False

        # AUTO RETRIEVE RELEVANT MEMORIES
        mem_list = self.mem.toggle_auto_retrive_memory_entries(
            is_auto_mem_rtve=is_auto_mem_rtve,
            prompt=prompt,
            prompt_embdings=prompt_embdings
        )

        # AUTO RETRIEVE RELEVANT SESSION DOCUMENTS
        doc_list = self.doc_kw_bs.toggle_auto_retrieve_sess_docs(
            is_auto_doc_rtve=is_auto_doc_rtve,
            prompt=prompt,
            prompt_embdings=prompt_embdings
        )

        # UPLOADED ATTACHMENTS (OPTIONAL)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )

        # AUTO WEB SEARCH
        # if _is_connected() and is_auto_web_sear:

        # ALL CONTEXT COMBINED (WON'T BE SAVED TO CHAT HISTORY)
        cmbind_prompt = format_context.build_prompt(
            prompt=prompt, mem_list=mem_list, doc_list=doc_list, attchmnt_dict=attchmnt_dict
        )
        msgs.append(llm.user_message(cmbind_prompt))

        # === MODEL ANSWER ======================================
        response = llm.model_response(model=MODEL, msgs=msgs)
        if not response:
            return
        ans, p_tkns, o_tkns = response

        # Calculate total tokens
        total_p_tkns = prompt_tkns + p_tkns
        total_o_tkns = o_tkns

        # === SAVE MESSAGES =====================================
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=ans,
            state="external",
            attchmnts=paths,
            p_tkns=total_p_tkns,
            o_tkns=total_o_tkns
        )

        # === STORE MEMORY(S) ===================================
        # self.memory.toggle_auto_store_memory_entries(
        #     is_auto_mem_store=is_auto_mem_store,
        #     model_max_tokens=self.get_model_max_tokens,
        #     context=self.chat.to_llm()
        # )

        # === STORE ATTACHMENT(S) ===============================
        if attchmnt_dict:
            self.doc_kw_bs.store_attachments(attchmnt_dict)
        return
        # // END HERE //
