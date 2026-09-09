from re import search
from agent import format_context
from pathlib import Path

from src.config import models
from src.config import prompts

from src.agent import (
    LLM,
    Embed,
    ChatLogs,
    build_prompt
)
from src.rag import (
    Memory,
    KnowledgeBase,
    DocumentKnowledgeBase,
    SearchAgent,
    generate_query
)
from src.logger import app_logger


app_log = app_logger(f"{__name__}.app")

MODEL                       = models.MODEL
MEM_RECALL_INTERPRET_PROMPT = prompts.MEM_RECALL_INTERPRET_PROMPT


class SlashCmds:
    def __init__(
        self,
        conn,
        chat_logs: ChatLogs,
        sess_name: str | None = None,
        project: str | None = None
    ):
        self.conn       = conn
        self.sess_name  = sess_name
        self.project    = project
        self.chat_logs  = chat_logs

        self.embed      = Embed()

        self.mem = Memory(
            conn=self.conn, chat_logs=self.chat_logs, project=self.project
        )

        self.kw_bs = KnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )

        self.doc_kw_bs = DocumentKnowledgeBase(
            conn=self.conn, chat_logs=self.chat_logs, sess_name=self.sess_name
        )

        self.sear_agt = SearchAgent(
            conn=self.conn, sess_name=self.sess_name
        )


    # ========================================================
    # MEMORY
    # ========================================================

    def cmd_memorise(
        self,
        prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None = None
    ) -> str | None:
        """
        Extract key info from user prompt and attachments (optional),
        save extracted memory entries to database.
        NOTE: User's question will be saved directly, this function
              will then generate a pre-written assistant message and
              saved.
        """
        if not prompt:
            app_log.warning("Command '/memorise' aborted: No prompt was provided")
            return "Please specify what to memorize."

        msgs = self.chat_logs.get_actv_convs()

        # === FULL CONTEXT ==========================================

        # Attachments (manual call by user)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )

        cmbind_prompt = build_prompt(prompt=prompt, attchmnt_dict=attchmnt_dict)
        msgs.append({"role": "user", "content": cmbind_prompt})
        app_log.debug("Appended new message to current messages")

        # === EXTRACT AND STORE MEMORY(S) ===========================

        print(f"Extracting content from user's input...")
        response = self.mem.extract_and_store_mem_from_conv(
            extraction="manual", prompt=prompt
        )
        if not response:
            return "Error: No data was extracted by the model."
        created_ids, p_tkns, o_tkns, emb_tkns = response

        # == PRINT TO TERMINAL ======================================

        mem_dict = self.mem.get_mem_content_from_ids(created_ids)
        if mem_dict:
            print("CONTENT SAVED:")
            for cont, ctgry in mem_dict.items():
                print(f"[{ctgry}] {cont}\n\n")
        else:
            return "Error: Unable to retrieve entry(s) from memory."

        # User confirm options
        choice = input("Press [Enter] to continue or type [u] to undo:")
        if choice == "u":
            app_log.debug("User selected [u]: Undoing saved memory(s)")
            self.mem.delete_mem(created_ids)
            return "Entry deleted."

        # /////////////////////////////////////////////////////
        # Token usage for memory extraction: p_tkns, o_tkns
        # /////////////////////////////////////////////////////

        # Save messages
        mock_resp = "Important information(s) has been extracted added to database."
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=mock_resp,
            state="external",
            attchmnts=paths,
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )

        # === STORE ATTACHMENT(S) ===================================

        if attchmnt_dict:
            app_log.info(
                "Storing %d uploaded attachment(s) to session '%s' knowledge base",
                len(attchmnt_dict),
                self.sess_name
            )
            for doc_path, data in attchmnt_dict.items():
                cont = data["content"]
                format = data["format"]
                count = self.doc_kw_bs.embed_and_add_to_kw_bs(
                    path=doc_path, cont=cont, format=format
                )
                if not count:
                    app_log.warning(
                        "Failed to store attachment '%s' to session '%s' knowledge base",
                        doc_path,
                        self.sess_name
                    )
                    print("Error: Failed to embed/store attachment to knowledge base")
                    continue
                app_log.info(
                    "Stored attachment '%s' as %s chunks to session '%s' knowledge base",
                    doc_path,
                    count,
                    self.sess_name
                )
                print("Attachment stored to session knowledge base")
                # /////////////////////////////////////////////
                # Embedding token count: emb_tkns + tkn_used
                # /////////////////////////////////////////////
        return


    def cmd_recall(
        self,
        prompt: str,
    ) -> str | None:
        """Retrieve and print relevant entries according to user prompt."""
        if not prompt:
            app_log.warning("Command '/recall' aborted: No prompt was provided")
            return "Please specify what to recall."

        msgs = self.chat_logs.get_actv_convs()

        # === FULL CONTEXT ==========================================

        # Retrieve memory(s)
        prompt, prompt_embeddings, emb_tkns = self.embed.embedding_content(prompt)
        mem_list = self.mem.query_similar_content(prompt, prompt_embeddings)

        cmbind_prompt = build_prompt(
            prompt=prompt, mem_list=mem_list
        )

        # === MODEL ANSWER ==========================================

        # Model interpret recalled memory(s)
        answer, p_tkns, o_tkns = LLM.response_memory_recall_format(
            model=self.mem.model,
            system_prompt=MEM_RECALL_INTERPRET_PROMPT,
            prompt=cmbind_prompt,
            context=msgs
        )

        # ///////////////////////////////////////////////////////////////////
        # Token usage for answering from recalled entries: p_tkns, o_tkns
        # ///////////////////////////////////////////////////////////////////

        # Save messages
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=answer,
            state="external",
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )
        return


    # ========================================================
    # COMPRESS
    # ========================================================

    def cmd_compress(
        self,
        prompt: str,
    ) -> str | None:
        """Retrieve and print relevant entries according to user prompt."""
        if not prompt:
            app_log.warning("Command '/compress' aborted: No prompt was provided")
            return "Please specify compression instructions."

        self.chat_logs.compress_active_conv(prompt)
        return

    # ========================================================
    # SEARCH
    # ========================================================

    def cmd_search(
        self,
        prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None = None
    ) -> str | None:
        """Generates, search and answer query based on user prompt."""
        if not prompt:
            app_log.error("Command '/search' aborted: No prompt was provided")
            return "Please specify what to search."

        msgs = self.chat_logs.get_actv_convs()

        # === FULL CONTEXT ==========================================

        # Attachments (optional)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, attch_paths=paths
        )

        # Get search results
        response = self.sear_agt.query_surface_content(
            context=msgs, prompt=prompt
        )
        if not response:
            return "No results found"
        sear_results, gen_qry_p_tkns, gen_qry_o_tkns = response

        cmbind_prompt = build_prompt(
            prompt=prompt, attchmnt_dict=attchmnt_dict, sear_results=sear_results
        )
        msgs.append({"role": "user", "content": cmbind_prompt})

        # === MODEL ANSWER ==========================================

        # Model interpret search results and answer user's questions
        answer, ans_p_tkns, ans_o_tkns = LLM.model_response(
            model=MODEL,
            msgs=msgs
        )

        # Save messages
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=answer,
            state="external",
            p_tkns=gen_qry_p_tkns + ans_p_tkns,
            o_tkns=gen_qry_o_tkns + ans_o_tkns
        )
        return
