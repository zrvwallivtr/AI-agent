from re import search
from agent import format_context
from pathlib import Path

from src.agent.models.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat_logs import ChatLogs
from src.agent.memory import Memory
from src.agent import format_context as fmt_cont
from src.tools.document_knowledge_base import DocumentKnowledgeBase
# from src.tools.search import is_connected, SearchAgent
from src import config
from src.logger import get_logger
from tools.knowledge_base import KnowledgeBase


logger = get_logger(__name__)


class Command:
    def __init__(
        self,
        model: str | None = None,
        sess_name: str | None = None,
        project: str | None = None
    ):
        self.model      = model
        self.sess_name  = sess_name
        self.project    = project
        self.chat_logs  = ChatLogs(sess_name=self.sess_name)
        self.mem        = Memory(project=self.project)
        self.kw_bs      = KnowledgeBase(sess_name=self.sess_name)
        self.doc_kw_bs  = DocumentKnowledgeBase(sess_name=self.sess_name)
        # self.search_agent   = SearchAgent()

    # ========================================================
    # Memory
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
        Note: User's question will be saved directly, this function
              will then generate a pre-written assistant message and
              saved.
        """
        if not prompt:
            logger.warning("Command '/memorise' aborted: No prompt was provided")
            return "Please specify what to memorize."

        messages = self.chat_logs.get_entire_conv()

        # == FULL CONTEXT ==========================================

        # Attachments (manual call by user)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, doc_paths=paths
        )

        cmbind_prompt = fmt_cont.build_prompt(prompt=prompt, attchmnt_dict=attchmnt_dict)
        messages.append({"role": "user", "content": cmbind_prompt})

        # == EXTRACT AND STORE MEMORY(S) ===========================

        print(f"Extracting content from user's input...")
        created_ids, p_tkns, o_tkns = self.mem.extract_and_store_mem_from_conv(
            extraction="manual", prompt=prompt
        )

        # Print to terminal
        mem_dict = self.mem.get_mem_content_from_ids(created_ids)
        if mem_dict:
            print("CONTENT SAVED:")
            for cont, ctgry in mem_dict.items():
                print(f"[{ctgry}] {cont}\n\n")
        else:
            return "Error: No data was extracted by the model."

        # User confirm options
        choice = input("Press [Enter] to continue or type [u] to undo:")
        if choice == "u":
            self.mem.delete_mem(created_ids)
            return "Entry deleted."

        # Save messages
        confirmation = "Important information(s) has been extracted added to database."
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=confirmation,
            state="external",
            attchmnts=paths,
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )

        # == STORE ATTACHMENT(S) ===================================

        if attchmnt_dict:
            for doc_path, cont in attchmnt_dict.items():
                notify = self.doc_kw_bs.embed_txt_and_add_doc_to_kw_bs(doc_path, cont)
                print(notify)
        return

    def cmd_recall(
        self,
        prompt: str,
        is_attchmnt: bool,
        paths: list[Path] | None = None
    ) -> str | None:
        """Retrieve and print relevant entries according to user prompt."""
        if not prompt:
            logger.warning("Command '/recall' aborted: No prompt was provided")
            return "Please specify what to recall."

        messages = self.chat_logs.get_entire_conv()

        # == FULL CONTEXT ==========================================

        # Retrieve memory(s)
        mem_list = self.mem.query_similar_content(prompt)

        # Attachments (manual call by user)
        attchmnt_dict = self.doc_kw_bs.get_attachments_content(
            is_attchmnt=is_attchmnt, doc_paths=paths
        )

        cmbind_prompt = fmt_cont.build_prompt(
            prompt=prompt, mem_list=mem_list, attchmnt_dict=attchmnt_dict
        )

        # == MODEL ANSWER ==========================================

        # Model interpret recalled memory(s)
        answer, p_tkns, o_tkns = LLM.response_memory_recall_format(
            model=self.mem.model,
            system_prompt=config.MEM_RECALL_INTERPRET_PROMPT,
            prompt=cmbind_prompt,
            context=messages
        )

        # Save messages
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=answer,
            state="external",
            attchmnts=paths,
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )

        # == STORE ATTACHMENT(S) ===================================

        if attchmnt_dict:
            for doc_path, cont in attchmnt_dict.items():
                notify = self.doc_kw_bs.embed_txt_and_add_doc_to_kw_bs(doc_path, cont)
                print(notify)
        return

    # ========================================================
    # Search
    # ========================================================

    # def cmd_search(
    #     self,
    #     prompt: str,
    #     enable_attachments: bool,
    #     file_paths: list[Path] | None = None
    # ) -> str | None:
    #     """Generates, search and answer query based on user prompt."""
    #     if not prompt:
    #         logger.error("Command '/search' aborted: No prompt was provided")
    #         return "Please specify what to search."

    #     messages = self.chat.to_llm()

    #     # Read attachments (optional)
    #     attachments_content = self.attachments.files(
    #         messages=messages,
    #         enable_attachments=enable_attachments,
    #         file_paths=file_paths,
    #     ) if enable_attachments == True else None

    #     # Model generates query
    #     cmbind_prompt = (
    #         f"# User prompt\n\n"
    #         f"{prompt}\n\n"
    #         f"---\n\n"
    #         f"# Attachment(s)\n\n"
    #         f"{attachments_content}"
    #         if attachments_content else prompt
    #     )
    #     print("Generating query...")
    #     query, p_tkns, o_tkns = self.search_agent.generates_query(
    #         context=messages,
    #         prompt=cmbind_prompt
    #     )

    #     # Search
    #     response, query_with_urls, p_tkns, o_tkns = self.search_agent.web_search_and_response(
    #         query=query,
    #         context=self.chat.to_llm(),
    #         prompt=cmbind_prompt,
    #         max_results=config.MAX_RESULTS
    #     )
    #     logger.info("Processed web search response")

    #     # Save messages
    #     self.chat.append_user_message_with_metadata(content=prompt, state="external")
    #     self.chat.append_assistant_message_with_metadata(
    #         content=response,
    #         state="external",
    #         query_with_urls=query_with_urls,
    #         search=True,
    #         p_tkns=p_tkns,
    #         o_tkns=o_tkns
    #     )
    #     return
