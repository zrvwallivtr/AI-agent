from re import search
import ollama
from pathlib import Path

from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.agent.chat_logs import ChatLogs
from src.agent.memory_embeddings import MemoryEmbed
from src.agent.attachments import Attachments
from src.tools.search import is_connected, SearchAgent
from src.tools.doc_knowledge_base import DocKnowledgeBase
from src import config
from src.logger import get_logger


logger = get_logger(__name__)


class Command:
    def __init__(
        self,
        model: str | None = None,
        sess_name: str | None = None,
        project: str | None = None
    ):
        self.model          = model
        self.sess_name      = sess_name
        self.project        = project
        self.chat_logs      = ChatLogs(sess_name=self.sess_name)
        self.mem_emb        = MemoryEmbed(project=self.project)
        # self.file_reader    = DocKnowledgeBase(session=self.sess_name)
        # self.attachments    = Attachments(session=self.sess_name)
        # self.search_agent   = SearchAgent()

    # ========================================================
    # Memory
    # ========================================================

    # def cmd_forget(self, prompt: str) -> str:
    #     """
    #     Find exact match in memory from user prompt and remove said entry.
    #     Note: User's question won't be saved.
    #     """
    #     if not prompt:
    #         logger.warning("Command '/forget' aborted: No prompt was provided")
    #         return "Please specify what to forget."

    #     match_data = self.memory.get_exact_match(prompt)

    #     if not match_data:
    #         logger.warning("Command '/forget' failed: No matching memory entry found")
    #         return f"Error: No matching memory found."

    #     target_id, matched_content = match_data

    #     # User confirm options
    #     print(f"Warning: [ChromaDB] Initializing delte sequence for: '{matched_content}...'")
    #     choice = input("Press [Enter] to confirm deletion or type [c] to cancel:")
    #     if choice == "c":
    #         logger.info("Memory deletion cancelled by the user")
    #         return "Deletion cancelled."
    #     self.memory.delete_from_db([target_id])
    #     logger.info("Deleted memory entry from database (target_id=%s)", target_id)
    #     return "Entry deleted."

    def cmd_memorise(
        self,
        prompt: str,
        # enable_attachments: bool,
        # file_paths: list[Path] | None = None
    ) -> str:
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

        # Read attachments (optional)
        # attachments_content = None
        # if enable_attachments == True:
        #     attachments_content = self.attachments.files(
        #         messages=messages,
        #         enable_attachments=enable_attachments,
        #         file_paths=file_paths,
        #     )
        #     logger.info("Retrieved extra context from uploaded files")

        # Format messages and extract memory
        # cmbind_prompt = (
        #     f"User prompt\n\n"
        #     f"{prompt}\n\n"
        #     f"---\n\n"
        #     f"Attachment(s)\n\n"
        #     f"{attachments_content}"
        # ) if attachments_content else prompt

        messages.append({"role": "user", "content": prompt})
        print(f"Extracting content from user's input...")
        created_ids, p_tkns, o_tkns = self.mem_emb.extract_and_store_mem_from_conv(extraction="manual", prompt=prompt)

        # Print to terminal
        mem_dict = self.mem_emb.get_mem_content_from_ids(created_ids)
        if mem_dict:
            print("CONTENT SAVED:")
            for cont, ctgry in mem_dict.items():
                print(f"[{ctgry}] {cont}\n\n")
        else:
            return "Error: No data was extracted by the model."

        # User confirm options
        choice = input("Press [Enter] to continue or type [u] to undo:")
        if choice == "u":
            self.mem_emb.delete_mem(created_ids)
            return "Entry deleted."

        # Save messages
        confirmation = "Important information(s) has been extracted added to database."
        self.chat_logs.add_conv_turn(
            prompt=prompt,
            response=confirmation,
            state="external",
            p_tkns=p_tkns,
            o_tkns=o_tkns
        )
        return "Memory saved to database"

    def cmd_recall(
        self,
        prompt: str,
        # enable_attachments: bool,
        # enable_auto_memory_retrieve: bool,
        # file_paths: list[Path] | None = None
    ) -> str | None:
        """
        Retrieve and print relevant entries according to user prompt.

        Model interpret:
        {attachments (optional)} + {memory entries}
        """
        if not prompt:
            logger.warning("Command '/recall' aborted: No prompt was provided")
            return "Please specify what to recall."

        messages = self.chat_logs.get_entire_conv()

        # Read attachments (optional)
        # attachments_content = None
        # if enable_attachments == True:
        #     attachments_content = self.attachments.files(
        #         messages=messages,
        #         enable_attachments=enable_attachments,
        #         file_paths=file_paths,
        #     )
        #     logger.info("Retrieved extra context from uploaded files")

        # Retrieve memory(s)
        mem_entries = self.mem_emb.query_similar_content(prompt)
        if isinstance(mem_entries, str):
            return mem_entries
        mem_recalled = ""
        for entry in mem_entries:
            cont = entry["content"]
            ctgry = entry["category"]
            score = entry["similarity"]
            mem_recalled += f"* [{ctgry}] {cont} (similarity: {score:.2f})\n"

        # Model interpret recalled memory(s)
        mem_sect = (
            f"# Recalled entry(s)\n\n"
            f"{mem_recalled}\n\n"
            f"--\n\n"
        ) if mem_recalled else ""
        # attachments_sect = (
        #     f"# Attachment(s)\n\n"
        #     f"{attachments_content}\n\n"
        #     if attachments_content else ""
        # )
        # cmbind_prompt = mem_sect + attachments_sect
        cmbind_prompt = mem_sect
        answer, p_tkns, o_tkns = LLM.response_memory_recall_format(
            model=self.mem_emb.model,
            system_prompt=config.MEM_RECALL_INTERPRET_PROMPT,
            recalled=cmbind_prompt,
            prompt=prompt,
            context=messages
        )

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
