import json
import uuid
import ollama
import chromadb
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Literal, get_args

from src import config
from src.agent.chat import Chat
from src.agent.llm import LLM
from src.agent.tokens_handler import Tokens
from src.logger import get_logger


logger = get_logger(__name__)

CATEGORY_TYPES = Literal["preference", "stack", "fact", "project", "instruction", "correction"]

CATEGORIES = list(get_args(CATEGORY_TYPES))


class Memory:
    def __init__(self, session: str | None = None, project: str | None = None):
        self.session            = session
        self.project            = project
        self.model              = config.MODEL
        self.embed_model        = config.EMBED_MODEL
        self.mem_prompt         = config.MEM_PROMPT
        self.mem_manual_prompt  = config.MEM_MANUAL_PROMPT
        self.path               = config.CHROMADB_DIR
        self.tokens             = Tokens(self.embed_model)
        self.chat               = Chat(session=session)

        # Initialize local ChromaDB client
        self.chroma_client  = chromadb.PersistentClient(path=self.path)

        # Decide database location (project_name / global_memory)
        # Note: 'project' must be specified in order for memory to stored in project path.
        collection_name = f"project_{project}" if self.project else "global_memory"

        # Grabs or dynamically initializes collection
        self.vector_db = self.chroma_client.get_or_create_collection(name=collection_name)

    # ============================================================
    # To database
    # ============================================================

    def _append_to_db(
        self,
        content: str,
        category: CATEGORY_TYPES,
        source: str
    ) -> str | None:
        """
        Appends new memory entry incrementally without wiping database.
        Silently overwrite an old entry if the text is duplicated, or 
        add a fresh record if it's completely new and returns its unique
        entry id.
        """
        content_trimmed = content.strip()

        # Check if token count exceeds max tokens of embedded model
        if not self.tokens.check_fit(content_trimmed):
            max_chars = self.tokens.model_max_tokens * 4
            logger.warning(
                "Content exceeds token budget: Current characters=%d, maximum characters=%d",
                len(content_trimmed),
                max_chars
            )
            print("Error: Message exceeds model token count limits")
            return

        # Generate entry id based only its content (case insensitive)
        entry_id = f"mem_{hashlib.md5(content_trimmed.lower().encode()).hexdigest()[:10]}"

        # Generate vector embedding
        response = ollama.embeddings(model=self.embed_model, prompt=content_trimmed)
        if not response:
            logger.warning("Failed to generate embedding for memory entry")
            print("Error: Model failed to generate vector embedding")
            return
        logger.debug("Generated vector embedding for content")

        # Add to ChromaDB
        embedding = response["embedding"]
        self.vector_db.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content_trimmed],
            metadatas=[{
                "category": category,
                "extraction": source,
                "created_at": datetime.now().isoformat()
            }]
        )
        logger.info("Memory content(s) added to the database: id=%s, category=%s", entry_id, category)
        return entry_id

    def _format_and_append_to_db(self, extracted_output: str, source: str) -> list[str]:
        """
        Format every other memory entry to the next line, trims
        out unnecessary spaces and symbols and return a list
        of created database IDs.
        """
        created_ids = []

        # Slice model output into line-by-line format
        for line in extracted_output.split("\n"):
            new_line = line.strip().lstrip("-*• ")

            # Group 1 ([a-zA-Z\s_/]+):
            # - Category name.
            #
            # Group 2 (.*):
            # - Everything after the brackets.
            match = re.search(r"\[([a-zA-Z\s_/]+)\]\s*(.*)", new_line)

            if match:
                category_tag    = match.group(1).strip().lower()
                actual_content  = match.group(2).strip()

                if not actual_content:
                    continue

                category = category_tag if category_tag in CATEGORIES else "fact"

                # Save to database with category
                entry_id = self._append_to_db(
                    content=actual_content,
                    category=category,
                    source=source
                )
                if entry_id:
                    created_ids.append(entry_id)

        return created_ids

    # ============================================================
    # From database
    # ============================================================

    def _grab_category_and_content(self, results: dict):
        """Retrieves 'category' and 'content' in each entry."""
        retrieved_entries = []
        if results and results["documents"] and results["documents"][0]:
            # Zips matching documents and metadata
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                # Appends the extracted "category" and "content"
                retrieved_entries.append({
                    "category": metadata.get("category", "fact"),
                    "content": doc
                })

        return retrieved_entries

    def retrieve_relevant_entry(
        self,
        prompt: str,
        limit: int = config.RETRIEVE_MEM_ENTRY_LIMIT
    ) -> list[dict]:
        """Queries ChromaDB using semantics."""
        # Returns nothing if database is empty
        if self.vector_db.count() == 0:
            logger.debug("Vector search skipped: Database is empty")
            return []

        # Generate embedding for the question
        response = ollama.embeddings(model=self.embed_model, prompt=prompt)
        if not response:
            logger.error("Failed to generate embedding search prompt")
            print("Error: Model failed to generate vector embedding")
            return []
        logger.debug("Generated query embedding for search")

        # Search ChromaDB for top matches with question vector
        query_embedding = response["embedding"]
        results = self.vector_db.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        # Reformat retrieved entries
        retrieved_entries = self._grab_category_and_content(results)

        return retrieved_entries

    def add_memory_entries(self, prompt: str) -> str | None:
        """
        Queries long-term vector storage and adds context
        into user's prompt.
        """
        logger.debug("Retrieving relevant vector memory for prompt context")
        memory = self.retrieve_relevant_entry(prompt, limit=config.RETRIEVE_MEM_ENTRY_LIMIT)

        if not memory:
            logger.info("No relevant memory entry found for prompt context")
            return ""

        # Reformat for model readability
        memory_text = "\n".join(
            f"- [{entry['category']}] {entry['content']}"
            for entry in memory
        )
        logger.info("Retrieved %d relevant memory entry(s)", len(memory))
        return memory_text

    def toggle_auto_retrive_memory_entry(
        self,
        enable_auto_memory_retrieve: bool,
        prompt: str
    ) -> str | None:
        """
        Auto memory entry ability, returns memory entries if its toggled on.

        Model decides from {prompt} --> {memory_entries}
        """
        if enable_auto_memory_retrieve == True:
            return self.add_memory_entries(prompt)
        return ""

    def get_entries_by_ids(self, ids: list[str] | None) -> list[dict]:
        """Retrieves documents and metadata directly from ChromaDB using IDs."""
        if not ids:
            logger.warning("Cannot retrieve entries: No IDs provided")
            return []

        if self.vector_db.count() == 0:
            logger.warning("Cannot retrieve entries: Database is emtpy")
            return []

        results = self.vector_db.get(ids=ids)

        if not results:
            logger.warning("No matches found for provided IDs")
            return []

        if not "documents" in results and not results["documents"]:
            logger.warning("Database response missing 'documents' payload")
            return []

        # Returns lists for ids, documents and metadatas
        retrieved_entries = []
        for entry_id, doc, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            retrieved_entries.append({
                "id": entry_id,
                "category": metadata.get("category", "fact") if metadata else "fact",
                "content": doc
            })
            logger.info("Retrieved %d memory entry(s) by ID", len(retrieved_entries))

        return retrieved_entries

    def get_exact_match(self, query: str) -> tuple[str, str] | None:
        """Finds the single closest memory matching the query."""
        if self.vector_db.count() == 0:
            logger.warning("Exact match query skipped: Database is emtpy")
            return None

        # Embed query to find the target
        response = ollama.embeddings(model=self.embed_model, prompt=query)
        if not response:
            logger.error("Failed to generate embedding for exact match query")
            print("Error: Model failed to generate vector embedding")
            return None
        logger.info("Vector embedding generated")

        # Look for top 1 exact match
        query_embedding = response["embedding"]
        results = self.vector_db.query(query_embeddings=[query_embedding], n_results=1)

        if not results:
            logger.warning("Exact match query returned empty results structure")
            return None

        if not results["ids"] and not results["ids"][0] and not len(results["ids"][0]) > 0:
            logger.warning("No matching IDs found for exact match query")
            return None

        target_id = results["ids"][0][0]
        matched_content = results["documents"][0][0]
        logger.info("Found exact memory match, target_id=%s", target_id)

        return target_id, matched_content

    def delete_from_db(self, ids: list[str]):
        """Removes a list vector ID reference key directly from database."""
        if not ids:
            return
        self.vector_db.delete(ids=ids)
        logger.info("Deleted %d memory entry(s) from database: ids=%s", len(ids), ids)

    # ============================================================
    # Store memory
    # ============================================================

    def extract_and_store_memory_entries(
        self, 
        context: list[dict], 
        source: Literal["manual", "automatic"],
    ) -> tuple[list[str], int, int]:
        """
        No prompts needed, process conversation logs,
        extracts standalone atomic facts via LLM, 
        and saved to database.

        Depends of the system prompt to decide whether to
        extract memory automatically or manually.
        """
        if source == "manual":
            system_prompt = self.mem_manual_prompt # User ask model with prompt
        else:
            system_prompt = self.mem_prompt # Extract memory automatically

        try:
            # Extract memory
            previous_entries = self.chat.get_trimmed_previous_entries(context)
            last_two_entries = self.chat.get_last_two_entries_roles(context)
            formatted_prompt = f"# All previous conversations\n\n{previous_entries}\n\n---\n\n# New conversations\n\n{last_two_entries}"
            extracted_output, p_tkns, o_tkns = LLM.response_with_new_sys_prompt_and_context(
                model=self.model,
                system_prompt=system_prompt,
                prompt=formatted_prompt
            )

            created_ids = self._format_and_append_to_db(extracted_output, source)
            if created_ids:
                logger.info("Extracted and saved %d memory entry(s)", len(created_ids))
                print(f"Memory saved")
            return created_ids, p_tkns, o_tkns

        except Exception as e:
            logger.error("Memory extraction synthesis failed: error=%s", e, exc_info=True)
            return [], 0, 0

    def toggle_auto_store_memory_entries(
        self,
        enable_auto_memory_store: bool,
        model_max_tokens: int,
        context: list[dict],
    ):
        """
        Auto store memory no prompts needed, 
        should be at the end of every conversations.
        """
        if enable_auto_memory_store == False:
            return

        if model_max_tokens < config.AUTO_MEMORY_STORE_TOKENS:
            return

        created_ids, p_tkns, o_tkns = self.extract_and_store_memory_entries(
            context=context,
            source= "automatic"
        )
