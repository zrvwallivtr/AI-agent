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


CATEGORY_TYPES = Literal["preference", "stack", "fact", "project", "instruction", "correction"]

CATEGORIES = list(get_args(CATEGORY_TYPES))


class Memory:
    def __init__(self, project: str | None = None):
        self.model              = config.MODEL
        self.embed_model        = config.EMBED_MODEL
        self.mem_prompt         = config.MEM_PROMPT
        self.mem_manual_prompt  = config.MEM_MANUAL_PROMPT
        self.path               = config.CHROMADB_DIR
        self.project            = project
        self.tokens             = Tokens(self.embed_model)

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
        if not content_trimmed:
            return

        # Check if token count exceeds max tokens of embedded model
        if not self.tokens.check_fit(content_trimmed):
            max_chars = self.tokens.model_max_tokens * 4
            print(f"Error: Message exceeds model token count limits, current characters = {len(content_trimmed)}, max chars = {max_chars}")
            return None

        # Generate entry id based only its content (case insensitive)
        entry_id = f"mem_{hashlib.md5(content_trimmed.lower().encode()).hexdigest()[:10]}"

        # Generate vector embedding
        response    = ollama.embeddings(model=self.embed_model, prompt=content_trimmed)
        embedding   = response["embedding"]

        # Add to ChromaDB
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

    def retrieve_relevant_entry(self, prompt: str, limit: int = config.RETRIEVE_MEM_ENTRY_LIMIT) -> list[dict]:
        """Queries ChromaDB using semantics."""
        # Returns nothing if database is empty
        if self.vector_db.count() == 0:
            return []

        # Generate embedding for the question
        response = ollama.embeddings(model=self.embed_model, prompt=prompt)

        # Extracts raw vector array
        query_embedding = response["embedding"]

        # Search ChromaDB for top matches with question vector
        results = self.vector_db.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        # Reformat retrieved entries
        retrieved_entries = self._grab_category_and_content(results)

        return retrieved_entries

    def add_memory_entries(self, prompt: str) -> str:
        """
        Queries long-term vector storage and adds context
        into user's prompt.
        """
        memory = self.retrieve_relevant_entry(prompt, limit=config.RETRIEVE_MEM_ENTRY_LIMIT)

        if memory:
            # Reformat for model readability
            memory_text = "\n".join(
                f"- [{entry['category']}] {entry['content']}"
                for entry in memory
            )

            # Combined found memories and user's question
            new_message = f"## Relevant context from memory\n\n{memory_text}\n\n---\n\n"
        else:
            # If no memories found
            new_message = ""

        return new_message

    def get_entries_by_ids(self, ids: list[str]) -> list[dict]:
        """Retrieves documents and metadata directly from ChromaDB using IDs."""
        if not ids or self.vector_db.count() == 0:
            return []

        results = self.vector_db.get(ids=ids)

        retrieved_entries = []
        if results and "documents" in results and results["documents"]:
            # Returns lists for ids, documents and metadatas
            for entry_id, doc, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
                retrieved_entries.append({
                    "id": entry_id,
                    "category": metadata.get("category", "fact") if metadata else "fact",
                    "content": doc
                })

        return retrieved_entries

    def get_exact_match(self, query: str) -> tuple[str, str] | None:
        """Finds the single closest memory matching the query."""
        if self.vector_db.count() == 0:
            return None

        # Embed query to find the target
        response        = ollama.embeddings(model=self.embed_model, prompt=query)
        query_embedding = response["embedding"]

        # Look for top 1 exact match
        results = self.vector_db.query(query_embeddings=[query_embedding], n_results=1)

        if results and results["ids"] and results["ids"][0] and len(results["ids"][0]) > 0:
            target_id = results["ids"][0][0]
            matched_content = results["documents"][0][0]
            return target_id, matched_content

        return None

    def delete_from_db(self, ids: list[str]):
        """Removes a list vector ID reference key directly from database."""
        if ids:
            self.vector_db.delete(ids=ids)

    # ============================================================
    # Model integration
    # ============================================================

    def extract_entries_and_store_to_db(
        self, 
        context: list[dict], 
        source: Literal["manual", "automatic"],
        prompt: str | None = None,
    ) -> tuple[list[str], int, int]:
        """
        Process conversation logs, extracts standalone atomic facts
        via LLM, and preserves them in long-term vector storage.

        Depends of the system prompt to decide whether to
        extract memory automatically or manually.
        """
        # Extract memory manually (User ask model with prompt)
        if source == "manual":
            system_prompt = self.mem_manual_prompt

        # Extract memory automatically
        else:
            system_prompt = self.mem_prompt

        try:
            # Extract memory
            response = LLM.response_auto_memory_store_format(
                model=self.model,
                system_prompt=system_prompt,
                context=context,
            )
            extracted_output, prompt_tokens, output_tokens = response

            if not extracted_output or not extracted_output.strip():
                return [], 0, 0

            created_ids = self._format_and_append_to_db(extracted_output, source)
            if created_ids:
                print(f"Memory saved to '{self.path}'.")
            return created_ids, prompt_tokens, output_tokens

        except Exception as e:
            print(f"Background memory synthesis encountered an error: {e}")
            return [], 0, 0

    def toggle_auto_retrive_memory_entry(
        self,
        enable_auto_memory_retrieve: bool,
        prompt: str
    ) -> str:
        """
        Auto memory entry ability, returns memory entries if its toggled on.

        Model decides from {prompt} --> {memory_entries}
        """
        if enable_auto_memory_retrieve == True:
            return self.add_memory_entries(prompt)

        else:
            return ""

    def toggle_auto_store_memory_entry(
        self,
        enable_auto_memory_store: bool,
        model_max_tokens: int,
        context: list[dict],
    ):
        if enable_auto_memory_store == True and model_max_tokens > config.AUTO_MEMORY_STORE_TOKENS:
            created_ids, prompt_tokens, output_tokens = self.extract_entries_and_store_to_db(
                context=context,
                source= "automatic"
            )
