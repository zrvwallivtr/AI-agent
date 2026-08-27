# AI Agent

A local Command-Line Interface (CLI) AI assistant featuring long-term memory, file context injection, (isolated web crawling / search and automated token management).

---

## Table of Contents

- [Features](#features)
    - [Core Capabilities](#core-capabilities)
    - [Context & Chat History](#context--history-management)
    - [Long-Term Memory](#long-term-memory)
    - [Document Processing](#document-processing)
    - [Web Search & Crawling](#web-search--crawling)
    - [Security](#security)
- [Command Reference](#command-reference)
- [Configuration](#configuration)
- [Data Directory Structure](#data-directory-structure)
- [Docker Services](#docker-services)
- [License](#license)

---

## Features

### Core Capabilities

* **Markdown Terminal Output:** All model responses are rendered in markdown format.
* **CLI Flags Support:** Built-in command-line arguments.
* **Local Data Storage:** User data, session histories, chat logs, memories and document/web content embeddings are stored in a single **PostgreSQL** database (with the **pgvector** extension).

### Context & History Management

* **Logging:** Conversation turns (prompt, response, token counts, tool calling) are persisted per session to the `chat_logs` table, linked to a `chat_sessions` table via `session_id`.
* **Automated Session Compression:** (needs update) Dynamically summarizes older conversations history when context limits are reached, preventing context-window crashes.
* **Multi-session support:** Create, store and switch between chat sessions. Session names and IDs are managed in a dedicated `chat_sessions` table.

### Long-Term Memory

The agent automatically retrieves relevant memories using **pgvector** cosine-similarity search.
Memory is **global and cross-session**, saved automatically or manually by `/memorise`.
Each memory entry is tagged with a category (`preference`, `stack`, `fact`, `project`, `instruction`, `correction`), with an extraction method (`manual` or `auto`).

### Document Processing

Converts different file formats content into clean plaintext/markdown context strings, automatically injecting them into the active chat context and persisting them to the knowledge base for future retrieval.

* **Plain Text:** `.txt`
* **Data & Configuration Formats:** `.csv`, `.xlsx`, `.yaml`, `.yml`, `.toml`, `.xml`
* **Documents:** `.pdf`, `.docx`, `.epub`
* **Code & Scripts:** `.py`, `.js`, `.ts`, `.tsx`, `.json`, `.md`, `.sh`, `.html`, `.css`, `.rs`, `.go`
* **Fallback Behavior:** Any unlisted text-based format defaults to be read as plain-text.
* **Deduplication:** Documents are hashed (SHA-256) before embedding; re-uploading identical content is detected and skipped rather than re-embedded.
* **Path Safety:** All document reads are resolved and validated against a fixed uploads directory to prevent path traversal outside the allowed folder.

### Web Search & Crawling (Still in development)

Web access runs through an isolated **Crawl4AI** service (self-hosted via Docker) paired with a self-hosted **SearXNG** instance for search.
Queries local or remote search engines.

* **Search (`SearXNG`):** Turns model generated query into a list of candidate URLs, without depending on any third-party search API.
* **Crawling (`Crawl4AI`):** Fetches and cleans page content into Markdown
    * **No-LLM-first extraction:**
    * **LLM fallback:**
* **Caching:** Crawled page content is cached in the `knowledge_base` table with a `content_hash` (change detection) and `expires_at`
* Executes web search/crawl based on model triggers or explicit user requests.
* Synthesizes and presnets only the finalized, context-aware answer in the main chat output.

### Security

Since the agent fetches and processes untrusted external content, there are several built in protections:

* **SSRF protection:**
* **Container hardening:**
* **API authentication:**
* **Prompt-injection defense:**
All Services (Postgres, pgAdmin, Crawl4AI) are bound to `127.0.0.1` only - never exposed beyond the local machine.

---

## Command Reference

| Command | Description |
| --- | --- |
| `/memorise <prompt>` | Instruct agent to extract and save key facts to the database. |
| `/recall <prompt>` | Retrieves relevant memories from ChromaDB. |
| `/search <prompt>` | Enable web search and generates a response based on the results. |

---

## Configuration

All agent behavior, models and feature toggles are managed through `~/.agent_app/config.toml`:

```toml
[models]
chat                = "ministral_3b:latest"
memory              = "nomic-embed-text"
project_manager     = "gemma3:1b"

# Optional token limits
# chat_max_tokens   =
# pm_max_tokens     =

[load_system_prompt]
chat            = "system"

[memory]
retrieve_entry_limit                        = 3
auto_memory_store_enable_at_model_tokens    = 128000
enable_auto_memory_retrieve                 = true
enable_auto_memory_store                    = true

[file_reader]
auto_read_dropbox_enable_at_model_tokens    = 128000
enable_auto_read_dropbox                    = true

[search]
engine                                  = "http://localhost:8080/search"
max_results                             = 3
max_char_per_page                       = 3000
auto_web_search_enable_at_model_tokens  = 128000
enable_auto_web_search                  = true
```

Secrets (database password, `CRAWL4AI_API_TOKEN`, pgAdmin credentials) are kept in a separate `.env` file, never in `config.toml`.

### Storage

All data stores in `~/.agent_app`:

```
~/.agent_app
├── config.toml             # Primary application configuration
├── .env                    # Secrets: DB password, CRAWL4AI_API_TOKEN, pgAdmin creds
├── docker-compose.yaml
├── logs
│   └── agent.log
├── searxng
│   └── settings.yml
└── uploads/                # User document uploads
```

## Docker Services

Backing services are managed via `docker compose` and are all bound to `127.0.0.1` (never exposed to the network):

| Service | Purpose | Port | Notes |
| --- | --- | --- | --- |

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
