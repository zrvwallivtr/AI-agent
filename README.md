# AI Agent

A local Command-Line Interface (CLI) AI assistant featuring ChromaDB long-term memory, file context injection, local web search and automated token management.

---

## Table of Contents

- [Features](#features)
    - [Core Capabilities](#core-capabilities)
    - [Context & Chat History](#context--history-management)
    - [Long-Term Memory](#long-term-memory)
    - [Document Processing](#document-processing)
    - [Web Search Integration](#web-search-integraton)
- [Command Reference](#command-reference)
- [Configuration](#configuration)
- [Data Directory Structure](#data-directory-structure)
- [License](#license)

---

## Features

### Core Capabilities

* **Markdown Terminal Output:** All model responses are rendered in markdown format.
* **CLI Flags Support:** Built-in command-line arguments.
* **Local Data Storage:** User data, session histories, vector stores are kept locally in `~/.agent_app/data`.

### Context & History Management

* **Logging:** Segregates active session processing `chat.json` (read by the model) and `chat_history.json` (Contains all unedited chat history).
* **Automated Session Compression:** Dynamically summarizes older conversations history when context limits are reached, preventing context-window crashes.
* **Multi-session support:** Create, store and switch between chat sessions.

### Long-Term Memory (ChromaDB integration)

The agent automatically retrieves relevant contextual memories using an embedding model. Manual memory management is also available via dedicated slash commands.

### Document Processing

Converts different file formats content into clean plaintext context strings, automatically injecting them into the active chat context.

* **Plain Text:** `.txt`
* **Data & Configuration Formats:** `.csv`, `.xlsx`, `.yaml`, `.yml`, `.toml`, `.xml`
* **Documents:** `.pdf`, `.docx`, `.epub`
* **Code & Scripts:** `.py`, `.js`, `.ts`, `.tsx`, `.json`, `.md`, `.sh`, `.html`, `.css`, `.rs`, `.go`
* **Fallback Behavior:** Any unlisted text-based format defaults to be read as plain-text.

### Web Search Integration

Queries local or remote search engines.

* Executes web search queries based on model triggers or explicit user requests.
* Safely dumps all search results into a temporary `.json` file for model analysis.
* Synthesizes and presents only the finalized, context-aware answer in the main chat output.

---

## Command Reference

| Command | Description |
| --- | --- |
| `/forget <prompt>` | Removes matching entries from the long-term memory database. |
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

### Storage

All data stores in `~/.agent_app`:

```
~/.agent_app
├── config.toml                     # Primary application configuration
└── data/
    ├── chats/                      # Active and archived session history
    │   ├── default_session/
    │   │   ├── chat_history.json
    │   │   └── chat.json
    │   └── custom_session/
    │       ├── chat_history.json
    │       └── chat.json
    ├── chroma/                     # Local ChromaDB sqlite vector index
    │   └── chroma.sqlite3
    ├── dropbox/                    # User document uploads
    │   ├── chat/
    │   │   ├── file.pdf
    │   │   ├── file.txt
    │   │   └── file_metadata.json
    │   └── custom_session/
    │       ├── file.py
    │       └── file_metadata.json
    ├── projects/                   # Workspace project files
    └── prompts/                    # System prompt templates
```

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
