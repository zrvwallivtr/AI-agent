Features
--------

### Core Capabilities

- **Markdown terminal output:** All model responses are rendered in markdown format.
- **CLI flags support:** Built-in command-line arguments.
- **Data storage:** All user files, sessions and logs are stored in `~/.agent_app/data`.

### Context & Chat History

- **Logging:** Segregates active session processing `chat.json` (read by the model) and `chat_history.json` (Contains all unedited chat history).
- **Automated token compression:** automatically summarises older conversations when token counts reaches certain threshold to prevent context-window crashes.
- **Multi-session support:** Switch between separate chat sessions.

### Long-Term Memory (ChromaDB integration)

The agent automatically fetches relevant contextual memories using an embedding model. Manually interface with its memory is also supported using commands:

- `/forget <prompt>` - Removes specified entries from the database.
- `/memorise <prompt>` - Instruct agent to extract and save specified context to the database.
- `/recall <prompt>` - Searches and retrieves matches from database.

### Document processing

Converts different file formats content into clean plaintext context strings to append into the active chat session.

**Officially supported formats**
- **Plain text:** `.txt`
- **Data & configuration formats:** `.csv`, `.xlsx`, `.yaml`, `.yml`, `.toml`, `.xml`
- **Text documents:** `.pdf`, `.docx`, `.epub`
- **Programming:** `.py`, `.js`, `.ts`, `.tsx`, `.json`, `.md`, `.sh`, `.html`, `.css`, `.rs`, `.go`

- Any other formats that are not on the list will be fallback to be read as plain text.

### Web search

Queries local or remote search engines.

- `/search <prompt>` to enable ability.
- Safely dumps all search results into a temporary `.json` file for model to read.
- Generates an accurate answer based on the search results and user prompt, ensuring only the finalized answer enters chat log.

### Configuration

- All operational aspects of the agent are managed via `config.toml`:
```toml
[models]
chat                = "ministral_3b:latest" # Primary conversational LLM
memory              = "nomic-embed-text"    # Embedding model for vector operations
project_manager     = "gemma3:1b"           # Secondary processing model

# Configurable model token thresholds
# chat_max_tokens     = 4096
# pm_max_tokens       = 2048

[search]
engine      = "http://localhost:8080/search"    # Search engine URL
max_results = 3                                 # Maximum search results per query

[load_system_prompt]
chat            = "system"
memory          = "memory"
project_manager = "project_manager"
```

### Storage

All data stores in `~/.agent_app`:

```
~/.agent_app
├── config.toml
└── data
    ├── chats                           (All session history)
    │   ├── chat.json
    │   ├── chat_history.json
    │   ├── custom.json
    │   └── custom_chat_history.json
    ├── chroma                          (Chroma vector data)
    │   └── chroma.sqlite3
    ├── dropbox                         (All user uploads)
    ├── projects
    └── prompts
```

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
