import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live


console = Console()

def get_last_two_entries_roles(context: list[dict]) -> str:
    """
    Get the last two entries of the entire message dictionary, ensure
    the last two entries are ordered in 'user' then 'assistant'. Else,
    only returns 'user' entry.
    """
    last_two_entries = context[-2:]

    # If conversation contains more than two entries
    if len(last_two_entries) == 2:
        roles = [msg.get("role") for msg in last_two_entries]
        
        # "user" then "assistant"
        if roles == ["user", "assistant"]:
            return "\n".join(
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in last_two_entries
            )

        # "assistant" then "user"
        if roles == ["assistant", "user"]:
            user_msg = last_two_entries[1]
            return f"{user_msg.get('role', 'user')}: {user_msg.get('content', '')}"

    # If conversation contains only 1 entry
    if len(last_two_entries) == 1:
        return "Error: Format incorrect."

    return ""

def get_trimmed_previous_entries(context: list[dict]) -> str:
    """
    Get all previous entries excluding the system prompt and the
    new entries starting with 'user'.
    """
    last_two_entries = context[-2:]

    # If conversation contains more than two entries
    if len(last_two_entries) == 2:
        last_two_roles = [msg.get("role") for msg in last_two_entries]

        # "user" then "assistant"
        if last_two_roles == ["user", "assistant"]:
            return "\n".join(
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in context[:-2] # Excluding the last two entries
                if msg.get("role") != "system" # Excluding system prompt
            )

        # "assistant" then "user"
        if last_two_roles == ["assistant", "user"]:
            user_msg = last_two_entries[1]
            return "\n".join(
                f"{user_msg.get('role', 'user')}: {user_msg.get('content', '')}"
                for msg in context[:-1] # Excluding the last one entry
                if msg.get("role") != "system" # Excluding system prompt
            )

    # If conversation contains only 1 entry
    if len(last_two_entries) == 1:
        return "Error: Format incorrect."

    return ""


class LLM:

    # =====================================================================
    # Roles
    # =====================================================================

    @staticmethod
    def system(content: str) -> dict:
        system_entry = {"role": "system", "content": content}
        return system_entry

    @staticmethod
    def assistant(content: str) -> dict:
        assistant_entry = {"role": "assistant", "content": content}
        return assistant_entry

    @staticmethod
    def user(content: str) -> dict:
        user_entry = {"role": "user", "content": content}
        return user_entry

    # =====================================================================
    # General model response (streamed onto the terminal)
    # =====================================================================

    @staticmethod
    def model_response(messages: list[dict], model: str) -> tuple[str, int, int]:
        """Response using specified model, streams output."""
        response        = ""
        prompt_tokens   = 0
        output_tokens   = 0

        # Send message to the model
        stream = ollama.chat(model=model, messages=messages, stream=True)

        # Stream model output in markdown
        with Live(
            Markdown(response),
            console=console,
            refresh_per_second=11,
            vertical_overflow="visible"
        ) as live:
            for chunk in stream:
                token = chunk.message.content
                response += token
                live.update(Markdown(response))

                # Get token count when done
                if chunk.done:
                    prompt_tokens = chunk.prompt_eval_count or 0
                    output_tokens = chunk.eval_count or 0

        return response, prompt_tokens, output_tokens

    # =====================================================================
    # Modified model response for customised system prompt and context
    # =====================================================================

    @staticmethod
    def response_with_new_sys_prompt_and_context(
        model: str,
        system_prompt: str,
        prompt: str,
        context: list[dict] | None = None
    ) -> ollama.ChatResponse:
        """Combines a system prompt, context and question with model response."""
        if context:
            formatted_question              = f"# User input\n{prompt}"
            context_without_system_prompt   = [msg for msg in context if msg["role"] != "system"]
            messages                        = [LLM.system(system_prompt)] + context_without_system_prompt + [LLM.user(formatted_question)]

        else:
            messages = [LLM.system(system_prompt)] + [LLM.user(prompt)]

        return ollama.chat(model=model, messages=messages)

    # =====================================================================
    # Memory response formats
    # =====================================================================

    @staticmethod
    def response_memory_recall(
        model: str,
        system_prompt: str,
        recalled: str,
        prompt: str,
        context: list[dict] | None = None
    ) -> ollama.ChatResponse:
        """Model reponse with memory recall system prompt and format."""
        if context:
            formatted_question              = f"{recalled}\n{prompt}"
            context_without_system_prompt   = [msg for msg in context if msg["role"] != "system"]
            messages                        = [LLM.system(system_prompt)] + context_without_system_prompt + [LLM.user(formatted_question)]

        else:
            messages = [LLM.system(system_prompt)] + [LLM.user(prompt)]

        response        = ollama.chat(model=model, messages=messages)
        content         = response.message.content
        prompt_tokens   = getattr(response, "prompt_eval_cound", 0) or 0
        output_tokens   = getattr(response, "eval_cound", 0) or 0

        return LLM.model_response(messages, model)

    @staticmethod
    def response_auto_memory_store_format(
        model: str,
        system_prompt: str,
        context: list[dict]
    ) -> tuple[str, int, int]:
        """
        Combines a system prompt, context and question with model response,
        the formatting is modified suitable for storing memory functions.
        """
        trimmed_previous_entries_str    = get_trimmed_previous_entries(context)
        last_two_entries_str            = get_last_two_entries_roles(context)

        formatted_prompt    = f"# All previous conversations\n{trimmed_previous_entries_str}\n\n---\n\n" + f"# New conversations\n{last_two_entries_str}"
        messages            = [LLM.system(system_prompt)] + [LLM.user(formatted_prompt)]

        response        = ollama.chat(model=model, messages=messages)
        content         = response.message.content
        prompt_tokens   = getattr(response, "prompt_eval_cound", 0) or 0
        output_tokens   = getattr(response, "eval_cound", 0) or 0

        return content, prompt_tokens, output_tokens
