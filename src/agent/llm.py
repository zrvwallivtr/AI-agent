import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live


console = Console()


class LLM:

    # ===========================
    # Roles
    # ===========================

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

    # ===========================
    # Execution
    # ===========================

    @staticmethod
    def model_response(messages: str, model: str) -> tuple[str, int, int]:
        response        = ""
        prompt_tokens   = 0
        output_tokens   = 0

        # Send message to the model
        stream = ollama.chat(model=model, messages=messages, stream=True)

        # Stream model output
        for chunk in stream:
            token = chunk.message.content
            print(token, end="", flush=True)
            response += token

            # Get token count when done
            if chunk.done:
                prompt_tokens = chunk.prompt_eval_count or 0
                output_tokens = chunk.eval_count or 0
        print("\n")

        return response, prompt_tokens, output_tokens

    @staticmethod
    def response_with_new_sys_prompt_and_context(
        model: str,
        system_prompt: str,
        context: list[dict],
        prompt: str
    ) -> ollama.ChatResponse:
        """Combines a system prompt, context and question with model response."""
        formatted_question              = f"User input:\n{prompt}"
        context_without_system_prompt   = [msg for msg in context if msg["role"] != "system"]

        messages                        = [LLM.system(system_prompt)] + context_without_system_prompt + [LLM.user(formatted_question)]

        return ollama.chat(model=model, messages=messages)

