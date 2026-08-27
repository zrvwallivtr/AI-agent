import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax


console = Console()


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
    def model_response(msgs: list[dict], model: str) -> tuple[str, int, int]:
        """Response using specified model, streams output."""
        response = ""
        line_buffer = ""
        in_code_block = False
        current_lang = "text"
        p_tkns   = 0
        o_tkns   = 0

        # Send message to the model
        stream = ollama.chat(model=model, messages=msgs, stream=True)

        # Stream model output in markdown
        for chunk in stream:
            token = chunk.message.content
            response += token
            line_buffer += token

            # Render and flush completed lines as markdown
            if "\n" in line_buffer:
                lines = line_buffer.split("\n")
                for line in lines[:-1]:

                    # Detect start/end of code fence
                    if line.strip().startswith("```"):
                        if not in_code_block:
                            current_lang = line.strip().lstrip("`").strip() or "text"
                            in_code_block = True
                        else:
                            in_code_block = False
                            current_lang = "text"
                        continue

                    if in_code_block:
                        syntax = Syntax(
                            line,
                            current_lang,
                            theme="monokai",
                            word_wrap=True
                        )
                        console.print(syntax)
                    else:
                        if line.strip():
                            console.print(Markdown(line))
                        else:
                            console.print()

                line_buffer = lines[-1]

            if chunk.done:
                p_tkns = chunk.prompt_eval_count or 0
                o_tkns = chunk.eval_count or 0

        if line_buffer:
            if in_code_block:
                console.print(Syntax(line_buffer, current_lang, theme="monokai", word_wrap=True))
            elif line_buffer.strip():
                console.print(Markdown(line_buffer))

        return response, p_tkns, o_tkns


    # =====================================================================
    # Modified model response for customised system prompt and context
    # =====================================================================

    @staticmethod
    def response_with_new_sys_prompt_and_context(
        model: str,
        system_prompt: str,
        prompt: str,
        context: list[dict] | None = None
    ) -> tuple[str, int, int]:
        """Combines a system prompt, context and question with model response."""
        if context:
            context_without_system_prompt = [msg for msg in context if msg["role"] != "system"]
            msgs = (
                [LLM.system(system_prompt)]
                + context_without_system_prompt
                + [LLM.user(prompt)]
            )
        else:
            msgs = [LLM.system(system_prompt)] + [LLM.user(prompt)]

        response        = ollama.chat(model=model, messages=msgs)
        content         = response.message.content
        p_tkns   = getattr(response, "prompt_eval_cound", 0) or 0
        o_tkns   = getattr(response, "eval_cound", 0) or 0
        return content, p_tkns, o_tkns


    # =====================================================================
    # Memory response formats
    # =====================================================================

    @staticmethod
    def response_memory_recall_format(
        model: str,
        system_prompt: str,
        prompt: str,
        context: list[dict] | None = None
    ) -> ollama.ChatResponse:
        """Model reponse with memory recall system prompt and format."""
        if context:
            context_without_system_prompt = [msg for msg in context if msg["role"] != "system"]
            msgs = (
                [LLM.system(system_prompt)]
                + context_without_system_prompt
                + [LLM.user(prompt)]
            )
        else:
            msgs = [LLM.system(system_prompt)] + [LLM.user(prompt)]
        return LLM.model_response(msgs, model)
