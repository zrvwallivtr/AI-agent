import ollama

from src.agent.models.ollama import ollama_clt
from src.logger import app_logger, prompt_logger


app_log     = app_logger(f"{__name__}.app")
prompt_log  = prompt_logger(f"{__name__}.prompt")


# =====================================================================
# Roles
# =====================================================================

def system_prompt(content: str) -> dict:
    system_entry = {"role": "system", "content": content}
    return system_entry


def assistant_message(content: str) -> dict:
    assistant_entry = {"role": "assistant", "content": content}
    return assistant_entry


def user_message(content: str) -> dict:
    user_entry = {"role": "user", "content": content}
    return user_entry


# =====================================================================
# General model response (streamed onto the terminal)
# =====================================================================

def model_response(model: str, msgs: list[dict]) -> tuple[str, int, int] | None:
    """Response using specified model, streams output."""
    response = ""
    p_tkns   = 0
    o_tkns   = 0

    if msgs:
        prompt_log.info(msgs[-1].get("content", ""))

    app_log.debug("Sending messages to model '%s'", model)

    stream = ollama_clt.chat(model=model, messages=msgs, stream=True)

    for chnk in stream:
        tkn = chnk.message.content
        response += tkn
        print(tkn, end="", flush=True)

        if chnk.done:
            p_tkns = chnk.prompt_eval_count or 0
            o_tkns = chnk.eval_count or 0

    print()

    if not response:
        app_log.warning("Model failed to generate response from the provided messages")
        return

    app_log.debug(
        "%d prompt tokens and %d output tokens for generating the response: Total token used = %d tokens",
        p_tkns,
        o_tkns,
        p_tkns + o_tkns
    )
    return response, p_tkns, o_tkns


# =====================================================================
# Modified model response for customised system prompt and context
# =====================================================================

def response_with_new_sys_prompt_and_context(
    model: str,
    sys_prompt: str,
    prompt: str | None,
    contxt: list[dict] | None = None
) -> tuple[str, int, int] | None:
    """Combines a system prompt, context and question with model response."""
    if prompt:
        usr_prompt = [user_message(prompt)]
    else:
        usr_prompt = []

    if contxt:
        context_without_system_prompt = [msg for msg in contxt if msg["role"] != "system"]
        msgs = [system_prompt(sys_prompt)] + context_without_system_prompt + usr_prompt
    else:
        msgs = [system_prompt(sys_prompt)] + usr_prompt
    
    prompt_log.info(system_prompt)
    prompt_log.info(msgs[-1].get("content", ""))

    response = ollama_clt.chat(model=model, messages=msgs)
    cont = response.message.content
    p_tkns = getattr(response, "prompt_eval_count", 0) or 0
    o_tkns = getattr(response, "eval_count", 0) or 0

    if not response:
        app_log.warning("Model failed to generate response from the provided messages")
        return

    app_log.debug(
        "%d prompt tokens and %d output tokens for generating the response: Total token used = %d tokens",
        p_tkns,
        o_tkns,
        p_tkns + o_tkns
    )
    return cont, p_tkns, o_tkns


# =====================================================================
# MEMORY RESPONSE FORMATS
# =====================================================================

def response_memory_recall_format(
    model: str,
    sys_prompt: str,
    prompt: str,
    context: list[dict] | None = None
) -> ollama.ChatResponse:
    """Model reponse with memory recall system prompt and format."""
    if context:
        context_without_system_prompt = [msg for msg in context if msg["role"] != "system"]
        msgs = (
            [system_prompt(sys_prompt)]
            + context_without_system_prompt
            + [user_message(prompt)]
        )
    else:
        msgs = [system_prompt(sys_prompt)] + [user_message(prompt)]

    prompt_log.info(sys_prompt)
    return model_response(model, msgs)
