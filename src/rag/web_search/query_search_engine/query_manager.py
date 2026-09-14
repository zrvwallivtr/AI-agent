import datetime
import re

from src.config import models
from src.config import prompts

from src.agent import llm


MODEL                   = models.MODEL
SEARCH_OR_NOT_PROMPT    = prompts.SEARCH_OR_NOT_PROMPT
QUERY_PROMPT            = prompts.QUERY_PROMPT


def search_or_not(
    model: str,
    context: list[dict],
    prompt: str
) -> tuple[bool, int, int] | None:
    """
    Query model to decide whether a question requires search or not.
    Returns either 'True' or 'False'.
    """
    response = llm.response_with_new_sys_prompt_and_context(
        model=model, sys_prompt=SEARCH_OR_NOT_PROMPT, prompt=prompt
    )
    if not response:
        return
    output, p_tkns, o_tkns = response

    if 'true' in output.lower():
        return True, p_tkns, o_tkns
    else:
        return False, p_tkns, o_tkns


def _query_check(search_query: str) -> str:
    """
    Check format of the search query, remove '"' if it
    exist at the start and end of the query.
    """
    if not search_query:
        return ""

    # Get only the first line of output
    search_query = search_query.strip().split('\n')[0]

    # Remove advanced operators that trigger WAF blocks
    bad_operators = ["inurl:", "site:", "intitle:", "filetype:", "sorted:newest", "sorted:"]
    for operator in bad_operators:
        search_query = search_query.replace(operator, "")

    # Remove quotes, colons and stray punctuation
    search_query = search_query.replace('"', "").replace("'", "").replace(":", "")
    search_query = search_query.strip('`* ')

    # Replace any double space into single space
    search_query = re.sub(r"\s+", " ", search_query)

    return search_query


def generate_query(model: str, contxt: list[dict], prompt: str) -> tuple[str, int, int] | None:
    """Generate query from user input with dynamic date injection."""
    # Update {{current_date}} in the system prompt to actual date
    current_date = datetime.datetime.now().strftime("%A, %d %B %Y")
    live_qry_sys_prmpt = QUERY_PROMPT.replace("{{current_date}}", current_date)

    response = llm.response_with_new_sys_prompt_and_context(
        model=model, sys_prompt=live_qry_sys_prmpt, contxt=contxt, prompt=prompt
    )
    if not response:
        return
    qry, p_tkns, o_tkns = response

    return _query_check(qry), p_tkns, o_tkns
