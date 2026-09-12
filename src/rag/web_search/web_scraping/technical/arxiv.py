import feedparser
import urllib.parse
import time
import json
from typing import Literal

from src.agent.models.llm import LLM


# ==============================================================
# API QUERY
# ==============================================================

API_FIELD_MAP = {
    "All fields":           "all",
    "Title":                "ti",
    "Author":               "au",
    "Abstract":             "abs",
    "Comment":              "co",
    "Journal reference":    "jr",
    "Subject category":     "cat",
    "Report number":        "rn",
    "arXiv identifier":     "id"
}

SORT_BY_MAP = {
    "Relevance":            "relevance",
    "Last updated date":    "lastUpdatedDate",
    "Submission date":      "submittedDate"
}

SORT_ORDER_MAP = {
    "Descending (newest/most relevant first)":  "descending",
    "Ascending (oldest first)":                 "ascending"
}


def _build_field_query(field: str, value: str) -> str:
    """Build a single 'prefix:value' clause for search_query."""
    prefix = API_FIELD_MAP.get(field)
    if not prefix:
        raise ValueError(f"'{field}' is not supported by the arXiv API")
    return f"{prefix}:{value}"


def _api_query_with_filters(
    clauses: list[tuple[str, str]],
    combine: Literal["AND", "OR", "ANDNOT"],
    sort_by: str = "Relevance",
    sort_order: str = "Descending (newest/most relevant first)",
    max_results: int = 10,
    start: int = 0,
) -> str:
    """Build full arXiv API query URL with boolean-combined field filters."""
    field_clauses = [_build_field_query(field, value) for field, value in clauses]
    combined = f" {combine} ".join(field_clauses)

    params = {
        "search_query": combined,
        "start": start,
        "max_results": max_results,
        "sortBy": SORT_BY_MAP[sort_by],
        "sortOrder": SORT_ORDER_MAP[sort_order]
    }
    query_str = urllib.parse.urlencode(params)
    return f"http://export.arxiv.org/api/query?{query_str}"


def _parse_api_results(url: str, max_results: int = 10) -> list[dict]:
    """Query arXiv's official API to get structured results."""
    feed = feedparser.parse(url)
    time.sleep(3)

    return [
        {
            "title": entry.title,
            "summary": entry.summary,
            "authors": [a.name for a in entry.authors],
            "url": entry.id,
            "pdf_url": next((l.href for l in entry.links if l.type == "application/pdf"), None),
            "published": entry.published
        }
        for entry in feed.entries
    ]


# ==============================================================
# LLM CALL
# ==============================================================

INSTRUCT = """You convert a user's paper search request into a JSON object. Follow these rules exactly.

OUTPUT FORMAT:
Return ONLY a single JSON object. No explanation, no markdown, no code fences, no extra text
before or after it.

JSON SCHEMA:
{
  "clauses": [["<field>", "<value>"], ...],
  "combine": "<AND|OR|ANDNOT>",
  "sort_by": "<Relevance|Last updated date|Submission date>",
  "sort_order": "<Descending (newest/most relevant first)|Ascending (oldest first)>",
  "max_results": <integer>,
  "start": <integer>
}

ALLOWED VALUES FOR "<field>" — use EXACTLY these strings, nothing else:
"All fields", "Title", "Author", "Abstract", "Comment", "Journal reference",
"Subject category", "Report number", "arXiv identifier"

RULES:
1. "clauses" must have at least 1 item. Each item is a two-element list: [field, value].
   The field must be one of the ALLOWED VALUES above, spelled and capitalized exactly as shown.
2. If the user does not clearly specify a field, use "All fields" with their search term as value.
3. If the user mentions more than one topic/author joined by "or", use "combine": "OR".
   If joined by "and"/"also"/multiple required conditions, use "combine": "AND".
   Default to "AND" if unclear.
4. "sort_by" defaults to "Relevance" unless the user asks for newest, latest, or recent results
   (use "Submission date" then) or explicitly asks to sort by update date.
5. "sort_order" defaults to "Descending (newest/most relevant first)" unless the user explicitly
   asks for oldest first.
6. "max_results" defaults to 10. Only change it if the user gives a specific number.
7. "start" is always 0 unless the user explicitly asks for a later page of results.
8. Never invent a field name outside the ALLOWED VALUES list. If the user's request doesn't map
   to any allowed field, use "All fields" instead.
9. Do not add keys beyond the schema. Do not omit any key.

EXAMPLES:

User: papers by von Braun about rocket propulsion
{"clauses": [["Author", "von Braun"], ["All fields", "rocket propulsion"]], "combine": "AND", "sort_by": "Relevance", "sort_order": "Descending (newest/most relevant first)", "max_results": 10, "start": 0}

User: latest papers on hypersonic vehicle design in aerospace engineering category
{"clauses": [["All fields", "hypersonic vehicle design"], ["Subject category", "physics.flu-dyn"]], "combine": "AND", "sort_by": "Submission date", "sort_order": "Descending (newest/most relevant first)", "max_results": 10, "start": 0}

User: find 25 papers about either satellite constellations or space debris mitigation
{"clauses": [["All fields", "satellite constellations"], ["All fields", "space debris mitigation"]], "combine": "OR", "sort_by": "Relevance", "sort_order": "Descending (newest/most relevant first)", "max_results": 25, "start": 0}

User: oldest papers mentioning orbital mechanics in the title
{"clauses": [["Title", "orbital mechanics"]], "combine": "AND", "sort_by": "Submission date", "sort_order": "Ascending (oldest first)", "max_results": 10, "start": 0}

User: papers about reentry heat shields, not related to Mars missions
{"clauses": [["All fields", "reentry heat shields"], ["All fields", "Mars missions"]], "combine": "ANDNOT", "sort_by": "Relevance", "sort_order": "Descending (newest/most relevant first)", "max_results": 10, "start": 0}"""


def _parse_llm_query_to_json(llm_out: str) -> dict:
    """
    Validate LLM JSON output and return as JSON format.

    JSON SCHEMA:
    {
      "clauses": [["<field>", "<value>"], ...],
      "combine": "<AND|OR|ANDNOT>",
      "sort_by": "<Relevance|Last updated date|Submission date>",
      "sort_order": "<Descending (newest/most relevant first)|Ascending (oldest first)>",
      "max_results": <integer>,
      "start": <integer>
    }
    """
    llm_out = llm_out.strip()

    if llm_out.startswith("```"):
        llm_out = llm_out.strip("`").removeprefix("json").strip()

    try:
        data = json.loads(llm_out)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}")

    # Required keys
    rqired = {"clauses", "combine", "sort_by", "sort_order", "max_results", "start"}
    if not rqired.issubset(data.keys()):
        raise ValueError(f"Missing keys: {rqired - data.keys()}")

    # Validate clauses against the pre-written map
    valid_fields = set(API_FIELD_MAP.keys())
    clean_clauses = []
    for pair in data["clauses"]:
        if not (isinstance(pair, list) and len(pair) == 2):
            raise ValueError(f"Malformed clauses: {pair}")

        field, value = pair
        if field not in valid_fields:
            field = "All fields" # Fallback to 'All fields' option rather than crash

        clean_clauses.append((field, str(value)))

    # Fallbacks
    if data["combine"] not in ("AND", "OR", "ANDNOT"):
        data["combine"] = "AND"
    if data["sort_by"] not in SORT_BY_MAP:
        data["sort_by"] = "Relevance"
    if data["sort_order"] not in SORT_ORDER_MAP:
        data["sort_order"] = "Descending (newest/most relevant first)"

    return {
        "clauses": clean_clauses,
        "combine": data["combine"],
        "sort_by": data["sort_by"],
        "sort_order": data["sort_order"],
        "max_results": int(data.get("max_results"), 10),
        "start": int(data.get("start", 0))
    }


# ==============================================================
# LLM QUERY API
# ==============================================================

def llm_search_arxiv(model:str, context: list[dict], prompt: str) -> tuple[list[dict], int, int] | None:
    """User text -> LLM JSON -> validate args -> arXiv API call."""
    response = LLM.response_with_new_sys_prompt_and_context(
        model=model, system_prompt=INSTRUCT, prompt=prompt, context=context
    )
    if not response:
        return
    cont, p_tkns, o_tkns = response

    parsed = _parse_llm_query_to_json(cont)
    url = _api_query_with_filters(**parsed)
    return _parse_api_results(url), p_tkns, o_tkns
