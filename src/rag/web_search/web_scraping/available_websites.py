AVA_WEBS_LIST = {
    "arxiv": "arXiv"
}


def clean_website_name(w_name: str) -> str | None:
    """
    Clean up noises from the given website name,
    ensure the website name matches existing list.
    """
    if not w_name:
        return
    
    clean_w_name = w_name.strip().strip("\"'`*.:;,")
    clean_w_name = clean_w_name.splitlines()[0].strip()
    lower_w_name = clean_w_name.lower()

    if lower_w_name in ("none", ""):
        return None

    return AVA_WEBS_LIST.get(lower_w_name)
