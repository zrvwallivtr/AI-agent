from typing import Any


def memory_section(mem_list: list[dict[str, Any]]) -> str:
    """Rearrange all memory entries into markdown format."""
    mems = [
        (
            f"## Memory\n"
            f"Content: {item['content']}\n"
            f"Similarity score: {item['similarity']}\n"
        ) for item in mem_list
    ]
    return "# Retrieved memory(s)\n\n" + "---\n\n".join(mems)

def attachment_section(attchmnt_dict: dict) -> str:
    """Rearrange all attachments into markdown format."""
    attchmnts = [
        (
            f"## Document name: {doc_path.name}\n"
            f"Path: {doc_path}"
            f"Content:\n"
            f"{cont}\n\n"
        ) for doc_path, cont in attchmnt_dict.items()
    ]
    return "# Attachment(s)\n\n" + "---\n\n".join(attchmnts)

def user_prompt_section(prompt: str) -> str:
    """Rearrange user prompt into markdown format."""
    return (
        f"# User prompt\n\n"
        f"{prompt}"
    )
