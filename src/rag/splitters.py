from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language


txt_spltr = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "",
    ]
)


md_spltr = MarkdownHeaderTextSplitter([
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
])
