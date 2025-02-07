from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document

def clean_text(text):
    """
    Perform basic cleaning on the input text.

    Args:
        text (str): The raw text.

    Returns:
        str: The cleaned text.
    """
    # Example cleaning steps
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())
    return text

def text_to_documents(text, metadata, chunk_size=1000, chunk_overlap=100):
    """
    Convert text into LangChain Document chunks with metadata.

    Args:
        text (str): The text to convert.
        metadata (dict): Metadata to associate with each document.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: A list of LangChain Documents.
    """
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)

    documents = [
        Document(page_content=chunk, metadata=metadata) for chunk in chunks
    ]
    return documents
