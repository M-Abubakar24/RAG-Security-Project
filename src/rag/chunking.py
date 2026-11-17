
from typing import Any


def split_text(
    text: str,
    chunk_size: int = 120,
    overlap: int = 20
) -> list[str]:
    """
    Split text into word-based chunks with overlapping context.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 120,
    overlap: int = 20
) -> list[dict[str, Any]]:
    """
    Convert loaded documents into smaller chunks.

    Original document metadata is preserved.
    """

    chunked_documents = []

    for document in documents:

        passage_id = str(document["passage_id"])
        text = document["text"]

        text_chunks = split_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(text_chunks):

            chunk_document = {
                **document,
                "chunk_id": f"{passage_id}_chunk_{index:03d}",
                "passage_id": passage_id,
                "chunk_index": index,
                "text": chunk
            }

            chunked_documents.append(chunk_document)

    return chunked_documents


if __name__ == "__main__":
    from pathlib import Path
    from document_loader import load_passages

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "data"
        / "synthetic"
        / "synthetic_benign_passages.csv"
    )

    documents = load_passages(dataset_path)

    chunks = chunk_documents(
        documents,
        chunk_size=120,
        overlap=20
    )

    print("\nChunking completed successfully!")
    print("Original documents:", len(documents))
    print("Generated chunks:", len(chunks))

    print("\nFirst chunk:")
    print(chunks[0])
