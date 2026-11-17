
from typing import Any

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32
) -> list[list[float]]:
    """
    Convert a list of text strings into numerical embeddings.
    """

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    if not texts:
        return []

    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("All input texts must be non-empty strings.")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.tolist()


def embed_documents(
    documents: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 32
) -> list[dict[str, Any]]:
    """
    Generate embeddings for chunked documents.

    Each document retains its original metadata and receives
    an additional embedding field.
    """

    if not documents:
        return []

    texts = [document["text"] for document in documents]

    embeddings = generate_embeddings(
        texts=texts,
        model=model,
        batch_size=batch_size
    )

    embedded_documents = []

    for document, embedding in zip(documents, embeddings):
        embedded_document = {
            **document,
            "embedding": embedding
        }

        embedded_documents.append(embedded_document)

    return embedded_documents


if __name__ == "__main__":
    from pathlib import Path

    from src.rag.document_loader import load_passages
    from src.rag.chunking import chunk_documents

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "data"
        / "synthetic"
        / "synthetic_benign_passages.csv"
    )

    print("Loading documents...")

    documents = load_passages(dataset_path)

    print("\nChunking documents...")

    chunks = chunk_documents(
        documents,
        chunk_size=120,
        overlap=20
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(DEFAULT_MODEL)

    print("\nGenerating embeddings...")

    embedded_documents = embed_documents(
        documents=chunks,
        model=model,
        batch_size=32
    )

    print("\nEmbedding generation completed!")

    print("Original documents:", len(documents))
    print("Generated chunks:", len(chunks))
    print("Embedded documents:", len(embedded_documents))
    print("Embedding dimensions:", len(embedded_documents[0]["embedding"]))

    print("\nFirst embedding preview:")
    print(embedded_documents[0]["embedding"][:10])
