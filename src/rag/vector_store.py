
from pathlib import Path
from typing import Any
import math

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    """
    Persistent ChromaDB vector store for RAG documents.
    """

    def __init__(
        self,
        database_path: str | Path,
        collection_name: str = "rag_documents"
    ):
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.database_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _prepare_metadata(
        document: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Keep only metadata types supported by ChromaDB.
        Exclude text, embeddings, and missing values.
        """

        metadata = {}

        for key, value in document.items():

            if key in {"text", "embedding"}:
                continue

            if value is None:
                continue

            if hasattr(value, "item"):
                value = value.item()

            if isinstance(value, float) and not math.isfinite(value):
                continue

            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value

        return metadata

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 500
    ) -> int:
        """
        Insert or update embedded documents in ChromaDB.
        """

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        if not documents:
            return 0

        for document in documents:
            if not document.get("chunk_id"):
                raise ValueError("A document is missing its chunk_id.")

            if not document.get("text"):
                raise ValueError("A document is missing its text.")

            embedding = document.get("embedding")

            if not isinstance(embedding, (list, tuple)) or not embedding:
                raise ValueError("A document has an invalid embedding.")

        total_added = 0

        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]

            self.collection.upsert(
                ids=[str(doc["chunk_id"]) for doc in batch],
                documents=[str(doc["text"]) for doc in batch],
                embeddings=[doc["embedding"] for doc in batch],
                metadatas=[
                    self._prepare_metadata(doc) for doc in batch
                ]
            )

            total_added += len(batch)

        return total_added

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents for a query.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_documents = []

        if not results["ids"] or not results["ids"][0]:
            return retrieved_documents

        for index, document_id in enumerate(results["ids"][0]):
            retrieved_documents.append({
                "chunk_id": document_id,
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index]
            })

        return retrieved_documents

    def count(self) -> int:
        """Return the number of stored chunks."""

        return self.collection.count()


if __name__ == "__main__":
    from src.rag.document_loader import load_passages
    from src.rag.chunking import chunk_documents

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "data"
        / "synthetic"
        / "synthetic_benign_passages.csv"
    )

    database_path = project_root / "data" / "chroma_db"

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

    texts = [document["text"] for document in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embedded_documents = []

    for document, embedding in zip(chunks, embeddings):
        embedded_documents.append({
            **document,
            "embedding": embedding.tolist()
        })

    print("\nInitializing ChromaDB...")

    store = VectorStore(
        database_path=database_path,
        collection_name="synthetic_benign_passages"
    )

    print("\nStoring documents...")

    stored_count = store.add_documents(embedded_documents)

    print("Documents processed:", stored_count)
    print("Total stored chunks:", store.count())

    query = "Which railway station is located in the northern region?"

    print("\nSearch query:", query)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = store.search(
        query_embedding=query_embedding,
        top_k=3
    )

    print("\nRetrieved results:")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Chunk ID:", result["chunk_id"])
        print("Distance:", result["distance"])
        print("Text:", result["text"])
