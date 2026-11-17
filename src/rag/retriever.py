
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from src.rag.vector_store import VectorStore, DEFAULT_MODEL


class Retriever:
    """
    Convert user queries into embeddings and retrieve
    semantically relevant documents from ChromaDB.
    """

    def __init__(
        self,
        database_path: str | Path,
        collection_name: str,
        model: SentenceTransformer | None = None
    ):
        self.model = model or SentenceTransformer(DEFAULT_MODEL)

        self.vector_store = VectorStore(
            database_path=database_path,
            collection_name=collection_name
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for a query.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.vector_store.count() == 0:
            return []

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    database_path = project_root / "data" / "chroma_db"

    retriever = Retriever(
        database_path=database_path,
        collection_name="synthetic_benign_passages"
    )

    test_queries = [
        "Which railway station is in the northern region?",
        "Where is Ardent Station located?",
        "Which station is in the central plateau?"
    ]

    for query in test_queries:

        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("=" * 60)

        results = retriever.retrieve(
            query=query,
            top_k=3
        )

        if not results:
            print("No documents found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print("Chunk ID:", result["chunk_id"])
            print("Distance:", round(result["distance"], 4))
            print("Passage:", result["text"])
