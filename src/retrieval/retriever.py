"""
What Is a Retriever?

A retriever is the component responsible for:
	Taking a user's query and returning the most relevant pieces of information 
	from the knowledge base.

Conceptually:
        Retriever
            │
            ├── Query processing
            ├── Query embedding
            ├── Vector search
            ├── Top-K selection
            ├── Score filtering
            └── Result formatting

"""

from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.faiss_store import FAISSVectorStore


class Retriever:

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        score_threshold: float | None = None
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str):
        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding,
            top_k = self.top_k
        )

        if self.score_threshold is not None:

            results = [
                result
                for result in results
                if result["score"]
                >= self.score_threshold
            ]

        return results
    