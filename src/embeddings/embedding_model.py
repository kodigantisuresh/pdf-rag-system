from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def embed_text(self, text: str):

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32
    ):

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def embed_query(self, query: str):

        return self.model.encode(
            query,
            normalize_embeddings=True
        )