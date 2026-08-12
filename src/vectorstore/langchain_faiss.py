from langchain_community.vectorstores import FAISS

from src.embeddings.langchain_embeddings import (
    create_embedding_model
)


def create_vector_store(
    documents
):

    embeddings = create_embedding_model()

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store