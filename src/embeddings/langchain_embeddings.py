from langchain_huggingface import HuggingFaceEmbeddings

def create_embedding_model():

    return HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5",
        encode_kwargs = {
            "normalize_embeddings": True
        }
    )