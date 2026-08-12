from src.embeddings.embedding_model import EmbeddingModel


embedding_model = EmbeddingModel()

text = (
    "Employees are eligible for "
    "18 days of annual paid leave."
)

vector = embedding_model.embed_text(text)

print("Vector shape:", vector.shape)
print("Vector dimension:", len(vector))
print("First 10 values:", vector[:10])