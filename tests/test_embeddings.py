from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

texts = [
    "Employees receive 18 days of annual paid leave.",
    "Workers are entitled to eighteen days of vacation.",
    "The company provides laptops to all employees.",
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Shape:", embeddings.shape)