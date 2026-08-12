"""
FAISS
 │
 ├── IndexFlatIP
 │      └── Exact search
 │
 ├── HNSW
 │      └── Graph-based ANN
 │
 ├── IVF
 │      └── Cluster-based ANN
 │
 ├── PQ
 │      └── Vector compression
 │
 └── IVF-PQ
        └── Clustering + compression
"""

from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.faiss_store import FAISSVectorStore

pdf_path = "data/raw/sample.pdf"

# 1. Load PDF
loader = PDFLoader(pdf_path)
documents = loader.load()

# 2. Chunk
chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50
)

chunks = chunker.chunk_documents(documents)

# 3. create embeddings
texts = [
    chunk["page_content"]
    for chunk in chunks
]

embedding_model = EmbeddingModel()
embeddings = embedding_model.embed_documents(texts)

# 4. Create FAISS index
dimension = embeddings.shape[1]
vector_store = FAISSVectorStore(dimension=dimension)

# Add vectors
vector_store.add(embeddings, chunks)
print("Vectors in FAISS:", vector_store.index.ntotal)

# 6. query
query = (
    "How many annual leave days "
    "do employees receive?"
)

query_embedding = embedding_model.embed_query(query)

# 7. Search
results = vector_store.search(query_embedding, top_k=3)

# 8. Display results
for rank, result in enumerate(results, start=1):
    document=result["document"]
    print("\n==============================")
    print(f"RESULT {rank}")
    print("==============================")

    print(
        "Score:",
        result["score"]
    )
    print(
        "Source:",
        document["metadata"]["source"]
    )

    print(
        "Page:",
        document["metadata"]["page"]
    )

    print(
        "Chunk ID:",
        document["metadata"]["chunk_id"]
    )

    print(
        "Text:",
        document["page_content"]
    )