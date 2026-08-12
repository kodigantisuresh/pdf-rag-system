"""

"""

from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.faiss_store import FAISSVectorStore
from src.retrieval.retriever import Retriever

PDF_PATH = "data/raw/sample.pdf"

# 1. Load
loader = PDFLoader(PDF_PATH)
documents = loader.load()

# 2. Chunk
chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50
)
chunks = chunker.chunk_documents(documents)

# 3. Embeddings
embedding_model = EmbeddingModel()
texts = [
    chunk["page_content"]
    for chunk in chunks
]

embeddings = embedding_model.embed_documents(texts)

# 4. vectore store
dimension = embeddings.shape[1]

vector_store = FAISSVectorStore(dimension=dimension)
vector_store.add(embeddings, chunks)

# 5. Retriever
retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model,
    top_k=3
)

# 6. Query
query = (
    "How many annual leave days "
    "do employees receive?"
)

results = retriever.retrieve(query)

# 7. Display
for rank, result in enumerate(
    results,
    start=1
):
    document = result["document"]
    print("\n" + "=" * 70)
    print(
        f"Rank: {rank}"
    )
    print(
        f"Score: {result['score']:.4f}"
    )
    print(
        f"Source: "
        f"{document['metadata']['source']}"
    )
    print(
        f"Page: "
        f"{document['metadata']['page']}"
    )
    print(
        f"Chunk: "
        f"{document['metadata']['chunk_id']}"
    )
    print(
        f"\nText:\n"
        f"{document['page_content']}"
    )