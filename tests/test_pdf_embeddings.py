from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker
from src.embeddings.embedding_model import EmbeddingModel


pdf_path = "data/raw/sample.pdf"


# 1. Load PDF
loader = PDFLoader(pdf_path)

documents = loader.load()


# 2. Chunk documents
chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50
)

chunks = chunker.chunk_documents(documents)


# 3. Extract text
texts = [
    chunk["page_content"]
    for chunk in chunks
]


# 4. Generate embeddings
embedding_model = EmbeddingModel()

embeddings = embedding_model.embed_documents(
    texts
)


print("Number of pages:", len(documents))
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Embedding dimensions:", embeddings.shape[1])