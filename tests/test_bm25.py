from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker
from src.retrieval.bm25_retriever import BM25Retriever


PDF_PATH = "data/raw/sample.pdf"


# ============================================================
# 1. LOAD PDF
# ============================================================

loader = PDFLoader(PDF_PATH)

documents = loader.load()


# ============================================================
# 2. CHUNK DOCUMENTS
# ============================================================

chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50
)

chunks = chunker.chunk_documents(
    documents
)


print(
    f"\nTotal chunks: {len(chunks)}"
)


# ============================================================
# 3. CREATE BM25 RETRIEVER
# ============================================================

retriever = BM25Retriever(
    chunks
)


print(
    "BM25 retriever created."
)


# ============================================================
# 4. TEST QUERY
# ============================================================

query = (
    "How many annual leave days "
    "do employees receive?"
)


results = retriever.search(
    query,
    top_k=5
)


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)

print(
    f"Query: {query}"
)

print("=" * 80)


for rank, result in enumerate(
    results,
    start=1
):

    document = result["document"]

    score = result["score"]

    metadata = document["metadata"]

    print(
        f"\nRank: {rank}"
    )

    print(
        f"Score: {score:.4f}"
    )

    print(
        f"Source: "
        f"{metadata.get('source', 'N/A')}"
    )

    print(
        f"Page: "
        f"{metadata.get('page', 'N/A')}"
    )

    print(
        f"Chunk ID: "
        f"{metadata.get('chunk_id', 'N/A')}"
    )

    print(
        "Text:"
    )

    print(
        document["page_content"]
    )

    print("-" * 80)