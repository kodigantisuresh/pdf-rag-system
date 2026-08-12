from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker

pdf_path = "data/raw/sample.pdf"

loader = PDFLoader(pdf_path)
documents = loader.load()
print(f"Pages loaded: {len(documents)}")

chunker = DocumentChunker(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = chunker.chunk_documents(documents)
print(f"Total chunks: {len(chunks)}")

for chunk in chunks[:5]:  # Print the first 5 chunks for brevity

    print("\n==============================")
    print(
        "Chunk ID:",
        chunk["metadata"]["chunk_id"]
    )
    print(
        "Source:",
        chunk["metadata"]["source"]
    )
    print(
        "Page:",
        chunk["metadata"]["page"]
    )
    print(
        "Text:",
        chunk["page_content"][:300]
    )