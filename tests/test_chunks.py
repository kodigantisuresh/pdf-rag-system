from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker


PDF_PATH = "data/raw/sample.pdf"


loader = PDFLoader(PDF_PATH)

documents = loader.load()


chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50
)

chunks = chunker.chunk_documents(
    documents
)


print("\n" + "=" * 80)
print("ALL CHUNKS")
print("=" * 80)


for index, chunk in enumerate(chunks):

    print("\n" + "-" * 80)

    print(f"Chunk Index : {index}")

    print(
        f"Chunk ID    : "
        f"{chunk['metadata']['chunk_id']}"
    )

    print(
        f"Page        : "
        f"{chunk['metadata']['page']}"
    )

    print(
        f"Characters  : "
        f"{len(chunk['page_content'])}"
    )

    print("\nText:")

    print(chunk["page_content"])