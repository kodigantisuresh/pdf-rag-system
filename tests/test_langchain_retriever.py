from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker

from src.retrieval.langchain_documents import (
    convert_to_langchain_documents
)

from src.vectorstore.langchain_faiss import (
    create_vector_store
)


PDF_PATH = "data/raw/sample.pdf"


# -----------------------------------
# 1. Load PDF
# -----------------------------------

loader = PDFLoader(PDF_PATH)

documents = loader.load()


# -----------------------------------
# 2. Chunk
# -----------------------------------

chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50
)

chunks = chunker.chunk_documents(
    documents
)


# -----------------------------------
# 3. Convert to LangChain Documents
# -----------------------------------

langchain_documents = (
    convert_to_langchain_documents(
        chunks
    )
)


# -----------------------------------
# 4. Create Vector Store
# -----------------------------------

vector_store = create_vector_store(
    langchain_documents
)


print(
    "LangChain FAISS vector store created."
)


# -----------------------------------
# 5. Create Retriever
# -----------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# -----------------------------------
# 6. Query
# -----------------------------------

query = (
    "How many annual leave days "
    "do employees receive?"
)


# -----------------------------------
# 7. Retrieve
# -----------------------------------

results = retriever.invoke(
    query
)


# -----------------------------------
# 8. Display
# -----------------------------------

for rank, document in enumerate(
    results,
    start=1
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"Rank: {rank}"
    )

    print(
        f"Source: "
        f"{document.metadata.get('source')}"
    )

    print(
        f"Page: "
        f"{document.metadata.get('page')}"
    )

    print(
        f"Chunk: "
        f"{document.metadata.get('chunk_id')}"
    )

    print(
        f"\nText:\n"
        f"{document.page_content}"
    )