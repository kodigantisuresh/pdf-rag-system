from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker

from src.retrieval.langchain_documents import (
    convert_to_langchain_documents
)

from src.vectorstore.langchain_faiss import (
    create_vector_store
)


PDF_PATH = "data/raw/sample.pdf"


TEST_CASES = [

    {
        "query": (
            "How many annual leave days "
            "do employees receive?"
        ),
        "expected_page": 2
    },

    {
        "query": (
            "What are the rules for "
            "working remotely?"
        ),
        "expected_page": 2
    },

    {
        "query": (
            "How should phishing incidents "
            "be reported?"
        ),
        "expected_page": 3
    }
]


# Load
loader = PDFLoader(PDF_PATH)

documents = loader.load()


# Chunk
chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50
)

chunks = chunker.chunk_documents(
    documents
)


# LangChain documents
langchain_documents = (
    convert_to_langchain_documents(
        chunks
    )
)


# Vector store
vector_store = create_vector_store(
    langchain_documents
)


# Evaluation
correct = 0


for test_case in TEST_CASES:

    query = test_case["query"]

    expected_page = test_case[
        "expected_page"
    ]

    results = vector_store.similarity_search(
        query,
        k=3
    )

    retrieved_pages = [
        document.metadata.get("page")
        for document in results
    ]

    hit = expected_page in retrieved_pages

    if hit:
        correct += 1

    print("\n" + "=" * 70)

    print(
        f"Query: {query}"
    )

    print(
        f"Expected page: {expected_page}"
    )

    print(
        f"Retrieved pages: {retrieved_pages}"
    )

    print(
        f"Hit@3: {hit}"
    )


recall_at_3 = (
    correct / len(TEST_CASES)
)


print("\n" + "=" * 70)

print(
    f"Recall@3: {recall_at_3:.2%}"
)