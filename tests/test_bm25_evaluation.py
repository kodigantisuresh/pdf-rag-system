from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker
from src.retrieval.bm25_retriever import BM25Retriever

from tests.retrieval_dataset import TEST_CASES


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
# 4. EVALUATION FUNCTION
# ============================================================

def evaluate_recall(top_k: int):

    hits = 0

    print("\n")
    print("=" * 80)
    print(f"BM25 RECALL@{top_k}")
    print("=" * 80)

    for test_case in TEST_CASES:

        query = test_case["query"]

        expected_pages = (
            test_case["expected_pages"]
        )

        # ----------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------

        results = retriever.search(
            query,
            top_k=top_k
        )

        # ----------------------------------------------------
        # Extract retrieved pages
        # ----------------------------------------------------

        retrieved_pages = []

        print("\nQuery:")
        print(query)

        print("\nRetrieved Results:")

        for rank, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            score = result["score"]

            metadata = document["metadata"]

            page = metadata["page"]

            chunk_id = metadata.get(
                "chunk_id",
                "N/A"
            )

            retrieved_pages.append(
                page
            )

            print(
                f"Rank {rank} | "
                f"Score: {score:.4f} | "
                f"Page: {page} | "
                f"Chunk: {chunk_id}"
            )

        # ----------------------------------------------------
        # Check hit
        # ----------------------------------------------------

        hit = any(
            page in expected_pages
            for page in retrieved_pages
        )

        if hit:
            hits += 1

        print(
            f"\nExpected Pages: "
            f"{expected_pages}"
        )

        print(
            f"Retrieved Pages: "
            f"{retrieved_pages}"
        )

        print(
            f"Result: "
            f"{'PASS' if hit else 'FAIL'}"
        )

    # ========================================================
    # Calculate Recall
    # ========================================================

    recall = hits / len(TEST_CASES)

    print("\n" + "-" * 80)

    print(
        f"Hits: {hits}/{len(TEST_CASES)}"
    )

    print(
        f"BM25 Recall@{top_k}: "
        f"{recall:.2%}"
    )

    print("-" * 80)

    return recall


# ============================================================
# 5. RUN EVALUATION
# ============================================================

recall_1 = evaluate_recall(1)

recall_3 = evaluate_recall(3)

recall_5 = evaluate_recall(5)


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("BM25 RETRIEVAL EVALUATION SUMMARY")
print("=" * 80)

print(
    f"BM25 Recall@1 : {recall_1:.2%}"
)

print(
    f"BM25 Recall@3 : {recall_3:.2%}"
)

print(
    f"BM25 Recall@5 : {recall_5:.2%}"
)

print("=" * 80)