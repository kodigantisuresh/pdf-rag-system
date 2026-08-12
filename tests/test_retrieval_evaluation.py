from src.ingestion.pdf_loader import PDFLoader
from src.chunking.document_chunker import DocumentChunker
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.faiss_store import FAISSVectorStore

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
# 3. CREATE EMBEDDINGS
# ============================================================

embedding_model = EmbeddingModel()

texts = [
    chunk["page_content"]
    for chunk in chunks
]

embeddings = embedding_model.embed_documents(
    texts
)


print(
    f"Embedding dimension: "
    f"{embeddings.shape[1]}"
)


# ============================================================
# 4. CREATE FAISS VECTOR STORE
# ============================================================

dimension = embeddings.shape[1]

vector_store = FAISSVectorStore(
    dimension=dimension
)

vector_store.add(
    embeddings,
    chunks
)


print(
    f"Vectors in FAISS: "
    f"{vector_store.index.ntotal}"
)


# ============================================================
# 5. RETRIEVAL EVALUATION
# ============================================================

def evaluate_recall(top_k: int):

    hits = 0

    print("\n")
    print("=" * 80)
    print(f"RECALL@{top_k}")
    print("=" * 80)

    for test_case in TEST_CASES:

        query = test_case["query"]

        expected_pages = (
            test_case["expected_pages"]
        )

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        query_embedding = (
            embedding_model.embed_query(
                query
            )
        )

        # ----------------------------------------------------
        # Search FAISS
        # ----------------------------------------------------

        results = vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # ----------------------------------------------------
        # Extract retrieved pages
        # ----------------------------------------------------

        retrieved_pages = []

        print("\nRetrieved Results:")

        for rank, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            page = document["metadata"]["page"]

            score = result["score"]

            metadata = document.get("metadata", {})

            chunk_id = document["metadata"].get(
                "chunk_id",
                "N/A"
            )

            retrieved_pages.append(page)

            print(
                f"Rank {rank} | "
                f"Score: {score:.4f} | "
                f"Page: {page} | "
                f"Chunk: {chunk_id}"
            )

        # ----------------------------------------------------
        # Check whether expected page was retrieved
        # ----------------------------------------------------

        hit = any(
            page in expected_pages
            for page in retrieved_pages
        )

        if hit:
            hits += 1

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("\nQuery:")
        print(query)

        print(
            f"Expected Pages: "
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
        f"Recall@{top_k}: "
        f"{recall:.2%}"
    )

    print("-" * 80)

    return recall


# ============================================================
# 6. RUN EVALUATION
# ============================================================

recall_1 = evaluate_recall(1)

recall_3 = evaluate_recall(3)

recall_5 = evaluate_recall(5)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("RETRIEVAL EVALUATION SUMMARY")
print("=" * 80)

print(
    f"Recall@1 : {recall_1:.2%}"
)

print(
    f"Recall@3 : {recall_3:.2%}"
)

print(
    f"Recall@5 : {recall_5:.2%}"
)

print("=" * 80)