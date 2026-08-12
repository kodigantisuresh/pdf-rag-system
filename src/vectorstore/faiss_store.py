"""
What Is FAISS(Facebook AI Similarity Search)?


	It's a library designed for:

		Efficient similarity search over dense vectors.
		FAISS provides optimized vector indexes for doing this efficiently
	Our First FAISS Index: faiss.IndexFlatIP
		Index
  		↓
		Flat
 		 ↓
		IP
	Flat Means:
		Store the vectors directly without an approximate index structure.
	IP(Inner Product):
		inner product becomes equivalent to cosine similarity.

	So IndexFlatIP:
		performs exact inner-product search.
	index.search(query, k=5) means:
		Find the 5 nearest vectors to the query.
		This is our Top-K retrieval.
	FAISS returns: scores, indices
		scores: Similarity/distance values.
		indices: Positions of the matching vectors.

"""

import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension: int):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.documents = []

    def add(
        self,
        embeddings,
        documents
    ):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding,
        top_k: int = 5
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(
                1, -1
            )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append(
                {
                    "score": float(score),
                    "document": self.documents[index]
                }
            )

        return results
