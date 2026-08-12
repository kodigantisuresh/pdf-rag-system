"""
What BM25Okapi Does: 

    This:
        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        ---> builds the BM25 index/statistics.

    Then:
        scores = self.bm25.get_scores(
            tokenized_query
        )

        ---> calculates a score for every chunk.
"""


from rank_bm25 import BM25Okapi

class BM25Retriever:  # ← Make sure class name is exactly this
    
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_documents = [
            self._tokenize(self._get_text(document))
            for document in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_documents)
    
    def _get_text(self, document):
        return document["page_content"]
    
    def _tokenize(self, text):
        return text.lower().split()
    
    def search(self, query: str, top_k: int = 5):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )
        
        results = []
        for index in ranked_indices[:top_k]:
            results.append({
                "score": float(scores[index]),
                "document": self.documents[index]
            })
        
        return results