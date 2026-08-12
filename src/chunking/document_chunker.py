from typing import List, Dict


class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(
        self,
        documents: List[Dict]
    ) -> List[Dict]:

        all_chunks = []

        for document in documents:

            text = document["page_content"].strip()

            metadata = document["metadata"]

            chunks = self._chunk_text(text)

            for chunk_index, chunk_text in enumerate(
                chunks,
                start=1
            ):

                chunk_metadata = metadata.copy()

                chunk_metadata["chunk_id"] = (
                    f"{metadata['source']}"
                    f"_p{metadata['page']}"
                    f"_c{chunk_index}"
                )

                all_chunks.append(
                    {
                        "page_content": chunk_text,
                        "metadata": chunk_metadata
                    }
                )

        return all_chunks

    def _chunk_text(
        self,
        text: str
    ) -> List[str]:

        if not text:
            return []

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:

            # Calculate the end position
            end = min(
                start + self.chunk_size,
                text_length
            )

            # Extract chunk
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # We reached the end of the document
            if end >= text_length:
                break

            # Move forward while maintaining overlap
            next_start = end - self.chunk_overlap

            # Safety check: prevent infinite loops
            if next_start <= start:
                raise RuntimeError(
                    "Chunking failed to make progress. "
                    "Check chunk_size and chunk_overlap."
                )

            start = next_start

        return chunks