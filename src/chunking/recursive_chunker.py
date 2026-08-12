class RecursiveChunker:

    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 50
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def chunk_test(self, text: str):
        text = text.strip()
        return self._split_recursive(
            text,
            self.separators
        )
    def _split_recursive(
            self,
            text: str,
            separators: list[str]
    ):
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0]
        parts = text.split(separator)
        chunks = []
        current_chunk = ""

        for part in parts:

            candidate = (
                current_chunk + separator + part
                if current_chunk
                else part
            )

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(
                        current_chunk.strip()
                    )
                current_chunk = part

        if current_chunk:
            chunks.append(
                current_chunk.strip()
            )
        return chunks
    