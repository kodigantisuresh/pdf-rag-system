"""
    The constructor:
        def __init__(self, chunk_size: int = 500):
            means our default chunk size is 500 characters

        ex:
            chunk_size = 500
                Then:
                    start = 0
                    start = 500
                    start = 1000
                    start = 1500
                    start = 2000
            And we extract: text[start:start + self.chunk_size]
                So:

                    0:500
                    500:1000
                    1000:1500
                    1500:2000
                    2000:2500 ---> The last one simply ends at the end of the string

    But we can change it like; "chunker = FixedSizeChunker(chunk_size=1000)"
"""

"""
    But LLMs don't fundamentally process "characters", They process "tokens".
    ThereFore: 500 characters ≠ 500 tokens
    All the chunks are converted into tokens before being processed by the model.
"""

"""
    Recursive Character Chunking:
        The idea is to split the text into chunks based on a hierarchy of separators.
        For example, we might first try to split by paragraphs, then by sentences, and finally by words.
        This way, we can ensure that the chunks are semantically meaningful and not just arbitrary slices of text.
        
        The idea is:
            Try to split the text at natural boundaries before resorting to smaller boundaries.

        Conceptually:
            Paragraph
            ↓
            Sentence
            ↓
            Phrase
            ↓
            Word

"""

"""
        A simplified hierarchy looks like:
            "\n\n" ---> Paragraph boundary
            ↓
            "\n" ---> Sentence boundary or Line boundary
            ↓
            " "  ---> Word boundary
            ↓
            ""  ---> Character boundary or character-level split
"""

class FixedSizeChunker:
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

    def chunk_text(self, text: str):
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += step

        return chunks
