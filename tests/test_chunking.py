"""
    Now let's introduce one of the most important concepts in RAG is 
    Chunk Overlap:
        Instead of:
            Chunk 1
                AAAAAAAAAA
            Chunk 2
                BBBBBBBBBB
        we use:
            Chunk 1
                AAAAAAAAAABBBB
            Chunk 2
                BBBCCCCCCCC

# The overlapping region preserves context.

"""

from src.chunking.fixed_chunker import FixedSizeChunker

text = """
Sample Company provides employees with 18 days of annual paid leave.
Employees should normally submit leave requests through the HR portal
at least three working days before the requested start date.
Employees working remotely are responsible for protecting company
information and remaining available during agreed working hours.
"""

chunker = FixedSizeChunker(chunk_size=100)
chunks = chunker.chunk_text(text)
print(f"Number of chunks: {len(chunks)}")

for index, chunk in enumerate(chunks):

    print("\n==============================")
    print(f"CHUNK {index + 1}")
    print("==============================")

    print(chunk)

