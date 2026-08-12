"""
    PyMuPDF: 
        We'll use PyMuPDF first because it gives us direct control over pages, 
        text extraction, and metadata.
"""

"""
    Why page_number + 1?:
    
        In Python, indexing starts at 0. So, when we enumerate over the pages of the document, 
        the first page is indexed as 0. To display the page number in a more human-readable format, 
        we add 1 to the index. This way, the first page will be displayed as "Page 1" instead of "Page 0".

        ex: 
            Python       Human
            -------------------
            0            Page 1
            1            Page 2
            2            Page 3

            So:
                page_number + 1 ---> converts Python's zero-based index into a human-friendly page number.
"""

import fitz      #its librarie of PyMuPDF

pdf_path = "data/raw/sample.pdf"

document = fitz.open(pdf_path)
print(f"Number of pages: {len(document)}")

for page_number, page in enumerate(document):
    text = page.get_text()

    print("=" * 50)
    print(f"Page {page_number + 1}")
    print("=" * 50)

    print(text[:1000])  # Print the first 1000 characters of the page text

document.close() 