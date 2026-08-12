from src.ingestion.pdf_loader import PDFLoader

pdf_path = "data/raw/sample.pdf"
loader = PDFLoader(pdf_path)
pages = loader.load()

print(f"Load pages: {len(pages)}")

for page in pages:
    print("\n----------------------------")
    print("source:", page["metadata"]["source"])
    print("page:", page["metadata"]["page"])

    print("Text:")
    print(page["page_content"][:500])  # Print the first 500 characters of the page text
