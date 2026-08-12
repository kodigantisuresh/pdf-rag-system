from src.ingestion.pdf_loader import PDFLoader


PDF_PATH = "data/raw/sample.pdf"

loader = PDFLoader(PDF_PATH)

documents = loader.load()


print("\n" + "=" * 80)
print("RAW PDF EXTRACTION")
print("=" * 80)


for document in documents:

    print("\n" + "-" * 80)

    print(
        f"Page: "
        f"{document['metadata']['page']}"
    )

    print(
        f"Characters: "
        f"{len(document['page_content'])}"
    )

    print("\nTEXT:")

    print(document["page_content"])