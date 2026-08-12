from langchain_core.documents import Document


def convert_to_langchain_documents(
    chunks: list[dict]
) -> list[Document]:

    documents = []

    for chunk in chunks:

        document = Document(
            page_content=chunk["page_content"],
            metadata=chunk["metadata"]
        )

        documents.append(document)

    return documents