from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


def load_all_documents(data_dir: str) -> List[Document]:
    data_path = Path(data_dir).resolve()
    print(f"Loading documents from: {data_path}")

    documents = []

    # Load PDFs, one document per page
    pdf_files = list(data_path.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        try:
            loaded_docs = PyMuPDFLoader(str(pdf_file)).load()
            print(f"Loaded {len(loaded_docs)} pages from {pdf_file}")
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")

    # Load Markdown files, one document per file
    markdown_files = list(data_path.rglob("*.md"))
    markdown_files += list(data_path.rglob("*.markdown"))
    print(f"Found {len(markdown_files)} Markdown files")

    for markdown_file in markdown_files:
        try:
            text = markdown_file.read_text(encoding="utf-8")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(markdown_file), "file_type": "markdown"},
                    )
                )
                print(f"Loaded Markdown file: {markdown_file}")
        except Exception as e:
            print(f"Error loading Markdown file {markdown_file}: {e}")

    return documents

# example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Total documents loaded: {len(docs)}")
    print(f"First document content: {docs[100].page_content if docs else 'No documents loaded'}")