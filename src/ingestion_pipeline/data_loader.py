from pathlib import Path
from typing import List, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

def load_all_documents(data_dir: str) -> List[Document]:

    data_path = Path(data_dir).resolve()
    print(f"Loading documents from: {data_path}")
    documents = []

    # pdf files
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files : {[str(file) for file in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"Loading PDF file: {pdf_file}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded_docs = loader.load()
            print(f"Loaded {len(loaded_docs)} documents from {pdf_file}")
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading PDF file {pdf_file}: {e}")


    return documents

# example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Total documents loaded: {len(docs)}")
    print(f"First document content: {docs[100].page_content if docs else 'No documents loaded'}")