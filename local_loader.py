import os
from pathlib import Path

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

def load_txt_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob('**/*.txt')
    for path in paths:
        print(f"Loading {path}")
        loader = TextLoader(path)
        docs.extend(loader.load())
    return docs


def load_csv_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


def get_document_text(uploaded_file, title=None):
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text
        doc_text = uploaded_file.read().decode()
        docs.append(doc_text)

    return docs
