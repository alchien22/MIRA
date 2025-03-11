from pathlib import Path

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
    """Loads all CSV files and returns documents with metadata."""
    docs = []
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        print(f"Loading {path}")
        loader = CSVLoader(
            file_path=path,
            csv_args={"delimiter": ",", "quotechar": '"'},
            metadata_columns=["note_id", "subject_id", "storetime"],
            content_columns=["text"],
        )
        docs.extend(loader.load())

    return docs