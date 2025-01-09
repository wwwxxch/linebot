from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# load documents
def load_documents(directory_path: str):

    logger.info(f"start loading, from: {directory_path}")

    # txt
    txt_loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},  # for chinese
    )

    # csv
    csv_loader = DirectoryLoader(
        directory_path,
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "csv_args": {
                "fieldnames": ["index", "title", "date", "author", "link", "content"],
                "delimiter": ",",
            },
            "content_columns": "content",
        },
    )

    documents = txt_loader.load() + csv_loader.load()
    logger.info(f"successfully loading {len(documents)} dcouments")
    return documents


# split documents
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    logger.info("start splitting documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", "", ","],  # 適合中文的分割符號
    )

    splits = text_splitter.split_documents(documents)
    logger.info(f"document split to {len(splits)} chunks")
    return splits


# create vector store
def create_vector_store(splits, persist_directory: str):
    logger.info("start creating vector store")

    # init embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

    # init vector store
    vector_store = Chroma(
        collection_name="cat_care", embedding_function=embeddings, persist_directory=persist_directory
    )
    vector_store.add_documents(splits)

    logger.info(f"vector store saved to {persist_directory}")

    return vector_store


def main():
    DOCS_DIRECTORY = os.path.join(os.getcwd(), "input_docs")
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    DB_DIRECTORY = os.path.join(os.getcwd(), "cat_care_db")

    try:
        # 1. load documents
        documents = load_documents(DOCS_DIRECTORY)

        # 2. split documents
        splits = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        # 3. create vector store
        _ = create_vector_store(splits, DB_DIRECTORY)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
