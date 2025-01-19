from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# from langchain_chroma.vectorstores import Chroma
import logging
from dotenv import load_dotenv
import os

load_dotenv()

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
    csv_loader_ptt = CSVLoader(
        file_path=os.path.join(directory_path, "ptt_cat.csv"),
        encoding="utf-8",
        csv_args={"fieldnames": ["index", "title", "date", "author", "link", "content"], "delimiter": ","},
        content_columns="content",
    )

    csv_loader_CatTestResult = CSVLoader(
        file_path=os.path.join(directory_path, "花花血檢結果.csv"),
        encoding="utf-8",
        csv_args={
            "fieldnames": ["date", "test_item", "test_result", "test_range", "test_unit"],
            "delimiter": ",",
        },
    )

    # pdf
    pdf_loader = UnstructuredLoader(
        file_path=os.path.join(directory_path, "花花檢查結果_20240608.pdf"),
        strategy="hi_res",
        partition_via_api=True,
    )

    documents = (
        txt_loader.load() + csv_loader_ptt.load() + csv_loader_CatTestResult.load() + pdf_loader.load()
    )
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
def create_vector_store(splits):
    logger.info("start creating vector store")

    # init embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

    # init vector store
    # vector_store = Chroma(
    #     collection_name="cat_care", embedding_function=embeddings, persist_directory=persist_directory
    # )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("catlinebot")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_documents(splits)

    # logger.info(f"vector store saved to {persist_directory}")
    logger.info("Documents successfully added to Pinecone")

    return vector_store


def main():
    DOCS_DIRECTORY = os.path.join(os.getcwd(), "input_docs")
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    # DB_DIRECTORY = os.path.join(os.getcwd(), "cat_care_db")

    try:
        # 1. load documents
        documents = load_documents(DOCS_DIRECTORY)

        # 2. split documents
        splits = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        # 3. create vector store
        _ = create_vector_store(splits)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
