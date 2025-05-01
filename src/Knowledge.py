import os
import logging
from datetime import datetime
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from TextCleaner import clean_pdf_documents

# Load environment variables from .env file
load_dotenv()

# Configure logging
current_script_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(current_script_dir, f"cleaning_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(filename=log_filename, filemode="a", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the directory containing the books
BOOKS_DIR = os.path.join(os.getenv("HOME"), "Books")

# Define supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt"}

# Initialize the text splitter
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4"), ("#####", "Header 5")], strip_headers=False)

# Initialize the embedding model
# Uncomment the desired embedding model

# OpenAI Embeddings
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

# Google Generative AI Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))

# PostgreSQL connection string
connection_string = os.getenv("DB_CONNECTION")


def load_and_clean_file(filepath: str) -> List[Document]:
    """
    Loads and cleans a single file, returning a list of Document objects.
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyMuPDF4LLMLoader(file_path=filepath, mode="single")
            docs = loader.load()
        elif ext == ".epub":
            loader = UnstructuredEPubLoader(filepath, mode="elements")
            docs = loader.load()
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            docs = [Document(page_content=content, metadata={"source": filepath})]
        else:
            logging.info(f"Skipped unsupported file: {filepath}")
            return []

        cleaned_docs, _ = clean_pdf_documents(docs, min_content_length=20, verbose=False)
        return cleaned_docs
    except Exception as e:
        logging.error(f"Failed to process {filepath}: {e}")
        return []


def process_and_upload_file(filepath: str):
    """
    Processes a single file: loads, cleans, chunks, and uploads to the vector store.
    """
    filename = os.path.basename(filepath)
    collection_name = os.path.splitext(filename)[0]  # Use filename without extension as collection name

    docs = load_and_clean_file(filepath)
    if not docs:
        logging.warning(f"No documents returned from {filename}")
        return

    chunks = []
    for doc in docs:
        split_chunks = text_splitter.split_text(doc.page_content)
        for chunk in split_chunks:
            chunk.metadata.update(doc.metadata)
            chunks.append(chunk)

    if not chunks:
        logging.warning(f"No chunks generated from {filename}")
        return

    try:
        vector_store = PGVector.from_documents(embedding=embedding_model, documents=chunks, collection_name=collection_name, connection=connection_string, use_jsonb=True)
        logging.info(f"Successfully uploaded {len(chunks)} chunks from {filename} to collection '{collection_name}'")
    except Exception as e:
        logging.error(f"Failed to upload chunks from {filename}: {e}")


def main():
    """
    Main function to process and upload all supported files in the BOOKS_DIR.
    """
    if not os.path.isdir(BOOKS_DIR):
        logging.error(f"Books directory does not exist: {BOOKS_DIR}")
        return

    book_files = [os.path.join(BOOKS_DIR, f) for f in os.listdir(BOOKS_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    if not book_files:
        logging.info("No supported book files found to process.")
        return

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_and_upload_file, filepath): filepath for filepath in book_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing books"):
            filepath = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing file {filepath}: {e}")

    print(f"Processing complete. Logs saved to {log_filename}")


if __name__ == "__main__":
    main()
