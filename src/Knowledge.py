import os
import json
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

import textwrap
from pprint import pprint
from sqlalchemy.engine.url import make_url
from IPython.display import Markdown
from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector
from sqlalchemy.engine.url import make_url
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import JSONLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain.load import dumps, loads
from tqdm import tqdm

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from TextCleaner import clean_pdf_documents

USE_BOOKS = "books_pdf"

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
connection_string = os.getenv("DB_CONNECTION")

# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))  # make sure it's in your .env

current_script_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_script_dir, USE_BOOKS)
book_docs: list[Document] = []

# Setup logger
log_filename = f"cleaning_{datetime.now().strftime('%Y%m%d')}.log"
log_filename = os.path.join(current_script_dir, log_filename)
logging.basicConfig(filename=log_filename, filemode="a", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  # 'w' to overwrite, 'a' to append

# Define document collections to be stored in PostgreSQL
collections = {
    "Books": None,  # PDF documents converted to Markdown,
}

# Chunk according to Markdown
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4"), ("#####", "Header 5")], strip_headers=False)


# Convert PDFS to Markdown
def load_and_clean_book(filepath: str) -> list[Document]:
    try:
        docs = PyMuPDF4LLMLoader(file_path=filepath, mode="single").load()
        cleaned_docs, _ = clean_pdf_documents(docs, min_content_length=20, verbose=False)
        return cleaned_docs
    except Exception as e:
        logging.error(f"Failed to process {filepath}: {e}")
        return []


def convert_books() -> list[Document]:
    book_files = [os.path.join(books_dir, f) for f in os.listdir(books_dir) if os.path.splitext(f)[1].lower() in {".pdf", ".txt"}]

    all_docs = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(load_and_clean_book, fp) for fp in book_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing books"):
            all_docs.extend(future.result())

    print(f"Number of books: {len(all_docs)}")

    chunks = []
    for b in tqdm(all_docs, desc="Chunking books by markdown header"):
        new_chunks = text_splitter.split_text(b.page_content)
        for c in new_chunks:
            c.metadata.update(b.metadata)
        chunks.extend(new_chunks)

    logging.info("Finished processing books.")
    print(f"Cleaning stats saved to {log_filename}")
    return chunks


# def convert_books() -> list[Document]:
#     global book_docs
#     for book in tqdm(os.listdir(books_dir), desc="chunking docs"):
#         print(book)
#         fp = os.path.join(books_dir, book)
#         book_name, filetype = os.path.splitext(book)
#         if filetype == '.pdf' or filetype == ".txt":
#             docs = PyMuPDF4LLMLoader(file_path=os.path.join(books_dir, book), mode='single').load()
#             book_docs.extend(docs)
#             print(f"Number of documents: {len(docs)}")

#     print(f"Number of books: {len(book_docs)}")

#     # Clean PDFs
#     book_docs, cleaning_stats = clean_pdf_documents(book_docs, min_content_length=20, verbose=False)

#     # This takes 4 minutes
#     chunks = []
#     for b in tqdm(book_docs, desc="chunking books by markdown header"):
#         new_chunks = text_splitter.split_text(b.page_content)
#         for c in new_chunks:
#             c.metadata.update(b.metadata)
#         chunks.extend(new_chunks)

#     # Log the cleaning stats
#     logging.info("Cleaning statistics:")
#     logging.info(cleaning_stats)

#     print(f"Cleaning stats saved to {log_filename}")
#     return chunks


# Load documents into PostgreSQL vector database
def load_documents(docs: List[Document]) -> PGVector:
    global collections
    databases = {}
    print(f"Number of docs attempted to be loaded: {len(docs)}")
    for name, docs in collections.items():
        pg_vector_args = dict(embedding=embedding_model, documents=docs, collection_name=name, connection=connection_string, use_jsonb=True)  # OpenAI embeddings for vectorization  # The documents to be stored  # The name of the collection in PostgreSQL  # Connection string to the PostgreSQL database  # Store metadata as JSONB for efficient querying
        print(pg_vector_args)

        book_data_vector_store = PGVector.from_documents(**pg_vector_args)
        databases[name] = book_data_vector_store
    # Extract the database name from the connection string
    database_name = make_url(connection_string).database

    # Display confirmation message
    print(f" Successfully loaded {len(docs)} chunks into '{name}' collection in '{database_name}'.")
    return book_data_vector_store


def main():
    global collections
    chunks = convert_books()
    collections["Books"] = chunks

    print(f"Number of chunks: {len(chunks)}")
    vector_db = load_documents(collections["Books"])


if __name__ == "__main__":
    main()
