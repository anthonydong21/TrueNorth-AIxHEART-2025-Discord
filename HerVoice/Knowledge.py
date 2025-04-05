
import os
from urllib import response
from dotenv import load_dotenv
from pprint import pprint
from llama_index.core.llms.chatml_utils import DEFAULT_SYSTEM_PROMPT
from sqlalchemy.engine.url import make_url
from IPython.display import Markdown
from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector
from sqlalchemy.engine.url import make_url
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser 
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.load import dumps, loads

# Extract the database name from the connection string
database_name = make_url(connection_string).database 

# Define document collections to be stored in PostgreSQL
collections = {
    "University_Events": semantic_split_web_docs,    # Web documents chunked semantically
    "Current_Student_Content": semantic_split_pdf_docs           # Book pages cleaned and used as chunks
}

# Load documents into PostgreSQL vector database
databases = {}
for name, docs in collections.items():
    databases[name] = PGVector.from_documents(
        embedding=embedding_model,      # OpenAI embeddings for vectorization
        documents=docs,                 # The documents to be stored 
        collection_name=name,           # The name of the collection in PostgreSQL
        connection=connection_string,   # Connection string to the PostgreSQL database
        use_jsonb=True      # Store metadata as JSONB for efficient querying
    )

    # Display confirmation message
    print(f" Successfully loaded {len(docs)} chunks into '{name}' collection in '{database_name}'.")
