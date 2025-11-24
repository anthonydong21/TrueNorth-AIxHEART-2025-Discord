import os
import sys
import traceback
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document  # Standard document format used in LangChain pipelines
from langchain.load import dumps, loads  # Serialize/deserialize LangChain objects
from langchain_community.vectorstores import FAISS  # FAISS vector store replacement for PGVector

from dotenv import load_dotenv

from .state import ChatState
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.llm import call_llm, get_embedding_model
from truenorth.utils.metaprompt import vectorstore_content_summary

load_dotenv()
logger = get_caller_logger()

# FAISS vector store path (replace database connection)
VECTOR_STORE_PATH = Path(__file__).parent.parent.parent / "vector_store" / "truenorth_kb_vectorstore/"

# Define the multi-query generation prompt
multi_query_generation_prompt = PromptTemplate.from_template(
    """
You are an AI assistant helping improve document retrieval in a vector-based search system.

---
                                                             
**Context about the database**
The vectorstore contains the following content:
{vectorstore_content_summary}

Your goal is to help retrieve **more relevant documents** by rewriting a user's question from multiple angles.
This helps compensate for the limitations of semantic similarity in vector search.

---

**Instructions**:
Given the original question and the content summary above:
1. Return the **original user question** first.
2. Then generate {num_queries} **alternative versions** of the same question.
    - Rephrase using different word choices, structure, or focus.
    - Use synonyms or shift emphasis slightly, but keep the original meaning.
    - Make sure all rewrites are topically relevant to the database content.

Format requirements:
- Do **not** include bullet points or numbers.
- Each version should appear on a **separate newline**.
- Return **exactly {num_queries} + 1 total questions** (1 original + {num_queries} new ones).  

---                                              

**Original user question**: {question}
"""
)


# Reciprocal Rank Fusion (RRF) Implementation
def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}  # Dictionary to store cumulative RRF scores for each document

    # Iterate through each ranked list of documents
    for docs in results:
        for i, doc in enumerate(docs):
            doc_str = dumps(doc)  # Convert document to a string format (JSON) to use as a dictionary key

            # Initialize the document's fused score if not already present
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            # Apply RRF scoring: 1 / (rank + k), where rank is 1-based
            rank = i + 1  # Adjust rank to start from 1 instead of 0
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort by cumulative RRF score (descending)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # Convert JSON strings back to Document objects and store RRF scores in metadata
    reranked_documents = []
    for doc_str, score in reranked_results:
        doc = loads(doc_str)  # Convert back to Document object
        doc.metadata["rrf_score"] = score  # Track how the document was ranked
        reranked_documents.append(doc)

    # Return the list of documents with scores embedded in metadata
    return reranked_documents


def load_faiss_vectorstore(embedding_model):
    """
    Load the FAISS vector store with proper error handling.

    Args:
        embedding_model: The embedding model to use for loading the vector store

    Returns:
        FAISS: Loaded FAISS vector store

    Raises:
        FileNotFoundError: If the vector store doesn't exist
        Exception: If there's an error loading the vector store
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. " "Please run Knowledge.py first to create the vector store.")

    try:
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"✅ Loaded FAISS vector store from {VECTOR_STORE_PATH}")
        logger.info(f"📊 Vector store contains {vectorstore.index.ntotal} documents")
        return vectorstore
    except Exception as e:
        logger.error(f"❌ Failed to load FAISS vector store: {e}")
        raise


def retrieve_documents(state: ChatState) -> ChatState:
    """
    Retrieves documents relevant to the user's question using multi-query RAG fusion.

    This node performs the following steps:
    - Reformulates the original user question into multiple diverse sub-queries.
    - Executes MMR-based retrieval for each reformulated query.
    - Applies Reciprocal Rank Fusion (RRF) to combine and rerank results.
    - Filters out metadata fields that are internal (like RRF scores).
    - Prepares and returns a list of LangChain `Document` objects to be used in downstream nodes.

    Args:
        state (ChatState): The current state of the LangGraph, containing the user's question.

    Returns:
        ChatState: Updated state containing a cleaned list of relevant `Document` objects.
    """
    logger.info("\n---QUERY TRANSLATION AND RAG-FUSION---")

    # Get embedding model
    embedding_model = get_embedding_model(model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"])

    # Load FAISS Vector Store that contains book data
    try:
        book_data_vector_store = load_faiss_vectorstore(embedding_model)
    except FileNotFoundError as e:
        logger.error(str(e))
        # Return state with empty documents if vector store not found
        return state
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return state

    logger.info(f"State Information: {state}")

    # Initialize LLM
    llm = lambda prompt: call_llm(prompt=prompt, model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"], pydantic_model=None, agent_name="document_retriever", verbose=False)

    question = state.question

    # Multi-query generator chain
    multi_query_generator = multi_query_generation_prompt | llm | (lambda x: [line.strip() for line in str(x.content).split("\n") if line.strip()])

    # RAG fusion chain with MMR retrieval
    retrieval_chain_rag_fusion_mmr = (
        multi_query_generator | book_data_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 15, "lambda_mult": 0.5}).map() | reciprocal_rank_fusion  # Final number of documents to return per query  # Initial candidate pool (larger for better diversity)  # Balances relevance (0) and diversity (1)  # Apply MMR retrieval to each reformulated query  # Rerank the combined results using RRF
    )

    # Run multi-query RAG + MMR + RRF pipeline to get relevant results
    try:
        rag_fusion_mmr_results = retrieval_chain_rag_fusion_mmr.invoke({"question": question, "num_queries": 3, "vectorstore_content_summary": vectorstore_content_summary})

        # Display summary of where results came from
        logger.info(f"Total number of results: {len(rag_fusion_mmr_results)}")
        for i, doc in enumerate(rag_fusion_mmr_results, start=1):
            excerpt = doc.page_content[:200].replace("\n", " ") + "..."  # first 200 characters
            source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            logger.info(f"     Document {i} from `{source}`, page {page}")
            logger.info(f"     Excerpt: {excerpt}")

        # Convert retrieved documents into Document objects with metadata and page_content only
        formatted_doc_results = [Document(metadata={k: v for k, v in doc.metadata.items() if k != "rrf_score"}, page_content=doc.page_content) for doc in rag_fusion_mmr_results]  # Remove rrf score and document id

        state.documents.extend(formatted_doc_results)
        logger.info(f"✅ Added {len(formatted_doc_results)} documents to state")

    except Exception as e:
        logger.error(f"❌ Error during document retrieval: {e}")
        
        # Get detailed traceback info
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_traceback:
            # Extract the last frame (where the error occurred)
            tb = traceback.extract_tb(exc_traceback)
            if tb:
                last_call = tb[-1]
                filename = last_call.filename
                line_number = last_call.lineno
                line_content = last_call.line
                
                logger.error(f"Location: {filename}, line {line_number}")
                logger.error(f"Code: {line_content}")
                
        # Continue with empty documents rather than failing completely
        pass
    return state


def get_vectorstore_info():
    """
    Utility function to get information about the current vector store.
    Useful for debugging and monitoring.

    Returns:
        dict: Information about the vector store
    """
    try:
        # Try to load with a dummy embedding model to check if store exists
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        dummy_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))

        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, dummy_embeddings, allow_dangerous_deserialization=True)

        return {"exists": True, "path": VECTOR_STORE_PATH, "total_documents": vectorstore.index.ntotal, "index_type": type(vectorstore.index).__name__}
    except Exception as e:
        return {"exists": False, "path": VECTOR_STORE_PATH, "error": str(e)}


# Optional: Migration helper function
def verify_vectorstore_compatibility(embedding_model):
    """
    Verify that the vector store is compatible with the current embedding model.
    This is useful when switching between different embedding providers.

    Args:
        embedding_model: The embedding model to test compatibility with

    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        vectorstore = load_faiss_vectorstore(embedding_model)
        # Try a simple similarity search to verify compatibility
        test_results = vectorstore.similarity_search("test query", k=1)
        return True
    except Exception as e:
        logger.warning(f"Vector store compatibility check failed: {e}")
        return False
