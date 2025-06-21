import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document  # Standard document format used in LangChain pipelines
from langchain.load import dumps, loads  # Serialize/deserialize LangChain objects
from langchain_postgres.vectorstores import PGVector  # Integration with Postgres + pgvector for vector storage

from sqlalchemy.engine.url import make_url  # Used to parse and construct database URLs

from dotenv import load_dotenv

from .state import ChatState
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.llm import call_llm, get_embedding_model
from truenorth.utils.metaprompt import vectorstore_content_summary

load_dotenv()
logger = get_caller_logger()

connection_string = os.getenv("DB_CONNECTION")

shared_connection_string = make_url(connection_string).render_as_string(hide_password=False)

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
        state (GraphState): The current state of the LangGraph, containing the user's question.

    Returns:
        dict: A dictionary containing a cleaned list of relevant `Document` objects under the key `"documents"`.
    """
    logger.info("\n---QUERY TRANSLATION AND RAG-FUSION---")
    embedding_model = get_embedding_model(model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"])

    # Connect to the PGVector Vector Store that contains book data.
    book_data_vector_store = PGVector(
        embeddings=embedding_model,
        collection_name="Books",  # Name of the collection/table in the vector DB
        connection=shared_connection_string,  # Use shared DB connection from earlier
        use_jsonb=True,
    )

    logger.info(f"State Information: {state}")
    llm = lambda prompt: call_llm(prompt=prompt, model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"], pydantic_model=None, agent_name="document_retriever", verbose=False)

    question = state.question

    multi_query_generator = (
        multi_query_generation_prompt
        | llm
        | (lambda x: [line.strip() for line in str(x.content).split("\n") if line.strip()])
    )
    retrieval_chain_rag_fusion_mmr = (
        multi_query_generator
        | book_data_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 15, "lambda_mult": 0.5}).map()  # Use MMR retrieval to enhance diversity in retrieved documents  # Final number of documents to return per query  # Initial candidate pool (larger for better diversity)  # Balances relevance (0) and diversity (1)  # Apply MMR retrieval to each reformulated query
        | reciprocal_rank_fusion  # Rerank the combined results using RRF
    )

    # Run multi-query RAG + MMR + RRF pipeline to get relevant results
    rag_fusion_mmr_results = retrieval_chain_rag_fusion_mmr.invoke({"question": question, "num_queries": 3, "vectorstore_content_summary": vectorstore_content_summary})

    # Display summary of where results came from (for teaching purposes)
    logger.info(f"Total number of results: {len(rag_fusion_mmr_results)}")
    for i, doc in enumerate(rag_fusion_mmr_results, start=1):
        excerpt = doc.page_content[:200].replace("\n", " ") + "..."  # first 200 characters
        logger.info(f"     Document {i} from `{doc.metadata['source']}`, page {doc.metadata['page']}")
        logger.info(f"     Excerpt: {excerpt}")

    # Convert retrieved documents into Document objects with metadata and page_content only
    formatted_doc_results = [Document(metadata={k: v for k, v in doc.metadata.items() if k != "rrf_score"}, page_content=doc.page_content) for doc in rag_fusion_mmr_results]  # Remove rrf score and document id

    state.documents.extend(formatted_doc_results)

    return state
