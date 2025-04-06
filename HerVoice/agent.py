# Environment Configuration
from dotenv import load_dotenv  
import os
from sqlalchemy.engine.url import make_url # Used to parse and construct database URLs
from langchain_postgres.vectorstores import PGVector # Integration with Postgres + pgvector for vector storage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LLM and Core LangChain Tools
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.load import dumps, loads  # Serialize/deserialize LangChain objects
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.documents import Document # # Standard document format used in LangChain pipelines

from typing_extensions import TypedDict # Define structured types for state management
from typing import List  # Specify types for list inputs or outputs
import asyncio # Support asynchronous execution for parallel LLM calls
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END # LangGraph tools to define stateful workflows 

# Visualization and Display Utilities
import textwrap
from IPython.display import Markdown, Image
from pprint import pprint

# Web Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults

google_api_key = os.getenv("GEMINI_API_KEY")
connection_string = os.getenv("DB_CONNECTION")
tavily_api_key =os.getenv("TAVILY_API_KEY")

# embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY"),  # make sure it's in your .env
    task_type="semantic_similarity"
)

# Initialize the Google Gemini-2.0-Flash model
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_retries=2,
    google_api_key= google_api_key
)

hervoice_vectorstore = PGVector(
    embeddings = embedding_model,
    collection_name="Books",  # Unified collection name
    connection=connection_string,
    use_jsonb=True,
)

from langchain.prompts import PromptTemplate

# Define the routing prompt for HerVoice
query_router_prompt_template = PromptTemplate.from_template("""
You are an expert at analyzing user questions and deciding which data source is best suited to answer them. You must choose **one** of the following options:

1. **Vectorstore**: Use this if the question can be answered by the **existing** content in the vectorstore. 
   The vectorstore contains information about **{vectorstore_content_summary}**.
                                                            
---                                                         

2. **Websearch**: Use this if the question is **within scope** (see below) but meets **any** of the following criteria:
    - The answer **cannot** be found in the local vectorstore
    - The question requires **more detailed or factual information** than what's available in HerVoice’s internal documents
    - The topic is **time-sensitive**, **news-based**, or depends on recent events or external sources    

---                                                         

3. **Chitter-Chatter**: Use this if the question:
   - Is **not related** to the scope below, or
   - Is too **broad, casual, or off-topic** to be answered using vectorstore or websearch.
   
   Chitter-Chatter is a fallback agent that gives a warm, supportive response and gently guides users back to relevant STEM-related topics.

---

Scope Definition:
Relevant questions are those related to **women in STEM** including academic support, career development, mentorship, scholarship opportunities, workplace equity, and challenges faced by women in technical fields.

---

Your Task:
Analyze the user's question. Return a JSON object with one key `"Datasource"` and one value: `"Vectorstore"`, `"Websearch"`, or `"Chitter-Chatter"`.

""")

# Define a summary of what's in the vectorstore
vectorstore_content_summary = """
HerVoice’s internal documents including resources on women in STEM, mentorship guidance, scholarship databases, 
career navigation strategies, overcoming workplace bias, empowerment programs, anonymous discussion forums, 
and tools for building confidence and equity in technical fields.
"""

# Define the topical scope of the system
relevant_scope = """STEM-related support for women, including career and education guidance, 
mentorship, leadership challenges, equity in the workplace, and related professional development resources."""

# Format the prompt using the content summary and scope
query_router_prompt = query_router_prompt_template.format(
    relevant_scope = relevant_scope,
    vectorstore_content_summary = vectorstore_content_summary
)


# Define the multi-query generation prompt
multi_query_generation_prompt = PromptTemplate.from_template("""
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
""")

# HerVoice-specific vectorstore content summary
vectorstore_content_summary = """
The vectorstore contains curated resources for women in STEM, including mentorship guides, leadership advice, 
career development materials, scholarship and opportunity listings, workplace bias, and anonymous experiences shared by women 
in technology, science, and academic fields. These documents are designed to support and empower women across all stages of their STEM journey.
"""

# Number of alternative queries to generate
num_queries = 3

# Create the query generation pipeline
multi_query_generator = (
    multi_query_generation_prompt.partial(
        vectorstore_content_summary=vectorstore_content_summary,
        num_queries=num_queries
    )
    | llm_gemini  # Replace with your Gemini model
    | StrOutputParser()
    | (lambda x: x.split("\n"))  # Split into a list of queries
)

# === Reciprocal Rank Fusion (RRF) Implementation for HerVoice Chatbot ===

def reciprocal_rank_fusion(results, k=60):
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion (RRF).

    Parameters:
    - results: List of ranked lists of LangChain Document objects
    - k: Constant that reduces the impact of lower-ranked documents

    Returns:
    - List of re-ranked LangChain Document objects with "rrf_score" in metadata
    """

    fused_scores = {}  # Store cumulative RRF scores per document

    for docs in results:
        for i, doc in enumerate(docs):
            doc_str = dumps(doc)  # Convert Document to JSON string to use as a key

            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            # RRF formula: 1 / (rank + k), with rank starting from 1
            rank = i + 1
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort documents by total fused RRF score (descending)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # Convert back to Document objects with RRF score in metadata
    reranked_documents = []
    for doc_str, score in reranked_results:
        doc = loads(doc_str)
        doc.metadata["rrf_score"] = score
        reranked_documents.append(doc)

    return reranked_documents


# === Define a retrieval chain for Multi-Query RAG Fusion with MMR (for HerVoice Chatbot) ===

retrieval_chain_rag_fusion_mmr = (
    multi_query_generator  # Reformulate the user query in multiple ways
    | hervoice_vectorstore.as_retriever(
        search_type="mmr",  # Use Max Marginal Relevance for diverse retrieval
        search_kwargs={
            'k': 3,              # Final top results to return per query
            'fetch_k': 15,       # Total candidate documents to consider
            "lambda_mult": 0.5   # Balance between relevance (0) and diversity (1)
        }
    ).map()  # Apply MMR retrieval to each reformulated query
    | reciprocal_rank_fusion  # Rerank all combined results using Reciprocal Rank Fusion
)

# === Define the Relevance Grader Prompt for HerVoice Chatbot ===

relevance_grader_prompt_template = PromptTemplate.from_template("""
You are a relevance grader evaluating whether a retrieved document is helpful in answering a user question related to supporting women in STEM.

---

**Retrieved Document**: 
{document}

**User Question**: 
{question}

---

**Your Task**:                                                          
Carefully and objectively assess whether the document contains any **keyword overlap**, **emotional support context**, or **semantic meaning** that is relevant to the question.
You do not need a complete answer—some partial relevance or useful guidance is enough to pass.

Return your decision as a JSON object with one key: `"binary_score"`.  
The `"binary_score"` should be either `"pass"` or `"fail"` to indicate relevance.
""")


# === Define the Prompt Template for Answer Generation (HerVoice Chatbot) ===
answer_generator_prompt_template = PromptTemplate.from_template("""
You are an AI assistant supporting women in STEM by answering their questions based on guidance documents and real experiences.

---

**Context**:
Use the following information extracted from the HerVoice knowledge base to answer the question:
{context}

**User Question**:
{question}
                                                                
---
                                                                
**Instructions**:
1. Base your answer strictly on the context provided above.
2. If the answer is **not found** in the context, clearly say: 
   "The current HerVoice knowledge base does not contain this information."
3. Keep the answer **concise**, **supportive**, and **directly relevant** to the question.
4. At the end, include a **reference section** (if possible):
    - Mention the **resource name**, **topic**, or **page number** (if available).

---

**Answer**:
""")


# === Define the Hallucination Checker Prompt for HerVoice ===
hallucination_checker_prompt_template = PromptTemplate.from_template("""
You are an AI grader evaluating whether an AI-generated answer is factually grounded in the provided HerVoice resource documents.

---

**Grading Criteria**:
- **Pass**: The answer is clearly based on the provided HerVoice content and does not contain made-up or inaccurate claims.
- **Fail**: The answer includes fabricated, incorrect, or unsupported information that is not reflected in the source materials.

---

**Reference Materials** (from HerVoice Knowledge Base):
{documents}

**AI-Generated Answer**:
{generation}

---

**Output Instructions**:
Return a JSON object with two keys: 
- `"binary_score"`: either `"pass"` or `"fail"` 
- `"explanation"`: a brief justification explaining why the answer passed or failed the factuality check.
""")

from langchain.prompts import PromptTemplate

# === Define the Answer Verifier Prompt for HerVoice Chatbot ===

answer_verifier_prompt_template = PromptTemplate.from_template("""
You are an AI grader verifying whether an AI-generated answer appropriately addresses the user's question based on guidance and support topics 
for women in STEM.

---

**Grading Criteria**:
- **Pass**: The answer clearly and directly responds to the user’s question. Providing supportive or related context is acceptable.
- **Fail**: The answer is off-topic, vague, overly generic, or does not meaningfully address the intent of the question.

---

**User Question**: 
{question}

**AI-Generated Answer**: 
{generation}

---

**Output Instructions**:
Return a JSON object with the following keys:
- "binary_score": either `"pass"` or `"fail"` 
- "explanation": a short justification explaining your grading decision
""")



# === Define the Query Rewriter Prompt for HerVoice Chatbot ===

query_rewriter_prompt_template = PromptTemplate.from_template("""
You are a query optimization expert tasked with rewriting user questions to improve document retrieval accuracy 
from the HerVoice vector knowledge base.

---

**Context**:
- Original Question: {question}
- Previous Answer (incomplete or unhelpful): {generation}

**Vectorstore Summary**:
{vectorstore_content_summary}

Note: The summary describes the general content of the HerVoice knowledge base, but it is not exhaustive.

---

**Your Task**:
Analyze the original question and the weak answer to determine:
1. What essential information was missing from the original query
2. Any ambiguous language or unclear intent
3. Specific topics (e.g., mentorship, scholarships, workplace bias, leadership, mental health) that should have been included
4. More effective keywords or rephrased structure to better match the knowledge base content

---

**Output Format**:
Return a JSON object with the following keys:
- "rewritten_question": A revised version of the question optimized for HerVoice document search
- "explanation": A short explanation describing how your rewrite improves clarity, relevance, or specificity
""")


# === Define the Chitter-Chatter Prompt for HerVoice Chatbot ===

chitterchatter_prompt_template = PromptTemplate.from_template("""
You are a friendly assistant designed to keep conversations within the current scope of the HerVoice knowledge base, 
while maintaining a warm, supportive tone that empowers women in STEM.

---

**Current Scope**:
{relevant_scope}

Your job is to respond conversationally while gently guiding the user toward meaningful, empowering, and relevant discussions 
based on the resources in the HerVoice knowledge base.

---

**Response Guidelines**:

1. **Casual Chit-Chat**:
  - Respond warmly to greetings or casual exchanges.
  - Keep the tone encouraging and human-like.
  - Be an empathetic listener if the user opens up.

2. **Off-Topic Questions**:
  - Politely acknowledge the question.
  - Mention that it falls outside your current scope.
  - Redirect to a related topic such as mentorship, leadership challenges, scholarships, navigating bias, or career growth in STEM.
  - Avoid saying "I don't know" without offering supportive redirection.

3. **In-Scope but Unanswerable Questions**:
  - If the question fits the mission but lacks enough detail to answer confidently:
    - Acknowledge the gap without guessing.
    - Gently ask for clarification or guide the user to rephrase the question.

---

**Important**:
Never invent or guess answers using general world knowledge.  
Your role is to **maintain trust** and offer emotionally supportive, mission-aligned responses.

Always end with a helpful suggestion, encouraging message, or thoughtful follow-up related to women in STEM.
""")



web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",        # Uses advanced search depth for more accurate results
    include_answer=True,            # Include a short answer to original query in the search results.
    tavily_api_key= tavily_api_key  # You have defined this API key in the .env file.
)


# === Define the state structure used across the LangGraph in HerVoice ===

class GraphState(TypedDict):
    question: str                         # Reformatted or current working version of the user query
    original_question: str                # Raw user input as originally received
    generation: str                       # Final Gemini-generated answer
    datasource: str                       # Selected route: "Vectorstore", "Websearch", or "Chitter-Chatter"
    hallucination_checker_attempts: int   # Number of hallucination checks performed
    answer_verifier_attempts: int         # Number of times we verified relevance/clarity
    documents: List[str]                  # Chunks retrieved from vectorstore
    checker_result: str                   # Final score (e.g., "pass" or "fail") from the factuality/relevance check

# ------------------------ Document Retriever Node ------------------------
def document_retriever(state):
    print("\n---QUERY TRANSLATION AND RAG-FUSION---")

    question = state["question"]

    rag_fusion_mmr_results = retrieval_chain_rag_fusion_mmr.invoke({
        "question": question,
        "num_queries": 3,
        "vectorstore_content_summary": vectorstore_content_summary
    })

    print(f"Total number of results: {len(rag_fusion_mmr_results)}")
    for i, doc in enumerate(rag_fusion_mmr_results, start=1):
        print(f"     Document {i} from `{doc.metadata.get('source', 'unknown')}`")

    formatted_doc_results = [
        Document(
            metadata={k: v for k, v in doc.metadata.items() if k != 'rrf_score'},
            page_content=doc.page_content
        ) 
        for doc in rag_fusion_mmr_results
    ]

    return {"documents": formatted_doc_results}


# ------------------------ Answer Generator Node ------------------------
def answer_generator(state):
    print("\n---ANSWER GENERATION---")

    documents = state["documents"]
    original_question = state.get("original_question", None)
    question = original_question if original_question else state["question"]

    documents = [
        Document(metadata=doc["metadata"], page_content=doc["page_content"])
        if isinstance(doc, dict) else doc
        for doc in documents
    ]

    answer_generator_prompt = answer_generator_prompt_template.format(
        context=documents,
        question=question
    )

    answer_generation = llm_gemini.invoke(answer_generator_prompt)
    print("Answer generation has been generated.")

    return {"generation": answer_generation.text}


# ------------------------ Web Searcher Node ------------------------
def web_search(state):
    print("\n---WEB SEARCH---")

    question = state["question"]
    documents = state.get("documents", [])

    web_results = web_search_tool(question)

    formatted_web_results = [
        Document(
            metadata={"source": "web", "url": "https://example.com"},
            page_content=web_results
        )
    ]

    documents = [
        Document(metadata=doc["metadata"], page_content=doc["page_content"])
        if isinstance(doc, dict) else doc
        for doc in documents
    ]

    documents.extend(formatted_web_results)
    print(f"Total number of web search documents: {len(formatted_web_results)}")
    return {"documents": documents}


# ------------------------ Chitter-Chatter Node ------------------------
def chitter_chatter(state):
    print("\n---CHIT-CHATTING---")
    question = state["question"]

    chitterchatter_response = llm_gemini.invoke(
        [
            SystemMessage(str(chitterchatter_prompt_template)),
            HumanMessage(question)
        ]
    )

    return {"generation": chitterchatter_response.text}


# ------------------------ Adaptive Query Rewrite Node ------------------------
def query_rewriter(state):
    print("\n---QUERY REWRITE---")

    original_question = state.get("original_question", None)
    question = original_question if original_question else state["question"]
    generation = state["generation"]

    query_rewriter_prompt = query_rewriter_prompt_template.format(
        question=question,
        generation=generation,
        vectorstore_content_summary=vectorstore_content_summary
    )

    query_rewriter_result = llm_gemini.with_structured_output(method="json_mode").invoke(query_rewriter_prompt)

    return {
        "question": query_rewriter_result['rewritten_question'],
        "original_question": question
    }


# ------------------------ Retry Counter Nodes ------------------------
def hallucination_checker_tracker(state):
    num_attempts = state.get("hallucination_checker_attempts", 0)
    return {"hallucination_checker_attempts": num_attempts + 1}


def answer_verifier_tracker(state):
    num_attempts = state.get("answer_verifier_attempts", 0)
    return {"answer_verifier_attempts": num_attempts + 1}
from pydantic import BaseModel
class RouteResponse(BaseModel):
    Datasource: str
# ------------------------ Routing Decision ------------------------
def route_question(state):
    print("---ROUTING QUESTION---")
    question = state["question"]
    route_question_response = llm_gemini.with_structured_output(schema=RouteResponse).invoke(
        [SystemMessage(query_router_prompt), HumanMessage(question)]
    )

    print(f"Response: {route_question_response}")
    print(f"Response type: {type(route_question_response)}")
    parsed_router_output = route_question_response.Datasource

    if parsed_router_output == "Websearch":
        print("---ROUTING QUESTION TO WEB SEARCH---")
        return "Websearch"
    elif parsed_router_output == "Vectorstore":
        print("---ROUTING QUESTION TO VECTORSTORE---")
        return "Vectorstore"
    elif parsed_router_output == "Chitter-Chatter":
        print("---ROUTING QUESTION TO CHITTER-CHATTER---")
        return "Chitter-Chatter"


# ------------------------ Async document relevance grading ------------------------
import asyncio

async def grade_documents_parallel(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    async def grade_document(doc, question):
        relevance_grader_prompt = relevance_grader_prompt_template.format(
            document=doc,
            question=question
        )
        grader_result = await llm_gemini.with_structured_output(method="json_mode").ainvoke(relevance_grader_prompt)
        return grader_result

    tasks = [grade_document(doc, question) for doc in documents]
    results = await asyncio.gather(*tasks)

    filtered_docs = []
    for i, score in enumerate(results):
        if score["binary_score"].lower() == "pass":
            print(f"---GRADE: DOCUMENT RELEVANT--- {score['binary_score']}")
            filtered_docs.append(documents[i])
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    total_docs = len(documents)
    relevant_docs = len(filtered_docs)

    if total_docs > 0:
        filtered_out_percentage = (total_docs - relevant_docs) / total_docs
        checker_result = "fail" if filtered_out_percentage >= 0.5 else "pass"
        print(f"---FILTERED OUT {filtered_out_percentage*100:.1f}% OF IRRELEVANT DOCUMENTS---")
        print(f"---**{checker_result}**---")
    else:
        checker_result = "fail"
        print("---NO DOCUMENTS AVAILABLE, WEB SEARCH TRIGGERED---")

    return {"documents": filtered_docs, "checker_result": checker_result}


# ------------------------ Decide whether to generate or fallback ------------------------
def decide_to_generate(state):
    print("---CHECK GENERATION CONDITION---")
    checker_result = state["checker_result"]

    if checker_result == "fail":
        print("---DECISION: DOCUMENTS INSUFFICIENT, USE WEB SEARCH---")
        return "Websearch"
    else:
        print("---DECISION: DOCUMENTS SUFFICIENT, PROCEED TO GENERATE---")
        return "generate"


# ------------------------ Final Answer Validation ------------------------
def check_generation_vs_documents_and_question(state):
    print("---CHECK HALLUCINATIONS WITH DOCUMENTS---")

    question = state.get("original_question") or state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_checker_attempts = state.get("hallucination_checker_attempts", 0)
    answer_verifier_attempts = state.get("answer_verifier_attempts", 0)

    hallucination_checker_prompt = hallucination_checker_prompt_template.format(
        documents=documents,
        generation=generation
    )

    hallucination_checker_result = llm_gemini.with_structured_output(method="json_mode").invoke(hallucination_checker_prompt)

    def ordinal(n):
        return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')}"

    if hallucination_checker_result['binary_score'].lower() == "pass":
        print("---DECISION: ANSWER IS GROUNDED---")

        print("---VERIFYING RELEVANCE TO QUESTION---")
        answer_verifier_prompt = answer_verifier_prompt_template.format(
            question=question,
            generation=generation
        )
        answer_verifier_result = llm_gemini.with_structured_output(method="json_mode").invoke(answer_verifier_prompt)

        if answer_verifier_result['binary_score'].lower() == "pass":
            print("---DECISION: ANSWER IS RELEVANT---")
            return "useful"
        elif answer_verifier_attempts > 1:
            print("---DECISION: MAX RETRIES REACHED FOR RELEVANCE---")
            return "max retries"
        else:
            print("---DECISION: ANSWER NOT RELEVANT, TRY QUERY REWRITE---")
            print(f"This is the {ordinal(answer_verifier_attempts + 1)} attempt.")
            return "not useful"

    elif hallucination_checker_attempts > 1:
        print("---DECISION: MAX RETRIES REACHED FOR HALLUCINATION---")
        return "max retries"
    else:
        print("---DECISION: ANSWER IS NOT GROUNDED, RETRY---")
        print(f"This is the {ordinal(hallucination_checker_attempts + 1)} attempt.")
        return "not supported"


# ------------------------ Verify Answer Usefulness ------------------------
def verify_answer_usefulness(state):
    print("\n---VERIFYING GENERAL ANSWER USEFULNESS---")

    question = state["question"]
    generation = state["generation"]

    answer_verifier_prompt = answer_verifier_prompt_template.format(
        question=question,
        generation=generation
    )

    answer_verifier_result = llm_gemini.with_structured_output(method="json_mode").invoke(
        answer_verifier_prompt
    )

    print(f"Usefulness Score: {answer_verifier_result['binary_score']}")
    print(f"Explanation: {answer_verifier_result['explanation']}")

    return {
        "answer_usefulness_score": answer_verifier_result["binary_score"],
        "answer_usefulness_explanation": answer_verifier_result["explanation"]
    }

# === Initialize Graph ===
workflow = StateGraph(GraphState)

# === Add Agent Nodes ===
workflow.add_node("WebSearcher", web_search)
workflow.add_node("DocumentRetriever", document_retriever)
workflow.add_node("RelevanceGrader", grade_documents_parallel)
workflow.add_node("AnswerGenerator", answer_generator)
workflow.add_node("QueryRewriter", query_rewriter)
workflow.add_node("ChitterChatter", chitter_chatter)
workflow.add_node("AnswerUsefulnessVerifier", verify_answer_usefulness)

# === Retry Tracker Nodes ===
workflow.add_node("HallucinationCheckerFailed", hallucination_checker_tracker)
workflow.add_node("AnswerVerifierFailed", answer_verifier_tracker)

# === Entry Point Routing ===
workflow.set_conditional_entry_point(
    route_question,
    {
        "Websearch": "WebSearcher",
        "Vectorstore": "DocumentRetriever",
        "Chitter-Chatter": "ChitterChatter",
    },
)

# === Node Transitions ===
workflow.add_edge("DocumentRetriever", "RelevanceGrader")
workflow.add_edge("WebSearcher", "AnswerGenerator")
workflow.add_edge("AnswerGenerator", "AnswerUsefulnessVerifier")
workflow.add_edge("HallucinationCheckerFailed", "AnswerGenerator")
workflow.add_edge("AnswerVerifierFailed", "QueryRewriter")
workflow.add_edge("QueryRewriter", "DocumentRetriever")
workflow.add_edge("ChitterChatter", END)

# === Conditional Routing After Document Relevance Grading ===
workflow.add_conditional_edges(
    "RelevanceGrader",
    decide_to_generate,
    {
        "Websearch": "WebSearcher",
        "generate": "AnswerGenerator",
    },
)

workflow.add_conditional_edges(
    "AnswerUsefulnessVerifier",
    check_generation_vs_documents_and_question,
    {
        "not supported": "HallucinationCheckerFailed",
        "useful": END,
        "not useful": "AnswerVerifierFailed",
        "max retries": "ChitterChatter"
    },
)

# === Compile the Graph ===
graph = workflow.compile()



