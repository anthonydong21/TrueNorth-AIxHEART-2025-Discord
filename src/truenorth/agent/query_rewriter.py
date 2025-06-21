# from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .state import ChatState
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.metaprompt import vectorstore_content_summary
from truenorth.utils.llm import call_llm

logger = get_caller_logger()

# Define the query rewriter prompt
query_rewriter_prompt_template = PromptTemplate.from_template(
    """
You are a query optimization expert tasked with rewriting questions to improve vector database retrieval accuracy.

---

**Context**:
- Original Question: {question}
- Previous Answer (incomplete or unhelpful): {generation}

**Vectorstore Summary**:
{vectorstore_content_summary}
                                                              
Note: The summary provides context about what's in the database but should not be treated as exhaustive.                                                       

---

**Your Task**:
Analyze the original question and the failed answer to identify:
1. What key information the original question was missing
2. Any ambiguities or unclear phrasing
3. Missing context or specialized terminology that should be included
4. Better keywords, phrasing, or terms to improve retrieval

---
                                                              
**Output Format**:
Return a JSON object with keys: "rewritten_question" and "explanation".
- "rewritten_question":  A refined version of the user's question optimized for vector search
- "explanation": A short explanation of how the rewrite improves coverage or clarity
"""
)


class QueryRewriteOutput(BaseModel):
    rewritten_question: str = Field(..., description="Refined version of the user's question optimized for vector search")
    explanation: str = Field(..., description="Short explanation of how the rewrite improves coverage or clarity")


def rewrite_query(state: ChatState) -> str:
    """
    Rewrites the original question if the answer was hallucinated or unhelpful.

    This node helps improve retrieval quality in the second attempt by:
    - Identifying gaps between the original query and the generated answer
    - Generating a clearer, more focused version of the question
    - Keeping a copy of the original for fallback comparison

    Args:
        state (GraphState): Contains the original and current question, and the LLM's previous answer.

    Returns:
        state: Updated state with the rewritten question and preserved original.
    """
    logger.info("Starting rewrite query agent...")

    logger.info("\n---QUERY REWRITE---")

    # Use original question if available, otherwise fall back to input
    original_question = state.original_question
    if original_question != 0:
        question = original_question
    else:
        question = state.question

    generation = state.generation

    # Create prompt and invoke the query rewriter
    query_rewriter_prompt = query_rewriter_prompt_template.format(question=question, generation=generation, vectorstore_content_summary=vectorstore_content_summary)

    response = call_llm(
        prompt=[query_rewriter_prompt],
        model_name=state.metadata.get("model_name"),
        model_provider=state.metadata.get("model_provider"),
        pydantic_model=QueryRewriteOutput,
        agent_name="query_rewriter",
        max_retries=1,
        verbose=True,
    )
    state.original_question = state.question
    state.question = response.rewritten_question

    return state
