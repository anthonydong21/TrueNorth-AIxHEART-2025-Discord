import asyncio
from langchain_core.prompts import PromptTemplate
from .state import ChatState
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.llm import call_llm
from pydantic import BaseModel, StrictBool

logger = get_caller_logger()


class RelevanceGrade(BaseModel):
    value: StrictBool


# Single document prompt
single_relevance_grader_prompt_template = PromptTemplate.from_template(
    """
You are a relevance grader evaluating whether a retrieved document is helpful in answering a user question.

---

**Retrieved Document**:
{document}

**User Question**:
{question}

---

**Instructions**:
- Use `true` if the document contains related or helpful information, even if partial.
- Use `false` only if completely unrelated.
- Do not include any extra text, explanation, or formatting — only `true` or `false`.
"""
)


async def check_relevance(state: ChatState) -> ChatState:
    logger.info("---CHECK DOCUMENT RELEVANCE (ONLY RAW TRUE/FALSE GRADING)---")

    if not state.documents:
        logger.info("---NO DOCUMENTS AVAILABLE, WEB SEARCH TRIGGERED---")
        state.metadata["relevance_score"] = "fail"
        return state

    question = state.question
    documents = state.documents
    model_name = state.metadata.get("model_name")
    model_provider = state.metadata.get("model_provider")

    async def grade_document(doc):
        prompt_text = single_relevance_grader_prompt_template.format(
            document=doc.page_content,
            question=question,
        )
        return call_llm(
            prompt=[prompt_text],
            model_name=model_name,
            model_provider=model_provider,
            pydantic_model=RelevanceGrade,
            agent_name="relevance_grader",
            max_retries=1,
            verbose=True,
        )

    grading_results = await asyncio.gather(*[grade_document(doc) for doc in documents])

    # Pretty logging
    logger.info("--- Document Grading Results ---")
    filtered_documents = []
    for idx, (doc, result) in enumerate(zip(documents, grading_results), start=1):
        mark = "✔️" if result.value else "❌"
        snippet = doc.page_content.strip().replace("\n", " ")[:100]
        logger.info(f'{mark} Document {idx}: {result.value} - "{snippet}..."')
        if result.value:
            filtered_documents.append(doc)

    # Determine checker result
    total_docs = len(documents)
    kept_docs = len(filtered_documents)
    filtered_out_pct = (total_docs - kept_docs) / total_docs if total_docs > 0 else 1.0
    checker_result = "pass" if filtered_out_pct < 0.5 else "fail"

    logger.info(f"Filtered out {filtered_out_pct:.1%}: {checker_result.upper()}")

    # Assign sequential citation numbers to filtered documents
    citation_map = {}
    for idx, doc in enumerate(filtered_documents, start=1):
        # Add citation number to document metadata
        doc.metadata["citation_num"] = idx
        
        # Build citation map for reference
        author = doc.metadata.get("author", "Unknown Author")
        title = doc.metadata.get("title", "Unknown Title")
        year = doc.metadata.get("year", "n.d.")
        page = doc.metadata.get("page", "")
        
        citation_key = f"{author} ({year}). {title}"
        if page:
            citation_key += f", p. {page}"
        
        citation_map[idx] = citation_key
    
    logger.info(f"--- Assigned {len(citation_map)} citation numbers ---")
    for num, citation in citation_map.items():
        logger.info(f"[{num}] {citation[:80]}...")

    # Update the state appropriately
    state.documents = filtered_documents
    state.metadata["relevance_score"] = checker_result
    state.metadata["citation_map"] = citation_map
    return state
