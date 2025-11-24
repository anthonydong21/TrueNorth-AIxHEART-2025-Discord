import time
import os
from datetime import datetime
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from truenorth.utils.llm import call_llm
from truenorth.agent.state import show_agent_reasoning
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.metaprompt import goals_as_str, system_relevant_scope

logger = get_caller_logger()

# Get API base URL from environment, default to production
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.mytruenorth.app")


class CitedSource(BaseModel):
    source_id: int = Field(description="The source ID number used in the text (e.g., 1 for [1])")
    quote: str = Field(description="The exact relevant quote from the source text that supports the statement.")


class AnswerResponse(BaseModel):
    answer: str = Field(description="The natural language answer to the user's question, including [ID] citations.")
    citations: Optional[List[CitedSource]] = Field(default_factory=list, description="List of sources cited in the answer with the relevant quotes.")


def create_citation_context(documents):
    """
    Creates a context string with numbered sources for the LLM to reference.

    Args:
        documents: List of Document objects with metadata

    Returns:
        tuple: (source_summary, formatted_context, references_dict)
    """
    if not documents:
        return "", "", {}

    sources_summary = []
    formatted_context = []
    references_dict = {}

    # Use existing citation numbers if available, otherwise assign them
    # document_relevence_checker assigns citation_num, so we should respect it
    # But we need to ensure they are sorted

    # Sort documents by citation_num if present, otherwise by index
    docs_with_num = [d for d in documents if d.metadata.get("citation_num")]
    docs_without_num = [d for d in documents if not d.metadata.get("citation_num")]

    sorted_docs = sorted(docs_with_num, key=lambda x: x.metadata.get("citation_num")) + docs_without_num

    # If no numbers were assigned (e.g. bypassed checker), assign 1..N
    current_num = 1

    for doc in sorted_docs:
        metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
        page_content = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")

        # Clean page content
        page_content = page_content.strip()

        # Helper to safely get and clean metadata
        def get_clean_meta(key, default=""):
            val = metadata.get(key)
            if val is None:
                return default
            # Remove BOM and strip whitespace
            return str(val).replace("\ufeff", "").strip() or default

        # Use assigned citation number or fallback
        source_num = metadata.get("citation_num") or current_num
        if not metadata.get("citation_num"):
            metadata["citation_num"] = source_num
            current_num += 1

        # Extract metadata with fallbacks
        author = get_clean_meta("author", "Unknown Author")
        # Remove BibTeX curly brackets from author names
        if author.startswith("{") and author.endswith("}"):
            author = author[1:-1]
        title = get_clean_meta("title", "Unknown Title")
        year = get_clean_meta("year") or (get_clean_meta("creationdate", "n.d.")[:4] if get_clean_meta("creationdate", "") else "n.d.")
        file_path = get_clean_meta("file_path", get_clean_meta("source", ""))
        page = get_clean_meta("page", get_clean_meta("page_number", get_clean_meta("page_num", "")))
        url = get_clean_meta("url", "")

        # Determine if it's a web source or file source
        is_web = "url" in metadata or url

        citation_display = ""

        if is_web:
            if title == "Unknown Title":
                citation_display = url
            else:
                citation_display = title

            citation_link = f"[{title}]({url})" if title != "Unknown Title" else f"{url}"
        else:
            # Book/PDF source
            citation_display = f"{author} ({year}) - {title}"
            if page:
                citation_display += f", p. {page}"

            filename = file_path.split("/")[-1] if file_path else ""
            api_url = f"{API_BASE_URL}/api/pdf/{filename}/pages?page={page}"
            citation_link = f"{author} ({year}). [{title}]({api_url})"

            sources_summary.append(f"[{source_num}] {citation_display}")
            references_dict[source_num] = citation_link

        # Add to formatted context for the LLM
        formatted_context.append(f"Source [{source_num}]:\nMetadata: {citation_display}\nContent: {page_content}\n")

    return "\n".join(sources_summary), "\n---\n".join(formatted_context), references_dict


def format_references_from_dict(references_dict):
    """
    Formats the references dictionary into a clean references section.
    """
    if not references_dict:
        return ""

    references_text = "\n**References**\n\n"
    for num in sorted(references_dict.keys()):
        references_text += f"*   [{num}] {references_dict[num]}\n"

    return references_text


# Define the structured prompt template for answer generation
answer_generator_prompt_template = PromptTemplate.from_template(
    """
Today is {current_datetime}.
                                                                
You are an assistant for question-answering tasks.

Here are your goals:
{goals_as_str}

You do not replace a therapist, legal counsel, or HR department, but you can provide emotional support, educational context, helpful language, and confidential documentation tools.

**Available Sources for Citation:**
{source_context}

Use the above numbered sources to support your answer. When referencing information, use the format [ID] (e.g., [1], [2]) in the text.

---

**Background Knowledge**:
Use the following background information to help answer the question:
{context}

**User Question**:
{question}
                                                                
---
                                                                
**Instructions**:
1. Base your answer primarily on the background knowledge provided above.
2. Use numbered citations in the text when referencing specific information (e.g., [1], [2]).
3. If the answer is **not present** in the knowledge, say so explicitly.
4. Be **comprehensive**, **accurate**, and **focused**. Do not give short, generic answers.
5. **Explain acronyms** (like PERMA+4) if you use them. Do not just list them; explain what they mean and how to apply them.
6. Provide **concrete, actionable advice** that an individual can use to improve their situation, not an organizational solution.
7. Only answer questions relevant to STEM, workplace support, or academic guidance.
8. **CRITICAL**: You must extract the exact quote from the source text that supports each citation you use. This will be used to display "inspirational" citation cards.
9. Return your response in the specified structured format, including the answer text and a list of citations with their quotes.

---
**Important**:
- Never invent or guess answers using general world knowledge.  
- Your role is to **maintain trust** and offer emotionally supportive, mission-aligned responses.
- Always keep a lighthearted but **thorough** manner of speaking while providing a helpful answer to the question.
"""
)


# ------------------------ Answer Generator Node ------------------------
def answer_generator(state):
    """
    Generates an answer based on the retrieved documents and user question.
    """
    logger.info("\n---ANSWER GENERATION---")

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    documents = state.documents

    # Use original_question if available
    if state.original_question:
        question = state.original_question
    else:
        question = state.question

    # Ensure all documents are LangChain Document objects
    documents = [Document(metadata=doc["metadata"], page_content=doc["page_content"]) if isinstance(doc, dict) else doc for doc in documents]

    # Create citation context
    source_context, formatted_context, references_dict = create_citation_context(documents)

    # Format the prompt
    prompt = answer_generator_prompt_template.format(current_datetime=current_datetime, context=formatted_context, question=question, goals_as_str=goals_as_str, source_context=source_context)

    logger.info(f"Answer generator prompt: {prompt}")

    # Force use of Gemini 2.5 Pro for answer generation
    # While preserving other state metadata
    answer_model_name = "gemini-2.5-pro"

    # Call LLM with structured output using Gemini 2.5
    response_obj = call_llm(prompt=prompt, model_name=answer_model_name, model_provider="Gemini", pydantic_model=AnswerResponse, agent_name="answer_generator_agent")  # Always Gemini for this model

    show_agent_reasoning(response_obj, f"Answer Generator Response | " + answer_model_name)

    # Extract components
    if isinstance(response_obj, AnswerResponse):
        generation_text = response_obj.answer
        structured_citations = response_obj.citations or []
    else:
        # Fallback if structured output failed
        generation_text = str(response_obj.content) if hasattr(response_obj, "content") else str(response_obj)
        structured_citations = []

    # Store the clean answer (without references section) for the web frontend
    state.metadata["clean_answer"] = generation_text

    # Append references section text for backward compatibility / Discord bot
    # We construct this using the STRUCTURED citations to ensure quotes match

    # Filter references_dict to only include used citations
    used_ids = set()
    structured_map = {c.source_id: c.quote for c in structured_citations}

    for cit in structured_citations:
        used_ids.add(cit.source_id)

    # Also check text for [N] markers as a backup
    import re

    text_ids = set(int(x) for x in re.findall(r"\[(\d+)\]", generation_text))
    used_ids.update(text_ids)

    # Build references section manually to include the correct quotes
    references_text_parts = ["\n**References**\n"]
    for num in sorted(list(used_ids)):
        ref_link = references_dict.get(num, "")
        if ref_link:
            quote = structured_map.get(num, "")
            # Format: * [1] Author...
            #         > "Quote"
            entry = f"*   [{num}] {ref_link}"
            if quote:
                entry += f"\n    > '{quote}'"
            references_text_parts.append(entry)

    references_section = "\n".join(references_text_parts)

    final_text = generation_text + "\n" + references_section

    # Update state
    # We store the structured citations in metadata so app.py can use them
    state.generation = final_text
    state.metadata["references_dict"] = references_dict
    state.metadata["structured_citations"] = [c.dict() for c in structured_citations]

    # Add message to history
    from langchain_core.messages import AIMessage

    state.messages.append(AIMessage(content=final_text))

    logger.info(f"Current state: {state}")
    logger.info(f"Response with integrated references: {state.generation}")

    return state
