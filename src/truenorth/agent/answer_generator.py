# answer-generator.py
import time
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from truenorth.utils.llm import call_llm
from truenorth.agent.state import show_agent_reasoning
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.metaprompt import goals_as_str, system_relevant_scope

logger = get_caller_logger()


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
    source_num = 1

    for doc in documents:
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

        # Extract metadata with fallbacks
        author = get_clean_meta("author", "Unknown Author")
        title = get_clean_meta("title", "Unknown Title")
        year = get_clean_meta("creationdate", "n.d.")[:4]
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

            # Create summary entry
            sources_summary.append(f"[{source_num}] {citation_display}")

            # Create full citation for references dict (internal use)
            citation_link = f"[{title}]({url})" if title != "Unknown Title" else f"{url}"
            references_dict[source_num] = f"{citation_link}"

        else:
            # Book/PDF source
            citation_display = f"{author} ({year}) - {title}"
            if page:
                citation_display += f", p. {page}"

            # Create summary entry
            sources_summary.append(f"[{source_num}] {citation_display}")

            # Create full citation for references dict (internal use)
            citation_link = f"{author} ({year}). [{title}](file://{file_path}#{page})"
            references_dict[source_num] = citation_link

        # Add to formatted context for the LLM
        # This explicitly links the Source ID to the Content
        formatted_context.append(f"Source [{source_num}]:\nMetadata: {citation_display}\nContent: {page_content}\n")

        source_num += 1

    return "\n".join(sources_summary), "\n---\n".join(formatted_context), references_dict


def format_references_from_dict(references_dict):
    """
    Formats the references dictionary into a clean references section.

    Args:
        references_dict: Dictionary mapping source numbers to citations

    Returns:
        str: Formatted references section
    """
    if not references_dict:
        return ""

    references_text = "\n**References**\n\n"
    for num in sorted(references_dict.keys()):
        references_text += f"*   [{num}] {references_dict[num]}\n"

    return references_text


# Define the enhanced prompt template for answer generation
answer_generator_prompt_template = PromptTemplate.from_template(
    """
Today is {current_datetime}.
                                                                
You are an assistant for question-answering tasks.

Here are your goals:
{goals_as_str}

You do not replace a therapist, legal counsel, or HR department, but you can provide emotional support, educational context, helpful language, and confidential documentation tools.

**Available Sources for Citation:**
{source_context}

Use the above numbered sources to support your answer. When referencing information, use the format [1], [2], etc. corresponding to the source numbers above.

---

**Background Knowledge**:
Use the following background information to help answer the question:
{context}

**User Question**:
{question}
                                                                
---
                                                                
**Instructions**:
1. Base your answer primarily on the background knowledge provided above.
2. Use numbered citations when referencing specific information (e.g., [1], [2]).
3. If the answer is **not present** in the knowledge, say so explicitly.
4. Keep the answer **concise**, **accurate**, and **focused** on the question.
5. End your response with a **References** section that includes:
   - Full citations with clickable links to the source
   - Meaningful quotes from the page content that support your answer
   - For books with multiple relevant sections, show different page links and quotes
6. Format references as:
   - Single section: `*   [i] Author (Year). [Title](file://path#page)\n    > "Meaningful quote from content"`
   - Multiple sections: `*   [i] Author (Year). *Title*\n    - [Page X](file://path#X): "Quote from page X"\n    - [Page Y](file://path#Y): "Quote from page Y"`
   - Web sources: `*   [i] [Title](URL)\n    > "Meaningful quote from content"`
7. Only answer questions relevant to STEM, workplace support, or academic guidance. For all other queries, politely decline.

---
**Important**:
Never invent or guess answers using general world knowledge.  
Your role is to **maintain trust** and offer emotionally supportive, mission-aligned responses.

Always keep a lighthearted, concise manner of speaking while providing a helpful answer to the question.
                                                                
**Answer**:
"""
)


# ------------------------ Answer Generator Node ------------------------
def answer_generator(state):
    """
    Generates an answer based on the retrieved documents and user question.

    This node prepares a prompt that includes:
    - The original or rewritten user question
    - A list of relevant documents (from vectorstore or web search)

    It invokes the main LLM to synthesize a concise and grounded response, returning the result
    for use in later hallucination and usefulness checks.

    Args:
        state (GraphState): The current LangGraph state containing documents and question(s).

    Returns:
        state (GraphState): The updated LangGraph state with the following values.
            - "question": The input question used in generation
            - "generation": The generated answer (str)
            - "references_table": Formatted references section (str)
            - "token_count": Number of tokens used in generation
            - "response_time": Time taken to generate the response (in seconds)
            - "total_cost": Cost incurred (if metered by API provider)
    """
    logger.info("\n---ANSWER GENERATION---")

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    documents = state.documents

    # Use original_question if available (after rewriting), otherwise default to input question
    if state.original_question:
        question = state.original_question
    else:
        question = state.question

    # Ensure all documents are LangChain Document objects (convert from dicts if needed)
    documents = [Document(metadata=doc["metadata"], page_content=doc["page_content"]) if isinstance(doc, dict) else doc for doc in documents]

    # Create citation context and references mapping
    source_context, formatted_context, references_dict = create_citation_context(documents)

    # Format the prompt for the answer generator
    prompt = answer_generator_prompt_template.format(current_datetime=current_datetime, context=formatted_context, question=question, goals_as_str=goals_as_str, source_context=source_context)

    logger.info(f"Answer generator prompt: {prompt}")
    response = call_llm(prompt=prompt, model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"], pydantic_model=None, agent_name="answer_generator_agent")

    show_agent_reasoning(response, f"Answer Generator Response | " + state.metadata["model_name"])

    # The LLM should now include references in its response, but we store the mapping for potential use
    state.messages.append(response)
    state.generation = str(response.content)
    state.metadata["references_dict"] = references_dict

    logger.info(f"Current state: {state}")
    logger.info(f"Response with integrated references: {state.generation}")

    return state
