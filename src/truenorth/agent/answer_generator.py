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
        tuple: (context_string, references_dict) where references_dict maps source numbers to full citations with quotes
    """
    if not documents:
        return "", {}

    sources = []
    references_dict = {}
    source_num = 1

    # Create individual numbered citations for each document instead of grouping
    for doc in documents:
        metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
        page_content = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")

        # Create source identifier
        if "author" in metadata and "title" in metadata:
            # Book/PDF source
            author = metadata.get("author", "Unknown Author")
            title = metadata.get("title", "Unknown Title")
            year = metadata.get("creationdate", "")[:4] if metadata.get("creationdate") else "n.d."
            file_path = metadata.get("file_path", metadata.get("source", ""))
            page = metadata.get("page", metadata.get("page_number", metadata.get("page_num", "Unknown")))

            # Create a short citation for the context
            sources.append(f"[{source_num}] {author} ({year}) - {title}, p. {page}")

            # Extract meaningful quote (first 100 characters of content, cleaning up formatting)
            meaningful_quote = page_content.strip().replace("\n", " ").replace("###", "").replace("##", "")
            meaningful_quote = meaningful_quote[:100] + "..." if len(meaningful_quote) > 100 else meaningful_quote

            # Create full citation for references
            citation = f"{author} ({year}). [{title}](file://{file_path}#{page})"
            citation += f'\n    > "{meaningful_quote}"'
            references_dict[source_num] = citation
#getting a change for fixing the potential double link
        elif "url" in metadata:
            # Web source
            url = metadata.get("url", "")
            title = metadata.get("title") or "Title N/A"

            if title == "Title N/A":
                sources.append(f"[{source_num}]{url}")
            else:
                sources.append(f"[{source_num}] {title}")

            meaningful_quote = page_content.strip()[:100] + "..." if len(page_content.strip()) > 100 else page_content.strip()
            citation = f"[{title}]({url})" if title != "Title N/A" else f"{url}"
            citation += f'\n    > "{meaningful_quote}"'
            references_dict[source_num] = citation

        source_num += 1

    context_string = "\n".join(sources)
    return context_string, references_dict


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
    source_context, references_dict = create_citation_context(documents)

    # Format the prompt for the answer generator
    prompt = answer_generator_prompt_template.format(current_datetime=current_datetime, context=documents, question=question, goals_as_str=goals_as_str, source_context=source_context)

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
