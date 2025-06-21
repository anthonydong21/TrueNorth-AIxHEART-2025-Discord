import time
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from .state import ChatState
from truenorth.utils.llm import call_llm, extract_json_from_response
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.metaprompt import goals_as_str, system_relevant_scope
from pydantic import BaseModel
from typing import List

logger = get_caller_logger()


class ReferenceSummary(BaseModel):
    summary: str
    key_quote: str


create_helpful_table_prompt_template = PromptTemplate.from_template(
    """
Today is {current_datetime}.

You are an assistant helping summarize articles into structured reference entries that are most useful to a user's question.

---

**Context Documents**:
{context}

---

**User Question**:
{question}

---

1. Extract the main idea or theme of the document as a short summary.
2. Select a key quote that represents the tone or message of the piece.
3. Return a JSON object with two keys: summary, and key_quote.
 
---

Your task:
Only return a single valid JSON object, nothing else.
                                                                    

"""
)


def create_reference_table(state: ChatState) -> ChatState:
    """
    Synthesizes information from a list of web-sourced documents into structured references.
    """
    references = []

    for doc in state.documents:
        prompt = create_helpful_table_prompt_template.format(current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question=state.question, context=doc)  # will stringify if needed

        try:
            # This returns a ReferenceSummary object directly
            response = call_llm(prompt=prompt, model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"], pydantic_model=ReferenceSummary, agent_name="create_reference_table", verbose=True)

            # Combine structured summary with doc metadata
            if response:
                combined = {"summary": response.summary, "key_quote": response.key_quote, "title": doc.metadata.get("title", "Untitled"), "url": doc.metadata.get("url", "Unknown URL")}
                references.append(combined)

        except Exception as e:
            logger.warning(f"❌ Failed to summarize document (page {doc.metadata.get('page')}): {e}")
            continue

    # Save to state metadata
    state.metadata["Reference Table"] = references
    return state
