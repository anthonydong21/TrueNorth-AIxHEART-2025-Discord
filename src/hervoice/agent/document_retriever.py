from langchain_core.tools import tool
from .state import ChatState
from hervoice.utils.logging import get_caller_logger


@tool
def retrieve_documents(state: ChatState) -> ChatState:
    """Document Retriever

    Args:
        state (ChatState): current conversation state

    Returns:
        ChatState: new conversation state
    """
    logger.info(f"[document_retriever] Q: {state.question}")
    docs = ["Sample retrieved doc 1", "Sample retrieved doc 2"]
    state.documents = docs
    logger.info(f"[document_retriever] Retrieved: {len(docs)} docs")
    return state
