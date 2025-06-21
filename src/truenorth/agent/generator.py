from langchain_core.tools import tool
from .state import ChatState
from truenorth.utils.logging import get_caller_logger


@tool
def generate_answer(state: ChatState) -> ChatState:
    """Answer Generator

    Args:
        state (ChatState): current conversation state

    Returns:
        ChatState: new conversation state
    """
    # logger.info(f"[generator] Generating answer for: {state.question}")
    # prompt_context = "\n".join(state.documents or [])
    # state.generation = f"Generated answer using context:\n{prompt_context[:100]}..."
    # logger.info(f"[generator] Output: {state.generation[:80]}")
    return state
