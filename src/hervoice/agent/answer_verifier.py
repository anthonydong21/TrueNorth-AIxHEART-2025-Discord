from langchain_core.tools import tool
from .state import ChatState
from hervoice.utils.logging import get_caller_logger


@tool
def check_relevance(state: ChatState) -> str:
    """Relevance Checker

    Args:
        state (ChatState): current conversation state

    Returns:
        str: current conversation state
    """
    # logger.info("[answer_verifier] Verifying answer relevance...")
    # if "Generated answer" in (state.generation or ""):
    #     logger.info("[answer_verifier] Result: pass")
    #     return "useful"
    # state.answer_verifier_attempts += 1
    # logger.info("[answer_verifier] Result: not useful")
    # return "not_useful"
    return state