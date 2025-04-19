from langchain_core.tools import tool
from .state import ChatState
from hervoice.utils.logging import get_caller_logger


@tool
def check_hallucination(state: ChatState) -> str:
    """Hallucination Checker

    Args:
        state (ChatState): current conversation state

    Returns:
        str: current conversation state
    """
    # logger.info("[hallucination_checker] Starting check...")
    # if "not grounded" in (state.generation or ""):
    #     state.hallucination_checker_attempts += 1
    #     logger.info("[hallucination_checker] Result: fail")
    #     return "retry_generation" if state.hallucination_checker_attempts < 2 else "hallucinated"
    # logger.info("[hallucination_checker] Result: pass")
    # return "check_relevance"
    return state
