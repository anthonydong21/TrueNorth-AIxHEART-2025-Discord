from langchain_core.tools import tool
from .state import ChatState
from hervoice.utils.logging import get_caller_logger


@tool
def evaluate_answer_by_design(state: ChatState) -> str:
    """Evaluate answer by design principles.

    Args:
        state (ChatState): current conversation state

    Returns:
        str: current conversation state
    """
    logger.info("[hallucination_checker] Starting check...")
    if "not grounded" in (state.generation or ""):
        state.hallucination_checker_attempts += 1
        logger.info("[hallucination_checker] Result: fail")
        return "retry_generation" if state.hallucination_checker_attempts < 2 else "hallucinated"
    logger.info("[hallucination_checker] Result: pass")
    return "check_relevance"
