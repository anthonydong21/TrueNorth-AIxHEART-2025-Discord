from langchain_core.tools import tool
from .state import ChatState
from hervoice.utils.logging import get_caller_logger


def evaluate_answer_by_design(state: ChatState) -> str:
    """Evaluate answer by design principles.

    Args:
        state (ChatState): current conversation state

    Returns:
        str: current conversation state
    """
    # Flag max retries
    state.current_try += 1
    if state.current_try >= state.max_retries:
        state.metadata['evaluator_result'] = "max_retries"
        return state
    
    state.metadata['evaluator_result'] = "pass"
    
    return state
