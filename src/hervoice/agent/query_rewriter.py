# from langchain_core.tools import tool
from .state import ChatState
from hervoice.utils.logging import get_caller_logger

logger = get_caller_logger()


def rewrite_query(state: ChatState) -> str:
    """Query Rewriter

    Args:
        state (ChatState): current conversation state

    Returns:
        str: current conversation state
    """
    logger.info("Starting rewrite query agent...")

    return state
