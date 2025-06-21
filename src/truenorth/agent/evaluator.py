from .state import ChatState
from truenorth.utils.logging import get_caller_logger

logger = get_caller_logger()


def evaluate_answer(state: ChatState) -> str:
    """Evaluates the generated answer based on retry count and hallucination-related reasons."""

    logger.info("[evaluator] Evaluating answer...")

    # Increment attempt count
    state.current_try += 1
    logger.info(f"[evaluator] Attempt {state.current_try}/{state.max_retries}")

    # Check for specific hallucination reasons
    evaluator_reason = state.metadata.get("evaluator_reason")

    # Check retry count
    if state.current_try >= state.max_retries:
        logger.warning("[evaluator] Max retries reached.")
        state.metadata["evaluator_result"] = "max_retries"
        return state

    if evaluator_reason == "hallucination":
        logger.warning("[evaluator] Marked as hallucinated.")
        state.metadata["evaluator_result"] = "hallucinated"
        return state

    if evaluator_reason == "invalid_links":
        logger.warning("[evaluator] Broken links detected.")
        state.metadata["evaluator_result"] = "broken_links"
        return state

    # All clear
    state.metadata["evaluator_result"] = "pass"
    state.generation = state.messages[-1].content
    logger.info("[evaluator] Evaluation passed.")
    return state
