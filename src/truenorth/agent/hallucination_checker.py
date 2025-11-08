import re
import requests
from .state import ChatState
from truenorth.utils.logging import get_caller_logger

logger = get_caller_logger()

def check_hallucination(state: ChatState) -> ChatState:
    logger.info("[hallucination_checker] Starting hallucination check...")

    generation = state.generation or ""
    if not generation.strip():
        state.metadata["hallucination_check_result"] = "fail"
        state.metadata["evaluator_reason"] = "hallucination"
        return state

    hallucination_triggers = [
        "as an AI", "I don't know", "not in the provided text",
        "based on general knowledge", "no information available"
    ]

    if any(trigger in generation.lower() for trigger in hallucination_triggers):
        state.metadata["hallucination_check_result"] = "fail"
        state.metadata["evaluator_reason"] = "hallucination"
        return state

    # ✅ NEW: Check links
    ''' testing to move broken link checker elsewhere
    urls = re.findall(r'(https?://\S+)', generation)
    invalid_links = []

    for url in urls:
        try:
            r = requests.head(url, allow_redirects=True, timeout=3)
            if r.status_code >= 400:
                invalid_links.append(url)
        except Exception as e:
            logger.warning(f"[hallucination_checker] Failed to validate URL: {url} ({e})")
            invalid_links.append(url)

    if invalid_links:
        logger.warning(f"[hallucination_checker] Found broken links: {invalid_links}")
        state.metadata["hallucination_check_result"] = "fail"
        state.metadata["evaluator_reason"] = "invalid_links"
        state.metadata["broken_links"] = invalid_links
        return state

    state.metadata["hallucination_check_result"] = "pass"
    state.metadata["evaluator_reason"] = "grounded"
    return state
'''
