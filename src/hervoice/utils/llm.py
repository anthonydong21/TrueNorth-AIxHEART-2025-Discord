"""Helper functions for LLM"""
"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from hervoice.llm.models import get_model_info, get_model
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from hervoice.utils.progress import progress
from hervoice.utils.logging import get_caller_logger

logger = get_caller_logger()

T = TypeVar("T", bound=BaseModel)


def call_llm(prompt: Any, model_name: str, model_provider: str, pydantic_model: Type[T], agent_name: Optional[str] = None, max_retries: int = 3, default_factory=None, verbose=False) -> T:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model

    Async LLM call with retry logic and structured output handling.
    """
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)

    if verbose:
        logger.info(f"Using LLM: {llm}")

    # Attach structured output parsing if supported
    if pydantic_model is not None and (not model_info or model_info.has_json_mode()):
        llm = llm.with_structured_output(pydantic_model, method="json_mode")

    for attempt in range(1, max_retries + 1):
        logger.info(f"LLM call attempt #{attempt}")
        if verbose:
            logger.info(f"Prompt: {prompt}")

        try:
            result = llm.invoke(prompt)  # << IMPORTANT: async invoke
            if verbose:
                logger.info(f"LLM Result: {result}")

            if pydantic_model is not None:
                if model_info and not model_info.has_json_mode():
                    parsed = extract_json_from_response(result.content)
                    if parsed:
                        return pydantic_model(**parsed)
                else:
                    # Handle simple true/false model outputs
                    if hasattr(pydantic_model, "__pydantic_root_model__"):
                        raw_output = result.content.strip().lower()
                        if raw_output == "true":
                            return pydantic_model(root=True)
                        elif raw_output == "false":
                            return pydantic_model(root=False)
                        else:
                            raise ValueError(f"Unexpected non-JSON raw output: {result.content}")

            return result.content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            if agent_name:
                progress.update_status(agent_name, None, f"Retry {attempt}/{max_retries}")

            if attempt == max_retries:
                logger.error(f"Max retries reached after {max_retries} attempts.")
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    return create_default_response(pydantic_model)


def create_default_response(model_class: Optional[Type[T]]) -> Optional[T]:
    if model_class is None:
        logger.warning("No model_class provided for fallback response; returning None.")
        return None

    if hasattr(model_class, "__pydantic_root_model__"):
        logger.warning("Fallback default RootModel response: False")
        return model_class(root=False)

    default_fields = {field_name: ("Error" if field.type_ == str else 0 if field.type_ in (int, float) else {} if field.type_ == dict else None) for field_name, field in model_class.model_fields.items()}
    return model_class(**default_fields)


def extract_json_from_response(content: str) -> Optional[dict]:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_content = content[json_start + 7 :]
            json_end = json_content.find("```")
            if json_end != -1:
                json_content = json_content[:json_end].strip()
            return json.loads(json_content)
        else:
            return json.loads(content)  # fallback if no code block
    except Exception as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        return None


def get_embedding_model(model_name: str, model_provider: str) -> Optional[Any]:
    """
    Returns an embedding model instance based on model name and provider.

    Args:
        model_name: Name of the embedding model
        model_provider: Provider of the model (e.g., "OpenAI", "Gemini", "Ollama")

    Returns:
        An instance of the embedding model, or None if unsupported
    """
    from hervoice.llm.models import get_model_info
    from langchain_openai import OpenAIEmbeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_ollama import OllamaEmbeddings

    model_info = get_model_info(model_name)

    if not model_info:
        logger.error(f"Model info not found for {model_name}")
        return None

    try:
        if model_provider == "OpenAI":
            return OpenAIEmbeddings(model=model_name)
        elif model_provider == "Gemini":
            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        elif model_provider == "Ollama":
            return OllamaEmbeddings(model=model_name)
        else:
            logger.error(f"Embedding not supported for provider: {model_provider}")
            return None
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        return None
