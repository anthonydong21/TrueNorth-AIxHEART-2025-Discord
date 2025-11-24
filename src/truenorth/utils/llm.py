import os
import json
import traceback
from enum import Enum
import re
from typing import Tuple, List, Dict, Any, Optional, TypeVar, Type, Union

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from truenorth.utils.logging import get_caller_logger
from truenorth.utils.progress import progress

logger = get_caller_logger()
load_dotenv()

# ----------------------------------------
# Model Enums and Config
# ----------------------------------------


class ModelProvider(str, Enum):
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"


class LLMModel(BaseModel):
    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        return (self.display_name, self.model_name, self.provider.value)

    def has_json_mode(self) -> bool:
        if self.is_deepseek() or self.is_gemini():
            return False
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        return True

    def is_deepseek(self) -> bool:
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        return self.provider == ModelProvider.OLLAMA


AVAILABLE_MODELS = [
    LLMModel(display_name="[anthropic] claude-3.5-haiku", model_name="claude-3-5-haiku-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[anthropic] claude-3.5-sonnet", model_name="claude-3-5-sonnet-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[anthropic] claude-3.7-sonnet", model_name="claude-3-7-sonnet-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[deepseek] deepseek-r1", model_name="deepseek-reasoner", provider=ModelProvider.DEEPSEEK),
    LLMModel(display_name="[deepseek] deepseek-v3", model_name="deepseek-chat", provider=ModelProvider.DEEPSEEK),
    LLMModel(display_name="[gemini] gemini-2.0-flash", model_name="gemini-2.0-flash", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[gemini] gemini-2.5-pro", model_name="gemini-2.5-pro-exp-03-25", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[groq] llama-4-scout-17b", model_name="meta-llama/llama-4-scout-17b-16e-instruct", provider=ModelProvider.GROQ),
    LLMModel(display_name="[groq] llama-4-maverick-17b", model_name="meta-llama/llama-4-maverick-17b-128e-instruct", provider=ModelProvider.GROQ),
    LLMModel(display_name="[openai] gpt-4.5", model_name="gpt-4.5-preview", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] gpt-4o", model_name="gpt-4o", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] o1", model_name="o1", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] o3-mini", model_name="o3-mini", provider=ModelProvider.OPENAI),
]

OLLAMA_MODELS = [
    LLMModel(display_name="[ollama] smollm (1.7B)", model_name="smollm:1.7b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] phi3  (3.8B)", model_name="phi3", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] gemma3 (4B)", model_name="gemma3:4b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] qwen2.5 (7B)", model_name="qwen2.5", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] llama3.1 (8B)", model_name="llama3.1:latest", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] gemma3 (12B)", model_name="gemma3:12b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] mistral-small3.1 (24B)", model_name="mistral-small3.1", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] gemma3 (27B)", model_name="gemma3:27b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] qwen2.5 (32B)", model_name="qwen2.5:32b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] llama-3.3 (70B)", model_name="llama3.3:70b-instruct-q4_0", provider=ModelProvider.OLLAMA),
]

LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str) -> Optional[LLMModel]:
    return next((m for m in AVAILABLE_MODELS + OLLAMA_MODELS if m.model_name == model_name), None)


def get_model(model_name: str, model_provider: ModelProvider):
    if model_provider == ModelProvider.GROQ:
        return ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
    elif model_provider == ModelProvider.OPENAI:
        return ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    elif model_provider == ModelProvider.ANTHROPIC:
        return ChatAnthropic(model=model_name, api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif model_provider == ModelProvider.DEEPSEEK:
        return ChatDeepSeek(model=model_name, api_key=os.getenv("DEEPSEEK_API_KEY"))
    elif model_provider == ModelProvider.GEMINI:
        return ChatGoogleGenerativeAI(model=model_name, api_key=os.getenv("GEMINI_API_KEY"))
    elif model_provider == ModelProvider.OLLAMA:
        return ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


# ----------------------------------------
# LLM Call Handling
# ----------------------------------------

T = TypeVar("T", bound=BaseModel)


def is_root_model(model_class: Type[BaseModel]) -> bool:
    return list(model_class.model_fields.keys()) == ["root"]


def instantiate_model(model_class: Type[T], data: Any) -> T:
    if is_root_model(model_class):
        return model_class(root=data)
    elif isinstance(data, dict):
        return model_class(**data)
    else:
        field = list(model_class.model_fields.keys())[0]
        return model_class(**{field: data})


def extract_json_from_response(text: Union[str, bytes]) -> Optional[dict]:
    """
    Robustly extract JSON from text that might be wrapped in markdown or contain extra commentary.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    # 1. Try standard Markdown JSON block
    md_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_block:
        try:
            return json.loads(md_block.group(1))
        except json.JSONDecodeError:
            pass  # Fall through to other methods

    # 2. Try finding the first brace-enclosed object
    json_candidates = re.findall(r"(\{.*?\})", text, re.DOTALL)
    for candidate in json_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 3. Try parsing the whole text as JSON (cleaning start/end)
    try:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    return None


def create_default_response(model_class: Optional[Type[T]]) -> Optional[T]:
    if model_class is None:
        return None
    try:
        return instantiate_model(model_class, False)
    except Exception:
        default_fields = {k: ("Error" if v.annotation == str else 0 if v.annotation in (int, float) else {} if v.annotation == dict else None) for k, v in model_class.model_fields.items()}
        return model_class(**default_fields)


def call_llm(prompt: Any, model_name: str, model_provider: str, pydantic_model: Type[T], agent_name: Optional[str] = None, max_retries: int = 3, default_factory=None, verbose=False) -> T:
    """
    Unified LLM caller that handles structured output differences across providers.
    """
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)

    logger.info(f"LLM provider: {model_provider} | Model: {model_name}")

    if pydantic_model:
        if model_info:
            logger.info("Configuring structured output...")
            llm = llm.with_structured_output(pydantic_model)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"LLM call attempt #{attempt}")
            if verbose:
                logger.info(f"Prompt: {prompt}")

            result = llm.invoke(prompt)

            if verbose:
                logger.info(f"LLM Result Type: {type(result)}")
                # Only log content if it's not the object itself (to avoid massive logs)
                if not isinstance(result, pydantic_model):
                    logger.info(f"LLM Result Content: {result}")

            # === SUCCESS CASE 1: Direct Pydantic Object ===
            # Most providers (OpenAI, Anthropic, newer Gemini) via LangChain return the object directly
            if pydantic_model and isinstance(result, pydantic_model):
                return result

            # === FALLBACK HANDLING ===
            if pydantic_model:
                # Normalize result content
                result_content = ""
                if isinstance(result, dict):
                    # Sometimes returned as a dict instead of model instance
                    return instantiate_model(pydantic_model, result)
                elif hasattr(result, "content"):
                    result_content = str(result.content)
                else:
                    result_content = str(result)

                # Check for tool_calls (Common for Gemini/OpenAI structured output)
                if hasattr(result, "tool_calls") and result.tool_calls:
                    try:
                        # LangChain standardizes tool calls into a list of dicts
                        args = result.tool_calls[0]["args"]
                        return instantiate_model(pydantic_model, args)
                    except Exception as e:
                        logger.warning(f"Failed to parse tool_calls: {e}")

                # Try to extract JSON from text content
                parsed_json = extract_json_from_response(result_content)
                if parsed_json:
                    return instantiate_model(pydantic_model, parsed_json)

                # Gemini-Specific Text Fallback (For "AnswerResponse" type models)
                # When Gemini ignores JSON mode and returns chatty text with citations
                if model_provider == "Gemini" or model_info.is_gemini():
                    logger.warning("[Gemini] JSON extraction failed. Attempting text-based fallback.")
                    try:
                        fields = list(pydantic_model.model_fields.keys())

                        # Custom parser for AnswerResponse
                        if "answer" in fields and "citations" in fields:
                            import re

                            # Simple heuristic: Everything before "### Citations" or "**References**" is the answer
                            # If we find citation markers [1] "quote", we extract them

                            answer_text = result_content
                            citations = []

                            # Split on common citation section headers
                            split_patterns = [r"###\s*Citations", r"\*\*\s*References\s*\*\*", r"###\s*References"]
                            for pattern in split_patterns:
                                parts = re.split(pattern, result_content, flags=re.IGNORECASE)
                                if len(parts) > 1:
                                    answer_text = parts[0].strip()
                                    citation_text = parts[1].strip()

                                    # Extract citations: [1] "quote"
                                    citation_matches = re.findall(r'\[(\d+)\]\s*"(.*?)"', citation_text, re.DOTALL)
                                    for source_id, quote in citation_matches:
                                        citations.append({"source_id": int(source_id), "quote": quote.strip()})
                                    break

                            return instantiate_model(pydantic_model, {"answer": answer_text, "citations": citations})

                        elif "content" in fields:
                            return instantiate_model(pydantic_model, {"content": result_content})

                    except Exception as e:
                        logger.warning(f"Fallback parsing failed: {e}")

                raise ValueError(f"Failed to parse structured output from response: {result_content[:200]}...")

            return result

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"LLM call failed on attempt {attempt}: {e}")
            logger.debug(f"Full traceback:\n{tb}")

            if agent_name:
                progress.update_status(agent_name, None, f"Retry {attempt}/{max_retries}")

            if attempt == max_retries:
                return default_factory() if default_factory else create_default_response(pydantic_model)

    return create_default_response(pydantic_model)


def get_embedding_model(model_name: str, model_provider: str) -> Optional[Any]:
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
