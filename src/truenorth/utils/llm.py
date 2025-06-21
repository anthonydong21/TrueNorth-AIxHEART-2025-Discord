import os
import json
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
from langchain_core.utils.function_calling import convert_to_openai_function


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
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    md_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_block:
        raw_block = md_block.group(1)

        def fix_quotes(json_str: str) -> str:
            def escape_problem_quotes(match):
                key = match.group(1)
                value = match.group(2)
                # Escape inner double quotes but keep the outer ones intact
                value_escaped = re.sub(r'(?<!\\)"', r'\\"', value)
                return f'"{key}": "{value_escaped}"'

            # Only apply to fields like "explanation": "some string"
            return re.sub(r'"(\w+)":\s*"((?:[^"\\]|\\.)*?)"', escape_problem_quotes, json_str, flags=re.DOTALL)

        safe_block = fix_quotes(raw_block)
        try:
            return json.loads(safe_block)
        except json.JSONDecodeError as e:
            logger.warning(f"[extract_json_from_response] Failed to parse markdown JSON after sanitization: {e}")
            logger.warning(f"Offending block:\n{safe_block}")

    json_candidates = re.findall(r"(\{.*?\})", text, re.DOTALL)
    for candidate in json_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(text)
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
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)

    if pydantic_model:
        if model_info and pydantic_model:
            logger.info("Attempting structured output")
            llm = llm.with_structured_output(pydantic_model)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"LLM call attempt #{attempt}")
            if verbose:
                logger.info(f"Prompt: {prompt}")
            result = llm.invoke(prompt)
            if verbose:
                logger.info(f"LLM Result: {result}")

            if pydantic_model:
                if model_info and model_info.is_gemini():
                    if isinstance(result, pydantic_model):
                        return result
                    parsed = extract_json_from_response(result.content)
                    if not parsed:
                        raise ValueError(f"[Gemini] Failed to extract JSON from:\n{result.content}")
                    return instantiate_model(pydantic_model, parsed)
                elif model_info and not model_info.has_json_mode():
                    parsed = extract_json_from_response(result.content)
                    if parsed:
                        return instantiate_model(pydantic_model, parsed)
                else:
                    raw_output = result.content.strip().lower()
                    if raw_output in {"true", "false"}:
                        return instantiate_model(pydantic_model, raw_output == "true")
                    else:
                        raise ValueError(f"Unexpected non-JSON raw output: {result.content}")

            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            if agent_name:
                progress.update_status(agent_name, None, f"Retry {attempt}/{max_retries}")
            if attempt == max_retries:
                logger.error(f"Max retries reached after {max_retries} attempts.")
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
