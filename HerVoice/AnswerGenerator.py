import os
from urllib import response
from dotenv import load_dotenv
from pprint import pprint
from llama_index.core.llms.chatml_utils import DEFAULT_SYSTEM_PROMPT
from sqlalchemy.engine.url import make_url
from IPython.display import Markdown
from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector
from sqlalchemy.engine.url import make_url
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser 
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.load import dumps, loads
from langchain_postgres import PGVector
from nltk import usage
from sqlalchemy.engine.url import make_url

# Gemini AI Class
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage
from typing import List, Tuple, Any

from ModelUsage import usage_tracker

# Load environment variables from .env file
load_dotenv()


connection_string = os.getenv("DB_CONNECTION")
google_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Google Gemini-2.0-Flash model
llm_google = ChatVertexAI(
    model="gemini-2.0-flash",
    project=os.getenv("DEFAULT_GOOGLE_PROJECT"),
    seed=3452025,
    temperature=0,
    max_retries=2,
    timeout=120,
)

current_script_dir = os.path.dirname(os.path.abspath(__file__))
system_prompt_path = os.path.join(current_script_dir, 'system_prompt.txt')
with open(system_prompt_path, 'r', encoding='utf-8') as f:
    DEFAULT_SYSTEM_PROMPT = f.read()

# Google Gemini
def invoke_llm(
        question: str ,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT, 
) -> str:
    USAGE_SUMMARY = []

    with usage_tracker({"input": 0.1, "output": 0.4}) as tracker_gemini: # Currently free of charge for free tier users
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        response = llm_google.invoke(messages)

        tracker_gemini.track_usage(response.usage_metadata)
    
    USAGE_SUMMARY.append(tracker_gemini.get_summary())

    return str(response.content), USAGE_SUMMARY


def generate_answer(question: str) -> Tuple[str, List[Any]]:
    response, usage = invoke_llm(question)
    return response, usage