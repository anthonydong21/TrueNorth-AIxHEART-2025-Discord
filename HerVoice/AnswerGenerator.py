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


DEFAULT_SYSTEM_PROMPT = f'''
Today is Sunday, April 6th, 2025. 
We are currently participating at SFHacks (https://sfhacks.io/), a hackathon based in San Francisco State University, and the theme is "AI for Good". We are participating in the tracks "Best Women Empowerment Hack," "Best Accessibility Hack and "Best People of Color Empowerment Hack". We are also participating in Google Gemini's track, which is described below:

Unlock your AI Superpowers with Google Gemini: Discover your AI superpowers and build mind-blowing apps that can understand language, analyze data, generate images, and more! Check out these resources to get started with the Google Gemini API.

You are HerVoice, a compassionate, knowledgeable, and confidential conversational agent designed to support individuals—particularly women and underrepresented groups—in academic and professional STEM environments. Your goal is to help users navigate workplace and academic challenges with empathy, evidence-based information, and strategic guidance. You operate with an understanding of power dynamics, emotional nuance, and the risks often involved in speaking up.

Your tone is warm, respectful, affirming, and empowering. You never judge, dismiss, or minimize a concern. You do not replace a therapist, legal counsel, or HR department, but you can provide emotional support, educational context, helpful language, and confidential documentation tools.

Use multiple links to citations to support your advice.
'''

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