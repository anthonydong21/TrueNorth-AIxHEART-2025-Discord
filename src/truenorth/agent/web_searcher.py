import requests #url checker
import os
from colorama import Fore, Style
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from .state import ChatState
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.cleaner import clean_documents

from dotenv import load_dotenv


def validateURL(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout = 5) #5 seconds for a response for website
        return r.status_code < 400
    except:
        return False
    

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

logger = get_caller_logger()

# Limit how many results to preview in logging
max_preview = 3

web_search_tool = TavilySearchResults(max_results=5, days=365, search_depth="advanced", include_answer=True, tavily_api_key=tavily_api_key)  # Uses advanced search depth for more accurate results  # Include a short answer to original query in the search results.  # You have defined this API key in the .env file.


def search_web(state: ChatState) -> ChatState:
    """WebSearcher

    Args:
        state (ChatState): current conversation state

    Returns:
        ChatState: new conversation state
    """
    logger.info("\n---WEB SEARCH---")
    # Start time

    question = state.question
    web_results = web_search_tool.invoke(question)

    documents = [Document(metadata=dict(url=doc["url"], title=doc["title"]), page_content=doc["content"]) if isinstance(doc, dict) else doc for doc in web_results]

    documents, stats = clean_documents(documents, verbose=True)

    WorkingDocs = []
    for doc in documents:
        url = doc.metadata.get("url")
        if url and validateURL(url):
            WorkingDocs.append(doc)
        else:
            logger.warning(f"[web_searcher] Invalid URL: {url}")

    documents = WorkingDocs

    state.documents.extend(documents)

    for i, doc in enumerate(documents[:max_preview]):
        # print(type(doc))
        logger.info(f"{Fore.MAGENTA}🔹 Result {i+1}{Style.RESET_ALL}\n" f"   {Fore.GREEN}Title:{Style.RESET_ALL} {doc.metadata.get('title', 'N/A')}\n" f"   {Fore.BLUE}URL:{Style.RESET_ALL} {doc.metadata.get('url', 'N/A')}\n" f"   {Fore.YELLOW}Excerpt:{Style.RESET_ALL} {doc.page_content[:200].strip()}...\n")

    if len(documents) > max_preview:
        logger.info(f"{Fore.CYAN}...and {len(documents) - max_preview} more results not shown.{Style.RESET_ALL}")

    logger.info(f"Total number of web search documents: {len(documents)}")
    return state
