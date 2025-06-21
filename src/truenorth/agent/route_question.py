# Query Router Agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from truenorth.agent.state import ChatState, show_agent_reasoning
from truenorth.utils.llm import call_llm
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.metaprompt import vectorstore_content_summary, system_relevant_scope

logger = get_caller_logger()

query_router_prompt_template = PromptTemplate.from_template(
    """
You are an expert at analyzing user question and deciding which data source is best suited to answer them. You must choose **one** of the following options:

1. **Vectorstore**: Use this if the question can be answered by the **existing** content in the vectorstore. 
   The vectorstore contains information about **{vectorstore_content_summary}**.
                                                            
---                                                         
                                                                                                                          
2. **Websearch**: Use this if the question is **within scope** (see below) but meets **any** of the following criteria:
    - The answer **cannot** be found in the local vectorstore
    - The question requires **more detailed or factual information** than what's available in the books (e.g. exact birth date, current events or references)
    - The topic is **time-sensitive** , **current**, or depends on recent events or updates    
                                                                                                                
---                                                         
                                                              
3. **Chitter-Chatter**: Use this if the question:
   - Is **not related** to the scope below, or
   - Is too **broad, casual, or off-topic** to be answered using vectorstore or websearch.
   
   Chitter-Chatter is a fallback agent that gives a friendly response and a follow-up to guide users back to relevant topics.
                                                            
---
                                                            
Scope Definition:
Relevant questions are those related to **{system_relevant_scope}** and topics in **{vectorstore_content_summary}**

---                                                        

Your Task:
Analyze the user's question. Return a JSON object with one key `"signal"` and one value: `"Vectorstore"`, `"Websearch"`, or `"Chitter-Chatter"`.

"""
)

from pydantic import BaseModel
from typing_extensions import Literal


class QueryRouterSignal(BaseModel):
    signal: Literal["Websearch", "Vectorstore", "Chitter-Chatter"]


# ------------------------ Routing Decision  ------------------------
def query_router_agent(state: ChatState):
    """
    Routes the user question to the appropriate agent based on the Query Router's classification.

    Args:
       state (GraphState): Contains the user's input question.

    Returns:
       str: One of 'Vectorstore', 'Websearch', or 'Chitter-Chatter'.
    """
    logger.info("---ROUTING QUESTION---")
    question = state.messages[0].content
    logger.info(f"Q: {question}")
    query_router_prompt = query_router_prompt_template.format(system_relevant_scope=system_relevant_scope, vectorstore_content_summary=vectorstore_content_summary)

    def create_default_query_router_signal():
        return QueryRouterSignal(signal="Chitter-Chatter", confidence=0.0, reasoning="Error in generating analysis; defaulting to Chitter-Chatter.")

    prompt = [SystemMessage(query_router_prompt), question]
    response = call_llm(prompt=prompt, model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"], pydantic_model=QueryRouterSignal, agent_name="query_router", default_factory=create_default_query_router_signal, max_retries=1, verbose=False)
    show_agent_reasoning(response, f"Route Question Response | " + state.metadata["model_name"])

    signal = response.signal
    logger.info(f"---ROUTING QUESTION TO {signal.upper()}---")
    state.metadata["signal"] = signal
    return state
