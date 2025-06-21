import os
from typing import Tuple, List, Dict, Any, Optional
from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from truenorth.utils.logging import get_caller_logger


# def run_chatbot(question: str):

#     state = ChatState(question=question)
#     result = rag_graph.invoke(state)
#     save_graph_as_png(result)
#     print(f"Question: {state.question}")
#     print(f"Response: {result.generation}")
