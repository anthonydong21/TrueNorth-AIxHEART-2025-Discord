import sys
import os
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from enum import Enum
from typing_extensions import Callable
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style, init
from pydantic import ValidationError

from langchain_core.messages import HumanMessage

from llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from agent.state import ChatState
from utils.progress import progress
from graph import build_rag_graph, save_graph_as_png


# Set up logging
file_path = os.path.realpath(__file__)
log_dir = os.path.join(os.path.dirname(file_path), ".logs")
os.makedirs(log_dir, exist_ok=True)
workflow_png = os.path.join(log_dir, "graph.png")
backtest_result_json = os.path.join(log_dir, "backtest_result.json")

logger = logging.getLogger("backtester")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    console = logging.StreamHandler()
    log_file = logging.FileHandler(os.path.join(log_dir, "backtester.log"))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    log_file.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(log_file)

test_question = "What is the square root of love?"
test_answer = "The square root of love is a metaphor suggesting that love has infinite expressions—romantic, familial, platonic, self-love—each combining in unique ways. There's no single formula for love; it's complex, multidimensional, and deeply personal."
test = {"question": test_question, "ideal_answer": test_answer, "actual_answer": None}
init(autoreset=True)


class Metrics(str, Enum):
    SIMILARITY = cosine_similarity


def parse_analyst_response(message_content: str):
    decisions = None
    return decisions


def run_chatbot(model_name: str, model_provider: str, selected_analysts: List[Any], messages: List[HumanMessage], show_reasoning: bool):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow
        logger.info("Compiling workflow")
        workflow = build_rag_graph(selected_analysts)
        agent = workflow.compile()
        fp = save_graph_as_png(agent, workflow_png)
        logger.info(f"Saved mermaid graph to: {fp}")

        final_state = {"decisions": None, "analyst_signals": None}
        try:
            final_state = agent.invoke(
                {
                    "question": messages[0].content,
                    "messages": messages,
                    "data": [],
                    "metadata": {
                        "show_reasoning": show_reasoning,
                        "model_name": model_name,
                        "model_provider": model_provider,
                    },
                },
            )
        except ValidationError as exc:
            print(repr(exc.errors()))

        # return {
        #     "decisions": parse_analyst_response(final_state.messages[-1].content),
        #     "analyst_signals": final_state.data["analyst_signals"],
        # }
        return final_state
    finally:
        # Stop progress tracking
        progress.stop()


class Backtester:
    def __init__(
        self,
        agent: Callable,
        model_name: str = "smollm:1.7b",
        model_provider: str = "Ollama",
        selected_analysts: list[str] = [],
        question: str = test_question,
    ):
        self.agent = agent
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.test_question = question

        logger.info("Loaded backtester.")

    def parse_agent_response(self, agent_output):
        """Parse JSON output from the agent (fallback to 'hold' if invalid)."""
        import json

        try:
            decision = json.loads(agent_output)
            return decision
        except Exception:
            logger.error(f"Error parsing action: {agent_output}")
            return {"action": "hold", "quantity": 0}

    def run_backtest(self, question: str = None):
        logger.info("Starting backtest...")
        test_messages = [
            HumanMessage(
                content=self.test_question,
            )
        ]
        output = self.agent(model_name=self.model_name, model_provider=self.model_provider, selected_analysts=self.selected_analysts, messages=test_messages, show_reasoning=True)
        with open(backtest_result_json, "w+") as f:
            f.write(str(output))

        return output

    def analyze_performance(self):
        return


def main(args):
    # Select LLM model based on whether Ollama is being used

    model_name = args.model_name
    model_provider = args.model_provider
    if args.ollama:
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")
        model_name = "smollm:1.7b"
        model_provider = "Ollama"

    # Create and run the backtester
    backtester = Backtester(
        agent=run_chatbot,
        model_name=model_name,
        model_provider=model_provider,
    )

    state = backtester.run_backtest()
    # performance_df: pd.DataFrame = backtester.analyze_performance()
    print(state.get("generation"))
    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash", help='Name of the model to use (default: "gemini-2.0-flash")')
    parser.add_argument("--model_provider", type=str, default="Gemini", help='Provider of the model (default: "Gemini")')
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")
    args = parser.parse_args()

    main(args)
