import os
import uvicorn
import logging
from typing import Tuple, List, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from truenorth.utils.progress import progress
from truenorth.graph import build_rag_graph, save_graph_as_png

# === Logging Setup ===
file_path = os.path.realpath(__file__)
log_dir = os.path.join(os.path.dirname(file_path), ".logs")
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if os.getenv("LOG_TO_FILE", "true").lower() == "true":
        log_file = logging.FileHandler(os.path.join(log_dir, "server.log"))
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)

# === FastAPI App ===
app = FastAPI()

# Allow UI to make API requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Model Input/Output ===
class QueryInput(BaseModel):
    question: str
    chat_history: List[str] = []


class QueryOutput(BaseModel):
    response: str
    usage: Any


# === Core Functionality ===
async def invoke_llm(question: str) -> Tuple[str, List[Any]]:
    logger.info("Invoking chatbot...")
    progress.start()
    try:
        model_name = "gemini-2.0-flash"
        model_provider = "Gemini"
        # model_name = "smollm:1.7b"
        # model_provider = "Ollama"
        selected_analysts = []

        workflow = build_rag_graph(selected_analysts)
        agent = workflow.compile()
        save_graph_as_png(agent, os.path.join(log_dir, "graph.png"))

        try:
            final_state = await agent.ainvoke(
                {
                    "question": question,
                    "messages": [HumanMessage(content=question)],
                    "data": [],
                    "metadata": {
                        "show_reasoning": True,
                        "model_name": model_name,
                        "model_provider": model_provider,
                    },
                }
            )
        except Exception as e:
            logger.exception("🔥 LangGraph execution failed")
            raise e

        # final_state = await agent.ainvoke(
        #     {
        #         "question": question,
        #         "messages": [HumanMessage(content=question)],
        #         "data": [],
        #         "metadata": {
        #             "show_reasoning": True,
        #             "model_name": model_name,
        #             "model_provider": model_provider,
        #         }
        #     }
        # )

        response = final_state.get("generation") or str(final_state)
        return response, final_state
    finally:
        progress.stop()


@app.post("/query", response_model=QueryOutput)
async def get_chat_response(input_data: QueryInput):
    response, usage = await invoke_llm(input_data.question)
    return QueryOutput(response=response, usage=usage)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
