import time
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from hervoice.utils.llm import call_llm
from hervoice.agent.state import show_agent_reasoning
from hervoice.utils.logging import get_caller_logger
from hervoice.utils.metaprompt import goals_as_str, system_relevant_scope

logger = get_caller_logger()

# Define the prompt template for answer generation
answer_generator_prompt_template = PromptTemplate.from_template(
    """
Today is {current_datetime}.
                                                                
You are an assistant for question-answering tasks.

Here are your goals:
{goals_as_str}

You do not replace a therapist, legal counsel, or HR department, but you can provide emotional support, educational context, helpful language, and confidential documentation tools.

Use all available links to citations to support your answer.

---

**Background Knowledge**:
Use the following background information to help answer the question:
{context}

****User Question**:
{question}
                                                                
---
                                                                
**Instructions**:
1. Base your answer primarily on the background knowledge provided.
2. If the answer is **not present** in the knowledge, say so explicitly.
3. Keep the answer **concise**, **accurate**, and **focused** on the question.
4. At the end, include a **reference section**:
    - For book-based sources, use **APA-style citations** if possible and local URLs.
    - For web-based sources, include **page title and URL**.
5. Only answer questions relevant to STEM, workplace support, or academic guidance. For all other queries, politely decline.

---
**Important**:
Never invent or guess answers using general world knowledge.  
Your role is to **maintain trust** and offer emotionally supportive, mission-aligned responses.

Always keep a lighthearted, concise manner of speaking while providing a helpful answer to the question.
                                                                
**Answer**:
"""
)


# ------------------------ Answer Generator Node ------------------------
def answer_generator(state):
    """
    Generates an answer based on the retrieved documents and user question.

    This node prepares a prompt that includes:
    - The original or rewritten user question
    - A list of relevant documents (from vectorstore or web search)

    It invokes the main LLM to synthesize a concise and grounded response, returning the result
    for use in later hallucination and usefulness checks.

    Args:
        state (GraphState): The current LangGraph state containing documents and question(s).

    Returns:
        state (GraphState): The updated LangGraph state with the following values.
            - "question": The input question used in generation
            - "generation": The generated answer (str)
            - "token_count": Number of tokens used in generation
            - "response_time": Time taken to generate the response (in seconds)
            - "total_cost": Cost incurred (if metered by API provider)
    """
    logger.info("\n---ANSWER GENERATION---")

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    documents = state.documents

    # Use original_question if available (after rewriting), otherwise default to input question
    if state.original_question:
        question = state.original_question
    else:
        question = state.question

    # Ensure all documents are LangChain Document objects (convert from dicts if needed)
    documents = [Document(metadata=doc["metadata"], page_content=doc["page_content"]) if isinstance(doc, dict) else doc for doc in documents]

    # Format the prompt for the answer generator
    prompt = answer_generator_prompt_template.format(current_datetime=current_datetime, context=documents, question=question, goals_as_str=goals_as_str)

    logger.info(f"Answer generator prompt: {prompt}")
    response = call_llm(prompt=prompt, model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"], pydantic_model=None, agent_name="chitter_chatter_agent")

    show_agent_reasoning(response, f"Answer Generator Response | " + state.metadata["model_name"])

    state.messages.append(response)
    state.generation = str(response.content)
    # reference_table = state.metadata["Reference Table"]
    # if reference_table:
    #     state.generation = [state.generation, '**References:**', state.metadata["Reference Table"]]

    logger.info(f"Current state: {state}")
    logger.info(f"Response: {state.generation}")
    return state

    # End time
    # end = time.time()
    # response_time = round(end - start, 2)
    # state.metadata["response_time"] += response_time

    # state["token_count"] = token_count
    # state["total_cost"] = total_cost
    # state["response_time"] = response_time

    # # Prepare row
    # row = {
    #     "timestamp": datetime.utcnow().isoformat(),
    #     "question": question,
    #     "generation": answer_generation.content,
    #     "token_count": token_count,
    #     "response_time": response_time,
    #     "total_cost": total_cost
    # }

    # # Log to CSV with header if file doesn't exist
    # csv_file = os.path.join(log_dir, "answer_generator_metrics.csv")
    # df = pd.DataFrame([row])
    # df.to_csv(csv_file, mode="a", index=False, header=not os.path.exists(csv_file))

    # return {
    #     "question": question,
    #     "generation": answer_generation.content,
    #     "token_count": token_count,
    #     "response_time": response_time,
    #     "total_cost": total_cost
    # }
