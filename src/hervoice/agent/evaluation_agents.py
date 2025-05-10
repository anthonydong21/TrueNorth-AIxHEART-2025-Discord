from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from hervoice.agent.state import ChatState, show_agent_reasoning
from hervoice.utils.llm import call_llm
from hervoice.utils.logging import get_caller_logger

logger = get_caller_logger()


# ===== Pydantic Models =====
class EvaluationOutput(BaseModel):
    result: Literal["1 - strongly disagree", "2 - disagree", "3 - somewhat disagree", "4 - neither agree nor disagree", "5 - somewhat agree", "6 - agree", "7 - strongly agree"] = Field(..., description="Likert score for the evaluated construct")
    explanation: str = Field(..., description="1–2 sentence explanation for the rating")


# ===== Prompt Templates =====
PROMPTS = {
    "anthropomorphism": PromptTemplate.from_template(
        """
You are an evaluator judging whether a chatbot response feels like it's from a human.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Would a typical user interpret this response as coming from a person (rather than a machine)?

Look for:
- Emotional language, empathy, self-awareness
- Conversational nuance, personality, or humor

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - somewhat disagree", 
  "4 - neither agree nor disagree", "5 - somewhat agree", 
  "6 - agree", "7 - strongly agree"
- explanation: brief reason
"""
    ),
    "attractivity": PromptTemplate.from_template(
        """
You are evaluating the *visual appeal* of a chatbot's response based on user experience.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the response suggest that the chatbot response is visually appealing, pleasant, or inviting?

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - somewhat disagree", 
  "4 - neither agree nor disagree", "5 - somewhat agree", 
  "6 - agree", "7 - strongly agree"
- explanation: brief reason
"""
    ),
    "identification": PromptTemplate.from_template(
        """
You are evaluating how relatable or personally resonant the chatbot seems.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the response reflect shared experience, values, or language that would make a user say, "I relate to this chatbot"?

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - somewhat disagree", 
  "4 - neither agree nor disagree", "5 - somewhat agree", 
  "6 - agree", "7 - strongly agree"
- explanation: brief reason
"""
    ),
    "goal_facilitation": PromptTemplate.from_template(
        """
You are assessing whether the chatbot helps the user achieve their goal.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the chatbot show understanding of the user's intent and take steps to help them get there?

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - somewhat disagree", 
  "4 - neither agree nor disagree", "5 - somewhat agree", 
  "6 - agree", "7 - strongly agree"
- explanation: brief reason
"""
    ),
    "trustworthiness": PromptTemplate.from_template(
        """
You are evaluating whether the chatbot seems trustworthy.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the response show care, consistency, or respect for the user’s privacy, goals, or emotional needs?

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - somewhat disagree", 
  "4 - neither agree nor disagree", "5 - somewhat agree", 
  "6 - agree", "7 - strongly agree"
- explanation: brief reason
"""
    ),
    "usefulness": PromptTemplate.from_template(
        """
You are assessing how helpful the chatbot is in accomplishing a task.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Would a user say, "This made my task easier/faster/clearer"?

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - somewhat disagree", 
  "4 - neither agree nor disagree", "5 - somewhat agree", 
  "6 - agree", "7 - strongly agree"
- explanation: brief reason
"""
    ),
    "accessibility": PromptTemplate.from_template(
        """
You are evaluating whether the chatbot feels easy to access and engage with.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the response show that the chatbot is readily available, easy to find, and quick to respond?

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - somewhat disagree", 
  "4 - neither agree nor disagree", "5 - somewhat agree", 
  "6 - agree", "7 - strongly agree"
- explanation: brief reason
"""
    ),
}


# ===== Agent Function Template =====
def make_eval_agent(name: str):
    def agent(state: ChatState) -> ChatState:
        print(f"\n---{name.upper()} EVALUATION---")
        logger.info(f"[{name}_agent] Evaluating...")

        prompt = [HumanMessage(PROMPTS[name].format(question=state.question, generation=state.generation))]

        response = call_llm(prompt=prompt, model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"], pydantic_model=EvaluationOutput, agent_name=f"{name}_agent", verbose=True)

        show_agent_reasoning(response, f"{name.title()} Response | {state.metadata['model_name']}")

        # Save result in metadata
        state.metadata[f"{name}_score"] = response.result
        state.metadata[f"{name}_explanation"] = response.explanation
        return state

    return agent


# ===== Named Agents =====
anthropomorphism_agent = make_eval_agent("anthropomorphism")
attractivity_agent = make_eval_agent("attractivity")
identification_agent = make_eval_agent("identification")
goal_facilitation_agent = make_eval_agent("goal_facilitation")
trustworthiness_agent = make_eval_agent("trustworthiness")
usefulness_agent = make_eval_agent("usefulness")
accessibility_agent = make_eval_agent("accessibility")
