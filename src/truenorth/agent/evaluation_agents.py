# evaluation_agents.py
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from truenorth.agent.state import ChatState, show_agent_reasoning
from truenorth.utils.llm import call_llm
from truenorth.utils.logging import get_caller_logger

logger = get_caller_logger()


# ===== Pydantic Models =====
class EvaluationOutput(BaseModel):
    result: Literal["1 - strongly disagree", "2 - disagree", "3 - neutral", "4 - agree", "5 - strongly agree"] = Field(..., description="Likert score for the evaluated construct (1-5 scale)")
    explanation: str = Field(..., description="1–2 sentence explanation for the rating")


# ===== Prompt Templates =====
PROMPTS = {
    # ✅ MR1: Foster Gender-Conscious Engagement
    "gender_consciousness": PromptTemplate.from_template(
        """
You are evaluating whether the chatbot engages in a way that is mindful of gender inclusion.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the response avoid reinforcing gender stereotypes and foster a sense of respect for gender diversity?

**CRITICAL**:
- The chatbot must NOT assume the user's gender.
- It should use gender-neutral language unless the user has explicitly stated their gender.

Output JSON:
- result: "1 - strongly disagree" to "5 - strongly agree"
- explanation: brief reason
    """
    ),
    # ✅ MR2: Provide Intuitive and Empathic Responses
    "empathic_intuition": PromptTemplate.from_template(
        """
You are evaluating whether the chatbot response is intuitive and emotionally aware.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the chatbot anticipate the user's emotional state and respond in a way that feels natural and caring?

Output JSON:
- result: "1 - strongly disagree" to "5 - strongly agree"
- explanation: brief reason
"""
    ),
    # ✅ MR3: Personal Experience and Visual Interaction
    "personal_visual_engagement": PromptTemplate.from_template(
        """
You are assessing whether the chatbot response reflects personalized or visually enriched engagement.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the chatbot acknowledge personal context or incorporate visual/interactive elements that enhance connection?

Output JSON:
- result: "1 - strongly disagree" to "5 - strongly agree"
- explanation: brief reason
"""
    ),
    # ✅ MR4: Establish Credible and Relatable Interactions
    "credibility_relatability": PromptTemplate.from_template(
        """
You are evaluating the credibility and relatability of the chatbot's response.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the chatbot provide reliable information in a tone and manner that feels relatable to the user?

Output JSON:
- result: "1 - strongly disagree" to "5 - strongly agree"
- explanation: brief reason
"""
    ),
    # ✅ MR5: Cultivate an Inclusive Community
    "inclusivity": PromptTemplate.from_template(
        """
You are evaluating whether the chatbot response supports inclusive community-building.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the chatbot encourage connection, belonging, or inclusion in a diverse environment?

**CRITICAL**:
- The chatbot must NOT assume the user belongs to a marginalized group (e.g., assuming the user is a minority, woman in STEM, etc.) unless explicitly stated.
- Inclusion should be broad and welcoming, not presumptive.

Output JSON:
- result: "1 - strongly disagree" to "5 - strongly agree"
- explanation: brief reason
"""
    ),
    # ✅ MR6: Enhance User Agency
    "user_agency": PromptTemplate.from_template(
        """
You are evaluating whether the chatbot empowers users to make informed, independent decisions.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the chatbot support the user's autonomy and sense of control in the interaction?

Output JSON:
- result: "1 - strongly disagree" to "5 - strongly agree"
- explanation: brief reason
"""
    ),
    # ✅ MR7: Simplify Information Processing
    "cognitive_simplicity": PromptTemplate.from_template(
        """
You are evaluating whether the chatbot simplifies complex information for easier user comprehension.

Context:
- User Question: {question}
- Chatbot Response: {generation}

Task:
Does the chatbot break down information clearly and reduce cognitive overload?

Output JSON:
- result: "1 - strongly disagree" to "5 - strongly agree"
- explanation: brief reason
"""
    ),
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
  "1 - strongly disagree", "2 - disagree", "3 - neutral", 
  "4 - agree", "5 - strongly agree"
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
  "1 - strongly disagree", "2 - disagree", "3 - neutral", 
  "4 - agree", "5 - strongly agree"
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
  "1 - strongly disagree", "2 - disagree", "3 - neutral", 
  "4 - agree", "5 - strongly agree"
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
  "1 - strongly disagree", "2 - disagree", "3 - neutral", 
  "4 - agree", "5 - strongly agree"
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
Does the response show care, consistency, or respect for the user's privacy, goals, or emotional needs?

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - neutral", 
  "4 - agree", "5 - strongly agree"
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

**CRITERIA FOR HIGH SCORE (4-5):**
- The answer gives **concrete, actionable steps** (e.g., "Schedule a weekly 15-minute check-in" instead of "communicate better").
- Any technical terms (especially PERMA+4) are **explicitly defined** with examples.
- The answer is **comprehensive** (at least 3-4 sentences of substance) and **self-contained**.

**CRITERIA FOR LOW SCORE (1-2):**
- The answer is too short (1-2 sentences) or generic.
- It mentions a framework (like PERMA+4) **without explaining what the acronym stands for** or how to use it.
- It relies on high-level jargon ("optimize wellbeing", "sustainable performance") without practical application.
- It simply points to citations without synthesizing the information.

Output JSON:
- result: one of: 
  "1 - strongly disagree", "2 - disagree", "3 - neutral", 
  "4 - agree", "5 - strongly agree"
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
  "1 - strongly disagree", "2 - disagree", "3 - neutral", 
  "4 - agree", "5 - strongly agree"
- explanation: brief reason
"""
    ),
}


# ===== Agent Function Template =====
def make_eval_agent(name: str):
    """
    AI-as-a-Judge Evaluation Agent
    """

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
gender_consciousness_agent = make_eval_agent("gender_consciousness")
empathic_intuition_agent = make_eval_agent("empathic_intuition")
personal_visual_engagement_agent = make_eval_agent("personal_visual_engagement")
credibility_relatability_agent = make_eval_agent("credibility_relatability")
inclusivity_agent = make_eval_agent("inclusivity")
user_agency_agent = make_eval_agent("user_agency")
cognitive_simplicity_agent = make_eval_agent("cognitive_simplicity")
