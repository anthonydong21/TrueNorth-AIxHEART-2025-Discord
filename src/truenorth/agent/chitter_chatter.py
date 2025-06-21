from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage


from truenorth.agent.state import ChatState, show_agent_reasoning
from truenorth.utils.llm import call_llm
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.metaprompt import goals_as_str, system_relevant_scope

logger = get_caller_logger()

# Prepare template
chitterchatter_prompt_template = PromptTemplate.from_template(
    """
Today is {current_datetime}. 

**Your goals are as following:**
{goals_as_str}

You do not replace a therapist, legal counsel, or HR department, but you can provide emotional support, educational context, helpful language, and confidential documentation tools.

Use multiple links to citations to support your advice.

---

**Current Scope**:
{system_relevant_scope}

Your job is to respond conversationally while gently guiding the user toward meaningful, empowering, and relevant discussions 
based on the resources in the knowledge base.

---

**Response Guidelines**:

1. **Casual Chit-Chat**:
  - Respond warmly to greetings or casual exchanges.
  - Keep the tone encouraging and human-like.
  - Be an empathetic listener if the user opens up.

2. **Off-Topic Questions**:
  - Politely acknowledge the question.
  - Mention that it falls outside your current scope.
  - Redirect to a related topic such as mentorship, leadership challenges, scholarships, navigating bias, or career growth in STEM.
  - Avoid saying "I don't know" without offering supportive redirection.

3. **In-Scope but Unanswerable Questions**:
  - If the question fits the mission but lacks enough detail to answer confidently:
    - Acknowledge the gap without guessing.
    - Gently ask for clarification or guide the user to rephrase the question.

---

**Important**:
Never invent or guess answers using general world knowledge.  
Your role is to **maintain trust** and offer emotionally supportive, mission-aligned responses.

Always keep a short and concise manner of speaking.
"""
)


def chitter_chatter_agent(state: ChatState) -> ChatState:
    """Chitter-Chatter Agent: Provides warm, fallback conversation when the input is off-topic or unclear."""

    print("\n---CHIT-CHATTING---")
    logger.info("[chitter_chatter_agent] Chit-chatting...")

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    last_user_message = state.question

    prompt = [SystemMessage(chitterchatter_prompt_template.format(current_datetime=current_datetime, goals_as_str=goals_as_str, system_relevant_scope=system_relevant_scope)), last_user_message]

    # logger.info(f"Chitter-chatter Prompt: {prompt}")

    # Call LLM (no pydantic model, expecting just text)
    response = call_llm(prompt=prompt, model_name=state.metadata["model_name"], model_provider=state.metadata["model_provider"], pydantic_model=None, agent_name="chitter_chatter_agent")

    show_agent_reasoning(response, f"Chitter-chatter Response | " + state.metadata["model_name"])

    # Update and return state
    state.generation = str(response.content)
    state.messages.append(response)
    return state
