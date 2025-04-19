import streamlit as st
from dotenv import load_dotenv
import os
import base64
from AnswerGenerator import generate_answer
from langchain_core.messages import HumanMessage, AIMessage

import streamlit as st

current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(current_script_dir)
media_dir = os.path.join(repo_dir, "img")

st.set_page_config(
    page_title="HerVoice – Chatbot for Women in STEM",
    page_icon="💬",
    layout="wide",
)

examples = [
    "I'm a junior software engineer who has been constantly interrupted and ignored at work. I feel so annoyed what should I do about it?",
    "I am a team member at my work and I just returned from my leave. For some reason I feel like I missed out on something because everyone is doing this project but me? They won't include me or anything it feels alienating what should I do????",
    "I'm a graduate student and in my lab meeting I got the feeling that someone slightly talked negative about me and it has really got my nerve. It feels wrong from them to have done that",
]

# Sidebar: HerVoice Mission Support
with st.sidebar:
    st.title("HerVoice")  # Sidebar title
    st.subheader("Our Mission 💜")  # Focused on your project mission

    st.write(
        """
    We are here to uplift and empower women and other voices in Science, Technology, Engineering, and Mathematics (STEM).

    👩‍🔬 Our chatbot offers:
    - A safe, anonymous space for sharing challenges  
    - Guidance on power dynamics and workplace bias  
    - Mentorship and career navigation support  
    - Resources for opportunities  

    HerVoice believes that every voice matters — especially yours.
    """
    )
    st.subheader("Focus Areas")
    st.write(
        """
    - 💼 Navigating Workplace Challenges  
    - 🧑‍🏫 Finding Mentorship  
    - 🧠 Building Confidence in STEM  
    - 🔍 Anonymous, Neutral Guidance
    """
    )

    # Reset conversation button
    if st.button("Reset Conversation 🔄", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


logo_base64 = get_image_base64(f"{media_dir}/minilogo.png")  # or logo.png


img_base64 = get_image_base64(f"{media_dir}/logo.png")

with st.container():
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; align-items: center; margin-top: -3rem; margin-bottom: 0rem;'>
            <img src='data:image/png;base64,{img_base64}' width='300'/>
        </div>
    """,
        unsafe_allow_html=True,
    )

# st.image(f"{media_dir}/bg.png")
img_bg = get_image_base64(f"{media_dir}/bg.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"]{{
    background-image: url("data:image/jpg;base64,{img_bg}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    backdrop-filter: blur(6px);

}}
[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"][data-testid="stChatInputTextArea"]{{
    background: rgba(0, 0, 0, 0);
}}

input[type="text"] {{
    background-color: rgba(255, 255, 255, 0.8) !important;
    color: #000 !important;
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
}}

button[kind="primary"] {{
    background-color: #f39ac4 !important;
    color: white !important;
    border-radius: 10px !important;
}}

[data-testid="stBottomBlockContainer"]{{
    background-color: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 8px;
    margin-top: 10px;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-pro")

# App settings


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Intro message
if not st.session_state.chat_history:
    with st.chat_message("ai", avatar=f"{media_dir}/minilogo.png"):
        st.markdown("**Hello, I'm HerVoice – your friendly STEM ally. How can I support you today?**")


# Chat input
user_query = st.chat_input("Enter your question or concern here...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("ai", avatar=f"{media_dir}/minilogo.png"):
        with st.spinner("HerVoice is thinking..."):
            # graph_output = graph.invoke({"question": str(user_query)})
            # final_answer = graph_output.get("generation", "I'm here to support you. How can I help?")
            # final_answer = final_answer()
            response, usage = generate_answer(str(user_query))
            final_answer = response
            with st.expander("📦 Full JSON Response", expanded=False):
                st.json(usage)

        st.markdown(final_answer)

        print(f"final answer: {final_answer}")
        print(f"final answer type: {type(final_answer)}")
        st.session_state.chat_history.append(AIMessage(final_answer))

        # st.session_state.chat_history.append(AIMessage(final_answer))
