import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import base64

# st.set_page_config(page_title="HerVoice - Women in STEM Chatbot", page_icon="💬", layout="centered")
# st.title("👩‍🔬 HerVoice - Empowering Women in STEM")



def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_image_base64("img/logo.png")

with st.container():
    st.markdown(f"""
        <div style='display: flex; justify-content: center; align-items: center; margin-top: -3rem; margin-bottom: 0rem;'>
            <img src='data:image/png;base64,{img_base64}' width='300'/>
        </div>
    """, unsafe_allow_html=True)



img_bg = get_image_base64("img/bg.jpg")

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
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# App settings

# with st.sidebar:
#     # st.image("img/stem_logo.png", width=200)
#     st.title("HerVoice Chatbot")
#     st.subheader("Your STEM Ally 💜")
#     st.write("""
#     Welcome to HerVoice, a supportive chatbot for women in Science, Technology, Engineering, and Mathematics (STEM).
    
#     🤝 I'm here to:
#     - Listen and guide through challenging experiences
#     - Share STEM career advice and resources
#     - Promote mentorship and equality
#     - Ensure a safe and anonymous space
    
#     Ask me anything—from handling power dynamics to finding scholarships and mentorships.
#     """)
    
#     if st.button("Reset Conversation 🔄", use_container_width=True):
#         st.session_state.chat_history = []
#         st.rerun()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to get response from Gemini
def get_hervoice_response(prompt, chat_history):
    messages = [{"role": "user", "parts": [msg.content]} if isinstance(msg, HumanMessage)
                else {"role": "model", "parts": [msg.content]}
                for msg in chat_history]

    messages.append({"role": "user", "parts": [prompt]})
    
    response = model.generate_content(messages)
    return response.text

# Display chat history
from langchain_core.messages import HumanMessage, AIMessage

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Intro message
if not st.session_state.chat_history:
    with st.chat_message("ai", avatar="img/minilogo.png"):
        st.markdown("**Hello, I'm HerVoice – your friendly STEM ally. How can I support you today?**")


# Chat input
user_query = st.chat_input("Enter your question or concern here...")


# if user_query:
#     st.session_state.chat_history.append(HumanMessage(user_query))
    
#     with st.chat_message("Human"):
#         st.markdown(user_query)
    
#     with st.chat_message("AI"):
#         with st.spinner("HerVoice is thinking..."):
#             ai_response = get_hervoice_response(user_query, st.session_state.chat_history)
#         st.markdown(ai_response)
    
#     st.session_state.chat_history.append(AIMessage(ai_response))
