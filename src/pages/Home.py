import os
import base64
import streamlit as st

st.title("Team HerVoice")

current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(os.path.dirname(current_script_dir))
media_dir = os.path.join(repo_dir, "img")


def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


img_base64 = get_image_base64(f"{media_dir}/group.png")  # or logo.png
img_bg = get_image_base64(f"{media_dir}/bg.jpg")

# st.image(f"{media_dir}/bg.png")
# img_bg = get_image_base64(f"{media_dir}/bg.jpg")

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

with st.container():
    st.markdown(
        f"""
            <img src='data:image/png;base64,{img_base64}' width='500'/>
    """,
        unsafe_allow_html=True,
    )

st.write(
    """
         HerVoice came to life with the collaboration of four students from SFSU and Claremont Graduate University! 

Thank you so much to SFHacks 2025 for hosting our visit! :)"""
)

st.subheader("Inspiration")

st.write(
    """
HerVoice was inspired by those moments when you’re not sure if you should speak up or just let it go. It’s made to be a supportive, smart sidekick that helps you figure things out without the pressure.
"""
)

st.subheader("What It Does")

st.write("""HerVoice is a private, no-judgment chatbot for anybody that can benefit from a wise ear at a critical moment. You can talk through tricky situations, ask anything, and get clear, supportive guidance—all totally confidential.""")

st.subheader("How we built it")
st.write("""Using Streamlit, PostgreSQL, LangChain, Google Gemini""")


st.subheader("Challenges we ran into")
st.write("""Learning about strict types with LangGraph, learning about Pydantic classes helped""")

st.subheader("Accomplishments that we're proud of")

st.write(
    """we made it all the way to San Francisco!!
We learned how to manipulate postgres datastores containing containing Google `models/text-embedding-004` embeddings"""
)

st.subheader("What we learned")

st.write("""we learned how to get structured data from an LLM""")

st.subheader("What's next for HerVoice")

st.write(
    """
public access would be awesome!
chromadb implementation would be great to move away from a local postgres database
"""
)
