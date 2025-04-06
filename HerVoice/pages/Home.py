import streamlit as st

st.title("Team HerVoice")
st.write('''HerVoice came to life with the collaboration of four students from SFSU and Claremont Graduate University! 

Thank you so much to SFHacks 2025 for hosting our visit! :)''')
    
st.subheader("Inspiration")

st.write('''
HerVoice was inspired by those moments when you’re not sure if you should speak up or just let it go. It’s made to be a supportive, smart sidekick that helps you figure things out without the pressure.
''')

st.subheader("What It Does")

st.write('''HerVoice is a private, no-judgment chatbot for anybody that can benefit from a wise ear at a critical moment. You can talk through tricky situations, ask anything, and get clear, supportive guidance—all totally confidential.''')

st.subheader("How we built it")
st.write('''Using Streamlit, PostgreSQL, LangChain, Google Gemini''')


st.subheader("Challenges we ran into")
st.write('''Learning about strict types with LangGraph''')

st.subheader("Accomplishments that we're proud of")

st.write('''we made it all the way to San Francisco!!
We learned how to manipulate postgres datastores containing containing Google `models/text-embedding-004` embeddings''')

st.subheader("What we learned")

st.write('''we learned how to get structured data from an LLM''')

st.subheader("What's next for HerVoice")

st.write('''
public access would be awesome!
chromadb implementation would be great to move away from a local postgres database
''')