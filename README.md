# 💜 HerVoice – AI Chatbot Empowering Women in STEM

HerVoice is an AI-powered support assistant designed to empower women in STEM (Science, Technology, Engineering, and Mathematics). It provides a safe and anonymous space for mentorship, career guidance, emotional support, and navigating workplace challenges.

---
## ✨ Features

- 🤖 **Conversational Agent** built with Google Gemini for empathetic and informative support  
- 🔍 **Multi-query Retrieval-Augmented Generation (RAG)** with MMR and RRF for accurate answers  
- 📚 **Custom Vector Database** using `PGVector` to index curated resources and books  
- 🌐 **Tavily Web Search** integration for current, real-world information  
- ✅ **Answer Verification** using hallucination and relevance grading  
- 🧠 **Query Rewriting** for improving search coverage  
- 🎨 **Streamlit Frontend** with a feminine and empowering aesthetic  

---

## 🛠️ Tech Stack

| Component    | Technology                              |
|--------------|------------------------------------------|
| LLM          | Google Gemini (via `langchain_google_genai`) |
| Vector Store | PostgreSQL + pgvector                    |
| Embeddings   | GoogleGenerativeAIEmbeddings             |
| Frontend     | Streamlit and FastAPI                    |
| Web Search   | Tavily API                               |
| Orchestration| LangGraph                                |

---

## Poetry Setup

Create a local virtual environment with Poetry using
```
 poetry config virtualenvs.in-project true
```
Then, `cd` into the directory that contains pyproject.toml and poetry.lock. Install the poetry environment with this terminal command:
```
poetry install
```
Now you can update your virtual environment that Poetry made:
```
source .venv/bin/activate
```

## Environment Setup
You need an `.env` file with these values set up:
```
export GEMINI_API_KEY="something"
export DB_CONNECTION= "postgresql+psycopg://something"
export DEFAULT_GOOGLE_PROJECT="something"
export GOOGLE_CLOUD_LOCATION=us-central1
...
```

...check out `.env.example` for a file you can copy.

# Setup
# You need to preprocess and embed PDFs into the vectorstore:

```python Knowledge.py```

This will take 20+ minutes if you have several PDFs. After you have converted your books to vectors and uploaded them to the PostreSQL database.

# Launch Chatbot API
This sets up a server that the Streamlit application can send calls to.
```poetry run uvicorn HerVoice.src.app:app --reload```

# Launch the Streamlit webpage
You need to have the above chatbot API launched in another terminal before running:
```streamlit run HerVoice.py```