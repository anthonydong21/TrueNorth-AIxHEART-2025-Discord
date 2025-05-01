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
## 📚 1. Preprocess & Embed  PDFs

Before running the chatbot, convert your PDFs into searchable vector embeddings:

```python Knowledge.py```

⏳ This process may take 20+ minutes depending on the number and size of PDFs. Data is uploaded to the PostgreSQL vectorstore (pgvector).


## 📡 2. Launch Chatbot API (FastAPI)

Start the backend server (runs on port 8000):

`poetry run uvicorn hervoice.app:app --reload`

### 🔗 API Docs

Once running, access the API docs and test endpoints at:

```http://localhost:8000/docs```

This Swagger UI lets you test HerVoice programmatically and inspect endpoint responses.

## 🌸 3. Launch the Streamlit UI

In a separate terminal, start the frontend (runs on port 8501):

```streamlit run HerVoice.py```

Then go to your browser:

`http://localhost:8501`

This interface connects to the API and provides an interactive chatbot experience.

# ✅ Makefile Usage

If `make` is installed, you can simplify operations with:
🔧 Local setup (no Docker)

```
make install        # Install dependencies using Poetry
make embed          # Preprocess PDFs into vector DB
make api            # Launch FastAPI backend (http://localhost:8000)
make ui             # Launch Streamlit frontend (http://localhost:8501)
make test           # "What is Langchain?" test question
```