# 💜 TrueNorth – PERMA+4 Enhanced Agentic RAG for Long-term Conversational Assistance and Retention Engagement

TrueNorth is an AI-powered support assistant designed to empower early-career professionals pioneering the cutting-edge STEM industry. It provides a safe and anonymous space for mentorship, career guidance, emotional support, and navigating high-stakes workplace situations.

---
## ✨ Features

- 🤖 **Conversational Agent** built with Google Gemini for empathetic and informative support  
- 🔍 **Multi-query Retrieval-Augmented Generation (RAG)** with MMR and RRF for accurate answers  
- 📚 **Custom Vector Database** using Langchain FAISS Local Vector Store to index curated resources and books  
- 🌐 **Tavily Web Search** integration for current, real-world information  
- ✅ **Answer Verification** using hallucination and relevance grading  
- 🧠 **Query Rewriting** for improving search coverage  
- 🎨 **Streamlit Frontend** with a feminine and empowering aesthetic  

---

## 🛠️ Tech Stack

| Component    | Technology                              |
|--------------|------------------------------------------|
| LLM          | Google Gemini (via `langchain_google_genai`)|
| Vector Store | Langchain FAISS Local Vector Store        |
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

`poetry run uvicorn truenorth.app:app --reload`

### 🔗 API Docs

Once running, access the API docs and test endpoints at:

```http://localhost:8000/docs```

This Swagger UI lets you test Streamlit UI programmatically and inspect endpoint responses.

## 🌸 3. Launch the Streamlit UI

In a separate terminal, start the frontend (runs on port 8501):

```streamlit run streamlit_ui.py```

Then go to your browser:

`http://localhost:8501`

This interface connects to the API and provides an interactive chatbot experience.

## 📐 4. Evaluation Workflows
### ✅ Similarity Evaluation

Evaluate response quality using cosine similarity:

`make test`

This runs chatbot responses and saves them to `answers_generated.json`.
### ✅ Agentic Evaluation (DP1–DP3)

Measure 7 design criteria:

- Anthropomorphism
- Attractivity
- Identification
- Goal Facilitation
- Trustworthiness
- Perceived Usefulness
- Accessibility

Run full evaluation:

`python src/design_evaluator.py`

Outputs are saved to:

`agentic_evaluation_results.csv`

Each row shows binary scores for each design quality.

# ✅ Makefile Usage

If `make` is installed, you can simplify operations with:

🔧 Local setup (no Docker)

```
make install        # Install dependencies using Poetry
make embed          # Preprocess PDFs into vector DB
make api            # Launch FastAPI backend (http://localhost:8000)
make ui             # Launch Streamlit frontend (http://localhost:8501)
make test           # Run chatbot API tests using predefined test cases and similarity matric
make eval           # Run evaluation metrics for features Anthropomorphism, Attractivity, Identification, Goal Facilitation, Trustworthiness, Usefulness, Accessibility
```
