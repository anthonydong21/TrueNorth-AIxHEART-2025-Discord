# HerVoice

## Poetry Setup

Create a local virtual environment with Poetry using
```
 poetry config virtualenvs.in-project true
```
Then, `cd` into the directory that contains pyproject.toml and poetry.lock. Install the poetry environment with this terminal command:
```
poetry install
```

## Gemini Setup
You need an `.env` file with these values set up:
```
export GEMINI_API_KEY="something"
export DB_CONNECTION= "postgresql+psycopg://something"
export GOOGLE_APPLICATION_CREDENTIALS="something.json"
export DEFAULT_GOOGLE_PROJECT="something"
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_GENAI_USE_VERTEXAI=True

```

You also need the JSON file that is mentioned in `GOOGLE_APPLICATION_CREDENTIALS`.