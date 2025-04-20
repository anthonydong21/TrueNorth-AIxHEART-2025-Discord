import json
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.evaluation import load_evaluator
from AnswerGenerator import generate_answer

# === Load environment variables ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Set up embedding model ===
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY,
)

# === 1. Batch Evaluation for Offline Testing ===
def run_evaluation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(current_dir, "test_cases.json")
    answers_file_path = os.path.join(current_dir, "answers_generated.json")

    # Load test cases
    print("Opening test cases...")
    with open(test_file_path, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    # Load actual responses
    print("Generating answers...")
    # generated_answers = []
    # for test_case in test_cases:
    #     user_query = test_case['query']
    #     response, usage = generate_answer(str(user_query))
    #     generated_answers.append(response)
    generated_answers = ["Not a good answer", ] * len(test_cases)

    # with open(answers_file_path, "r", encoding="utf-8") as file:
    #     generated_answers = json.load(file)

    # Load cosine evaluator
    cosine_evaluator = load_evaluator(
        "embedding_distance",
        embeddings=embedding_model,
        distance_metric="cosine"
    )

    results = []
    print("Computing cosine evaluation...")
    for case, generation in zip(test_cases, generated_answers):
        label = case["label"]
        query = case["query"]
        expected = case["expected_response"]
        actual = generation
        
        score = cosine_evaluator.evaluate_strings(
            prediction=actual,
            reference=expected
        )['score']
        similarity = 1 - score

        results.append({
            "Label": label,
            "Query": query,
            "Expected": expected,
            "Actual": actual,
            "Cosine Similarity": round(similarity, 4)
        })

    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df)

    return df

if __name__ == "__main__":
    run_evaluation()

# === 2. LangGraph Node for Real-time Cosine Evaluation ===
def compute_cosine_similarity_node(state):
    print("\n--- COSINE SIMILARITY CALCULATION ---")

    documents = state.get("documents", [])
    prediction = state.get("generation", None)

    if not documents or not prediction:
        print("No documents or generation available for cosine similarity.")
        return state

    ground_truth = documents[0].page_content.strip()

    cosine_evaluator = load_evaluator(
        "embedding_distance",
        embeddings=embedding_model,
        distance_metric="cosine"
    )

    score = cosine_evaluator.evaluate_strings(
        prediction=prediction,
        reference=ground_truth
    )["score"]

    cosine_sim = 1 - score
    print("Cosine Similarity Score:", cosine_sim)

    state["cosine_score"] = cosine_sim
    return state

