import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.evaluation import load_evaluator

# === Load environment variables ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Set up embedding model ===
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY,
)


# === Evaluation Function ===
def run_evaluation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(current_dir, "test_cases.json")
    answers_file_path = os.path.join(current_dir, "answers_generated.json")
    output_csv_path = os.path.join(current_dir, "evaluation_results.csv")
    heatmap_path = os.path.join(current_dir, "similarity_heatmap.png")

    # Load test cases
    print("Loading test cases...")
    with open(test_file_path, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    # Load generated answers
    print("Loading generated answers from test_examples.sh...")
    with open(answers_file_path, "r", encoding="utf-8") as file:
        generated_answers = json.load(file)

    # Cosine evaluator setup
    cosine_evaluator = load_evaluator("embedding_distance", embeddings=embedding_model, distance_metric="cosine")

    results = []
    similarities = []
    print("Computing cosine similarity scores...")
    for case, actual in zip(test_cases, generated_answers):
        label = case["label"]
        query = case["query"]
        expected = case["expected_response"]

        score = cosine_evaluator.evaluate_strings(prediction=actual, reference=expected)["score"]
        similarity = 1 - score
        similarities.append(similarity)

        results.append({"Label": label, "Query": query, "Expected": expected, "Actual": actual, "Cosine Similarity": round(similarity, 4)})

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nEvaluation results saved to {output_csv_path}")

    # Plot heatmap
    plt.figure(figsize=(max(6, len(df) * 0.6), 2))
    sns.heatmap([similarities], annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, xticklabels=df["Label"], yticklabels=["Similarity"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    print(f"Similarity heatmap saved to {heatmap_path}")

    return df


if __name__ == "__main__":
    run_evaluation()
