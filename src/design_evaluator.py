import pandas as pd
from tqdm import tqdm
import os
import json
from dotenv import load_dotenv

from hervoice.agent.state import ChatState
from hervoice.agent.evaluation_agents import (
    anthropomorphism_agent,
    attractivity_agent,
    identification_agent,
    goal_facilitation_agent,
    trustworthiness_agent,
    usefulness_agent,
    accessibility_agent,
)

load_dotenv()

def run_agentic_evaluation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(current_dir, "agentic_evaluation_results.csv")

    # Try to load existing evaluation results if available
    if os.path.exists(output_csv_path):
        print("\nExisting evaluation results found. Loading from CSV...")
        df = pd.read_csv(output_csv_path)
    else:
        test_file_path = os.path.join(current_dir, "test_cases.json")
        answers_file_path = os.path.join(current_dir, "answers_generated.json")

        with open(test_file_path, "r", encoding="utf-8") as file:
            test_cases = json.load(file)

        with open(answers_file_path, "r", encoding="utf-8") as file:
            generated_answers = json.load(file)

        agent_funcs = [
            anthropomorphism_agent,
            attractivity_agent,
            identification_agent,
            goal_facilitation_agent,
            trustworthiness_agent,
            usefulness_agent,
            accessibility_agent,
        ]

        results = []

        print("Running agentic evaluations...")
        for case, actual in tqdm(zip(test_cases, generated_answers), total=len(test_cases), desc="Evaluating", ncols=100):
            label = case["label"]
            question = case["query"]
            generation = actual
            theme = case.get("theme", "Unknown").strip()

            state = ChatState(
                question=question,
                generation=generation,
                metadata={"model_name": "gemini-2.0-flash", "model_provider": "Gemini"},
                messages=[]
            )

            for agent in agent_funcs:
                state = agent(state)

            row = {
                "Label": label,
                "Question": question,
                "Response": generation,
                "Theme": theme,
                "Anthropomorphism": state.metadata["anthropomorphism_score"],
                "Attractivity": state.metadata["attractivity_score"],
                "Identification": state.metadata["identification_score"],
                "Goal Facilitation": state.metadata["goal_facilitation_score"],
                "Trustworthiness": state.metadata["trustworthiness_score"],
                "Usefulness": state.metadata["usefulness_score"],
                "Accessibility": state.metadata["accessibility_score"],
            }

            results.append(row)

        df = pd.DataFrame(results)
        for col in [
            "Anthropomorphism", "Attractivity", "Identification",
            "Goal Facilitation", "Trustworthiness", "Usefulness", "Accessibility"
        ]:
            df[col] = df[col].map(likert_mapping).astype('Int64')
        df.to_csv(output_csv_path, index=False)
        print(f"\nAgentic evaluation results saved to {output_csv_path}")

    # Likert mapping and summary statistics
    likert_cols = [
        "Anthropomorphism",
        "Attractivity",
        "Identification",
        "Goal Facilitation",
        "Trustworthiness",
        "Usefulness",
        "Accessibility"
    ]

    likert_mapping = {
        "1 - strongly disagree": 1,
        "2 - disagree": 2,
        "3 - somewhat disagree": 3,
        "4 - neither agree nor disagree": 4,
        "5 - somewhat agree": 5,
        "6 - agree": 6,
        "7 - strongly agree": 7
    }

    df_numeric = df[likert_cols].replace(likert_mapping).apply(pd.to_numeric, errors='coerce')

    print("\n📊 Mean and Variance per evaluation metric (overall):")
    for col in likert_cols:
        series = df_numeric[col].dropna()
        mean = series.mean()
        var = series.var(ddof=1)
        print(f"{col}: Mean = {mean:.2f}, Variance = {var:.2f}")

    print("📚 Mean and Variance per evaluation metric grouped by theme:")
    print("📌 Number of questions per theme:")
    theme_counts = df['Theme'].value_counts().sort_index()
    for theme, count in theme_counts.items():
        print(f"  {theme}: {count} questions")
    df["Theme"] = df["Theme"].str.strip()

    example_questions = {
        "Anthropomorphism": "Does the Agent seem like a real person?",
        "Attractivity": "Do you find the Agent visually appealing?",
        "Identification": "Can you personally relate to the Agent?",
        "Goal Facilitation": "Does the Agent understand and help you achieve your goals?",
        "Trustworthiness": "Do you feel you can trust the Agent?",
        "Usefulness": "Does the Agent help you complete tasks more effectively?",
        "Accessibility": "Was the Agent easy to find and interact with?"
    }

    summary_rows = []

    # Collect overall stats
    for col in likert_cols:
        series = df_numeric[col].dropna()
        mean = series.mean()
        var = series.var(ddof=1)
        summary_rows.append({"Theme": "Overall", "Metric": col, "Mean": round(mean, 2), "Variance": round(var, 2), "Example Question": example_questions.get(col, "")})

    # Collect stats per theme
    for theme in df["Theme"].unique():
        print(f"\nTheme: {theme}")
        df_theme = df[df["Theme"] == theme].copy()
        df_theme_numeric = df_theme[likert_cols].replace(likert_mapping).apply(pd.to_numeric, errors='coerce')
        for col in likert_cols:
            series = df_theme_numeric[col].dropna()
            mean = series.mean()
            var = series.var(ddof=1)
            print(f"  {col}: Mean = {mean:.2f}, Variance = {var:.2f}")
            summary_rows.append({"Theme": theme, "Metric": col, "Mean": round(mean, 2), "Variance": round(var, 2), "Example Question": example_questions.get(col, "")})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(current_dir, "agentic_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📄 Summary statistics saved to {summary_path}")

    return df


if __name__ == "__main__":
    run_agentic_evaluation()
