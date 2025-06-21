import pandas as pd
from tqdm import tqdm
import os
import json
from dotenv import load_dotenv

from truenorth.agent.state import ChatState
from truenorth.agent.evaluation_agents import (
    anthropomorphism_agent,
    attractivity_agent,
    identification_agent,
    goal_facilitation_agent,
    trustworthiness_agent,
    usefulness_agent,
    accessibility_agent,
    gender_consciousness_agent,
    empathic_intuition_agent,
    personal_visual_engagement_agent,
    credibility_relatability_agent,
    inclusivity_agent,
    user_agency_agent,
    cognitive_simplicity_agent,
)


load_dotenv()


def run_agentic_evaluation():
    """
    We use Anthropic's Claude 3.7 Sonnet for evaluation.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(current_dir, "agentic_evaluation_results.csv")

    likert_mapping = {"1 - strongly disagree": 1, "2 - disagree": 2, "3 - somewhat disagree": 3, "4 - neither agree nor disagree": 4, "5 - somewhat agree": 5, "6 - agree": 6, "7 - strongly agree": 7}

    agent_funcs = {
        # DP1
        "Anthropomorphism": anthropomorphism_agent,
        "Identification": identification_agent,
        "Gender-Consciousness": gender_consciousness_agent,
        "Empathic Intuition": empathic_intuition_agent,
        # DP2
        "Trustworthiness": trustworthiness_agent,
        "Personal Visual Engagement": personal_visual_engagement_agent,
        "Credibility Relatability": credibility_relatability_agent,
        "Inclusivity": inclusivity_agent,
        # DP3
        "Goal Facilitation": goal_facilitation_agent,
        "Usefulness": usefulness_agent,
        "Accessibility": accessibility_agent,
        "Attractivity": attractivity_agent,
        "User Agency": user_agency_agent,
        "Cognitive Simplicity": cognitive_simplicity_agent,
    }

    dp_mapping = {
        "DP1": ["Anthropomorphism", "Identification", "Gender-Consciousness", "Empathic Intuition"],
        "DP2": ["Trustworthiness", "Personal Visual Engagement", "Credibility Relatability", "Inclusivity"],
        "DP3": ["Goal Facilitation", "Usefulness", "Accessibility", "Attractivity", "User Agency", "Cognitive Simplicity"],
    }

    if os.path.exists(output_csv_path):
        print("\nExisting evaluation results found. Loading from CSV...")
        df = pd.read_csv(output_csv_path)
    else:
        test_file_path = os.path.join(current_dir, "test_cases.json")
        answers_file_path = os.path.join(current_dir, "answers_generated.json")

        with open(test_file_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        with open(answers_file_path, "r", encoding="utf-8") as f:
            generated_answers = json.load(f)

        results = []
        print("Running agentic evaluations...")
        for case, actual in tqdm(zip(test_cases, generated_answers), total=len(test_cases), desc="Evaluating", ncols=100):
            label = case["label"]
            question = case["query"]
            generation = actual
            theme = case.get("theme", "Unknown").strip()

            state = ChatState(question=question, generation=generation, messages=[], metadata={"model_provider": "Anthropic", "model_name": "claude-3-7-sonnet-latest"})

            for name, agent in agent_funcs.items():
                state = agent(state)

            row = {
                "Label": label,
                "Question": question,
                "Response": generation,
                "Theme": theme,
            }
            for name in agent_funcs.keys():
                key = name.replace(" ", "_").lower()
                row[name] = state.metadata.get(f"{key}_score", "")

            results.append(row)

        df = pd.DataFrame(results)
        for col in agent_funcs:
            df[col] = df[col].map(likert_mapping).astype("Int64")
        df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Agentic evaluation results saved to {output_csv_path}")

    # GROUPED SUMMARY BY DP
    print("\n📊 Summary Statistics Grouped by Design Principle")

    summary_rows = []
    for dp, metrics in dp_mapping.items():
        dp_df = df[metrics].replace(likert_mapping).apply(pd.to_numeric, errors="coerce")
        means = dp_df.mean()
        variances = dp_df.var(ddof=1)
        print(f"\n{dp} - {len(metrics)} Metrics")
        for metric in metrics:
            mean = means[metric]
            var = variances[metric]
            print(f"  {metric}: Mean = {mean:.2f}, Variance = {var:.2f}")
            summary_rows.append({"Design Principle": dp, "Metric": metric, "Mean": round(mean, 2), "Variance": round(var, 2)})

    # Also output per-theme
    df["Theme"] = df["Theme"].str.strip()
    theme_counts = df["Theme"].value_counts()
    print("\n📌 Number of Questions per Theme:")
    for theme, count in theme_counts.items():
        print(f"  {theme}: {count}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(current_dir, "agentic_summary_by_dp.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📄 Summary stats saved to {summary_path}")

    return df


if __name__ == "__main__":
    run_agentic_evaluation()
