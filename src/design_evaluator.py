# design_evaluator.py
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
    # Meta-requirement evaluation agents
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
    Evaluation using Statkus et al. (2024) core metrics plus meta-requirements
    for gender-inclusive STEM support, aligned with TrueNorth design principles.
    Uses 1-5 Likert scale.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(current_dir, "agentic_evaluation_results.csv")

    # Updated to 1-5 scale as requested
    likert_mapping = {"1 - strongly disagree": 1, "2 - disagree": 2, "3 - neutral": 3, "4 - agree": 4, "5 - strongly agree": 5}

    # Core evaluation metrics from Statkus et al. (2024)
    core_metrics = {
        "Anthropomorphism": anthropomorphism_agent,
        "Attractivity": attractivity_agent,
        "Identification": identification_agent,
        "Goal Facilitation": goal_facilitation_agent,
        "Trustworthiness": trustworthiness_agent,
        "Usefulness": usefulness_agent,
        "Accessibility": accessibility_agent,
    }

    # Meta-requirements for gender-inclusive STEM support
    meta_requirements = {
        "Gender-Consciousness (MR1)": gender_consciousness_agent,
        "Empathic Intuition (MR2)": empathic_intuition_agent,
        "Personal Visual Engagement (MR3)": personal_visual_engagement_agent,
        "Credibility Relatability (MR4)": credibility_relatability_agent,
        "Inclusive Community (MR5)": inclusivity_agent,
        "User Agency (MR6)": user_agency_agent,
        "Cognitive Simplicity (MR7)": cognitive_simplicity_agent,
    }

    # Combine all evaluation agents
    all_agents = {**core_metrics, **meta_requirements}

    # Design Principle mapping based on TrueNorth framework
    dp_mapping = {
        "DP1 - Emotionally Intelligent & Stereotype-Neutral": {"Core Metrics": ["Anthropomorphism", "Identification"], "Meta-Requirements": ["Gender-Consciousness (MR1)", "Empathic Intuition (MR2)"]},
        "DP2 - Trustworthy & Personalized Community": {"Core Metrics": ["Trustworthiness"], "Meta-Requirements": ["Personal Visual Engagement (MR3)", "Credibility Relatability (MR4)", "Inclusive Community (MR5)"]},
        "DP3 - Empowering & Streamlined Interactions": {"Core Metrics": ["Goal Facilitation", "Usefulness", "Accessibility", "Attractivity"], "Meta-Requirements": ["User Agency (MR6)", "Cognitive Simplicity (MR7)"]},
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
            theme = case.get("theme", "Unknown").strip().rstrip(",")  # Remove trailing comma

            state = ChatState(question=question, generation=generation, messages=[], metadata={"model_provider": "Anthropic", "model_name": "claude-3-7-sonnet-latest"})

            # Run all evaluation agents
            for name, agent in all_agents.items():
                state = agent(state)

            row = {
                "Label": label,
                "Question": question,
                "Response": generation,
                "Theme": theme,
            }

            # Add scores for all metrics
            for name in all_agents.keys():
                key = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                row[name] = state.metadata.get(f"{key}_score", "")

            results.append(row)

        df = pd.DataFrame(results)

        # Convert Likert responses to numeric (1-5 scale)
        for col in all_agents:
            df[col] = df[col].map(likert_mapping).astype("Int64")

        df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Agentic evaluation results saved to {output_csv_path}")

    # GROUPED SUMMARY BY DESIGN PRINCIPLE
    print("\n📊 Summary Statistics Grouped by Design Principle")
    print("=" * 60)

    summary_rows = []

    for dp, categories in dp_mapping.items():
        print(f"\n{dp}")
        print("-" * len(dp))

        all_metrics_for_dp = categories["Core Metrics"] + categories["Meta-Requirements"]

        # Core Metrics
        if categories["Core Metrics"]:
            print("  Core Metrics (Statkus et al., 2024):")
            core_df = df[categories["Core Metrics"]].apply(pd.to_numeric, errors="coerce")
            core_means = core_df.mean()
            core_vars = core_df.var(ddof=1)

            for metric in categories["Core Metrics"]:
                mean = core_means[metric]
                var = core_vars[metric]
                print(f"    {metric}: Mean = {mean:.2f}, Variance = {var:.2f}")
                summary_rows.append({"Design Principle": dp, "Category": "Core Metric", "Metric": metric, "Mean": round(mean, 2), "Variance": round(var, 2)})

        # Meta-Requirements
        if categories["Meta-Requirements"]:
            print("  Meta-Requirements (Gender-Inclusive STEM):")
            meta_df = df[categories["Meta-Requirements"]].apply(pd.to_numeric, errors="coerce")
            meta_means = meta_df.mean()
            meta_vars = meta_df.var(ddof=1)

            for metric in categories["Meta-Requirements"]:
                mean = meta_means[metric]
                var = meta_vars[metric]
                print(f"    {metric}: Mean = {mean:.2f}, Variance = {var:.2f}")
                summary_rows.append({"Design Principle": dp, "Category": "Meta-Requirement", "Metric": metric, "Mean": round(mean, 2), "Variance": round(var, 2)})

        # Overall DP score
        dp_df = df[all_metrics_for_dp].apply(pd.to_numeric, errors="coerce")
        overall_mean = dp_df.mean().mean()
        print(f"  Overall {dp} Mean: {overall_mean:.2f}")

    # Theme analysis
    print(f"\n📌 Distribution of Questions by Theme")
    print("=" * 40)
    df["Theme"] = df["Theme"].str.strip()
    theme_counts = df["Theme"].value_counts()
    for theme, count in theme_counts.items():
        print(f"  {theme}: {count} questions")

    # Save detailed summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(current_dir, "agentic_summary_by_dp.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📄 Detailed summary saved to {summary_path}")

    # Additional analysis: Meta-requirement alignment with themes
    print(f"\n🎯 Meta-Requirement Performance by Theme")
    print("=" * 45)

    mr_cols = [col for col in df.columns if col.startswith("Gender-Consciousness") or col.startswith("Empathic") or col.startswith("Personal") or col.startswith("Credibility") or col.startswith("Inclusive") or col.startswith("User Agency") or col.startswith("Cognitive")]

    if mr_cols:
        theme_mr_analysis = df.groupby("Theme")[mr_cols].mean().round(2)
        print(theme_mr_analysis.to_string())

        theme_mr_path = os.path.join(current_dir, "theme_meta_requirements_analysis.csv")
        theme_mr_analysis.to_csv(theme_mr_path)
        print(f"\n📊 Theme-MR analysis saved to {theme_mr_path}")

    return df


if __name__ == "__main__":
    run_agentic_evaluation()
