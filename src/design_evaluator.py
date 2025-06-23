# design_evaluator.py
import pandas as pd
from tqdm import tqdm
import os
import json
import concurrent.futures
from functools import lru_cache
from dotenv import load_dotenv
import numpy as np  # Import numpy for NaN check

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


@lru_cache(maxsize=1000)
def cached_likert_mapping(response):
    """Cache Likert scale mappings for faster processing"""
    mapping = {"1 - strongly disagree": 1, "2 - disagree": 2, "3 - neutral": 3, "4 - agree": 4, "5 - strongly agree": 5}
    return mapping.get(response, None)


def evaluate_single_case(case_data):
    """
    Process a single test case - designed for parallel execution.
    """
    case, actual, all_agents = case_data

    label = case["label"]
    question = case["query"]
    generation = actual
    theme = case.get("theme", "Unknown").strip().rstrip(",")

    # state = ChatState(question=question, generation=generation, messages=[], metadata={"model_provider": "Gemini", "model_name": "gemini-2.0-flash"})
    # state = ChatState(question=question, generation=generation, messages=[], metadata={"model_provider": "Ollama", "model_name": "llama3.2"})
    state = ChatState(question=question, generation=generation, messages=[], metadata={"model_provider": "Anthropic", "model_name": "claude-3-5-haiku-latest"})

    # Run all evaluation agents sequentially for this case
    for name, agent in all_agents.items():
        state = agent(state)

    row = {
        "Label": label,
        "Question": question,
        "Response": generation,
        "Theme": theme,
    }

    # print("<<<<<<<<<<<<<< START STATE METADATA DEBUG >>>>>>>>>>>>>>")
    # print(f"DEBUG: state.metadata after all agents: {state.metadata}")
    # print("<<<<<<<<<<<<<<< END STATE METADATA DEBUG >>>>>>>>>>>>>>>")

    row = {
        "Label": label,
        "Question": question,
        "Response": generation,
        "Theme": theme,
        # Core Metrics - these seem to be handled correctly as they are populating
        "Anthropomorphism": state.metadata.get("anthropomorphism_score"),
        "Attractivity": state.metadata.get("attractivity_score"),
        "Identification": state.metadata.get("identification_score"),
        "Goal Facilitation": state.metadata.get("goal_facilitation_score"),
        "Trustworthiness": state.metadata.get("trustworthiness_score"),
        "Usefulness": state.metadata.get("usefulness_score"),
        "Accessibility": state.metadata.get("accessibility_score"),
        # Meta-Requirement Metrics
        "Gender-Consciousness (MR1)": state.metadata.get("gender_consciousness_score"),
        "Empathic Intuition (MR2)": state.metadata.get("empathic_intuition_score"),
        "Personal Visual Engagement (MR3)": state.metadata.get("personal_visual_engagement_score"),
        "Credibility Relatability (MR4)": state.metadata.get("credibility_relatability_score"),
        "Inclusive Community (MR5)": state.metadata.get("inclusivity_score"),
        "User Agency (MR6)": state.metadata.get("user_agency_score"),
        "Cognitive Simplicity (MR7)": state.metadata.get("cognitive_simplicity_score"),
    }
    return row


def run_agentic_evaluation(max_workers=4, test_run=False):
    """
    Evaluation using Statkus et al. (2024) core metrics plus meta-requirements
    for gender-inclusive STEM support, aligned with TrueNorth design principles.
    Uses 1-5 Likert scale. Optimized for speed with parallel processing.

    Args:
        max_workers: Number of parallel workers for evaluation (default: 4)
        test_run: If True, only one test case will be processed (default: False)
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

    # Always attempt to load from CSV first, unless it's a test run and we want to regenerate
    # For a test run, we always want to process the single case, so no need to load existing CSV
    if os.path.exists(output_csv_path) and not test_run:
        print("\nExisting evaluation results found. Loading from CSV...")
        df = pd.read_csv(output_csv_path)
        # Ensure numeric columns are Int64 for consistency
        for col in all_agents:
            if col in df.columns:
                df[col] = df[col].map(likert_mapping).astype("Int64")
    else:
        test_file_path = os.path.join(current_dir, "test_cases.json")
        answers_file_path = os.path.join(current_dir, "answers_generated.json")

        with open(test_file_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        with open(answers_file_path, "r", encoding="utf-8") as f:
            generated_answers = json.load(f)

        if test_run:
            print("\n🚀 Running in test mode: Processing only one test case.")
            test_cases = [test_cases[0]]
            generated_answers = [generated_answers[0]]

        results = []
        print("Running agentic evaluations...")
        # Prepare data for parallel processing
        cases_for_processing = [(case, actual, all_agents) for case, actual in zip(test_cases, generated_answers)]

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Wrap the executor.map with tqdm for a progress bar
            # Handle exceptions from workers
            futures = [executor.submit(evaluate_single_case, case_data) for case_data in cases_for_processing]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating", ncols=100):
                try:
                    results.append(future.result())  # This will re-raise any exception from evaluate_single_case
                except ValueError as e:
                    print(f"\n❌ Error during evaluation: {e}")
                    print("Stopping evaluation due to parsing error.")
                    # Re-raise the exception to stop the main process
                    raise
                except Exception as e:
                    print(f"\n❌ An unexpected error occurred: {e}")
                    print("Stopping evaluation due to unexpected error.")
                    raise

        df = pd.DataFrame(results)

        print(df)

        # Convert Likert responses to numeric (1-5 scale)
        for col in all_agents:
            if col in df.columns:
                # First, replace any empty strings with numpy NaN to ensure consistent handling
                df[col] = df[col].replace("", np.nan)

                # Map the Likert strings to numeric values. Non-matching strings will result in NaN.
                df[col] = df[col].map(likert_mapping)

                # Convert to numeric, coercing any remaining errors to NaN, then to nullable integer type
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

                # Optional: Add debug print to see if NaNs are introduced
                if df[col].isnull().any():
                    print(f"DEBUG: Column '{col}' contains NaN values after conversion.")
                    print(df[df[col].isnull()])  # Shows rows where this column is NaN
        # Always save results to CSV, even in test_run mode, to avoid NA issues downstream if reloaded
        df.to_csv(output_csv_path, index=False)
        if not test_run:
            print(f"\n✅ Agentic evaluation results saved to {output_csv_path}")
        else:
            print(f"\n✅ Test run completed. Results saved to {output_csv_path} (useful for debugging).")

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
            core_vars = core_df.var(ddof=1)  # ddof=1 for sample variance

            for metric in categories["Core Metrics"]:
                mean = core_means[metric]
                var = core_vars[metric]

                # Handle NaN for variance (e.g., when only one data point)
                mean_str = f"{mean:.2f}" if not pd.isna(mean) else "N/A"
                var_str = f"{var:.2f}" if not pd.isna(var) else "N/A (single point)"

                print(f"    {metric}: Mean = {mean_str}, Variance = {var_str}")
                summary_rows.append({"Design Principle": dp, "Category": "Core Metric", "Metric": metric, "Mean": (round(mean, 2) if not pd.isna(mean) else None), "Variance": (round(var, 2) if not pd.isna(var) else None)})

        # Meta-Requirements
        if categories["Meta-Requirements"]:
            print("  Meta-Requirements (Gender-Inclusive STEM):")
            meta_df = df[categories["Meta-Requirements"]].apply(pd.to_numeric, errors="coerce")
            meta_means = meta_df.mean()
            meta_vars = meta_df.var(ddof=1)  # ddof=1 for sample variance

            for metric in categories["Meta-Requirements"]:
                mean = meta_means[metric]
                var = meta_vars[metric]

                # Handle NaN for variance
                mean_str = f"{mean:.2f}" if not pd.isna(mean) else "N/A"
                var_str = f"{var:.2f}" if not pd.isna(var) else "N/A (single point)"

                print(f"    {metric}: Mean = {mean_str}, Variance = {var_str}")
                summary_rows.append({"Design Principle": dp, "Category": "Meta-Requirement", "Metric": metric, "Mean": (round(mean, 2) if not pd.isna(mean) else None), "Variance": (round(var, 2) if not pd.isna(var) else None)})

        # Overall DP score
        if not df[all_metrics_for_dp].empty and not df[all_metrics_for_dp].isnull().all().all():
            dp_df = df[all_metrics_for_dp].apply(pd.to_numeric, errors="coerce")
            overall_mean = dp_df.mean().mean()
            if not pd.isna(overall_mean):
                print(f"  Overall {dp} Mean: {overall_mean:.2f}")
            else:
                print(f"  Overall {dp} Mean: N/A (no valid data)")
        else:
            print(f"  No valid data to calculate Overall {dp} Mean for the selected metrics.")

    # Theme analysis
    print(f"\n📌 Distribution of Questions by Theme")
    print("=" * 40)
    df["Theme"] = df["Theme"].str.strip()
    theme_counts = df["Theme"].value_counts()
    for theme, count in theme_counts.items():
        print(f"  {theme}: {count} questions")

    # Always save summary for consistency
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(current_dir, "agentic_summary_by_dp.csv")
    summary_df.to_csv(summary_path, index=False)
    if not test_run:
        print(f"\n📄 Detailed summary saved to {summary_path}")
    else:
        print(f"\n📄 Detailed summary saved to {summary_path} (even for test run).")

    # Additional analysis: Meta-requirement alignment with themes
    print(f"\n🎯 Meta-Requirement Performance by Theme")
    print("=" * 45)

    mr_cols = [col for col in df.columns if col in meta_requirements.keys()]  # Use meta_requirements.keys() for exact match

    if mr_cols and not df.empty and not df[mr_cols].isnull().all().all():
        theme_mr_analysis = df.groupby("Theme")[mr_cols].mean().round(2)
        print(theme_mr_analysis.to_string())

        # Always save theme-MR analysis for consistency
        theme_mr_path = os.path.join(current_dir, "theme_meta_requirements_analysis.csv")
        theme_mr_analysis.to_csv(theme_mr_path)
        if not test_run:
            print(f"\n📊 Theme-MR analysis saved to {theme_mr_path}")
        else:
            print(f"\n📊 Theme-MR analysis saved to {theme_mr_path} (even for test run).")
    elif df.empty:
        print("No data available for Meta-Requirement Performance by Theme analysis.")
    else:
        print("No meta-requirement columns with valid data found for analysis.")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TrueNorth Agentic Evaluation")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1). Google Gemini supports up to 8.")
    parser.add_argument("--test-run", action="store_true", help="Run only one test case to save on costs")
    args = parser.parse_args()

    # The main call to run_agentic_evaluation
    run_agentic_evaluation(max_workers=args.workers, test_run=args.test_run)
