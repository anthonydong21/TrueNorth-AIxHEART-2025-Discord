from tqdm import tqdm
import os
import json
import pandas as pd
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
    test_file_path = os.path.join(current_dir, "test_cases.json")
    answers_file_path = os.path.join(current_dir, "answers_generated.json")
    output_csv_path = os.path.join(current_dir, "agentic_evaluation_results.csv")

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

        state = ChatState(
            question=question,
            generation=generation,
            metadata={"model_name": "gemini-2.0-flash", "model_provider": "Gemini"},
            messages=[]
        )

        for agent in agent_funcs:
            state = agent(state)

        results.append({
            "Label": label,
            "Question": question,
            "Response": generation,
            "Anthropomorphism": state.metadata["anthropomorphism_score"],
            "Attractivity": state.metadata["attractivity_score"],
            "Identification": state.metadata["identification_score"],
            "Goal Facilitation": state.metadata["goal_facilitation_score"],
            "Trustworthiness": state.metadata["trustworthiness_score"],
            "Usefulness": state.metadata["usefulness_score"],
            "Accessibility": state.metadata["accessibility_score"],
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nAgentic evaluation results saved to {output_csv_path}")
    return df


if __name__ == "__main__":
    run_agentic_evaluation()
