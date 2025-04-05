# Experiment setup

from AnswerGenerator import generate_answer


def main():
    question = "What day is it today?"
    print(f"User: {question}")
    response, usage = generate_answer(question)
    print(f"Agent Response: {response}")
    print(usage)
    print("Done!")
    return

if __name__ == "__main__":
    main()