import requests
from typing import Tuple, List, Any

# API_URL = "http://0.0.0.0:8000/query"
API_URL = "http://localhost:8000/query"



def generate_answer(prompt: str, chat_history: List[Any] = []) -> Tuple[str, Any]:
    payload = {"question": prompt, "chat_history": chat_history}  # You can add actual chat history here if needed

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["response"], data["usage"]
    except requests.RequestException as e:
        print(f"❌ Error talking to backend: {e}")
        return "Sorry, something went wrong.", []


# Example usage
if __name__ == "__main__":
    reply, usage = generate_answer("What is the square root of love?")
    print("Response:", reply)
    print("Usage:", usage)
