import json
import logging
from langchain_core.messages.base import BaseMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import operator


# Merge utility (you can use it manually during runtime)
def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {**a, **b}


# Define agent state as a Pydantic model
class ChatState(BaseModel):
    question: str
    original_question: str = None
    generation: str = None
    
    # NOTE: messages and documents are stored redundantly — they also live inside metadata variable
    messages: List[BaseMessage] = Field(default_factory=list)
    documents: List[Any] = Field(default_factory=list)

    # Contains all information carried between agents
    metadata: Dict[str, Any] = Field(default_factory=dict)

    max_retries : int = 2
    current_try : int = 0

    def merged_data(self, other: "ChatState") -> Dict[str, Any]:
        return merge_dicts(self.data, other.data)

    def merged_metadata(self, other: "ChatState") -> Dict[str, Any]:
        return merge_dicts(self.metadata, other.metadata)


# an old draft of this class
# class ChatState(BaseModel):
#     question: str
#     metadata: Optional[Dict[str, Any]] = {}
#     generation: Optional[str] = None
#     documents: Optional[List[str]] = None
#     hallucination_checker_attempts: int = 0
#     answer_verifier_attempts: int = 0
#     token_count: Optional[int] = None
#     response_time: Optional[float] = None


class HCResult(BaseModel):
    binary_score: str
    explanation: Optional[str] = None


class AVResult(BaseModel):
    relevance_score: str
    explanation: Optional[str] = None


def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    if isinstance(output, (dict, list)):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=4))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(str(output))
            print(json.dumps(parsed_output, indent=4))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            print(output)

    print("=" * 48)
