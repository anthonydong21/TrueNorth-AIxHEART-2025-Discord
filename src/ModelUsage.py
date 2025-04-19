# Define a class to track the usage of the model
from contextlib import contextmanager
import time
from typing import Any


class ModelUsageTracker:
    def __init__(self, price_per_1m_tokens):
        """
        Initializes a tracker for monitoring API usage.

        Parameters:
        - price_per_1m_tokens: Defines the $ cost per 1 million tokens.
            - Example 1: Separate input/output pricing: {"input": 10, "output": 30}
            - Example 2: A single flat rate for all tokens: 20

        Attributes:
        - start_time: timestamp when the API call starts.
        - end_time: timestamp when the API call ends.
        - prompt_tokens: number of input tokens used.
        - completion_tokens: number of output tokens generated.
        - total_tokens: sum of prompt and completion tokens.
        - total_cost: Cost calculated based on token usage.

        """
        self.price_per_1m_tokens = price_per_1m_tokens
        self.start_time = None
        self.end_time = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.response_time = 0.0

    def track_usage(self, usage):
        """
        Extracts token usage from API response and calculates the cost.
        """
        if usage:
            # Extract input (prompt) and output (completion) tokens
            self.prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            self.completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
            self.total_tokens = usage.get("total_tokens", self.prompt_tokens + self.completion_tokens)

            # Calculate the total cost based on pricing
            if isinstance(self.price_per_1m_tokens, dict):  # Separate princing for input/output tokens
                input_cost = (self.prompt_tokens / 1_000_000) * self.price_per_1m_tokens.get("input", 0)
                output_cost = (self.completion_tokens / 1_000_000) * self.price_per_1m_tokens.get("output", 0)
                self.total_cost = input_cost + output_cost
            elif isinstance(self.price_per_1m_tokens, (int, float)):  # Flat-rate pricing
                self.total_cost = (self.total_tokens / 1_000_000) * self.price_per_1m_tokens

    def get_summary(self) -> dict[str, Any]:
        """
        Return a dictionary with tracked details.
        """
        return {"Prompt Tokens": self.prompt_tokens, "Completion Tokens": self.completion_tokens, "Total Tokens": self.total_tokens, "Total Cost (USD)": round(self.total_cost, 6), "Response Time (s)": round(self.response_time, 2)}

    def display_summary(self):
        """
        Display usage summary in plain text format.
        """
        print(
            f"""
------------------------------------------------------------
Model Usage Summary
------------------------------------------------------------
- Prompt Tokens:      {self.prompt_tokens}
- Completion Tokens:  {self.completion_tokens}
- Total Tokens:       {self.total_tokens}
- Total Cost: $       {self.total_cost:.6f}
- Response Time:      {self.response_time:.2f} seconds
"""
        )


# Define a context manager to automatically track API call usage
@contextmanager
def usage_tracker(price_per_1m_tokens):
    """
    Context manager for tracking model usage, response time, and cost.
      - Automatically starts tracking before API call.
      - Stops tracking after API call completes and calculates response time.
      - Displays usage summary upon exit.
    """
    tracker = ModelUsageTracker(price_per_1m_tokens)
    tracker.start_time = time.time()  # Record start time before API call
    yield tracker  # Provide the tracker instance
    tracker.end_time = time.time()  # Record end time after API call
    tracker.response_time = tracker.end_time - tracker.start_time  # Compute response time
    # Display usage summary
    tracker.display_summary()
