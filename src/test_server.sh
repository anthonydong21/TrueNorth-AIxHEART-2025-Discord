#!/bin/bash

# Simple shell script to test chatbot API

QUESTION="I feel like it's just me struggling."

echo "Sending question: $QUESTION"

curl -X POST http://127.0.0.1:8000/query \
    -H "Content-Type: application/json" \
    -d '{
        "question": "'"$QUESTION"'",
        "chat_history": []
    }' | jq
