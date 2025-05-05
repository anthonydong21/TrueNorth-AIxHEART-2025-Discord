#!/bin/bash

set -e  # Exit the script if any command returns a non-zero status

# Path to the test cases file
TEST_CASES="src/test_cases.json"

# API endpoint
API_URL="http://127.0.0.1:8000/query"

# Output file to store generated responses
OUTPUT_FILE="src/answers_generated.json"

# Initialize empty array to store responses
responses=()

# Progress bar function
progress_bar() {
    local progress=$1
    local total=$2
    local width=40
    local percent=$(( 100 * progress / total ))
    local filled=$(( width * progress / total ))
    local empty=$(( width - filled ))

    printf "\rProgress: ["
    printf "%0.s#" $(seq 1 $filled)
    printf "%0.s-" $(seq 1 $empty)
    printf "] %d%% (%d/%d)" "$percent" "$progress" "$total"
}

# Get number of test cases
NUM_CASES=$(jq length "$TEST_CASES")

# Loop through each test case
for (( i=0; i<$NUM_CASES; i++ ))
do
    QUERY=$(jq -r ".[$i].query" "$TEST_CASES")
    LABEL=$(jq -r ".[$i].label" "$TEST_CASES")

    echo -e "\n[$LABEL] Sending question: $QUERY"

    JSON_PAYLOAD=$(jq -n --arg question "$QUERY" '{"question": $question, "chat_history": []}')

    # Capture both response and HTTP status code
    HTTP_RESPONSE=$(curl -s -w "HTTPSTATUS:%{http_code}" -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "$JSON_PAYLOAD")

    # Split raw response and status
    RAW_BODY=$(echo "$HTTP_RESPONSE" | sed -e 's/HTTPSTATUS\:.*//g')
    HTTP_STATUS=$(echo "$HTTP_RESPONSE" | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')

    echo "HTTP Status: $HTTP_STATUS"

    if [ "$HTTP_STATUS" -ne 200 ]; then
        echo "❌ Error: Received HTTP status $HTTP_STATUS"
        exit 1
    fi

    RESPONSE=$(echo "$RAW_BODY" | jq -r '.response')

    if [ "$RESPONSE" == "null" ] || [ -z "$RESPONSE" ]; then
        echo "⚠️ Error: No 'response' field in response"
        exit 1
    fi

    echo "Response: $RESPONSE"
    responses+=("$RESPONSE")

    # Update progress bar
    progress_bar $((i + 1)) $NUM_CASES
done

# Write responses to output file as JSON array
printf '%s\n' "$(jq -n '$ARGS.positional' --args "${responses[@]}")" > "$OUTPUT_FILE"

echo -e "\n✅ All responses saved to $OUTPUT_FILE."
