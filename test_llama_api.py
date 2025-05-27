curl -X POST https://api.llama.com/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer LLM|1136112611883997|GS5Wq1Fh7mjjZch__rmLEMPocBc" \
-d '{
    "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    "max_completion_tokens": 1024,
    "temperature": 0.7
}'