"""
ChatGPT Framework Module
Supports OpenAI's GPT-4 and compatible models via the OpenAI API.
"""

import streamlit as st
import requests
import streamlit as st
from typing import Dict, List, Optional

# Import API key utilities
from utils.api_keys import get_api_key, validate_api_key

def process_chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    framework_config: Optional[Dict[str, str]] = None
) -> Optional[dict]:
    """
    Process a chat using the OpenAI ChatGPT API (GPT-4, GPT-3.5-turbo, etc).

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name (e.g., 'gpt-4')
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        framework_config: Dictionary containing 'api_key_ref' and 'api_base_url'

    Returns:
        Dictionary with standard response format:
        {
            "content": str,  # The generated text response
            "prompt_tokens": int,  # Number of input tokens
            "completion_tokens": int,  # Number of output tokens
            "elapsed_time": float  # Time taken for the request in seconds
        }
    """
    if not framework_config:
        print("Error: ChatGPT framework_config not provided.")
        return {"error": "ChatGPT framework configuration not provided."}

    try:
        api_key_ref = framework_config.get("api_key_ref")
        api_base_url = framework_config.get("api_base_url")

        if not api_key_ref:
            return {"error": "API key reference (api_key_ref) not found in ChatGPT framework configuration."}
        if not api_base_url:
            return {"error": "API base URL (api_base_url) not found in ChatGPT framework configuration."}

            # Safely get and validate the API key
    api_key = get_api_key()
    is_valid, error_msg = validate_api_key(api_key, "API key")
    if not is_valid:
        return {'error': error_msg}

        if not api_key:
            return {"error": f"API key for reference '{api_key_ref}' not found in st.secrets."}

        # Construct the full URL
        url = f"{api_base_url.rstrip('/')}/chat/completions"

        # OpenAI expects messages as a list of {"role": ..., "content": ...}
        openai_messages = []
        for msg in messages:
            if msg["role"] not in ("system", "user", "assistant"):
                continue  # Ignore any unsupported roles
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "top_p": top_p
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed_time = time.time() - start_time
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        print(f"Error in ChatGPT API call: {str(e)}")
        return None
