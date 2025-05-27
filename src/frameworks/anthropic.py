"""
Anthropic Framework Module
Supports Claude 3 (Sonnet, Opus, Haiku) via Anthropic v1/messages API.
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
    Process a chat using the Anthropic Claude API (Claude 3 Sonnet, Opus, Haiku).

    Args:
        messages: List of message dicts with 'role' and 'content' (roles: 'user', 'assistant', 'system')
        model: Model name (e.g., 'claude-3-sonnet-20240229')
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        framework_config: Dictionary containing Anthropic framework configuration

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
        print("Error: Anthropic framework_config not provided.")
        return {"error": "Anthropic framework configuration not provided."}

    try:
        api_key_ref = framework_config.get("api_key_ref")
        api_base_url = framework_config.get("api_base_url")

        if not api_key_ref:
            return {"error": "API key reference (api_key_ref) not found in Anthropic framework configuration."}
        if not api_base_url:
            return {"error": "API base URL (api_base_url) not found in Anthropic framework configuration."}

            # Safely get and validate the API key
    api_key = get_api_key()
    is_valid, error_msg = validate_api_key(api_key, "API key")
    if not is_valid:
        return {'error': error_msg}

        if not api_key:
            return {"error": f"API key for reference '{api_key_ref}' not found in st.secrets."}

        url = api_base_url

        # Anthropic API expects roles as 'user' and 'assistant' (system prompt is a separate field)
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Anthropic v1/messages supports a system prompt field
                if not system_prompt:
                    system_prompt = msg["content"]
                continue
            if msg["role"] in ("user", "assistant"):
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        payload = {
            "model": model,
            "messages": filtered_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 1024  # Required by Anthropic API
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed_time = time.time() - start_time
        if response.status_code != 200:
            response.raise_for_status()
        data = response.json()

        content = data["content"][0]["text"] if data.get("content") else ""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        print(f"Error in Anthropic API call: {str(e)}")
        return None
