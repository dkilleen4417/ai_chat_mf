"""
Ollama Framework Module
Supports local Llama models via Ollama's OpenAI-compatible API.
"""

from typing import Any, Dict, List, Optional
import time

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
    framework_config: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None
) -> Optional[dict]:
    """
    Process a chat using Ollama's local OpenAI-compatible API for Llama models.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name (e.g., 'llama3', 'llama3:8b', etc.)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        framework_config: Configuration for the framework
        max_tokens: Maximum number of tokens to generate

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
        st.error("Ollama framework configuration not provided.")
        return {
            "error": "Ollama framework configuration not provided.",
            "content": "Error: Ollama framework configuration not provided.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }

    api_base_url = framework_config.get("api_base_url")
    if not api_base_url:
        st.error("API base URL (api_base_url) not found in Ollama framework configuration.")
        return {
            "error": "API base URL (api_base_url) not found in Ollama framework configuration.",
            "content": "Error: API base URL (api_base_url) not found in Ollama framework configuration.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }

    url = f"{api_base_url.rstrip('/')}/chat/completions"

    api_key_ref = framework_config.get("api_key_ref")
    api_key = None
    if api_key_ref:
        # Safely get and validate the API key
        api_key = get_api_key(api_key_ref)
        is_valid, error_msg = validate_api_key(api_key, "API key")
        if not is_valid:
            return {'error': error_msg}

    if not api_key:
            st.warning(f"API key for reference '{api_key_ref}' not found in st.secrets for Ollama. Proceeding without API key.")

    try:
        ollama_messages = []
        for msg in messages:
            if msg["role"] not in ("system", "user", "assistant"):
                continue
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        payload = {
            "model": model,
            "messages": ollama_messages,
            "temperature": temperature,
            "top_p": top_p
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed_time = time.time() - start_time
        response.raise_for_status()
        data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API request failed: {str(e)}")
        error_content = f"Error: Ollama API request failed. {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_content += f" Details: {error_details}"
            except ValueError:
                error_content += f" Status: {e.response.status_code}, Response: {e.response.text}"
        return {
            "error": str(e),
            "content": error_content,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": time.time() - start_time if 'start_time' in locals() else 0
        }
    except Exception as e:
        st.error(f"Error processing Ollama response: {str(e)}")
        return {
            "error": str(e),
            "content": f"Error processing Ollama response: {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": time.time() - start_time if 'start_time' in locals() else 0
        }
