"""
Meta Llama API Framework Module
Supports Meta's Llama models through their API.
"""

import os
import time
from typing import List, Dict, Optional, Any
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
) -> Dict[str, Any]:
    """
    Process a chat request using the Meta Llama API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name to use (e.g., 'llama-3-8b-instruct')
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Nucleus sampling parameter (0.0 to 1.0)
        framework_config: Dictionary containing 'api_key_ref' and 'api_base_url'
        max_tokens: Optional maximum number of tokens to generate
        
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
        return {
            "error": "Llama framework configuration not provided.",
            "content": "Error: Llama framework configuration not provided.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }

    api_key_ref = framework_config.get("api_key_ref")
    api_base_url = framework_config.get("api_base_url")
    # Force correct base URL if needed
    if api_base_url and api_base_url.startswith("https://api.llama.ai"):
        api_base_url = "https://api.llama.com/v1/chat/completions"

    if not api_key_ref:
        return {
            "error": "API key reference (api_key_ref) not found in Llama framework configuration.",
            "content": "Error: API key reference (api_key_ref) not found in Llama framework configuration.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }
    if not api_base_url:
        return {
            "error": "API base URL (api_base_url) not found in Llama framework configuration.",
            "content": "Error: API base URL (api_base_url) not found in Llama framework configuration.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }

        # Safely get and validate the API key
    api_key = get_api_key(api_key_ref)
    is_valid, error_msg = validate_api_key(api_key, "API key")
    if not is_valid:
        return {'error': error_msg}

    if not api_key:
        return {
            "error": f"API key for reference '{api_key_ref}' not found in st.secrets.",
            "content": f"Error: API key for reference '{api_key_ref}' not found in st.secrets.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }
    
    # Map model names if needed
    model_mapping = {
        # Add any model name mappings here if needed
        # "llama-3-8b": "llama-3-8b-instruct",
    }
    
    # Use mapped model name if it exists, otherwise use the provided model name
    model = model_mapping.get(model, model)
    
    # Prepare request headers: only Authorization per working curl
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Prepare messages (ensure they have the correct format)
    cleaned_messages = []
    for msg in messages:
        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
            error_msg = f"Invalid message format: {msg}"
            return {
                "error": error_msg,
                "content": f"Error: {error_msg}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "elapsed_time": 0
            }
        
        # Ensure valid role
        role = msg["role"]
        if role not in ["system", "user", "assistant"]:
            role = "user"  # Default to user for invalid roles
            
        cleaned_messages.append({
            "role": role,
            "content": msg["content"]
        })
    
    # Prepare request payload
    payload = {
        "model": model,
        "messages": cleaned_messages,
        "temperature": max(0.0, min(1.0, float(temperature))),  # Clamp to 0-1 range
    }
    
    # Add optional parameters if provided
    if max_tokens is not None:
        payload["max_completion_tokens"] = int(max_tokens)
    
    # Add any additional parameters from **params that are supported by the API
    # Example: payload.update({k: v for k, v in params.items() if k in ["stop", "n", "stream"]})
    
    # Make the API request
    start_time = time.time()
    try:
        # Debug: Print the full request details
        print(f"[DEBUG] Making request to: {api_base_url}")
        print(f"[DEBUG] Headers: {headers}")
        print(f"[DEBUG] Payload: {payload}")
        
        response = requests.post(
            api_base_url,  # Use the base URL directly from secrets
            headers=headers,
            json=payload,
            timeout=60  # 60 second timeout
        )
        
        # Debug: Print the response status and first 500 chars of response
        print(f"[DEBUG] Response status: {response.status_code}")
        print(f"[DEBUG] Response headers: {dict(response.headers)}")
        print(f"[DEBUG] Response content (first 500 chars): {response.text[:500]}")
        response.raise_for_status()
        data = response.json()
        
        # Extract the response content and token usage from the Llama API response format
        content = data.get("completion_message", {}).get("content", {})
        if isinstance(content, dict):
            content = content.get("text", "")
            
        # Extract token counts from metrics
        metrics = {m["metric"]: m["value"] for m in data.get("metrics", [])}
        
        return {
            "content": content,
            "prompt_tokens": metrics.get("num_prompt_tokens", 0),
            "completion_tokens": metrics.get("num_completion_tokens", 0),
            "elapsed_time": time.time() - start_time
        }
        
    except requests.exceptions.RequestException as e:
        # Debug: Print full exception details
        import traceback
        print(f"[DEBUG] Request exception: {str(e)}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        if hasattr(e, 'request'):
            print(f"[DEBUG] Request headers: {e.request.headers}")
            print(f"[DEBUG] Request body: {getattr(e.request, 'body', 'No body')}")
        error_msg = f"API request failed: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_msg = f"{error_msg} - {error_details}"
            except:
                error_msg = f"{error_msg} - {e.response.text}"
                
        return {
            "error": error_msg,
            "content": f"Error: {error_msg}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": time.time() - start_time
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return {
            "error": error_msg,
            "content": f"Error: {error_msg}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": time.time() - start_time
        }
