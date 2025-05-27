"""
ChatGPT Framework Module
Supports OpenAI's GPT-4 and compatible models via the OpenAI API.
"""

import time
import json
import traceback
import streamlit as st
import requests
from typing import Dict, List, Optional, Any, Tuple

# Import API key utilities
from utils.api_keys import get_api_key, validate_api_key

def log_error(error_type: str, message: str, details: Any = None) -> Dict[str, str]:
    """Helper function to log errors in a consistent format."""
    error_data = {
        "error": error_type,
        "message": message,
        "details": str(details) if details else None
    }
    print(f"[ChatGPT] ERROR: {json.dumps(error_data, indent=2)}")
    return error_data

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
        # Validate framework configuration
        if not framework_config:
            return log_error("invalid_config", "Framework configuration is missing or invalid")
            
        api_key_ref = framework_config.get("api_key_ref")
        api_base_url = framework_config.get("api_base_url")

        if not api_key_ref:
            return log_error("missing_config", "API key reference not found", "api_key_ref is required in framework configuration")
            
        if not api_base_url:
            return log_error("missing_config", "API base URL not found", "api_base_url is required in framework configuration")

        # Safely get and validate the API key
        print(f"[ChatGPT] Getting API key with reference: {api_key_ref}")
        api_key = get_api_key(api_key_ref)
        
        if not api_key:
            return log_error("missing_key", "API key not found", f"API key for reference '{api_key_ref}' not found in st.secrets")
            
        is_valid, error_msg = validate_api_key(api_key, "API key")
        if not is_valid:
            return log_error("invalid_key", "API key validation failed", error_msg)
            
        print("[ChatGPT] API key validation successful")

        # Construct the full URL - ensure we don't double-append /chat/completions
        base_url = api_base_url.rstrip('/')
        if not base_url.endswith('/chat/completions'):
            base_url = f"{base_url}/chat/completions"
        url = base_url

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
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": st.secrets.get("OPENAI_ID", ""),
        }
        
        print(f"[ChatGPT] Making request to: {url}")
        print(f"[ChatGPT] Headers: { {k: '***' if 'key' in k.lower() or 'auth' in k.lower() else v for k, v in headers.items()} }")

        try:
            # Make the API request
            start_time = time.time()
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Log response status and headers
            print(f"[ChatGPT] Response status: {response.status_code}")
            print(f"[ChatGPT] Response headers: {dict(response.headers)}")
            
            # Try to parse JSON response
            try:
                data = response.json()
                print(f"[ChatGPT] Response data: {json.dumps(data, indent=2)[:500]}...")  # Log first 500 chars
            except Exception as e:
                print(f"[ChatGPT] Error parsing JSON response: {e}")
                print(f"[ChatGPT] Response text: {response.text[:1000]}")
                raise
                
            response.raise_for_status()

            # Extract response data
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected response format: {type(data)}")
                
            if "choices" not in data or not data["choices"]:
                raise ValueError("No choices in response")
                
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_time": time.time() - start_time
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus Code: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f"\nError: {json.dumps(error_data, indent=2)}"
                except:
                    error_msg += f"\nResponse: {e.response.text[:500]}"
            return log_error("api_error", error_msg, traceback.format_exc())
        except Exception as e:
            return log_error("processing_error", f"Error processing API response: {str(e)}", traceback.format_exc())
    except Exception as e:
        print(f"Error in ChatGPT API call: {str(e)}")
        return None
