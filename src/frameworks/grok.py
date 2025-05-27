# frameworks/grok.py
"""
Grok API implementation for chat completions.
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
    temperature: float, 
    top_p: float, 
    framework_config: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Process a chat request using the Grok API
    
    Args:
        messages: List of message objects with 'role' and 'content'
        model: Model name to use
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        framework_config: Dictionary containing 'api_key_ref' and 'api_base_url'
        
    Returns:
        Dictionary with standard response format or error details.
    """
    if not framework_config:
        return {
            "error": "Grok framework configuration not provided.",
            "content": "Error: Grok framework configuration not provided.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }

    api_key_ref = framework_config.get("api_key_ref")
    api_base_url = framework_config.get("api_base_url")

    if not api_key_ref:
        return {
            "error": "API key reference (api_key_ref) not found in Grok framework configuration.",
            "content": "Error: API key reference (api_key_ref) not found in Grok framework configuration.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }
    if not api_base_url:
        return {
            "error": "API base URL (api_base_url) not found in Grok framework configuration.",
            "content": "Error: API base URL (api_base_url) not found in Grok framework configuration.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }

        # Safely get and validate the API key
    api_key = get_api_key()
    is_valid, error_msg = validate_api_key(api_key, "API key")
    if not is_valid:
        return {'error': error_msg}

    if not api_key:
        return {
            "error": f"API key for reference '{api_key_ref}' not found in st.secrets.",
            "content": f"Error: API key for reference '{api_key_ref}' not found in st.secrets.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }
    
    # Prepare request parameters (temperature and top_p are now direct args)

    # Make API request
    start_time = time.time()
    try:
        # Prepare the payload
        payload = {
            "model": model,
            "messages": messages,  # Grok accepts timestamp and other fields
            "temperature": temperature, # Use direct argument
            "top_p": top_p            # Use direct argument
        }
        
        # Make the API request
        response = requests.post(
            url=api_base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=payload,
            timeout=60  # Add timeout to prevent hanging
        )
        response.raise_for_status()  # Raise exception for HTTP errors
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: Unable to connect to Grok API. {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_msg += f" Details: {error_details}"
            except:
                if e.response.text:
                    error_msg += f" Response: {e.response.text}"
                    
        return {
            "error": str(e),
            "content": error_msg,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": time.time() - start_time
        }
    
    end_time = time.time()
    
    # Process response
    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Extract token usage
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": end_time - start_time
        }
    except (KeyError, ValueError) as e:
        return {
            "error": str(e),
            "content": f"Error: Invalid response from Grok API. {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": end_time - start_time
        } 