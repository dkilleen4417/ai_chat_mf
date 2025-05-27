# frameworks/groq.py
"""
Groq API implementation for chat completions.
"""
import streamlit as st
import requests
from typing import Dict, List, Optional
import time

# Import API key utilities
from utils.api_keys import get_api_key, validate_api_key

def process_chat(
    messages: List[Dict[str, str]], 
    model: str, 
    temperature: float, 
    top_p: float, 
    framework_config: Optional[Dict[str, str]] = None,
    max_tokens: int = 1024  
) -> Dict:
    """
    Process a chat request using the Groq API
    
    Args:
        messages: List of message objects with 'role' and 'content'
        model: Model name to use
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        framework_config: Dictionary containing 'api_key_ref' and 'api_base_url'
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary with standard response format or error details.
    """
    if not framework_config:
        return {
            "error": "Groq framework configuration not provided.",
            "content": "Error: Groq framework configuration not provided.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }

    api_key_ref = framework_config.get("api_key_ref")
    api_base_url = framework_config.get("api_base_url")

    if not api_key_ref:
        return {
            "error": "API key reference (api_key_ref) not found in Groq framework configuration.",
            "content": "Error: API key reference (api_key_ref) not found in Groq framework configuration.",
            "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": 0
        }
    if not api_base_url:
        return {
            "error": "API base URL (api_base_url) not found in Groq framework configuration.",
            "content": "Error: API base URL (api_base_url) not found in Groq framework configuration.",
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
    
    # Map model names if needed (in case app-specific model names don't match Groq's naming)
    groq_models = {
        "llama3-8b-8192": "llama3-8b-8192",
        "llama3-70b-8192": "llama3-70b-8192",
        "mixtral-8x7b-32768": "mixtral-8x7b-32768",
        "gemma-7b-it": "gemma-7b-it"
        # Add more mappings as needed
    }
    
    # If the model name doesn't match a Groq model, try to map it or use a default
    if model not in groq_models.values():
        # Check if we have a mapping for this model name
        if model in groq_models:
            groq_model = groq_models[model]
        else:
            # Default to a common Groq model if no mapping exists
            st.warning(f"Model '{model}' not recognized by Groq, using llama3-8b-8192 instead")
            groq_model = "llama3-8b-8192"
    else:
        groq_model = model
        
    # Map to appropriate Groq model
    
    # Prepare request parameters (temperature, top_p, max_tokens are now direct args)
    
    # Make API request
    start_time = time.time()
    try:
        # Clean and validate messages for Groq API
        cleaned_messages = []
        
        # Process each message to remove unsupported fields

        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                error_msg = f"Invalid message format: {msg}"
                # Invalid message format
                return {
                    "error": error_msg,
                    "content": f"Error: {error_msg}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed_time": 0
                }
            
            # Create a new clean message with only the fields Groq supports
            clean_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # Ensure only valid roles
            if clean_msg['role'] not in ['system', 'user', 'assistant']:
                clean_msg['role'] = 'user'  # Default to user for invalid roles
                
            cleaned_messages.append(clean_msg)
            
        # Prepare the request payload according to Groq API specs
        # See https://console.groq.com/docs/api-reference
        payload = {
            "model": groq_model,  # Use mapped/validated model name
            "messages": cleaned_messages,  # Use the cleaned messages without timestamp
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens # Use direct argument
        }
        
        # Send the payload to Groq API
        
        # Make the API request with better error handling
        response = requests.post(
            url=api_base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=payload,
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Check response status
        # Raise exception for HTTP errors
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: Unable to connect to Groq API. {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_msg += f" Details: {error_details}"
            except:
                error_msg += f" Status: {e.response.status_code}, Response: {e.response.text}"
        
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
            "content": f"Error: Invalid response from Groq API. {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_time": end_time - start_time
        }