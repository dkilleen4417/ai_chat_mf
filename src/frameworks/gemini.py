# frameworks/gemini.py

"""
Gemini Framework Module
This module handles chat processing using Google's Gemini API.
"""

import time
from typing import Dict, List, Optional
import google.generativeai as genai
import streamlit as st

# Import API key utilities
from utils.api_keys import get_api_key, validate_api_key

# Note: GEMINI_ENDPOINT is now imported from config for non-sensitive settings


def process_chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    framework_config: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, str]]:
    """
    Process a chat using the Gemini API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name (e.g., 'gemini-1.5-pro')
        temperature: Temperature parameter for response randomness
        top_p: Top-p parameter for nucleus sampling
        framework_config: Dictionary containing 'api_key_ref'
    
    Returns:
        The response text from the model, or None if an error occurs
    """
    if not framework_config:
        print("Error: Gemini framework_config not provided.")
        return {"error": "Gemini framework configuration not provided."}

    # Get API key from configuration
    api_key_ref = framework_config.get("api_key_ref")
    if not api_key_ref:
        return {"error": "API key reference (api_key_ref) not found in Gemini framework configuration."}

    # Safely get and validate the API key
    api_key = get_api_key(api_key_ref)
    is_valid, error_msg = validate_api_key(api_key, "Gemini API key")
    if not is_valid:
        return {"error": f"Invalid Gemini API key: {error_msg}"}

    try:
        # Configure Gemini client
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
    except Exception as e:
        return {"error": f"Failed to initialize Gemini client: {str(e)}"}

    # Prepare messages (Gemini does not support 'system' role)
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "system":
            continue  # Gemini does not support system role
        role = msg["role"]
        if role not in ["user", "assistant"]:
            raise ValueError("Invalid role. Only 'user' and 'assistant' roles are allowed.")
        if role == "assistant":
            role = "model"
        formatted_messages.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    # Start timing
    start_time = time.time()
    try:
        # Use Gemini's chat API for multi-turn conversations
        # The last user message is the new input, the rest is history
        if not formatted_messages:
            raise ValueError("No valid messages to send to Gemini.")
        # Separate history and latest user message
        history = formatted_messages[:-1]
        latest_message = formatted_messages[-1]["parts"][0]["text"] if formatted_messages[-1]["role"] == "user" else None
        if latest_message is None:
            raise ValueError("Last message must be from user for chat input.")
        chat = model_obj.start_chat(history=history)
        response = chat.send_message(latest_message, generation_config={
            "temperature": temperature,
            "top_p": top_p
        })
        elapsed_time = time.time() - start_time
        text = response.text
        # Estimate token counts (1 token ≈ 4 characters for English text)
        def estimate_tokens(s: str) -> int:
            return max(1, int(len(s) / 4))
        prompt_text = " ".join([msg["content"] for msg in messages if msg["role"] != "system"])
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(text)
        return {
            "content": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time
        }
    except Exception as api_error:
        print(f"[Gemini] API error: {api_error}")
        import traceback
        traceback.print_exc()
        return {"error": str(api_error)}
