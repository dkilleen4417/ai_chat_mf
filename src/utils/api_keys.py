"""
Utility functions for managing API keys.

This module provides functions to safely retrieve API keys from Streamlit secrets
with proper error handling and fallback mechanisms.
"""
import streamlit as st
from typing import Optional, Tuple

def get_api_key(key_name: str, secret_name: Optional[str] = None) -> Optional[str]:
    """
    Safely retrieve an API key from Streamlit secrets.
    
    Args:
        key_name: The name of the API key to retrieve
        secret_name: Optional name of the secret section (defaults to key_name in root)
        
    Returns:
        The API key if found, None otherwise
    """
    try:
        if secret_name:
            # Handle nested secrets (e.g., st.secrets["section"]["key"])
            if "." in secret_name:
                parts = secret_name.split(".")
                secret_value = st.secrets
                for part in parts:
                    secret_value = secret_value.get(part, {})
                return secret_value.get(key_name)
            return st.secrets.get(secret_name, {}).get(key_name)
        return st.secrets.get(key_name)
    except Exception as e:
        st.error(f"Error accessing secret {key_name}: {e}")
        return None

def validate_api_key(key: Optional[str], key_name: str) -> Tuple[bool, str]:
    """
    Validate that an API key exists and is in the correct format.
    
    Args:
        key: The API key to validate
        key_name: The name of the API key (for error messages)
        
    Returns:
        A tuple of (is_valid, error_message)
    """
    if not key:
        return False, f"{key_name} not found in secrets"
    if not isinstance(key, str):
        return False, f"{key_name} must be a string"
    if not key.strip():
        return False, f"{key_name} is empty"
    return True, ""
