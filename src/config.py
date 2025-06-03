"""
Application configuration settings.

This file contains non-sensitive configuration settings that can be safely versioned.
Sensitive information like API keys should be kept in secrets.toml.
"""

# API Endpoints (non-sensitive)
OPENAI_ENDPOINT = "https://api.openai.com/v1"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
WEATHERFLOW_API_ENDPOINT = "https://swd.weatherflow.com/swd/rest/observations/station"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1"
OLLAMA_ENDPOINT = "http://localhost:11434/v1"
CODESTRAL_CHAT_ENDPOINT = "https://codestral.mistral.ai/v1/chat/completions"
CODESTRAL_COMP_ENDPOINT = "https://codestral.mistral.ai/v1/fim/completions"
GROK_ENDPOINT = "https://api.x.ai/v1"  # xAI's Grok API endpoint

# Application Settings (non-sensitive)
WEATHERFLOW_STATION_ID = "137684"

# Default settings that might be overridden by environment variables or config files
DEFAULT_SETTINGS = {
    "ollama_model": "llama3",
    "ollama_temperature": 0.7,
    "openai_model": "gpt-4-turbo",
    "openai_temperature": 0.7,
    "gemini_model": "gemini-1.5-pro",
    "gemini_temperature": 0.7,
}

# Add any other non-sensitive configuration here
