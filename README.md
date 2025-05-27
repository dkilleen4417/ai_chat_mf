# AI Chat - Multi-Framework

A Streamlit-based AI chat application supporting multiple LLM providers with a focus on security, maintainability, and ease of use. This application provides a clean interface for interacting with various AI models while keeping your data secure and organized.

## 🚀 Key Features

- **Multi-Framework Support**: Seamlessly switch between different AI providers (OpenAI, Anthropic, Gemini, Groq, etc.)
- **Secure Configuration**: Sensitive data is managed through Streamlit's secrets system
- **Database Backed**: MongoDB integration for conversation history and application data
- **Backup System**: Built-in tools for backing up both database and source code
- **Modular Design**: Easy to extend with new AI frameworks and features
- **Responsive UI**: Clean, user-friendly interface with dark/light mode support

## 🏗️ Project Structure

```
.
├── .streamlit/               # Streamlit configuration
│   └── secrets.toml          # API keys and sensitive configuration (DO NOT COMMIT)
├── backups/                  # Auto-generated backup files
│   ├── ai_chat_mf_*/        # Database backups
│   └── projects_src_*.tar.gz # Source code backups
├── scripts/                  # Utility scripts
│   ├── backup_mongodb.py     # Database backup utility
│   └── backup_source.py      # Source code backup utility
├── src/                      # Source code
│   ├── frameworks/          # LLM provider implementations
│   │   ├── __init__.py
│   │   ├── anthropic.py
│   │   ├── chatgpt.py
│   │   ├── gemini.py
│   │   └── ...
│   ├── utils/               # Utility modules
│   │   ├── __init__.py
│   │   └── api_keys.py      # Secure API key management
│   ├── app.py               # Main application
│   └── config.py            # Non-sensitive configuration
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔒 Security Best Practices

1. **Secrets Management**
   - Store all sensitive information in `~/.streamlit/secrets.toml` (global) instead of project directory
   - Never commit secrets to version control
   - Rotate API keys if they are ever exposed
   - Use different API keys for different environments

2. **Backup System**
   - **Database Backup**: `python3 scripts/backup_mongodb.py`
   - **Source Code Backup**: `python3 scripts/backup_source.py`
   - **Backup Everything**: Run both commands above
   - Backups are stored in timestamped directories under `./backups/`

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- MongoDB (local or remote)
- API keys for desired AI providers

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dkilleen4417/ai_chat_mf.git
   cd ai_chat_mf
   ```

2. **Set up a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Secrets**
   Create or update your global Streamlit secrets file at `~/.streamlit/secrets.toml`:
   
   ```toml
   # Core Configuration
   MONGODB_URL = "mongodb://localhost:27017/"
   MONGODB_DB_NAME = "ai_chat_mf"
   
   # API Keys (replace with your actual keys)
   OPENAI_API_KEY = "your_openai_key"
   ANTHROPIC_API_KEY = "your_anthropic_key"
   GEMINI_API_KEY = "your_gemini_key"
   GROQ_API_KEY = "your_groq_key"
   SERPER_API_KEY = "your_serper_key"
   ```

### Running the Application

```bash
streamlit run src/app.py
```

### Managing Backups

#### Database Backup
```bash
python3 scripts/backup_mongodb.py
```

#### Source Code Backup
```bash
python3 scripts/backup_source.py
```

#### Backup Everything
```bash
python3 scripts/backup_mongodb.py
python3 scripts/backup_source.py
```

Backups are stored in the `./backups` directory with timestamps.

## 🛠️ Development

### Adding a New LLM Provider

1. **Create a new module** in `src/frameworks/` (e.g., `my_llm.py`)

2. **Implement the interface**:
   ```python
   from typing import Dict, List, Optional
   
   def process_chat(
       messages: List[Dict[str, str]],
       model: str,
       temperature: float,
       top_p: float,
       framework_config: Optional[Dict[str, str]] = None
   ) -> Dict[str, str]:
       """Process a chat request using the provider's API.
       
       Args:
           messages: List of message dicts with 'role' and 'content' keys
           model: Model identifier to use
           temperature: Sampling temperature (0.0 to 1.0)
           top_p: Nucleus sampling parameter
           framework_config: Provider-specific configuration
           
       Returns:
           Dictionary containing response and metadata
       """
       try:
           # Your implementation here
           return {
               "content": "Response from the model",
               "model": model,
               "provider": "my_llm"
           }
       except Exception as e:
           return {
               "error": str(e),
               "status": "error"
           }
   ```

3. **Add configuration** through the app's UI under "Manage Frameworks"

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- MongoDB for data persistence
- All the amazing AI providers and their teams

## 📝 Notes

- Always back up your data before making major changes
- Monitor your API usage to avoid unexpected charges
- Keep your dependencies updated for security and features