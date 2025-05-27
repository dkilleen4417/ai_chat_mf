import streamlit as st
from pymongo import MongoClient
import os

def get_database():
    """Get a database connection using credentials from Streamlit secrets."""
    try:
        mongodb_url = st.secrets["MONGODB_URL"]
        db_name = st.secrets["MONGODB_DB_NAME"]
        client = MongoClient(
            mongodb_url,
            maxPoolSize=50,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            retryWrites=True
        )
        # Test the connection
        client.server_info()
        return client[db_name]
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise

def main():
    # Load secrets from .streamlit/secrets.toml
    st.set_page_config(layout="wide")
    
    # Get database connection
    try:
        db = get_database()
        
        # Get all frameworks
        print("\n=== FRAMEWORKS ===")
        frameworks = list(db.frameworks.find({}, {"_id": 0}))
        for fw in frameworks:
            print(f"\nName: {fw.get('name')}")
            print(f"Display Name: {fw.get('display_name')}")
            print(f"API Base URL: {fw.get('api_base_url')}")
            print(f"API Key Ref: {fw.get('api_key_ref')}")
        
        # Get all models
        print("\n=== MODELS ===")
        models = list(db.models.find({}, {"_id": 0}))
        for model in models:
            print(f"\nName: {model.get('name')}")
            print(f"Framework: {model.get('framework')}")
            print(f"Model Name: {model.get('model_name')}")
            print(f"Max Tokens: {model.get('max_tokens')}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
