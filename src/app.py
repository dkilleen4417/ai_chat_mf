from time import time as current_time, sleep
import os
import sys
import requests
from pymongo import MongoClient
import streamlit as st

# Add the src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import configuration
from config import (
    OPENAI_ENDPOINT,
    GEMINI_ENDPOINT,
    LANGCHAIN_ENDPOINT,
    WEATHERFLOW_API_ENDPOINT,
    GROQ_ENDPOINT,
    OLLAMA_ENDPOINT,
    CODESTRAL_CHAT_ENDPOINT,
    CODESTRAL_COMP_ENDPOINT,
    WEATHERFLOW_STATION_ID,
    DEFAULT_SETTINGS
)

st.set_page_config(
    page_icon="💬", 
    layout="wide", 
    page_title="AI Chat - Multi-Framework",
    initial_sidebar_state="expanded",
    menu_items=None)

ss = st.session_state

import sys

def get_db_source():
    """
    Determine which database to use based on a command-line argument.
    Usage: streamlit run src/app.py -- --db_source atlas  (or 'local')
    Defaults to 'local' if not provided.
    """
    db_source = "local"
    for i, arg in enumerate(sys.argv):
        if arg == "--db_source" and i + 1 < len(sys.argv):
            db_source = sys.argv[i + 1].lower()
    return db_source

def get_database():
    """Get a database connection using credentials from Streamlit secrets and runtime parameter."""
    db_source = get_db_source()
    if db_source == "atlas":
        mongodb_url = st.secrets["MONGO_ATLAS_URI"].format(
            username=st.secrets["MONGO_ATLAS_USER"],
            password=st.secrets["MONGO_ATLAS_CLUSTER_PW"],
            host=st.secrets["MONGO_ATLAS_HOST"]
        )
        db_name = st.secrets["MONGO_ATLAS_DB_NAME"]
    else:
        mongodb_url = st.secrets["MONGO_LOCAL_URI"]
        db_name = st.secrets["MONGO_LOCAL_DB_NAME"]
    client = MongoClient(
        mongodb_url,
        maxPoolSize=50,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        retryWrites=True
    )
    client.server_info()  # Test the connection
    return client[db_name]

def initialize():
    ss.initialized = True
    ss.db = get_database()
    ss.show_metrics = True
    ss.llm_avatar = "🤖"
    ss.user_avatar = "😎"
    ss.use_web_search = False
    ss.web_search_results = 3
    ss.edit_model_name = None
    ss.edit_model_data = None
    ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
    active_model_name = ss.active_chat.get("model")
    model_info = ss.db.models.find_one({"name": active_model_name})
    ss.active_framework = model_info.get("framework") if model_info else None

def get_friendly_time(seconds_ago):
    time_actions = {
        lambda d: d < 60: lambda d: "Just now",
        lambda d: d < 3600: lambda d: f"{int(d / 60)}m ago",
        lambda d: d < 86400: lambda d: f"{int(d / 3600)}h ago",
        lambda d: d < 172800: lambda d: "Yesterday",
        lambda d: d < 604800: lambda d: f"{int(d / 86400)}d ago",
        lambda d: d < 2592000: lambda d: f"{int(d / 604800)}w ago",
        lambda d: True: lambda d: "Long ago"
    }
    for condition, action in time_actions.items():
        if condition(seconds_ago):
            return action(seconds_ago)
            
def update_active_framework():
    active_model_name = ss.active_chat.get("model")
    if not active_model_name:
        ss.active_framework = None
        return
        
    model_info = ss.db.models.find_one({"name": active_model_name})
    if not model_info:
        # If model doesn't exist, clear the active model
        ss.active_chat["model"] = None
        ss.active_framework = None
    else:
        ss.active_framework = model_info.get("framework")

def search_web(query):
    """Perform a web search using the Serper API.
    
    Args:
        query (str): The search query
        
    Returns:
        str: Formatted search results or empty string if no results
    """
    try:
        if "SERPER_API_KEY" not in st.secrets:
            st.error("SERPER_API_KEY not found in secrets")
            return ""
            
        # Using Serper API for Google Search results
        response = requests.get(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": st.secrets["SERPER_API_KEY"],
                "Content-Type": "application/json"
            },
            params={
                "q": query,
                "num": ss.web_search_results  # Use the user's preferred number of results
            }
        ) 
        if response.status_code != 200:
            st.error(f"Web search failed with status code: {response.status_code}")
            return ""
            
        data = response.json()
        results = []
        
        # Extract organic search results
        organic = data.get('organic', [])
        for result in organic[:ss.web_search_results]:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            link = result.get('link', '')
            # Serper organic results use 'snippet' for summary
            if title and snippet:
                if link:
                    results.insert(0, f"[{title}]({link})\n{snippet}")
                else:
                    results.insert(0, f"{title}\n{snippet}")
                
        if not results:
            st.info("No relevant search results found")
            return ""
            
        # Format results as a bulleted list
        context = "\n".join([f"- {result}" for result in results])
        return context        
    except Exception as e:
        st.error(f"Web search error: {str(e)}")
        return ""

def manage_sidebar():
    st.sidebar.markdown("### :blue[Active Chat] 🎯")
    st.sidebar.markdown(f"**Chat Name:** :blue[{ss.active_chat['name']}]")
    st.sidebar.markdown(f"**Model:** :blue[{ss.active_chat['model']}]")
    st.sidebar.markdown(f"**Framework:** :blue[{ss.active_framework}]")
    
    web_search = st.sidebar.toggle('Enable Web Search 🔍', value=ss.use_web_search, key='web_search_toggle', help="Enhance responses with web search")
    if web_search != ss.use_web_search:
        ss.use_web_search = web_search
        st.rerun()
    # Show number input only if web search is enabled
    if ss.use_web_search:
        st.sidebar.number_input(
            "Number of Web Search Results",
            min_value=1,
            max_value=10,
            value=3,  # Default to 3 results
            key='web_search_results',
            help="Number of web search results to display"
        )

    st.sidebar.markdown("### :blue[Select Chat] 📚")
    col1, col2 = st.sidebar.columns([7, 1])
    with col1:
        if st.button("💬 Scratch Pad", key="default_chat", use_container_width=True):
            ss.active_chat = ss.db.chats.find_one({"name": "Scratch Pad"})
            update_active_framework()
            st.rerun()
    with col2:
        if st.button("🧹", key="clear_default", help="Clear Scratch Pad history"):
            ss.db.chats.update_one({"name": "Scratch Pad"},{"$set": {"messages": []}})
            st.rerun()

    # Create and sense if clicked the current chat if not the default chat (Scratch Pad) And the clear button
    if ss.active_chat["name"] != "Scratch Pad":
        friendly_time = get_friendly_time(current_time() - ss.active_chat.get("updated_at", ss.active_chat.get("created_at", 0)))
        col1, col2 = st.sidebar.columns([7, 1])
        with col1:
            st.button(f"👉 {ss.active_chat['name']} • {friendly_time}", key="current_chat", use_container_width=True)
        with col2:
            if st.button("🧹", key="clear_current", help=f"Clear {ss.active_chat['name']} history"):
                ss.db.chats.update_one({"name": ss.active_chat['name']},{"$set": {"messages": []}})
                st.rerun()

    # Get list all the chats in the DB that are not archived, default, or active
    chats = list(ss.db.chats.find({"archived": False}, {"name": 1, "created_at": 1, "updated_at": 1}))
    other_chats = [c for c in chats if c["name"] not in ["Scratch Pad", ss.active_chat["name"]]]
    
    # Add a divider if there are other chats
    if other_chats:
        st.sidebar.divider()
        # Sort by updated_at, most recent first
        other_chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        for chat in other_chats:
            friendly_time = get_friendly_time(current_time() - chat.get("updated_at", chat.get("created_at", 0)))
            col1, col2 = st.sidebar.columns([7, 1])
            with col1:
                if st.button(f"💬 {chat['name']} • {friendly_time}", key=f"chat_{chat['name']}", use_container_width=True):
                    # Update updated_at field in DB
                    ss.db.chats.update_one(
                        {"name": chat["name"]},
                        {"$set": {"updated_at": current_time()}}
                    )
                    ss.active_chat = ss.db.chats.find_one({"name": chat["name"]})
                    update_active_framework()
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{chat['name']}", help=f"Delete {chat['name']}"):
                    ss.db.chats.delete_one({"name": chat["name"]})
                    st.rerun()

    # Web search results number input


def get_chat_response():
    fresh_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
    messages = fresh_chat["messages"].copy()
    messages.insert(0, {"role": "system", "content": fresh_chat["system_prompt"]})
    
    # Get the last user message
    last_user_message = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
    
    # Enhance with web search if enabled
    web_results = ""
    if ss.use_web_search and last_user_message:
        web_results = search_web(last_user_message["content"])
        if web_results:
            # Find the index of the last user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Prepend web search results to the user's message content
                    messages[i]["content"] = f"[Web search results]\n{web_results}\n\n" + messages[i]["content"]
                    break

    model_info = ss.db.models.find_one({"name": fresh_chat["model"]})
    if not model_info:
        st.error(f"Model '{fresh_chat['model']}' not found in database")
        return None
        
    # Get the framework information for this model
    framework_name = model_info.get("framework", "")
    
    if not framework_name:
        st.error(f"Model '{fresh_chat['model']}' has no associated framework")
        return None
    
    # Fetch the framework configuration from the database
    framework_config = ss.db.frameworks.find_one({"name": framework_name})
    if not framework_config:
        st.error(f"Framework configuration for '{framework_name}' not found in database.")
        return None
        
    # Ensure required fields exist in framework_config for Anthropic
    if framework_name.lower() == 'anthropic':
        if 'api_key_ref' not in framework_config:
            framework_config['api_key_ref'] = 'ANTHROPIC_API_KEY'
        if 'api_base_url' not in framework_config:
            framework_config['api_base_url'] = 'https://api.anthropic.com/v1/messages'

    start_time = current_time()
    
    # Import the appropriate framework module
    try:
        # Dynamic import of the framework module
        try:
            # First try the new path (src.frameworks)
            framework_module = __import__(f"src.frameworks.{framework_name}", fromlist=["process_chat"])
        except ModuleNotFoundError:
            # Fall back to the old path (frameworks)
            framework_module = __import__(f"frameworks.{framework_name}", fromlist=["process_chat"])
            
        # Get parameters for the model
        temperature = model_info.get("temperature", 0.7)
        top_p = model_info.get("top_p", 0.9)
        max_tokens = model_info.get("max_tokens", 1024)  # Fetch max_tokens, default 1024
        
    except ModuleNotFoundError as e:
        st.error(f"Framework module for '{framework_name}' not found. Error: {e}")
        st.error(f"Python path: {os.sys.path}")
        return "Error: Framework module not found or could not be loaded.", 0.0
    except ImportError:
        st.error(f"Framework module for '{framework_name}' not found")
        return None
        
    # Call the framework-specific processing function
    try:

        
        # Call the framework-specific processing function with only the expected parameters
        result = framework_module.process_chat(
            messages=messages,
            model=fresh_chat["model"],
            temperature=temperature,
            top_p=top_p,
            framework_config=framework_config  # Pass the fetched config
        )
        
        if not result or 'error' in result:
            st.error(f"Error from {framework_name}: {result.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error processing chat: {str(e)}")
        return None

    except Exception as e:
        st.error(f"Error processing chat: {str(e)}")
        return None

    # If the framework returned None (indicating an error)
    if result is None:
        st.error("Failed to get response from the model. Please check the framework configuration or API key.")
        return None

    # If the framework returned an error
    if isinstance(result, dict) and "error" in result:
        st.error(f"Error from {framework_name} framework: {result['error']}")
        return None

    # Defensive: ensure required keys are present
    required_keys = ["content", "prompt_tokens", "completion_tokens", "elapsed_time"]
    for k in required_keys:
        if k not in result:
            st.error(f"Framework '{framework_name}' returned incomplete response (missing {k}).")
            return None

    # Extract the response content
    content = result["content"]
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    elapsed_time = result["elapsed_time"]
    
    # Prepare both messages for the chat history
    messages_to_add = []

    # If we had search results, show them as a visible assistant message (for chat display)
    if web_results:
        web_results_message = {
            "role": "assistant",
            "content": f"🔍 Web Search Results:\n{web_results}",
            "timestamp": current_time(),
            "is_search_result": True
        }
        messages_to_add.append(web_results_message)

    # Add the AI response
    response_message = {"role": "assistant", "content": content, "timestamp": current_time()}
    messages_to_add.append(response_message)
    
    # Update MongoDB with all new messages
    ss.db.chats.update_one(
        {"name": ss.active_chat['name']}, 
        {"$push": {"messages": {"$each": messages_to_add}}}
    )
    
    # Calculate cost based on token usage
    tokens = prompt_tokens + completion_tokens
    model_pricing = ss.db.models.find_one({"name": ss.active_chat["model"]}, {"input_price": 1, "output_price": 1, "_id": 0})
    input_cost = (prompt_tokens / 1_000_000) * model_pricing.get("input_price", 0)
    output_cost = (completion_tokens / 1_000_000) * model_pricing.get("output_price", 0)
    total_cost = input_cost + output_cost
    messages_per_dollar = 100 / total_cost if total_cost > 0 else 0
    
    return {
        "text": content,
        "time": elapsed_time,
        "tokens": tokens,
        "rate": tokens / elapsed_time if elapsed_time > 0 else 0,
        "cost": total_cost,
        "messages_per_dollar": messages_per_dollar
    }

def render_chat_tab():
    message_container = st.container(height=565, border=True)
    ss.active_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
    update_active_framework()
    messages = ss.active_chat.get("messages", [])

    for msg in messages:
        avatar = ss.llm_avatar if msg["role"] == "assistant" else ss.user_avatar
        with message_container.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
    prompt = st.chat_input("Type your message...")
    if prompt:
        message = {"role": "user","content": prompt,"timestamp": current_time()}
        ss.db.chats.update_one({"name": ss.active_chat['name']}, {"$push": {"messages": message}})
        # Refresh the active_chat after adding the user message
        ss.active_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
        update_active_framework()
        with message_container.chat_message("user", avatar=ss.user_avatar):
            st.markdown(prompt)
        if response_data := get_chat_response():
            # Display any new messages that were added (search results and AI response)
            fresh_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
            # Get the last 2 messages if we have search results (identified by is_search_result flag)
            last_messages = fresh_chat["messages"][-2:]
            has_search = any(msg.get("is_search_result") for msg in last_messages)
            new_messages = last_messages if has_search else fresh_chat["messages"][-1:]
            
            for msg in new_messages:
                if msg.get("is_search_result"):
                    with message_container.chat_message("assistant", avatar="🔍"):
                        st.markdown(msg["content"])
                else:
                    with message_container.chat_message("assistant", avatar=ss.llm_avatar):
                        st.markdown(msg["content"])
            # Refresh the active_chat after getting response
            ss.active_chat = ss.db.chats.find_one({"name": ss.active_chat['name']})
            update_active_framework()
            if ss.show_metrics:
                st.info(
                    f"Time: {response_data['time']:.2f}s | "
                    f"Tokens: {response_data['tokens']} | "
                    f"Speed: {response_data['rate']:.1f} T/s | "
                    f"Cost: ${response_data['cost']:.4f} | "
                    f"Messages/Dollar: {format(response_data['messages_per_dollar'], ',.0f')}"
                )

def render_new_chat_tab():
    st.markdown("### Create New Chat 🆕")
    with st.form("new_chat_form", clear_on_submit=True):
        new_chat_name = st.text_input(
            "Chat Name",
            placeholder="Enter chat name...",
            help="Enter a unique name for your new chat"
        ).strip()
        try:
            # Get models with their framework information
            db_models = list(ss.db.models.find({}, {"name": 1, "framework": 1, "_id": 0}))
            # Create a list of tuples with (display_name, model_name)
            available_models = [
                (f"{model['name']} ({model.get('framework', 'No Framework')})", model['name'])
                for model in db_models
            ]
            # If no models found, show empty list
            available_models = available_models if db_models else []
        except Exception as e:
            st.error(f"Error fetching models: {str(e)}")
            available_models = []
        
        # Create a mapping of display names to model names
        model_mapping = {m[0]: m[1] for m in available_models} if available_models else {}
        
        # Show the dropdown with display names
        selected_display = st.selectbox(
            "Select Model",
            options=list(model_mapping.keys()) if model_mapping else ["No models available"],
            format_func=lambda x: x,
            help="Choose model - different models have different capabilities"
        )
        
        # Get the actual model name from the mapping
        model = model_mapping.get(selected_display) if model_mapping else None
        
        try:
            db_prompts = list(ss.db.prompts.find())
            available_prompts = [(p["name"], p["content"]) for p in db_prompts]
        except Exception as e:
            st.error(f"Error fetching prompts: {str(e)}")
            available_prompts = []
            
        selected_prompt = st.selectbox(
            "Select System Prompt",
            options=[p[0] for p in available_prompts],
            help="Choose the system prompt that defines how the AI should behave"
        )
        
        # Show the content of the selected prompt
        if selected_prompt:
            prompt_content = next((p[1] for p in available_prompts if p[0] == selected_prompt), "")
            st.text_area("System Prompt Content", value=prompt_content, disabled=True, height=150)
        
        submitted = st.form_submit_button("Create Chat", use_container_width=True)
        if submitted:
            if not new_chat_name:
                st.error("Please enter a chat name")
            elif ss.db.chats.find_one({"name": new_chat_name}):
                st.error("A chat with this name already exists")
            else:
                current_time_val = current_time()
                
                # Get the selected prompt content
                system_prompt = next((p[1] for p in available_prompts if p[0] == selected_prompt), "")
                
                new_chat = {
                    "name": new_chat_name,
                    "model": model,
                    "system_prompt": system_prompt,
                    "messages": [],
                    "created_at": current_time_val,
                    "updated_at": current_time_val,
                    "archived": False
                }
                ss.db.chats.insert_one(new_chat)
                new_chat = ss.db.chats.find_one({"name": new_chat_name})
                if new_chat:
                    ss.active_chat = new_chat
                    update_active_framework()
                    st.success(f"Chat '{new_chat_name}' created successfully!")
                    st.rerun()

def render_archive_tab():
    st.markdown("### Archive Management 📂")
    st.markdown("Toggle archive status for your chats. Archived chats won't appear in the sidebar.")
    st.divider()
    
    # Retrieve all chats except Scratch Pad
    all_chats = list(ss.db.chats.find({"name": {"$ne": "Scratch Pad"}}, {"name": 1, "archived": 1}))
    
    # Display each chat with its archive status
    for chat in all_chats:
        col1, col2, col3 = st.columns([3, 3, 2])
        archived_status = chat.get('archived', False)
        with col1:
            st.markdown(f"**Chat Name:** :blue[{chat['name']}]")
        with col2:
            st.markdown(f"**Archived:** :blue[{archived_status}]")
        with col3:
            # Use a checkbox to toggle the archived status
            toggle = st.checkbox("Archived", value=archived_status, key=f"toggle_{chat['name']}", help="Check to archive this chat")
            if toggle != archived_status:
                ss.db.chats.update_one({"_id": chat["_id"]}, {"$set": {"archived": toggle}})
                st.rerun()  # Refresh to update the list

def render_models_tab():
    st.markdown("### Model Management 🤖")
    
    # Horizontal radio button for model actions
    model_action = st.radio(
        "Select Model Action", 
        ["Add", "Edit", "Delete"], 
        horizontal=True
    )
    
    # Add Model functionality
    if model_action == "Add":
        with st.form("add_model_form", clear_on_submit=True):
            # Basic Model Information
            col_name, col_framework = st.columns(2)
            with col_name:
                model_name = st.text_input("Model Name", placeholder="Enter model name...")
            with col_framework:
                # Get frameworks from database
                try:
                    frameworks = list(ss.db.frameworks.find({}, {"name": 1, "display_name": 1, "_id": 0}))
                    framework_options = [fw["display_name"] for fw in frameworks]
                    framework_map = {fw["display_name"]: fw["name"] for fw in frameworks}
                    
                    if not framework_options:
                        st.error("No frameworks available. Please add frameworks first.")
                        framework_display_name = ""
                    else:
                        framework_display_name = st.selectbox("Framework", options=framework_options)
                except Exception as e:
                    st.error(f"Error loading frameworks: {str(e)}")
                    framework_display_name = ""
                    framework_map = {}
            
            # Model Capabilities
            col1, col2 = st.columns(2)
            with col1:
                text_input = st.checkbox("Text Input", value=True)
                image_input = st.checkbox("Image Input")
                text_output = st.checkbox("Text Output", value=True)
                image_output = st.checkbox("Image Output")
            
            with col2:
                tools = st.checkbox("Tools")
                functions = st.checkbox("Functions")
                thinking = st.checkbox("Thinking")
            
            # Model Parameters
            col3, col4 = st.columns(2)
            with col3:
                temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
            
            with col4:
                max_input_tokens = st.number_input("Max Input Tokens", min_value=0, value=131072)
                max_output_tokens = st.number_input("Max Output Tokens", min_value=0, value=8192)
            
            # Pricing
            col5, col6 = st.columns(2)
            with col5:
                input_price = st.number_input("Input Price (per million tokens)", min_value=0.0, value=2.0, format="%.2f")
            
            with col6:
                output_price = st.number_input("Output Price (per million tokens)", min_value=0.0, value=10.0, format="%.2f")
            
            submitted = st.form_submit_button("Add Model")
            
            if submitted:
                if not model_name:
                    st.error("Model Name is required!")
                else:
                    # Get framework name from selected display name
                    framework = framework_map.get(framework_display_name, "")
                    
                    if not framework:
                        st.error("Please select a valid framework")
                        return
                    
                    # Prepare model document
                    new_model = {
                        "name": model_name,
                        "framework": framework,  # Add the framework field
                        "temperature": temperature,
                        "top_p": top_p,
                        "input_price": input_price,
                        "output_price": output_price,
                        "text_input": text_input,
                        "image_input": image_input,
                        "text_output": text_output,
                        "image_output": image_output,
                        "tools": tools,
                        "functions": functions,
                        "thinking": thinking,
                        "max_output_tokens": max_output_tokens,
                        "max_input_tokens": max_input_tokens,
                        "created_at": current_time()
                    }
                    
                    # Check if model already exists
                    existing_model = ss.db.models.find_one({"name": model_name})
                    if existing_model:
                        st.error(f"Model '{model_name}' already exists!")
                    else:
                        # Insert new model
                        ss.db.models.insert_one(new_model)
                        st.success(f"Model '{model_name}' added successfully!")
                        st.balloons()
                        # Show message for 2 seconds before refresh
                        sleep(2)
                        st.rerun()
    
    # Edit Model functionality - Step 1: Select Model
    if model_action == "Edit":
        if 'edit_model_name' not in ss or ss.edit_model_name is None:
            # Step 1: Model Selection with Framework Info
            available_models = list(ss.db.models.find({}, {"name": 1, "framework": 1, "_id": 0}))
            
            if not available_models:
                st.warning("No models available for editing.")
                return
                
            # Create display names with framework info
            model_display_names = [
                f"{model['name']} ({model.get('framework', 'No Framework')})" 
                for model in available_models
            ]
            model_name_map = {display: model["name"] for display, model in zip(model_display_names, available_models)}
                
            selected_display = st.selectbox(
                "Select Model to Edit", 
                model_display_names,
                key="model_selector"
            )
            selected_model = model_name_map[selected_display]
            
            if st.button('Edit Selected Model'):
                ss.edit_model_name = selected_model
                ss.edit_model_data = ss.db.models.find_one({"name": selected_model})
                st.rerun()
            return
            
        # Step 2: Edit Form
        current_model = ss.edit_model_data
        model_to_edit = ss.edit_model_name
        
        with st.form("edit_model_form"):
            # Framework selection
            try:
                frameworks = list(ss.db.frameworks.find({}, {"name": 1, "display_name": 1, "_id": 0}))
                framework_options = [fw["display_name"] for fw in frameworks]
                framework_map = {fw["name"]: fw["display_name"] for fw in frameworks}
                reverse_framework_map = {fw["display_name"]: fw["name"] for fw in frameworks}
                
                current_framework = current_model.get("framework", "")
                
                current_framework_display = framework_map.get(current_framework, "")
                
                if not framework_options:
                    st.error("No frameworks available. Please add frameworks first.")
                    framework_display_name = ""
                else:
                    framework_display_name = st.selectbox(
                        "Framework", 
                        options=framework_options,
                        index=framework_options.index(current_framework_display) if current_framework_display in framework_options else 0
                    )
            except Exception as e:
                st.error(f"Error loading frameworks: {str(e)}")
                framework_display_name = ""
                reverse_framework_map = {}
            
            # Model Capabilities
            st.subheader("Model Capabilities")
            col1, col2 = st.columns(2)
            with col1:
                text_input = st.checkbox("Text Input", value=current_model.get("text_input", False))
                image_input = st.checkbox("Image Input", value=current_model.get("image_input", False))
                text_output = st.checkbox("Text Output", value=current_model.get("text_output", True))
                image_output = st.checkbox("Image Output", value=current_model.get("image_output", False))
            
            with col2:
                tools = st.checkbox("Tools", value=current_model.get("tools", False))
                functions = st.checkbox("Functions", value=current_model.get("functions", False))
                thinking = st.checkbox("Thinking", value=current_model.get("thinking", False))
            
            # Model Parameters
            st.subheader("Model Parameters")
            col3, col4 = st.columns(2)
            with col3:
                temperature = st.slider(
                    "Temperature", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=current_model.get("temperature", 0.7), 
                    step=0.05
                )
                top_p = st.slider(
                    "Top P", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=current_model.get("top_p", 0.9), 
                    step=0.05
                )
            
            with col4:
                max_input_tokens = st.number_input(
                    "Max Input Tokens", 
                    min_value=0, 
                    value=current_model.get("max_input_tokens", 4096)
                )
                max_output_tokens = st.number_input(
                    "Max Output Tokens", 
                    min_value=0, 
                    value=current_model.get("max_output_tokens", 4096)
                )
            
            # Pricing
            st.subheader("Pricing")
            col5, col6 = st.columns(2)
            with col5:
                input_price = st.number_input(
                    "Input Price (per million tokens)", 
                    min_value=0.0, 
                    value=current_model.get("input_price", 0.0), 
                    format="%.2f"
                )
            
            with col6:
                output_price = st.number_input(
                    "Output Price (per million tokens)", 
                    min_value=0.0,
                    value=current_model.get("output_price", 0.0),
                    format="%.2f"
                )
            
            # Form actions
            col7, col8 = st.columns(2)
            with col7:
                if st.form_submit_button("Update Model"):
                    # Get framework name from selected display name
                    framework = reverse_framework_map.get(framework_display_name, "")
                    
                    if not framework:
                        st.error("Please select a valid framework")
                    else:
                        # Update model document
                        update_data = {
                            "framework": framework,
                            "temperature": temperature,
                            "top_p": top_p,
                            "input_price": input_price,
                            "output_price": output_price,
                            "text_input": text_input,
                            "image_input": image_input,
                            "text_output": text_output,
                            "image_output": image_output,
                            "tools": tools,
                            "functions": functions,
                            "thinking": thinking,
                            "max_output_tokens": max_output_tokens,
                            "max_input_tokens": max_input_tokens,
                            "updated_at": current_time()
                        }
                        
                        # Update the model
                        ss.db.models.update_one(
                            {"name": model_to_edit},
                            {"$set": update_data}
                        )
                        st.success(f"Model '{model_to_edit}' updated successfully!")
                        st.balloons()  # Visual feedback
                        ss.edit_model_name = None
                        ss.edit_model_data = None
                        st.rerun()
            
            with col8:
                if st.form_submit_button("Cancel"):
                    ss.edit_model_name = None
                    ss.edit_model_data = None
                    st.rerun()
    
    # Delete Model functionality
    if model_action == "Delete":
        # Retrieve all models except the default with their frameworks
        available_models = list(ss.db.models.find(
            {"name": {"$ne": "grok-2-latest"}}, 
            {"name": 1, "framework": 1, "_id": 0}
        ))
        
        if not available_models:
            st.warning("No models available for deletion.")
        else:
            # Create display names with framework info
            model_display_names = [
                f"{model['name']} ({model.get('framework', 'No Framework')})" 
                for model in available_models
            ]
            model_name_map = {display: model["name"] for display, model in zip(model_display_names, available_models)}
            
            with st.form("delete_model_form", clear_on_submit=True):
                selected_display = st.selectbox(
                    "Select Model to Delete", 
                    model_display_names,
                    help="Note: 'grok-2-latest' cannot be deleted"
                )
                model_to_delete = model_name_map[selected_display]
                
                submitted = st.form_submit_button("Delete Model")
                
                if submitted:
                    # Double-check the model is not the default
                    if model_to_delete == "grok-2-latest":
                        st.error("Cannot delete the default model 'grok-2-latest'.")
                    else:
                        # Perform deletion
                        result = ss.db.models.delete_one({"name": model_to_delete})
                        
                        if result.deleted_count > 0:
                            # Clear any edit state if the deleted model was being edited
                            if ss.get('edit_model_name') == model_to_delete:
                                ss.edit_model_name = None
                                ss.edit_model_data = None
                            # Show success message and animation
                            st.success(f"Model '{model_to_delete}' deleted successfully!")
                            st.balloons()
                            # Show message for 2 seconds before refresh
                            sleep(2)
                            st.rerun()
                        else:
                            st.error(f"Could not delete model '{model_to_delete}'.")

def render_prompts_tab():
    st.markdown("### System Prompt Management 📝")
    
    # Horizontal radio button for prompt actions
    prompt_action = st.radio(
        "Select Prompt Action", 
        ["Add", "Edit", "Delete"], 
        horizontal=True
    )
    
    st.divider()
    
    # Add Prompt functionality
    if prompt_action == "Add":
        with st.form("add_prompt_form", clear_on_submit=True):
            # Basic Prompt Information
            prompt_name = st.text_input("Prompt Name", placeholder="Enter prompt name...")
            prompt_description = st.text_input("Description", placeholder="Brief description of the prompt...")
            prompt_content = st.text_area(
                "Prompt Content", 
                placeholder="Enter system prompt content...",
                height=300
            )
            
            submitted = st.form_submit_button("Add Prompt")
            
            if submitted:
                if not prompt_name:
                    st.error("Prompt Name is required!")
                elif not prompt_content:
                    st.error("Prompt Content is required!")
                else:
                    # Prepare prompt document
                    new_prompt = {
                        "name": prompt_name,
                        "description": prompt_description,
                        "content": prompt_content,
                        "created_at": current_time()
                    }
                    
                    # Check if prompt already exists
                    existing_prompt = ss.db.prompts.find_one({"name": prompt_name})
                    if existing_prompt:
                        st.error(f"Prompt '{prompt_name}' already exists!")
                    else:
                        # Insert new prompt
                        ss.db.prompts.insert_one(new_prompt)
                        st.success(f"Prompt '{prompt_name}' added successfully!")
    
    # Edit Prompt functionality
    if prompt_action == "Edit":
        # Retrieve all prompts
        available_prompts = list(prompt["name"] for prompt in ss.db.prompts.find())
        
        if not available_prompts:
            st.warning("No prompts available for editing.")
        else:
            with st.form("edit_prompt_form", clear_on_submit=False):
                # Prompt selection
                prompt_to_edit = st.selectbox(
                    "Select Prompt to Edit", 
                    available_prompts
                )
                
                # Retrieve current prompt details
                current_prompt = ss.db.prompts.find_one({"name": prompt_to_edit})
                
                # Prompt description
                prompt_description = st.text_input(
                    "Description", 
                    value=current_prompt.get("description", "")
                )
                
                # Prompt content
                prompt_content = st.text_area(
                    "Prompt Content", 
                    value=current_prompt.get("content", ""),
                    height=300
                )
                
                submitted = st.form_submit_button("Save Prompt")
                
                if submitted:
                    if not prompt_content:
                        st.error("Prompt Content is required!")
                    else:
                        # Prepare updated prompt document
                        updated_prompt = {
                            "name": prompt_to_edit,
                            "description": prompt_description,
                            "content": prompt_content,
                            "updated_at": current_time()
                        }
                        
                        # Preserve created_at if it exists
                        if "created_at" in current_prompt:
                            updated_prompt["created_at"] = current_prompt["created_at"]
                        
                        # Update the prompt in the database
                        ss.db.prompts.replace_one({"name": prompt_to_edit}, updated_prompt)
                        st.success(f"Prompt '{prompt_to_edit}' updated successfully!")
    
    # Delete Prompt functionality
    if prompt_action == "Delete":
        # Retrieve all prompts except the default
        available_prompts = list(prompt["name"] for prompt in ss.db.prompts.find({"name": {"$ne": "Default System Prompt"}}))
        
        if not available_prompts:
            st.warning("No prompts available for deletion.")
        else:
            with st.form("delete_prompt_form", clear_on_submit=True):
                prompt_to_delete = st.selectbox(
                    "Select Prompt to Delete", 
                    available_prompts,
                    help="Note: 'Default System Prompt' cannot be deleted"
                )
                
                submitted = st.form_submit_button("Delete Prompt")
                
                if submitted:
                    # Double-check the prompt is not the default
                    if prompt_to_delete == "Default System Prompt":
                        st.error("Cannot delete the 'Default System Prompt'.")
                    else:
                        # Confirm deletion
                        confirm = st.checkbox("I understand this action cannot be undone")
                        if confirm:
                            # Perform deletion
                            result = ss.db.prompts.delete_one({"name": prompt_to_delete})
                            
                            if result.deleted_count > 0:
                                st.success(f"Prompt '{prompt_to_delete}' deleted successfully!")
                            else:
                                st.error(f"Could not delete prompt '{prompt_to_delete}'.")

def render_publish_tab():
    st.markdown("### Publish 📢")
    st.warning("🚧 Publish Functionality is currently under construction. Stay tuned for exciting features!")
    
    st.markdown("#### Upcoming Features:")
    st.markdown("- AI Editor Review")
    st.markdown("  - Spelling and Grammar Corrections")
    st.markdown("  - Text Editing (Add, Remove, Reorder)")
    st.markdown("  - Content Outlining")
    st.markdown("- Audio Podcast Generation")
    st.markdown("  - Two-Party Discussion Conversion")
    
    st.info("We're working on transforming your chats into polished, professional content!")

def manage_frameworks():
    """Manage frameworks in the database."""
    st.header("Manage Frameworks")

    # Tabs for different actions
    tab1, tab2, tab3 = st.tabs(["Add", "Edit", "Delete"]) # Simplified tab names
    
    with tab1:
        with st.form(key="add_framework_form", clear_on_submit=True):
            st.subheader("Add New Framework")
            name = st.text_input("Framework Name", help="Internal name used in code (e.g., 'gemini')")
            display_name = st.text_input("Display Name", help="User-friendly name (e.g., 'Gemini')")
            api_base_url_input = st.text_input("API Base URL", help="Standardized field: api_base_url (e.g., 'https://api.example.com/v1'). Must start with http:// or https://")
            api_key_ref_input = st.text_input("API Key Environment Variable", help="Standardized field: api_key_ref. Name of env variable (e.g., 'GEMINI_API_KEY')")
            
            if st.form_submit_button("Add Framework"):
                if not name or not display_name or not api_base_url_input or not api_key_ref_input:
                    st.error("All fields are required.")
                elif not (api_base_url_input.startswith('http://') or api_base_url_input.startswith('https://')):
                    st.error("API Base URL must start with http:// or https://")
                elif ss.db.frameworks.find_one({"name": name}):
                    st.error(f"Framework with name '{name}' already exists.")
                else:
                    framework_data = {
                        "name": name,
                        "display_name": display_name,
                        "api_base_url": api_base_url_input, # Standardized field name
                        "api_key_ref": api_key_ref_input   # Standardized field name
                    }
                    ss.db.frameworks.insert_one(framework_data)
                    st.success(f"Framework '{display_name}' added successfully!")
                    st.rerun()
    
    with tab2:
        st.subheader("Existing Frameworks")
        frameworks = list(ss.db.frameworks.find({}, {"_id": 0}))
        if not frameworks:
            st.info("No frameworks found in the database.")
        else:
            framework_options = {
                f"{fw.get('display_name', 'Unnamed')} ({fw.get('name', 'Unknown')})": fw.get('name') 
                for fw in frameworks
            }
            selected_fw_display_name = st.selectbox(
                "Select Framework to Edit", 
                options=list(framework_options.keys()), 
                index=None, 
                placeholder="Choose a framework..."
            )

            if selected_fw_display_name:
                selected_fw_name = framework_options[selected_fw_display_name]
                fw = next((f for f in frameworks if f.get('name') == selected_fw_name), None)

                if fw:
                    st.markdown(f"--- ")
                    st.markdown(f"#### Editing: {fw.get('display_name', 'Unnamed Framework')} (`{fw.get('name', 'Unknown')}`)")
                    # st.json(fw) # Removed temporary debug line
                    
                    with st.form(key=f"edit_framework_{fw.get('name', 'unknown')}", clear_on_submit=True):
                        # Read with fallbacks for backward compatibility
                        val_display_name = fw.get('display_name')
                        val_api_url_to_use = fw.get('api_base_url') or fw.get('api_url') or fw.get('base_url')
                        val_api_key_ref_to_use = fw.get('api_key_ref') or fw.get('api_key_name')

                        edit_display_name = st.text_input("New Display Name", value=val_display_name if val_display_name is not None else '')
                        # Label clearly indicates the target standard field name
                        edit_api_base_url = st.text_input("New API Base URL", value=val_api_url_to_use if val_api_url_to_use is not None else '')
                        edit_api_key_ref = st.text_input(
                            "New API Key Environment Variable", 
                            value=val_api_key_ref_to_use if val_api_key_ref_to_use is not None else '', 
                            help="Name of the environment variable holding the API key (e.g., 'GEMINI_API_KEY')"
                        )
                        if st.form_submit_button("Update Framework"): 
                            update_payload = {
                                "display_name": edit_display_name,
                                "api_base_url": edit_api_base_url,  # Standardized field name
                                "api_key_ref": edit_api_key_ref     # Standardized field name
                            }
                            # Prepare to unset old/alternative field names
                            unset_payload = {}
                            if 'api_url' in fw: unset_payload['api_url'] = ""
                            if 'base_url' in fw: unset_payload['base_url'] = ""
                            if 'api_key_name' in fw: unset_payload['api_key_name'] = ""

                            operation = {"$set": update_payload}
                            if unset_payload:
                                # Ensure we don't try to unset a field if it's the same as the new standard one (edge case, but good practice)
                                if 'api_base_url' in unset_payload: del unset_payload['api_base_url'] 
                                if 'api_key_ref' in unset_payload: del unset_payload['api_key_ref']
                                if unset_payload: # only add $unset if there's something to unset
                                   operation["$unset"] = unset_payload

                            ss.db.frameworks.update_one(
                                {"name": fw.get('name', 'unknown')},
                                operation
                            )
                            st.success(f"Framework '{edit_display_name}' updated!")
                            # Clear selection or rerun to reflect changes
                            st.rerun()
                else:
                    st.error("Selected framework not found. This should not happen.")
    
    with tab3:
        st.subheader("Delete Existing Framework")
        st.warning("⚠️ Deleting a framework is a permanent action and cannot be undone.")

        frameworks_to_delete = list(ss.db.frameworks.find({}, {"_id": 0})) 
        if not frameworks_to_delete:
            st.info("No frameworks found in the database to delete.")
        else:
            framework_del_options = {
                f"{fw_del.get('display_name', 'Unnamed')} ({fw_del.get('name', 'Unknown')})": fw_del.get('name') 
                for fw_del in frameworks_to_delete
            }
            selected_fw_del_display_name = st.selectbox(
                "Select Framework to Delete", 
                options=list(framework_del_options.keys()), 
                index=None, 
                placeholder="Choose a framework...",
                key="delete_fw_selectbox" # Unique key for this selectbox
            )

            if selected_fw_del_display_name:
                selected_fw_del_name = framework_del_options[selected_fw_del_display_name]
                fw_del = next((f for f in frameworks_to_delete if f.get('name') == selected_fw_del_name), None)

                if fw_del:
                    fw_name = fw_del.get('name', 'Unknown')
                    fw_display_name = fw_del.get('display_name', 'Unnamed Framework')
                    
                    st.markdown(f"--- ")
                    st.markdown(f"#### Selected for Deletion: {fw_display_name} (`{fw_name}`)")
                    st.markdown(f"**API URL:** {fw_del.get('api_base_url', 'Not specified')}")
                    st.markdown(f"**API Key Ref:** {fw_del.get('api_key_ref', 'Not specified')}")

                    confirm_key = f"confirm_delete_{fw_name}"
                    if confirm_key not in ss:
                        ss[confirm_key] = False

                    if ss[confirm_key]:
                        st.error(f"Are you sure you want to delete the framework '{fw_display_name}' (`{fw_name}`)?")
                        col_confirm, col_cancel, _ = st.columns([1,1,3])
                        with col_confirm:
                            if st.button("Yes, Delete Permanently", key=f"confirm_delete_btn_{fw_name}", type="primary"):
                                result = ss.db.frameworks.delete_one({"name": fw_name})
                                if result.deleted_count > 0:
                                    st.success(f"Framework '{fw_display_name}' deleted successfully.")
                                else:
                                    st.error(f"Failed to delete framework '{fw_display_name}'. It might have been already deleted.")
                                ss[confirm_key] = False 
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_delete_btn_{fw_name}"):
                                ss[confirm_key] = False 
                                st.rerun()
                    else:
                        if st.button("Initiate Delete Framework", key=f"delete_fw_btn_{fw_name}"):
                            ss[confirm_key] = True
                            st.rerun()
                else:
                    st.error("Selected framework not found for deletion. This should not happen.")

def manage_menu():
    tab_actions = {
        "💬 Chat": render_chat_tab,
        "🆕 New Chat": render_new_chat_tab,
        "🗂️ Archive": render_archive_tab,
        "🤖 Models": render_models_tab,
        "📝 Prompts": render_prompts_tab,
        "📢 Publish": render_publish_tab,
        "🔩 Frameworks": manage_frameworks
    }
    
    tabs = st.tabs(list(tab_actions.keys()))
    
    for tab, (label, render_func) in zip(tabs, tab_actions.items()):
        with tab:
            render_func()

def main():
    if 'initialized' not in ss:
        initialize()
    manage_sidebar()
    manage_menu()

if __name__ == "__main__":
    main()