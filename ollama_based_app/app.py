import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        justify-content: flex-end;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .message-content {
        max-width: 70%;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Ollama URL input
    st.session_state.ollama_url = st.text_input(
        "Ollama Server URL",
        value=st.session_state.ollama_url,
        help="Default: http://localhost:11434"
    )
    
    # Model selection
    try:
        response = requests.get(f"{st.session_state.ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]
            if models:
                st.session_state.selected_model = st.selectbox(
                    "Select Model",
                    models,
                    key="model_selector"
                )
            else:
                st.warning("No models found. Please download a model using: `ollama pull model_name`")
        else:
            st.error("Failed to connect to Ollama server")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to Ollama. Make sure it's running on " + st.session_state.ollama_url)
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
    
    # Temperature slider
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat history cleared!")
    
    st.markdown("---")
    st.markdown("""
    ### 📌 How to use:
    1. Ensure Ollama is running
    2. Download a model: `ollama pull llama2`
    3. Select the model from dropdown
    4. Start chatting!
    
    ### Popular Models:
    - `llama2` - General purpose
    - `mistral` - Fast & efficient
    - `neural-chat` - Conversational
    - `openchat` - High quality
    """)

# Main chat interface
st.title("🤖 Ollama Chatbot")
st.markdown("Chat with your locally running Ollama models")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    if not st.session_state.selected_model:
        st.error("Please select a model first!")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response from Ollama
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                with st.spinner("Thinking..."):
                    response = requests.post(
                        f"{st.session_state.ollama_url}/api/generate",
                        json={
                            "model": st.session_state.selected_model,
                            "prompt": prompt,
                            "temperature": temperature,
                            "stream": True
                        },
                        stream=True,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line)
                                full_response += data.get("response", "")
                                message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                    else:
                        st.error(f"Error from Ollama: {response.status_code}")
                        full_response = "Error generating response"
            
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model may be taking too long.")
                full_response = "Error: Request timed out"
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Ollama server. Make sure it's running!")
                full_response = "Error: Connection failed"
            except Exception as e:
                st.error(f"Error: {str(e)}")
                full_response = f"Error: {str(e)}"
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
