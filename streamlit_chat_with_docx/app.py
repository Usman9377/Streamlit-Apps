import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime
# .venv environment made under python
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Chat with Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stChatMessage { border-radius: 10px; padding: 1rem; }
    .upload-section { background-color: #f0f2f6; padding: 2rem; border-radius: 10px; }
    .info-box { background-color: #e3f2fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #1976d2; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "document_content" not in st.session_state:
    st.session_state.document_content = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None

@st.cache_data
def load_document(file_content: bytes, filename: str) -> str:
    """Cache document content for faster retrieval"""
    return file_content.decode('utf-8')

def save_conversation(conv_id: str, messages: list, doc_name: str):
    """Save conversation to session state"""
    st.session_state.conversations[conv_id] = {
        "messages": messages,
        "document": doc_name,
        "created": datetime.now().isoformat()
    }

def restore_conversation(conv_id: str):
    """Restore a previous conversation"""
    if conv_id in st.session_state.conversations:
        conv = st.session_state.conversations[conv_id]
        st.session_state.current_conversation_id = conv_id
        st.session_state.document_name = conv["document"]
        st.rerun()

def create_system_prompt(document_content: str) -> str:
    """Create an optimized system prompt for document Q&A"""
    return f"""You are a helpful document assistant. You have been provided with the following document:

<document>
{document_content}
</document>

Your role is to answer questions about this document accurately and concisely. 
- If the answer is in the document, provide it with relevant context.
- If the question cannot be answered from the document, clearly state that.
- Keep responses focused and well-organized.
- Use bullet points or numbered lists when appropriate."""

def chat_with_document(messages: list, document_content: str) -> str:
    """Send messages to OpenAI with document context"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_message = create_system_prompt(document_content)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            *messages
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Sidebar for navigation and settings
with st.sidebar:
    st.title("📚 Document Chat Assistant")
    
    api_key = st.text_input("OpenAI API Key", type="password", 
                            value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    
    st.subheader("📋 Conversation History")
    if st.session_state.conversations:
        for conv_id, conv_data in st.session_state.conversations.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📄 {conv_data['document']}", key=f"conv_{conv_id}"):
                    restore_conversation(conv_id)
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    del st.session_state.conversations[conv_id]
                    if st.session_state.current_conversation_id == conv_id:
                        st.session_state.current_conversation_id = None
                    st.rerun()
    else:
        st.info("No conversations yet. Upload a document to start!")
    
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.current_conversation_id = None
        st.session_state.document_content = None
        st.session_state.document_name = None
        st.rerun()

# Main content area
st.title("💬 Chat with Your Documents")

# Upload section
if st.session_state.document_content is None:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload a Text Document")
    uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"])
    
    if uploaded_file:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Please enter your OpenAI API Key in the sidebar")
        else:
            with st.spinner("Processing document..."):
                content = load_document(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.document_content = content
                st.session_state.document_name = uploaded_file.name
                st.session_state.current_conversation_id = datetime.now().isoformat()
                st.session_state.conversations[st.session_state.current_conversation_id] = {
                    "messages": [],
                    "document": uploaded_file.name,
                    "created": datetime.now().isoformat()
                }
            st.success(f"✅ Document '{uploaded_file.name}' loaded successfully!")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Chat interface
else:
    st.markdown(f'<div class="info-box"><strong>📄 Current Document:</strong> {st.session_state.document_name}</div>', 
                unsafe_allow_html=True)
    
    # Get current conversation
    conv_id = st.session_state.current_conversation_id
    messages = st.session_state.conversations[conv_id]["messages"]
    
    # Display chat history
    st.subheader("Chat History")
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your document...")
    
    if user_input:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Please enter your OpenAI API Key in the sidebar")
        else:
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                try:
                    response = chat_with_document(messages, st.session_state.document_content)
                    messages.append({"role": "assistant", "content": response})
                    save_conversation(conv_id, messages, st.session_state.document_name)
                    
                    with st.chat_message("assistant"):
                        st.write(response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
