import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from rag_system import RAGSystem

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling for better readability and contrast
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f1f1f;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        color: #1f1f1f;
        line-height: 1.6;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
        border-left: 4px solid #4caf50;
    }
    
    .source-info {
        background-color: #fffde7;
        border: 1px solid #ffb300;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #333;
    }
    
    /* Ensure proper text color in messages */
    .chat-message strong {
        color: #1565c0;
    }
    
    .chat-message span {
        color: #1f1f1f !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "indexing_status" not in st.session_state:
        st.session_state.indexing_status = None
    
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False

def check_api_key():
    """Check if Azure OpenAI API key is configured."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or api_key == "your_api_key" or not endpoint:
        st.error("Azure OpenAI credentials not configured!")
        st.info("Please set your AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in the .env file.")
        return False
    
    st.session_state.api_key_valid = True
    return True

def initialize_rag_system():
    """Initialize the RAG system."""
    try:
        if st.session_state.rag_system is None:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem()
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return False

def display_chat_message(message: Dict, message_index: int, sources: List[Dict] = None):
    """Display a chat message with styling."""
    is_user = message["role"] == "user"
    
    css_class = "user-message" if is_user else "assistant-message"
    role_name = "You" if is_user else "Assistant"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{role_name}:</strong><br>
            <span style="line-height: 1.6;">{message["content"]}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources for assistant messages
        if not is_user and sources:
            with st.expander("Sources", expanded=False):
                for i, source in enumerate(sources):
                    file_path = Path(source['file'])
                    file_name = file_path.name
                    
                    st.markdown(f"""
                    <div class="source-info">
                        <strong>Source {i+1}:</strong> {file_name}<br>
                        <strong>Page:</strong> {source.get('page', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)

                    # Add a download button for the source file
                    if file_path.exists():
                        try:
                            with open(file_path, "rb") as fp:
                                st.download_button(
                                    label=f"Download {file_name}",
                                    data=fp,
                                    file_name=file_name,
                                    mime=f"application/{file_path.suffix.lstrip('.')}",
                                    key=f"download_{message_index}_{i}_{file_name}_{source.get('page', 'N/A')}"
                                )
                        except Exception as e:
                            st.error(f"Could not read file for download: {e}")
                    else:
                        st.warning(f"Source file not found: {file_name}")

def sidebar_content():
    """Render sidebar content."""
    
    # Document Management
    st.sidebar.subheader("Document Management")
    
    # Define and create Docs directory
    docs_dir = os.path.join(os.getcwd(), "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # Supported file extensions for consistency
    supported_extensions = ['.pdf', '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.yml', '.yaml']
    uploader_types = [ext.lstrip('.') for ext in supported_extensions]

    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        type=uploader_types,
        accept_multiple_files=True,
        help="Upload files to be indexed by the RAG system."
    )

    # State management for uploaded files to prevent re-saving
    if "saved_file_ids" not in st.session_state:
        st.session_state.saved_file_ids = set()

    if uploaded_files:
        new_files_to_save = []
        current_file_ids = set()
        
        for f in uploaded_files:
            file_id = f"{f.name}-{f.size}"
            current_file_ids.add(file_id)
            if file_id not in st.session_state.saved_file_ids:
                new_files_to_save.append(f)
        
        if new_files_to_save:
            saved_count = 0
            for uploaded_file in new_files_to_save:
                try:
                    file_path = os.path.join(docs_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
                    st.session_state.saved_file_ids.add(file_id)
                    saved_count += 1
                except Exception as e:
                    st.sidebar.error(f"Error saving {uploaded_file.name}: {e}")
            
            if saved_count > 0:
                st.sidebar.success(f"Successfully saved {saved_count} file(s).")
                # Rerun to update the file list below and button states
                st.rerun()
        
        # Sync session state with uploader
        st.session_state.saved_file_ids.intersection_update(current_file_ids)

    # Show files in Docs directory
    doc_files = []
    try:
        for file_path in Path(docs_dir).glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                doc_files.append(file_path.name)
        
        if not doc_files:
            st.sidebar.warning("No supported documents found. Upload files to get started.")
            
    except Exception as e:
        st.sidebar.error(f"Error scanning Docs folder: {e}")

    # Index documents button
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Index Documents", use_container_width=True, disabled=not doc_files):
            if st.session_state.rag_system:
                with st.spinner("Indexing documents from Docs folder..."):
                    result = st.session_state.rag_system.index_directory(docs_dir)
                    st.session_state.indexing_status = result
                st.rerun()
    
    with col2:
        if st.button("Force Reindex", use_container_width=True, disabled=not doc_files):
            if st.session_state.rag_system:
                with st.spinner("Force reindexing documents from Docs folder..."):
                    result = st.session_state.rag_system.index_directory(docs_dir, force_reindex=True)
                    st.session_state.indexing_status = result
                st.rerun()
    
    # Display indexing status
    if st.session_state.indexing_status:
        status = st.session_state.indexing_status
        if status["status"] == "success":
            st.sidebar.success(f"{status['message']}")
            st.sidebar.info(f"Files processed: {status.get('files_processed', 0)}")
            st.sidebar.info(f"Chunks created: {status.get('chunks_created', 0)}")
        else:
            st.sidebar.error(f"{status['message']}")
    
    st.sidebar.markdown("---")
    
    # Database Statistics
    st.sidebar.subheader("Database Stats")
    
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_database_stats()
        
        if "error" not in stats:
            st.sidebar.write(f"**Total documents:** {stats.get('total_documents', 0)}")
            st.sidebar.write(f"**Unique sources:** {stats.get('unique_sources', 0)}")
        else:
            st.sidebar.error("Error loading stats")
    
    st.sidebar.markdown("---")
    
    # Clear functions
    st.sidebar.subheader("Clear Data")
    
    if st.sidebar.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.sidebar.button("Clear Database", use_container_width=True):
        if st.session_state.rag_system:
            result = st.session_state.rag_system.clear_database()
            if result["status"] == "success":
                st.sidebar.success("Database cleared!")
                st.session_state.indexing_status = None
            else:
                st.sidebar.error(f"Error: {result['message']}")
        st.rerun()

def main_chat_interface():
    """Main chat interface."""
    st.markdown('<h1 class="main-header">RAG Chat Assistant</h1>', unsafe_allow_html=True)
    
    # Check if system is ready
    if not st.session_state.api_key_valid:
        st.warning("Please configure your Azure OpenAI API key to start chatting.")
        return
    
    if not st.session_state.rag_system:
        st.warning("RAG system not initialized. Please wait or check logs.")
        return
    
    # Check if documents are indexed
    stats = st.session_state.rag_system.get_database_stats()
    if stats.get('total_documents', 0) == 0:
        st.info("No documents indexed yet. Use the 'Index Documents' button in the sidebar to index documents from your 'Docs' folder.")
        return
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            sources = message.get("sources", []) if message["role"] == "assistant" else None
            display_chat_message(message, i, sources)
    
    # Chat input using form to avoid session state conflicts
    with st.container():
        st.markdown("---")
        
        # Initialize input counter to clear input after submission
        if "input_counter" not in st.session_state:
            st.session_state.input_counter = 0
        
        with st.form("chat_form", clear_on_submit=True):
            # Create columns for input and button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask a question about your documents:",
                    placeholder="What would you like to know?",
                    label_visibility="collapsed",
                    key=f"user_input_{st.session_state.input_counter}"
                )
            
            with col2:
                send_button = st.form_submit_button("Send", use_container_width=True)
            
            # Process user input
            if send_button and user_input.strip():
                # Increment counter to clear input on next render
                st.session_state.input_counter += 1
                
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Generate response
                with st.spinner("Thinking..."):
                    # Prepare chat history for context (last few messages)
                    chat_context = []
                    for msg in st.session_state.chat_history[-10:]:  # Last 10 messages
                        if msg["role"] in ["user", "assistant"]:
                            chat_context.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                    
                    response_data = st.session_state.rag_system.generate_response(
                        user_input, 
                        chat_context[:-1]  # Exclude current message
                    )
                
                # Add assistant response to history
                assistant_message = {
                    "role": "assistant",
                    "content": response_data["response"],
                    "sources": response_data.get("sources", [])
                }
                st.session_state.chat_history.append(assistant_message)
                
                # Rerun to refresh the interface
                st.rerun()

def main():
    """Main application function."""
    initialize_session_state()
    
    # Check API key
    if not check_api_key():
        return
    
    # Initialize RAG system
    if not initialize_rag_system():
        return
    
    # Create layout
    sidebar_content()
    main_chat_interface()

if __name__ == "__main__":
    main() 