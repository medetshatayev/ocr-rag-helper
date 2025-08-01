import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import mimetypes

from rag_system import RAGSystem

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Farmon AI Agent",
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

    div[data-testid="stForm"] {
        position: relative;
    }

    div[data-testid="stForm"] button {
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 20px !important;
        background-color: #2196f3 !important;
        color: white !important;
        border: none !important;
        position: absolute !important;
        right: 10px;
        top: 0;
        bottom: 0;
        margin-top: auto;
        margin-bottom: auto;
    }
    
    div[data-testid="stForm"] button:hover {
        background-color: #1976d2 !important;
    }

    /* Add padding to the right of the input to make space for the button */
    div[data-testid="stForm"] div[data-testid="stTextInput"] input {
        padding-right: 60px !important;
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
    
    if "processing_file" not in st.session_state:
        st.session_state.processing_file = None
    if "clear_error" not in st.session_state:
        st.session_state.clear_error = None
    if "clear_error_time" not in st.session_state:
        st.session_state.clear_error_time = None
    if "saved_file_ids" not in st.session_state:
        st.session_state.saved_file_ids = set()

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
        
        if not is_user and sources:
            with st.expander("Sources", expanded=False):
                for i, source_info in enumerate(sources[:5]):
                    file_name = source_info.get('source')
                    if not file_name:
                        continue
                    
                    file_path = Path("Docs") / file_name
                    
                    st.markdown(f"""
                    <div class="source-info">
                        <strong>Source {i+1}:</strong> {Path(file_name).name}<br>
                    </div>
                    """, unsafe_allow_html=True)

                    if file_path.exists():
                        try:
                            with open(file_path, "rb") as fp:
                                mime = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
                                st.download_button(
                                    label=f"Download {Path(file_name).name}",
                                    data=fp,
                                    file_name=Path(file_name).name,
                                    key=f"download_{message_index}_{i}",
                                    mime=mime
                                )
                        except Exception as e:
                            st.warning(f"Could not load file for download: {e}")

def sidebar_content():
    """Render sidebar content."""
    
    # Document Management
    st.sidebar.subheader("Document Management")
    
    # Define and create Docs directory
    docs_dir = os.path.join(os.getcwd(), "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # Supported file extensions for consistency
    supported_extensions = ['.pdf', '.txt', '.md', '.py', '.js', '.html', '.css', 
                            '.json', '.xml', '.csv', '.yml', '.yaml']
    uploader_types = [ext.lstrip('.') for ext in supported_extensions]

    # Initialize dynamic key to reset uploader
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    is_processing = bool(st.session_state.get("processing_file"))

    uploaded_file = st.sidebar.file_uploader(
        "Upload Document",
        type=uploader_types,
        key=f"uploader_{st.session_state.uploader_key}",
        disabled=is_processing
    )

    if is_processing:
        st.sidebar.info(f"Processing {st.session_state.processing_file}... Please wait.")

    if uploaded_file:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if file_id not in st.session_state.saved_file_ids:
            try:
                docs_dir = "./Docs"
                os.makedirs(docs_dir, exist_ok=True)
                
                file_path = os.path.join(docs_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.processing_file = uploaded_file.name
                st.session_state.saved_file_ids.add(file_id)
                
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Error saving {uploaded_file.name}: {e}")
                st.session_state.processing_file = None

    if is_processing:
        try:
            docs_dir = "./Docs"
            file_path = os.path.join(docs_dir, st.session_state.processing_file)
            if os.path.exists(file_path):
                result = st.session_state.rag_system.index_file(
                    file_path=file_path,
                    document_id="dir_Docs",
                    original_filename=st.session_state.processing_file
                )
                st.session_state.indexing_status = result
            else:
                st.session_state.indexing_status = {"status": "error", "message": f"File {st.session_state.processing_file} not found."}
        finally:
            st.session_state.processing_file = None
            st.session_state.uploader_key += 1
            st.rerun()
    
    # Display indexing status: show errors only, suppress success message
    if st.session_state.indexing_status:
        status = st.session_state.indexing_status
        if status.get("status") != "success":
            st.sidebar.error(f"{status.get('message', 'Unknown error')}")
    
    st.sidebar.markdown("---")
    
    # List all files in Docs/
    st.sidebar.subheader("Library Documents")
    if st.session_state.rag_system:
        docs_files = [f.name for f in Path(docs_dir).iterdir() if f.is_file()]
        if docs_files:
            for file in docs_files:
                st.sidebar.write(f"- {file}")
        else:
            st.sidebar.write("No documents in library.")
    else:
        st.sidebar.write("System not initialized.")
    
    st.sidebar.markdown("---")
    
    # Clear functions
    st.sidebar.subheader("Clear Data")
    
    if st.sidebar.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.sidebar.button("Clear Library", use_container_width=True):
        errors = []
        if st.session_state.rag_system:
            # Clear database
            db_clear_result = st.session_state.rag_system.clear_database()
            if db_clear_result["status"] != "success":
                errors.append(f"Database Clear Error: {db_clear_result['message']}")
            else:
                st.session_state.indexing_status = None

            # Clear Docs folder
            docs_clear_result = st.session_state.rag_system.clear_docs_folder(docs_dir)
            if docs_clear_result["status"] != "success":
                errors.append(f"Docs Folder Clear Error: {docs_clear_result['message']}")
            else:
                st.session_state.saved_file_ids = set()

        if errors:
            st.session_state.clear_error = "\\n".join(errors)
            st.session_state.clear_error_time = time.time()
        else:
            st.session_state.clear_error = None
        
        st.rerun()

def main_chat_interface():
    """Main chat interface."""
    st.markdown('<h1 class="main-header">Farmon AI Agent</h1>', unsafe_allow_html=True)
    
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
        st.info("No documents found, upload new file to start")
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
            user_input = st.text_input(
                "Ask a question about your documents:",
                placeholder="What would you like to know?",
                label_visibility="collapsed",
                key=f"user_input_{st.session_state.input_counter}"
            )
            
            send_button = st.form_submit_button("↑")
            
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
                        query=user_input,
                        document_id="dir_Docs",
                        chat_history=chat_context[:-1]  # Exclude current message
                    )
                
                # Add assistant response to history
                assistant_message = {
                    "role": "assistant",
                    "content": response_data["answer"],
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