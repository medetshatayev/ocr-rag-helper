import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict

from rag_system import RAGSystem

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
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        st.error("OpenAI API key not configured! Please set your OPENAI_API_KEY in the .env file.")
        st.info("Create a .env file in your project directory with: OPENAI_API_KEY=your_actual_api_key")
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

def display_chat_message(message: Dict, sources: List[Dict] = None):
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
                    st.markdown(f"""
                    <div class="source-info">
                        <strong>Source {i+1}:</strong> {Path(source['file']).name}<br>
                        <strong>Page:</strong> {source.get('page', 'N/A')}<br>
                        <strong>Type:</strong> {source.get('content_type', 'text')}<br>
                        <strong>Similarity:</strong> {source.get('similarity_score', 0):.3f}
                    </div>
                    """, unsafe_allow_html=True)

def sidebar_content():
    """Render sidebar content."""
    st.sidebar.title("RAG Assistant Settings")
    
    # API Key status
    if st.session_state.api_key_valid:
        st.sidebar.success("OpenAI API key configured")
    else:
        st.sidebar.error("OpenAI API key missing")
    
    st.sidebar.markdown("---")
    
    # Document Management
    st.sidebar.subheader("Document Management")
    
    # Docs directory info
    docs_dir = os.path.join(os.getcwd(), "Docs")
    docs_exists = os.path.exists(docs_dir)
    
    if docs_exists:
        st.sidebar.success(f"**Docs Directory Found:**\n{docs_dir}")
        
        # Show files in Docs directory
        try:
            doc_files = []
            for file_path in Path(docs_dir).glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.yml', '.yaml']:
                    doc_files.append(file_path.name)
            
            if doc_files:
                st.sidebar.info(f"Found {len(doc_files)} document(s):\n• " + "\n• ".join(doc_files[:5]))
                if len(doc_files) > 5:
                    st.sidebar.info(f"... and {len(doc_files) - 5} more")
            else:
                st.sidebar.warning("No supported documents found in Docs folder")
        except Exception as e:
            st.sidebar.error(f"Error scanning Docs folder: {e}")
    else:
        st.sidebar.error(f"**Docs Directory Not Found:**\nPlease create a 'Docs' folder and add your documents")
    
    # Index documents button
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Index Documents", use_container_width=True, disabled=not docs_exists):
            if st.session_state.rag_system and docs_exists:
                with st.spinner("Indexing documents from Docs folder..."):
                    result = st.session_state.rag_system.index_directory(docs_dir)
                    st.session_state.indexing_status = result
                st.rerun()
    
    with col2:
        if st.button("Force Reindex", use_container_width=True, disabled=not docs_exists):
            if st.session_state.rag_system and docs_exists:
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
            st.sidebar.write(f"**Content types:** {len(stats.get('content_types', {}))}")
            
            # Content type breakdown
            content_types = stats.get('content_types', {})
            if content_types:
                st.sidebar.write("**Content Types:**")
                for content_type, count in content_types.items():
                    st.sidebar.write(f"• {content_type}: {count}")
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
        st.warning("Please configure your OpenAI API key to start chatting.")
        return
    
    if not st.session_state.rag_system:
        st.warning("RAG system not initialized. Please check the sidebar for errors.")
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
            display_chat_message(message, sources)
    
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