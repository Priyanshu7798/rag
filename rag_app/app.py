from dotenv import load_dotenv
import streamlit as st
import mimetypes
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import tempfile
import os
from datetime import datetime
from qdrant_client import QdrantClient

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #006eea 0%, #786aa2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1a1a1a;
    }
    
    .assistant-message {
      background-color: #f3e5f5;
      border-left: 4px solid #9c27b0;
      color: #1a1a1a;
    }
    
    .file-upload-section {
      background-color: #f8f9fa;
      padding: 2rem;
      border-radius: 10px;
      border: 2px dashed #dee2e6;
      text-align: center;
      margin: 1rem 0;
    }
    
    [data-testid="stSidebar"] {
      background-color: #1e1e2f;
      color: #f1f1f1;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-processing {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e3e8ed;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }
    
    .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    .sidebar-header {
      color: #2c3h50;
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .status-container {
      background: #ffffff;
      padding: 1rem;
      border-radius: 8px;
      border: 1px solid #e9ecef;
      margin-top: 0.5rem;
    }
    
    .instruction-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .instruction-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .clear-btn-container {
        margin-top: 1rem;
    }
    
    .upload-area {
        border: 2px dashed #cbd5e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    #col-2 styling
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] > label {
        font-size: 14px;
        color: #333;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        margin: 0.5rem 0;
        font-size: 16px !important;
        font-weight: 600;
        color: #1f4e79;
    }
    .streamlit-expanderContent {
        margin: 0.5rem 0;
        background-color: #f9fafc;
        padding: 10px;
        border-radius: 8px;
    }

    /* Chat previews */
    .chat-preview {
        font-size: 13px;
        padding: 6px 10px;
        margin: 4px 0;
        background-color: #eef2f7;
        border-radius: 6px;
        display: block;
    }

    /* Section titles */
    h3 {
        color: #2b5876;
        margin-top: 20px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 5px;
    }

    /* Tips section */
    .tips {
        background-color: #e8f5e9;
        border-left: 4px solid #66bb6a;
        padding: 10px 15px;
        border-radius: 6px;
        font-size: 14px;
        color: #2e7d32;
    }

    /* Restore button */
    button[kind="secondary"] {
        margin-top: 8px;
        background-color: #ffffff;
        color: #1f4e79;
        border: 1px solid #1f4e79;
        border-radius: 8px;
        padding: 5px 10px;
    }
    
    .connection-status {
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .connection-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .connection-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Store previous chat sessions
if 'qdrant_client' not in st.session_state:
    st.session_state.qdrant_client = None

# Qdrant configuration function
@st.cache_resource
def get_qdrant_client():
    """Initialize and return Qdrant client"""
    try:
        # Get credentials from environment variables or Streamlit secrets
        qdrant_url = os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            st.error("❌ Qdrant URL or API Key not found. Please set QDRANT_URL and QDRANT_API_KEY environment variables or add them to Streamlit secrets.")
            return None
        
        # Create Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Test connection
        collections = client.get_collections()
        st.success(f"✅ Connected to Qdrant Cloud! Found {len(collections.collections)} collections.")
        return client
        
    except Exception as e:
        st.error(f"❌ Failed to connect to Qdrant: {str(e)}")
        return None

# Helper functions
def save_current_chat():
  """Save current chat session to history"""
  if st.session_state.messages and st.session_state.current_file:
    chat_session = {
        'file_name': st.session_state.current_file,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'messages': st.session_state.messages.copy(),
        'message_count': len(st.session_state.messages)
    }
    st.session_state.chat_history.insert(0, chat_session)  # Add to beginning
    # Keep only last 10 chat sessions to avoid memory issues
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[:10]

def get_file_type(file_name):
  file_type, _ = mimetypes.guess_type(file_name)
  return file_type

def extract_text_from_pdf(temp_path):
  loader = PyPDFLoader(temp_path)
  docs = loader.load()
  return docs

def process_document(uploaded_file):
    """Process uploaded document and create vector store"""
    try:
        # Get Qdrant client
        if not st.session_state.qdrant_client:
            st.session_state.qdrant_client = get_qdrant_client()
        
        if not st.session_state.qdrant_client:
            return None, None, None
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
        # Extract text
        extracted_text = extract_text_from_pdf(temp_path)
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        split_docs = text_splitter.split_documents(documents=extracted_text)
        
        # Create embeddings
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Get Qdrant credentials
        qdrant_url = os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY")
        
        # Create unique collection name based on filename and timestamp
        collection_name = f"doc_{uploaded_file.name.replace('.pdf', '').replace(' ', '_').lower()}_{int(datetime.now().timestamp())}"
        
        # Create vector store with hosted Qdrant
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            force_recreate=True  # This ensures a fresh collection for each document
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return vector_store, extracted_text, split_docs
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None, None, None

def get_ai_response(query, vector_store):
    """Get AI response based on query and vector store"""
    try:
        # Perform vector search
        search_results = vector_store.similarity_search(query=query, k=4)
        
        # Build context
        context = "\n\n".join([
            f"Content: {result.page_content}\nPage: {result.metadata.get('page', 'N/A')}\nSource: {result.metadata.get('source', 'Unknown')}"
            for result in search_results
        ])
        
        # System prompt
        system_prompt = f"""
        You are a helpful AI Assistant who answers user queries based on the available context
        retrieved from uploaded documents. Provide accurate, helpful responses and reference
        specific page numbers when possible. Always return the page number of the document.
        You are like a teacher to them who teaches them and give reference according to the pdf

        Context:
        {context}
        """
        
        # Generate response
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Document Assistant</h1>
    <p>Upload documents, ask questions, and get intelligent answers powered by AI</p>
    <p>Now powered by Qdrant Cloud for reliable vector storage!</p>
</div>
""", unsafe_allow_html=True)

# Initialize Qdrant connection check
if not st.session_state.qdrant_client:
    with st.spinner("🔌 Connecting to Qdrant Cloud..."):
        st.session_state.qdrant_client = get_qdrant_client()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h2 style="color: #2c3h50;font-size: 2rem; font-weight: 600; margin: 0;">📋 Document Hub</h2>
        <p style="color: #6c757d; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Manage your documents and chat settings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Connection status
    st.markdown('---')
    st.markdown('<div class="sidebar-header">🔌 Connection Status</div>', unsafe_allow_html=True)
    if st.session_state.qdrant_client:
        st.markdown('<div class="connection-status connection-success">✅ Qdrant Cloud Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="connection-status connection-error">❌ Qdrant Connection Failed</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.8rem; color: #6c757d;">Check your credentials and try refreshing</p>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('---')
    st.markdown('<div class="sidebar-header">📤 Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to start asking questions",
        label_visibility="collapsed",
        disabled=not st.session_state.qdrant_client  # Disable if no connection
    )
    if not uploaded_file:
        st.markdown('<p style="color: #6c757d; font-size: 0.85rem; margin: 0.5rem 0 0 0;">Drag and drop or click to upload</p>', unsafe_allow_html=True)
    
    if not st.session_state.qdrant_client:
        st.warning("⚠️ Upload disabled - Qdrant connection required")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document status
    st.markdown('---')
    st.markdown('<div class="sidebar-header">📊 Document Status</div>', unsafe_allow_html=True)
    if st.session_state.document_processed and st.session_state.current_file:
        st.markdown(f'<span class="status-badge status-success">✅ Ready</span>',unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.85rem; color: #495057; margin: 0.5rem 0 0 0;"><strong>File:</strong> {st.session_state.current_file}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #6c757d; margin: 0.25rem 0 0 0;">✓ Processed and indexed in Qdrant Cloud</p>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-processing">⏳ Waiting</span>',unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.85rem; color: #6c757d; margin: 0.5rem 0 0 0;">Upload a document to get started</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown('---')
    st.markdown('<div class="sidebar-header">💡 Quick Guide</div>', unsafe_allow_html=True)
    
    instructions = [
      "Upload a PDF document",
      "Wait for processing",
      "Ask questions about content",
      "Get answers with references"
    ]
    
    for i, instruction in enumerate(instructions, 1):
        st.markdown(f'''
          <div class="instruction-item">
              <div class="instruction-number">{i}</div>
              <div>{instruction}</div>
          </div>
          ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear chat button
    st.markdown('---')
    st.markdown('<div class="sidebar-header">🛠️ Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.document_processed:
        st.markdown('<p style="font-size: 0.8rem; color: #28a745; margin: 0.5rem 0 0 0; text-align: center;">💬 Ready to chat with your document!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Document processing
    if uploaded_file is not None and st.session_state.qdrant_client:
        if uploaded_file.name != st.session_state.current_file:
            # Save current chat before switching to new document
            save_current_chat()
            
            # Reset for new document
            st.session_state.current_file = uploaded_file.name
            st.session_state.document_processed = False
            st.session_state.messages = []  # Clear current chat
            
            with st.spinner("🔄 Processing document and uploading to Qdrant Cloud... This may take a moment."):
                progress_bar = st.progress(0)
                progress_bar.progress(25, "Extracting text...")
                
                vector_store, extracted_text, split_docs = process_document(uploaded_file)
                progress_bar.progress(75, "Creating embeddings and uploading to Qdrant...")
                
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.document_processed = True
                    progress_bar.progress(100, "Complete!")
                    st.success("✅ Document processed and uploaded to Qdrant Cloud successfully!")
                    
                    # Show document preview
                    with st.expander("📄 Document Preview", expanded=False):
                        preview_text = extracted_text[0].page_content[:1000] if extracted_text else "No content available"
                        st.text_area("Document Content (First 1000 characters)", 
                                   preview_text, height=200, disabled=True)
                        st.info(f"Document split into {len(split_docs)} chunks and stored in Qdrant Cloud")
                else:
                    st.error("❌ Failed to process document. Please check your Qdrant connection and try again.")
                
                progress_bar.empty()
    
    elif uploaded_file is not None and not st.session_state.qdrant_client:
        st.error("❌ Cannot process document - Qdrant connection required. Please check your credentials.")
    
    # Chat interface
    st.markdown("### 💬 Chat with your Document")
    
    if st.session_state.document_processed:
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_button = st.columns([4, 1])
            with col_input:
                user_query = st.text_input(
                    "Ask a question about your document:",
                    placeholder="e.g., What is the main topic of this document?",
                    label_visibility="collapsed"
                )
            with col_button:
                submit_button = st.form_submit_button("Send 🚀", use_container_width=True)
        
        if submit_button and user_query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                ai_response = get_ai_response(user_query, st.session_state.vector_store)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            st.rerun()
    
    else:
        if not st.session_state.qdrant_client:
            st.error("🚫 Qdrant connection required to use the chat feature. Please check your credentials.")
        else:
            st.info("👆 Please upload a PDF document to start chatting!")

with col2:
    # Statistics and info panel
    st.markdown("### 📈 Session Stats")
    
    stats_container = st.container()
    with stats_container:
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Messages", len(st.session_state.messages))
        with col_stat2:
            st.metric("Documents", 1 if st.session_state.document_processed else 0)
    
    # Recent Chats History
    if st.session_state.chat_history:
        st.markdown("### 🕒 Recent Chats")
        
        for i, chat_session in enumerate(st.session_state.chat_history):
            with st.expander(f"📄 {chat_session['file_name'][:20]}... ({chat_session['timestamp']})", expanded=False):
                st.write(f"**Messages:** {chat_session['message_count']}")
                st.write(f"**Time:** {chat_session['timestamp']}")
                
                # Show first few messages from the chat
                if chat_session['messages']:
                    st.write("**Sample conversation:**")
                    for msg in chat_session['messages'][:2]:  # Show first 2 messages
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        st.text(f"{role_icon} {content_preview}")
                
                # Button to restore this chat
                if st.button(f"🔄 Restore Chat #{i+1}", key=f"restore_{i}"):
                    # Save current chat first
                    save_current_chat()
                    # Restore selected chat
                    st.session_state.messages = chat_session['messages'].copy()
                    st.session_state.current_file = chat_session['file_name']
                    st.rerun()
    
    # Current session activity
    if st.session_state.messages:
        st.markdown("### 💬 Current Session")
        recent_messages = st.session_state.messages[-3:] if len(st.session_state.messages) > 3 else st.session_state.messages
        
        for i, msg in enumerate(recent_messages):
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            st.text(f"{role_icon} {content_preview}")
    
    st.markdown("---")
    # Tips
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ask specific questions for better answers
    - Reference page numbers will be provided
    - Try asking about summaries, key points, or specific topics
    - Use clear, concise language
    - Documents are now stored securely in Qdrant Cloud
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Built with ❤️ using Streamlit, LangChain, OpenAI, and Qdrant Cloud"
    "</div>", 
    unsafe_allow_html=True
)