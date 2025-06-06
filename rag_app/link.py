import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import time

load_dotenv()
nest_asyncio.apply()

# --------- Custom CSS Styling ---------
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .sidebar-logo {
        border-radius: 50%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .sidebar-title {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .sidebar-description {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .feature-title {
        color: #1e293b;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-text {
        color: #64748b;
        font-size: 0.85rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-right: 2rem;
    }
    
    .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .message-content {
        line-height: 1.6;
    }
    
    /* Input styling */
    .stChatInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-top: 3px solid #667eea;
        flex: 1;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #e2e8f0;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .processing-text {
        margin-left: 1rem;
        color: #667eea;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --------- Streamlit Config ---------
st.set_page_config(
    page_title="ChaiCode AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

# --------- Header ---------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧠 ChaiCode AI Assistant</h1>
    <p class="header-subtitle">Your intelligent companion for ChaiCode documentation</p>
</div>
""", unsafe_allow_html=True)

# --------- Sidebar ---------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <img src="https://avatars.githubusercontent.com/u/134247811?s=200&v=4" 
             class="sidebar-logo" width="80">
        <h2 class="sidebar-title">💬 ChaiCode Bot</h2>
        <p class="sidebar-description">
            Interact with comprehensive ChaiCode documentation using natural language queries
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🔍 Smart Search</div>
        <div class="feature-text">Advanced semantic search across all documentation</div>
    </div>
    
    <div class="feature-card">
        <div class="feature-title">⚡ Real-time Answers</div>
        <div class="feature-text">Get instant responses with source citations</div>
    </div>
    
    <div class="feature-card">
        <div class="feature-title">📚 Comprehensive Coverage</div>
        <div class="feature-text">HTML, Git, C, Django, SQL, DevOps and more</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick links
    st.markdown("### 🔗 Quick Links")
    st.link_button("📄 View Documentation", "https://docs.chaicode.com", use_container_width=True)
    st.link_button("👨‍💻 GitHub Repository", "https://github.com/chaicode", use_container_width=True)
    
    st.markdown("---")
    
    # Stats
    st.markdown("""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">45+</div>
            <div class="stat-label">Topics</div>
        </div>
    </div>
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">6</div>
            <div class="stat-label">Courses</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------- Documentation URLs ---------
urls = [
    "https://docs.chaicode.com/youtube/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
    'https://chaidocs.vercel.app/youtube/chai-aur-html/introduction/',
    'https://chaidocs.vercel.app/youtube/chai-aur-html/emmit-crash-course/',
    'https://chaidocs.vercel.app/youtube/chai-aur-html/html-tags/',
    'https://chaidocs.vercel.app/youtube/chai-aur-git/welcome/',
    'https://chaidocs.vercel.app/youtube/chai-aur-git/introduction/',
    'https://chaidocs.vercel.app/youtube/chai-aur-git/terminology/',
    'https://chaidocs.vercel.app/youtube/chai-aur-git/behind-the-scenes/',
    'https://chaidocs.vercel.app/youtube/chai-aur-git/branches/',
    'https://chaidocs.vercel.app/youtube/chai-aur-git/diff-stash-tags/',
    'https://chaidocs.vercel.app/youtube/chai-aur-git/managing-history/',
    'https://chaidocs.vercel.app/youtube/chai-aur-git/github/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/welcome/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/introduction/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/hello-world/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/variables-and-constants/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/data-types/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/operators/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/control-flow/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/loops/',
    'https://chaidocs.vercel.app/youtube/chai-aur-c/functions/',
    'https://chaidocs.vercel.app/youtube/chai-aur-django/welcome/',
    'https://chaidocs.vercel.app/youtube/chai-aur-django/getting-started/',
    'https://chaidocs.vercel.app/youtube/chai-aur-django/jinja-templates/',
    'https://chaidocs.vercel.app/youtube/chai-aur-django/tailwind/',
    'https://chaidocs.vercel.app/youtube/chai-aur-django/models/',
    'https://chaidocs.vercel.app/youtube/chai-aur-django/relationships-and-forms/',
    'https://chaidocs.vercel.app/youtube/chai-aur-sql/welcome/',
    'https://chaidocs.vercel.app/youtube/chai-aur-sql/introduction/',
    'https://chaidocs.vercel.app/youtube/chai-aur-sql/postgres/',
    'https://chaidocs.vercel.app/youtube/chai-aur-sql/normalization/',
    'https://chaidocs.vercel.app/youtube/chai-aur-sql/database-design-exercise/',
    'https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-and-keys/',
    'https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-exercise/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/welcome/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-vpc/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-nginx/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-rate-limiting/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-ssl-setup/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/node-nginx-vps/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/postgresql-docker/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/postgresql-vps/',
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/node-logger/',
]

# --------- Load, Chunk, Embed and Store ---------
@st.cache_resource(show_spinner=False)
def setup_vectorstore():
    with st.spinner("🔄 Initializing ChaiCode AI Assistant..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📥 Loading documentation...")
        progress_bar.progress(20)
        
        async def process_urls(urls):
            loader = WebBaseLoader(urls)
            loader.requests_per_second = 2
            docs = loader.aload()
            return docs

        docs = asyncio.get_event_loop().run_until_complete(process_urls(urls))
        
        status_text.text("✂️ Splitting documents...")
        progress_bar.progress(50)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        status_text.text("🧠 Creating embeddings...")
        progress_bar.progress(70)
        
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        
        status_text.text("🗄️ Setting up vector store...")
        progress_bar.progress(90)
        
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            url="http://localhost:6333",
            collection_name="chaicode_docs"
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Ready to help!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return vector_store, embedding_model

# Initialize the system
try:
    vector_db, embedding_model = setup_vectorstore()
    client = OpenAI()
    system_ready = True
except Exception as e:
    st.error(f"⚠️ System initialization failed: {str(e)}")
    st.info("Please ensure Qdrant is running and OpenAI API key is configured.")
    system_ready = False

# --------- Chat Interface ---------
if system_ready:
    # Chat State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Welcome message
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown("""
            👋 **Welcome to ChaiCode AI Assistant!**
            
            I'm here to help you explore ChaiCode's comprehensive documentation. You can ask me about:
            
            - **HTML & CSS** - Tags, styling, and web development basics
            - **Git & GitHub** - Version control, branching, and collaboration
            - **C Programming** - Syntax, data types, functions, and more
            - **Django** - Web framework, templates, models, and forms
            - **SQL & Databases** - Queries, joins, normalization, and design
            - **DevOps** - Server setup, Nginx, Docker, and deployment
            
            💡 *Try asking: "How do I create a Django model?" or "What are Git branches?"*
            """)
    
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            if role == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(message)
    
    # User input
    if user_query := st.chat_input("💬 Ask me anything about ChaiCode documentation..."):
        # Add user message to chat
        st.session_state.chat_history.append(("user", user_query))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"**You:** {user_query}")
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documentation..."):
                # Search for relevant documents
                search_results = vector_db.similarity_search(user_query, k=4)
                context = "\n\n".join([
                    f"{doc.page_content}\n📄 Source: {doc.metadata.get('source', 'N/A')}"
                    for doc in search_results
                ])

                # Prepare the prompt
                SYSTEM_PROMPT = f"""
You are ChaiCode AI Assistant, a helpful and knowledgeable assistant specializing in ChaiCode documentation.

Your role:
- Answer questions using ONLY the provided documentation context
- Be accurate, helpful, and conversational
- Always cite sources at the end of your response
- If information isn't available in the context, clearly state that
- Provide code examples when relevant
- Format your responses clearly with proper markdown

Documentation Context:
{context}

Guidelines:
- Use bullet points and formatting for clarity
- Include relevant code snippets from the documentation
- Mention specific source URLs when citing
- Be encouraging and supportive in your tone
"""

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ]

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.7
                    )

                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    
                    # Add to chat history
                    st.session_state.chat_history.append(("assistant", answer))
                    
                except Exception as e:
                    error_msg = f"⚠️ **Error generating response:** {str(e)}\n\nPlease check your OpenAI API configuration and try again."
                    st.error(error_msg)
                    st.session_state.chat_history.append(("assistant", error_msg))

    # Quick action buttons
    if not st.session_state.chat_history or len(st.session_state.chat_history) <= 1:
        st.markdown("### 🚀 Quick Start Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏷️ HTML Basics", use_container_width=True):
                st.session_state.next_query = "What are the basic HTML tags I should know?"
                st.rerun()
        
        with col2:
            if st.button("🌿 Git Fundamentals", use_container_width=True):
                st.session_state.next_query = "How do I get started with Git?"
                st.rerun()
        
        with col3:
            if st.button("🐍 Django Setup", use_container_width=True):
                st.session_state.next_query = "How do I set up a Django project?"
                st.rerun()
    
    # Handle quick start queries
    if hasattr(st.session_state, 'next_query'):
        user_query = st.session_state.next_query
        del st.session_state.next_query
        st.rerun()

else:
    st.warning("⚠️ System not ready. Please check the configuration and reload the page.")

# --------- Footer ---------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    Built with ❤️ using LangChain, Qdrant, OpenAI, and Streamlit<br>
    <small>© 2024 ChaiCode AI Assistant</small>
</div>
""", unsafe_allow_html=True)