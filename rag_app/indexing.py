from dotenv import load_dotenv
import streamlit as st
import asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()
st.set_page_config(page_title="🌐 RAG from Website Docs", layout="centered")

st.title("🌐 ChaiCode Docs Chatbot")
st.write("Paste documentation URLs and ask questions directly!")

# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# URL Input
url_input = st.text_area("🔗 Paste one or more documentation URLs (comma-separated):")
if st.button("📥 Ingest URLs"):
    urls = [url.strip() for url in url_input.split(",") if url.strip()]
    st.info("Loading and parsing docs...")
    
    async def process_urls(urls):
        loader = WebBaseLoader(urls)
        loader.requests_per_second = 2  # Control crawling rate
        docs = await loader.aload()
        return docs
    
    docs = asyncio.run(process_urls(urls))
    split_docs = text_splitter.split_documents(docs)

    # Save to Qdrant
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name="rag_docs"
    )

    st.success("✅ Docs ingested and indexed!")

# Ask a question
st.subheader("💬 Ask a question about the docs")

query = st.text_input("❓ What do you want to know?")
if query:
    # Load existing vector DB
    vector_db = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="rag_docs",
        embedding=embedding_model
    )

    # Perform search
    results = vector_db.similarity_search(query, k=4)

    context = "\n\n".join([
        f"{r.page_content}\n📄 Source: {r.metadata.get('source', 'Unknown')}" for r in results
    ])

    SYSTEM_PROMPT = f"""
    You are a helpful assistant that answers questions based only on the documentation below.
    Provide concise, clear answers and include the reference URL at the end.

    Documentation:
    {context}
    """

    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )

    st.markdown("### 🤖 Answer")
    st.write(response.choices[0].message.content)
