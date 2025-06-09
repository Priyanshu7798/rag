# 📄 RAG PDF Reader App

A simple Streamlit app that allows you to upload a PDF file and interact with it using Retrieval-Augmented Generation (RAG). Ask questions about the content of your PDF and get accurate, context-aware answers powered by language models and vector similarity search.

---

## 🚀 Features

- 📁 Upload and parse PDF documents
- 🔍 Chunking and embedding of PDF text
- 🤖 Ask questions based on the PDF content using RAG
- 🧠 Vector database support (e.g., FAISS)
- ⚡ Powered by modern LLMs (e.g., OpenAI, Cohere, Hugging Face, etc.)
- 🎨 Clean and interactive UI built with Streamlit

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/rag-pdf-reader.git
cd rag-pdf-reader
```
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

# 🧠 How It Works
PDF Upload: Upload a PDF document via the Streamlit UI.

Text Extraction: The app extracts and splits the text into manageable chunks.

Embeddings: It generates vector embeddings of each chunk using a language model.

Vector Store: Embeddings are stored in a FAISS index or another supported vector DB.

User Query: When you ask a question, the app retrieves the most relevant chunks using similarity search.

LLM Response: The selected chunks and your question are passed to the language model for answering.

# 🖥️ Usage
Run the Streamlit app locally:

```bash
streamlit run app.py
```

# 🛠️ Configuration
Create a .env file in the project root with the following:
```bash
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=openai/text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo
```


