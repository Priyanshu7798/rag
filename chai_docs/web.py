import asyncio
import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

# 1. URLs to load
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
    'https://chaidocs.vercel.app/youtube/chai-aur-devops/node-logger/'
]

# 2. Load and scrape website pages asynchronously
async def process_urls(urls):
    loader = WebBaseLoader(urls)
    loader.requests_per_second = 2
    docs =  loader.aload()
    return docs

docs = asyncio.get_event_loop().run_until_complete(process_urls(urls))
# print(docs[0])

# 3. Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = splitter.split_documents(docs)

# 4. Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# 5. Store in Qdrant DB
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="chaicode_docs"
)

print("✅ Docs ingested and indexed into Qdrant!\n")

# 6. Chat loop
client = OpenAI()
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="chaicode_docs",
    embedding=embedding_model
)

print("💬 Ask questions about ChaiCode Docs (type 'exit' to quit):")

while True:
    query = input("👤 > ").strip()
    if query.lower() == "exit":
        break

    search_results = vector_db.similarity_search(query, k=4)
    context = "\n\n".join([
        f"{doc.page_content}\n📄 Source: {doc.metadata.get('source', 'N/A')}"
        for doc in search_results
    ])

    SYSTEM_PROMPT = f"""
You are a helpful assistant answering user queries strictly using the following documentation context.

Respond clearly and concisely and include the source URL at the end.

Documentation:
{context}
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or gpt-3.5-turbo
        messages=messages
    )

    print(f"🤖: {response.choices[0].message.content}\n")
