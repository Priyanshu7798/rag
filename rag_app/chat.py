from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

client = OpenAI()


# Vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(
  url="http://localhost:6333",
  collection_name="rag_app",
  embedding=embedding_model
)


context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

    Context:
    {context}
"""


# Initialize messages list
messages = []

while True:
  query = input("👤 > ")
  
  # Perform vector search
  search_results = vector_db.similarity_search(
    query=query
  )

  # Build context from results
  context = "\n\n\n".join([
      f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nFile Location: {result.metadata.get('source', 'Unknown')}"
      for result in search_results
  ])

  # System prompt with context
  SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers user queries based on the available context
    retrieved from a PDF file along with page contents and page numbers.

    You should only answer the user based on the following context and guide them
    to open the right page number to know more.

    Context:
    {context}
  """

  # Update messages
  messages = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": query}
  ]

  # Generate response
  response = client.chat.completions.create(
    model="gpt-4.1-mini",  # or "gpt-3.5-turbo" if 4.1-mini isn't available
    messages=messages
  )

  print(f"🤖: {response.choices[0].message.content}")

