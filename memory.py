import os
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "agent-memory"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector store for long-term memory
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

def vector_search(query):
    return vectorstore.similarity_search(query, k=3)

def vector_add(text):
    vectorstore.add_texts([text])
    return "saved"

# Short-term conversation memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=10,
    return_messages=True
)
