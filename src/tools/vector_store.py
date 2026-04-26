import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone

def get_vector_store():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = "linga-synthesize"
    
    if not api_key:
        print("Warning: PINECONE_API_KEY not found. Falling back to in-memory store (ChromaDB simulator if needed).")
        return None

    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536, # OpenAI embeddings dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
            )
        )
    
    embeddings = OpenAIEmbeddings()
    vectorstore = LangChainPinecone.from_existing_index(index_name, embeddings)
    return vectorstore
