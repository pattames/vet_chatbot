import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import logging

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB with persistence
DB_PATH = "./vector_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Use Sentence Transformers embedding function
embedding_function = SentenceTransformerEmbeddingFunction(
   model_name="intfloat/multilingual-e5-base"
)

# Create or get collection in ChromaDB
# The embedding function will be used when adding documents
collection = chroma_client.get_or_create_collection(
   name="company_policies",
   embedding_function=embedding_function
)

# KNOWLEDGE_BASE
KNOWLEDGE_BASE = {
   
}

if __name__ == "__main__":
   print("tests:")