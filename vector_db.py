import os
import shutil
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import logging
from knowledge_base import KNOWLEDGE_BASE

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB with persistence
DB_PATH = "./vector_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Use Sentence Transformers embedding function
embedding_function = SentenceTransformerEmbeddingFunction(
   model_name="intfloat/multilingual-e5-base" # Multilingual for spanish
)

# Create or get collection in ChromaDB
# The embedding function will be used when adding documents
collection = chroma_client.get_or_create_collection(
   name="veterinary_knowledge",
   embedding_function=embedding_function,
   metadata={"hnsw:space": "cosine"} # Use Cosine Distance instead of default L2 distance (better for semantic similarity)
)

# Populate database with all veterinary knowledge
def insert_knowledge():
   """Store all Veterinary Knowledge in ChromaDB"""
   logger.info("\nIndexing Veterinary Knowledge into Unified Knowledge Base...")

   # Grab all existing IDs to avoid duplicates ahead
   existing_docs = collection.get()
   existing_ids = set(existing_docs.get("ids", []))

   for key, value in KNOWLEDGE_BASE.items():
      # Safe insertion
      try:
         # Skip document if it already exists (avoids duplicates)
         if key in existing_ids:
            logger.info(f"Knowledge already exists: {key}, skipping...")
            continue # Jump to the next iteration (code below doesn't execute for this iteration)

         collection.add(
            ids=[key],
            documents=[f"passage: {value}"], # Used for embedding and search (With prefix for E5 model usage)
            metadatas=[{"knowledge_key": key, "knowledge_content": value}] # Used for retrieval
         )
         logger.info(f"Stored knowledge: {key} → {value[:50]}...")

      except Exception as e: # Catch any exception that happens during insertion
         logger.error(f"Error inserting knowledge {key}: {str(e)}")

# Function to compare query to collection's values and return matching knowledge
def query_knowledge(query: str) -> str:
   """Query VectorDB for Veterinary Knowledge"""
   logger.info(f"Searching Unified Knowledge Base for query: \"{query}\" matches")

   try:
      # Vector similarity search
      # Compares the query embeddings to the collections values embeddings
      # Returns most similar values
      results = collection.query(
         query_texts=[f"query: {query}"], # Prefix for E5 model usage
         n_results=3, # Return top 3 values, even if not relevant (adjustable)
         include=["metadatas", "distances"] # Include id's (default), metadatas and distances in results
      )

      # Check for valid results
      if not results or not results.get("metadatas") or not results["metadatas"][0]:
         return "No relevant knowledge found."

      # Check unfiltered results
      # for metadata in results["metadatas"][0]:
      #    print(metadata.get("knowledge_key", ""))

      # Results that pass similarity threshold
      logger.info(f"Vector search results for query: {query}")
      filtered_knowledge = []
      # Process results to meet quality
      for i, metadata in enumerate(results["metadatas"][0]):
         # Get distance (lower is better for L2 distance)
         try:
            distance = results["distances"][0][i]
         except (IndexError, KeyError, TypeError):
            distance = float("inf") # If error, set distance to infinity (will fail threshold)

         # Get metadata's key and content (handle in case it's None or malformed)
         if metadata and isinstance(metadata, dict):
            knowledge_content = metadata.get("knowledge_content", "")
            knowledge_key = metadata.get("knowledge_key", "")
         else:
            knowledge_content = ""
            knowledge_key = ""
            logger.warning(f"Invalid metadata at index {i}: {metadata}")

         # Filter (very strict threshold because of very small knowledge base)
         if distance < 0.21: # Adjustable threshold (make it less strict as knowledgebase expands)
            filtered_knowledge.append(knowledge_content)
            logger.info(f"   ✓ Match {i+1} [{knowledge_key}]: {knowledge_content[:50]}... (Distance: {distance:.3f})")
         else:
            logger.info(f"   ✗ Weak Match {i+1} [{knowledge_key}]: {knowledge_content[:50]}... (Distance: {distance:.3f})")

   except Exception as e: # Catch any errors during search
      logger.error(f"Error querying knowledge: {str(e)}")
      return "An error occured while looking for knowledge"

# Utility to reset collection
def reset_collection(): # Use when modified knowledge base, changed embedding model or testing fresh installs
    """Utility function to reset the collection if needed."""
    try:
        shutil.rmtree(DB_PATH)
        logger.info("Collection deleted successfully.")
    except Exception as e:
        logger.info(f"Collection doesn't exist or couldn't be deleted: {e}")

if __name__ == "__main__":
   # Optional: Reset collection for fresh start (also delete folder inside vector_db):
   # reset_collection()

   # Check collection
   # print(collection.get())

   # Create collection and index Veterinary Knowledge
   # insert_knowledge()

   #Check
   query_knowledge("Taquicardia severa")