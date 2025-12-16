import json
import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# 1. Load environment variables
load_dotenv()

# This ensures OpenAIEmbeddings finds the key and url without issues
# LangChain looks for 'OPENAI_API_KEY' by default, even if we use LLMod
os.environ["OPENAI_API_KEY"] = os.getenv("LLMOD_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.llmod.ai/v1"

# --- Configuration ---
BATCH_SIZE = 100 
INDEX_NAME = "ted-rag" 

def upload_data():
    print("--- Starting Upload to Pinecone ---")
    
    # 2. Check if API keys exist
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: LLMOD_API_KEY not found in .env file (or loaded incorrectly).")
        return
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not found in .env file.")
        return

    # 3. Load the pre-processed chunks
    try:
        with open('chunks_ready_for_upload.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from file.")
    except FileNotFoundError:
        print("Error: 'chunks_ready_for_upload.json' not found. Run prepare_chunks.py first.")
        return

    # 4. Initialize Models
    print("Initializing Embedding Model...")
    try:
        embeddings = OpenAIEmbeddings(model="RPRTHPB-text-embedding-3-small")
        # Test embedding to ensure connection works before starting loop
        embeddings.embed_query("test")
        print("Embedding model connection confirmed.")
    except Exception as e:
        print(f"Error connecting to Embedding API: {e}")
        return

    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(INDEX_NAME)
        # Verify connection
        stats = index.describe_index_stats()
        print(f"Pinecone connected. Current stats: {stats}")
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return

    # 5. Upload in Batches
    total_chunks = len(chunks)
    print(f"Starting upload of {total_chunks} chunks in batches of {BATCH_SIZE}...")

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        
        batch_texts = [item['text'] for item in batch]
        # Create unique IDs for Pinecone: talkID_index
        batch_ids = [f"{item['id']}_{j}" for j, item in enumerate(batch)]
        
        try:
            # Generate Embeddings (This costs money!)
            vectors = embeddings.embed_documents(batch_texts)
            
            # Prepare format for Pinecone
            to_upsert = []
            for j, vector in enumerate(vectors):
                item = batch[j]
                metadata = {
                    "text": item['text'],
                    "title": item['title'],
                    "speaker": item['speaker'],
                    "url": item['url'],
                    "talk_id": item['id'],
                    # Adding extra metadata for filtering/sorting if needed
                    "topics": item.get('topics', ''),
                    "views": item.get('views', 0)
                }
                to_upsert.append((batch_ids[j], vector, metadata))
            
            # Upsert
            index.upsert(vectors=to_upsert)
            
            # Progress bar style log
            progress = (i + BATCH_SIZE) / total_chunks * 100
            print(f"Uploaded batch {i // BATCH_SIZE + 1} ({min(progress, 100):.1f}%)")
            
        except Exception as e:
            print(f"Error in batch starting at index {i}: {e}")
            time.sleep(1) # Wait a bit before retrying or moving on

    print("\n--- Upload Complete! ---")

if __name__ == "__main__":
    upload_data()