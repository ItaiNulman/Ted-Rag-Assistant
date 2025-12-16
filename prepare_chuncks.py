import pandas as pd
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os
import ast  # To parse the string representation of lists safely

# --- Configuration ---
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200 
EMBEDDING_COST_PER_1M_TOKENS = 0.02 

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def clean_topics(topics_str):
    """Parses columns like "['technology', 'culture']" into a clean string."""
    try:
        # Convert string representation of list to actual list
        topics_list = ast.literal_eval(topics_str)
        return ", ".join(topics_list)
    except:
        return str(topics_str)

def prepare_chunks():
    print("--- Starting Chunking Process (V3 - Full Metadata) ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('ted_talks_en.csv')
    except FileNotFoundError:
        print("Error: 'ted_talks_en.csv' not found.")
        return

    # Clean data
    df = df.dropna(subset=['transcript', 'title', 'speaker_1'])
    df = df[df['transcript'].str.strip().astype(bool)]
    
    print(f"Processing {len(df)} valid talks...")

    # 2. Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        model_name="gpt-4" 
    )

    total_chunks = []
    total_tokens = 0

    # 3. Iterate over each talk
    for index, row in df.iterrows():
        # Extract raw data
        transcript = row['transcript']
        title = row['title']
        speaker = row['speaker_1']
        url = row['url']
        
        # New Metadata extraction
        topics = clean_topics(row.get('topics', '[]'))
        description = row.get('description', '')
        views = row.get('views', 0)
        published = row.get('published_date', '')

        # Construct a rich context header
        # This header will be prepended to EVERY chunk
        context_header = (
            f"Title: {title}\n"
            f"Speaker: {speaker}\n"
            f"Topics: {topics}\n"
            f"Description: {description}\n"
            f"Stats: {views} views, Published: {published}\n"
        )

        # Split the transcript
        raw_chunks = text_splitter.split_text(transcript)
        
        for chunk_text in raw_chunks:
            if len(chunk_text) < 50:
                continue

            # Combine header + content
            final_text = f"{context_header}Content: {chunk_text}"
            
            chunk_data = {
                "id": str(row['talk_id']),
                "title": title,
                "speaker": speaker,
                "url": url,
                "topics": topics, # Saving separately for potential filtering later
                "views": int(views),
                "text": final_text, # This goes to vector DB
                "chunk_size": len(final_text)
            }
            total_chunks.append(chunk_data)
            
            # Count tokens
            total_tokens += count_tokens(final_text)

        if index % 500 == 0:
            print(f"Processed {index} talks...")

    # 4. Summary
    estimated_cost = (total_tokens / 1_000_000) * EMBEDDING_COST_PER_1M_TOKENS
    
    print("\n--- Summary ---")
    print(f"Total Chunks Created: {len(total_chunks)}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Estimated Cost: ${estimated_cost:.4f}")
    
    # 5. Save
    output_file = 'chunks_ready_for_upload.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(total_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to '{output_file}'. Now run inspect_json.py to verify the rich context.")

if __name__ == "__main__":
    prepare_chunks()