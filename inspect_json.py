import json

def inspect_chunks():
    print("--- Inspecting JSON Data ---")
    
    file_path = 'chunks_ready_for_upload.json'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    total = len(chunks)
    print(f"Total chunks found: {total}")

    # 1. Sanity Check: Print the first chunk to see structure
    print("\n--- Sample Chunk (First Item) ---")
    first_chunk = chunks[0]
    print(json.dumps(first_chunk, indent=2, ensure_ascii=False))

    # 2. Quality Check: Search for potential issues
    print("\n--- Quality Report ---")
    
    # Check for empty text
    empty_chunks = [c for c in chunks if not c.get('text') or len(c['text'].strip()) == 0]
    if empty_chunks:
        print(f"⚠️ WARNING: Found {len(empty_chunks)} empty chunks!")
    else:
        print("✅ No empty chunks found.")

    # Check for very short chunks (might be garbage/headers)
    short_chunks = [c for c in chunks if len(c.get('text', '')) < 50]
    if short_chunks:
        print(f"⚠️ WARNING: Found {len(short_chunks)} chunks shorter than 50 characters.")
        print(f"Sample short chunk: {short_chunks[0]['text']}")
    else:
        print("✅ No dangerously short chunks found.")

    # Check for missing metadata
    missing_meta = [c for c in chunks if not c.get('title') or not c.get('speaker')]
    if missing_meta:
        print(f"⚠️ WARNING: Found {len(missing_meta)} chunks with missing metadata.")
    else:
        print("✅ All chunks have Title and Speaker metadata.")

    print("\n--- Conclusion ---")
    if not empty_chunks and not missing_meta:
        print("Data looks GOOD for upload! 🚀")
    else:
        print("Please review warnings before uploading.")

if __name__ == "__main__":
    inspect_chunks()