from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
# Import our new RAG brain
from rag_agent import TedRagAgent

# Load env vars
load_dotenv()

app = Flask(__name__)
app.json.sort_keys = False
CORS(app)

# Initialize the RAG Agent once when the server starts
# This loads the models and connects to Pinecone
try:
    print("Initializing RAG Agent...")
    agent = TedRagAgent()
    print("RAG Agent ready!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize agent: {e}")
    agent = None

# --- Configuration Endpoint ---
@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "chunk_size": 1000,
        "overlap_ratio": 0.2,
        "top_k": 5 
    })

# --- Main Query Endpoint ---
@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
        
    user_question = data.get('question', '')

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    if not agent:
        return jsonify({"error": "Agent is not initialized. Check server logs."}), 500

    # REAL RAG LOGIC:
    # 1. Retrieve relevant chunks
    # 2. Construct prompt
    # 3. Get answer from GPT
    try:
        response_data = agent.ask(user_question)
        return jsonify(response_data)
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "Internal processing error"}), 500

if __name__ == '__main__':
    # Running on 5001 to avoid Mac AirPlay conflict
    app.run(port=5001, debug=True)