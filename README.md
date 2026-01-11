# 🎤 TED Talk RAG Assistant

An AI-powered assistant capable of answering questions, summarizing talks, and providing recommendations based on a database of over 4,000 TED Talks.

Built using **Retrieval-Augmented Generation (RAG)** architecture to ensure accurate, grounded responses without hallucinations.

## 🚀 Live Demo
**Base API URL:** `https://ted-rag-assistant-2n3c8s77p-itai-nulmans-projects.vercel.app`

* **Check Status:** [Click here to view API Stats](https://ted-rag-assistant-2n3c8s77p-itai-nulmans-projects.vercel.app/api/stats)
* **Interact:** Use cURL or Postman to send requests to `/api/prompt`.

## 🛠️ Tech Stack
* **Python 3.12** (Backend Logic)
* **Flask** (API Server)
* **LangChain** (RAG Framework & Chains)
* **Pinecone** (Vector Database for Semantic Search)
* **OpenAI API** (LLM for Generation & Embeddings)
* **Vercel** (Serverless Deployment)

## ✨ Key Features
* **Semantic Search:** Uses vector embeddings (`text-embedding-3-small`) to find relevant talks even if keywords don't match exactly.
* **Context-Aware Answers:** Uses a strict system prompt to answer *only* based on the provided TED dataset.
* **Metadata Filtering:** Enriched data allows the model to recommend talks based on views, speakers, and topics.
* **Hallucination Prevention:** The agent explicitly admits ignorance if the answer isn't found in the dataset (e.g., "I don't know based on the provided TED data").

## ⚙️ Architecture
1.  **Ingestion:** TED transcripts were processed, chunked, and enriched with metadata (Title, Speaker, Views).
2.  **Storage:** Vectors stored in a **Pinecone** index (Dimension: 1536).
3.  **Retrieval:** User query is converted to a vector, and top-k relevant chunks are fetched.
4.  **Generation:** The retrieved chunks + user query are sent to the LLM (`gpt-4o-mini` equivalent) to generate a grounded response.

## 💻 Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ItaiNulman/Ted-Rag-Assistant.git
    cd Ted-Rag-Assistant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your keys:
    ```env
    OPENAI_API_KEY=your_openai_key
    PINECONE_API_KEY=your_pinecone_key
    ```

4.  **Run the App:**
    ```bash
    python app.py
    ```

## 📡 API Usage Examples

### 1. Check Configuration (GET)
```bash
curl https://ted-rag-assistant-2n3c8s77p-itai-nulmans-projects.vercel.app/api/stats
