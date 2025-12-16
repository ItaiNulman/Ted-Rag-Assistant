import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- CRITICAL CONFIGURATION FOR LLMOD ---
# We explicitly set these so LangChain knows exactly where to look
os.environ["OPENAI_API_KEY"] = os.getenv("LLMOD_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.llmod.ai/v1"

class TedRagAgent:
    def __init__(self):
        # 1. Initialize API Keys
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = "ted-rag"
        self.top_k = 5  # As defined in API/Stats

        # 2. Initialize Models (using Course Specific Model Names)
        self.llm = ChatOpenAI(
            model="RPRTHPB-gpt-5-mini"
        )

        self.embeddings = OpenAIEmbeddings(
            model="RPRTHPB-text-embedding-3-small"
        )

        # 3. Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(self.index_name)
            # Warm up connection check
            self.index.describe_index_stats()
            print("Pinecone agent connection successful.")
        except Exception as e:
            print(f"Warning: Pinecone could not be initialized: {e}")
            self.index = None

        # 4. Define the System Prompt (Requirement from assignment)
        self.system_prompt_text = """
        You are a TED Talk assistant that answers questions strictly and
        only based on the TED dataset context provided to you (metadata
        and transcript passages).
        
        You must not use any external knowledge, the open internet, or information that is not explicitly
        contained in the retrieved context.
        
        If the answer cannot be determined from the provided context, respond: "I don't know
        based on the provided TED data."
        
        Always explain your answer using the given context, quoting or paraphrasing the relevant
        transcript or metadata when helpful.
        """

    def get_relevant_context(self, query):
        """Retrieves the top-k most relevant chunks from Pinecone."""
        if not self.index:
            return []

        # Convert query to vector
        query_vector = self.embeddings.embed_query(query)

        # Query Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )

        # Format context for the LLM and for the API response
        context_data = []
        for match in results['matches']:
            meta = match['metadata']
            context_data.append({
                "talk_id": meta.get('talk_id', 'unknown'),
                "title": meta.get('title', 'Unknown Title'),
                "chunk": meta.get('text', ''),
                "score": match['score']
            })
        
        return context_data

    def ask(self, user_question):
        """Main RAG Flow: Retrieve -> Augment -> Generate"""
        
        # Step 1: Retrieve Context
        context_items = self.get_relevant_context(user_question)
        
        # Create a single string of context for the prompt
        if not context_items:
            context_str = "No relevant context found in the database."
        else:
            context_str = "\n\n".join([f"--- Context Item ---\n{c['chunk']}" for c in context_items])

        # Step 2: Augment Prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_text),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])

        # Step 3: Generate Answer
        chain = prompt_template | self.llm | StrOutputParser()
        
        try:
            response_text = chain.invoke({
                "context": context_str,
                "question": user_question
            })
        except Exception as e:
            response_text = f"Error communicating with LLM: {e}"

        # Return full structure required by the assignment API
        return {
            "response": response_text,
            "context": context_items,
            "Augmented_prompt": {
                "System": self.system_prompt_text,
                "User": f"Context:\n{context_str}\n\nQuestion: {user_question}"
            }
        }