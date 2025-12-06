import os
import json
import faiss
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer



client = genai.Client(api_key="AIzaSyDJdJ9BtezaRkPYn9OJASes1qmowMudzGk")
MODEL = "gemini-2.5-flash"



INDEX_PATH = "faiss_store/index.faiss" 
META_PATH = "faiss_store/meta.json"
CHUNK_PATH = "faiss_store/chunks.json"

print("Loading FAISS index + metadata...")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

with open(CHUNK_PATH, "r") as f:
    all_chunks = json.load(f)



embed_model = SentenceTransformer("all-MiniLM-L6-v2") 

def embed_query(text):
    """Embed a user query for FAISS search."""
    return embed_model.encode([text], convert_to_numpy=True).astype("float32")



def semantic_search(query, k=4):
    q_vec = embed_query(query)
    D, I = index.search(q_vec, k)

    chunks = [all_chunks[idx] for idx in I[0]]

    
    if D[0][0] > 1.0:   
        return None

    useful_chunks = [c for c in chunks if len(c.strip()) > 50]

    return useful_chunks if useful_chunks else None




def generate_answer(query, retrieved_chunks):

    
    if not retrieved_chunks:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"""
No relevant context was found.

Answer the question normally using your general knowledge:
Question: {query}
"""
        )
        return response.text.strip()

    
    context = "\n\n".join(retrieved_chunks)

    rag_prompt = f"""
You have been given context passages.

Rules:
1. If the answer CAN be found in the context, use only the context.
2. If the context does NOT contain the answer, IGNORE the context and answer normally.
3. Do NOT hallucinate contradictory facts.

Context:
{context}

Question: {query}

Answer:
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=rag_prompt
    )

    return response.text.strip()


if __name__ == "__main__":
    print("RAG System Ready — Powered by Gemini 2.5 Flash\n")

    while True: 
        query = input("\nAsk: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            break

        retrieved = semantic_search(query)

        answer = generate_answer(query, retrieved)

        print("\n--- Final Answer ---")
        print(answer)
